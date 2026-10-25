# MiniCPM-o 4.5 原生能力与 AVM 连续语音契约

## 1. 边界、身份与验收状态

本契约定义 AVM 的目标架构：MiniCPM-o 4.5 原生规模、200 ms 认知周期、16 kHz
混合麦克风输入、25 Hz 连续声学 token 输出。不存在文本生成、时间 token、
SILENCE/SPEECH 分类头、EOS 驱动的发声结束或静默 PCM 旁路。静音也是完整监督的声音。

源模型、tokenizer、Token2Wav、固定音色、推理代码及 qualification corpus 必须以
不可变 revision、许可证和逐文件 SHA-256 锁定。缺失资产或失败资格报告必须显式失败；
旧 Mimi/CosyVoice 资产不具有替代资格。完整真实权重导入后的Core计数为
8,169,568,767（8.17B），含训练Value分支。2026-10-26四个完整源分片已通过来源哈希
与tensor审计，并完成原生规模CPU连续推理；这不替代冻结声学资格或正式训练验收。

真实声学结果见[2026-10-25资格实验](experiments/native-acoustic-qualification.md)：
固定初始化后的最终复验中，讲话峰值约1.069、全静音RMS约0.00384，声学与时延均未通过。
早期单次声学通过不能作为可复现资格；固定前缀本身不足以保证静音保真，正式运行保持关闭。
仓库已统一到`avm-model-v3`、200 ms原生张量路径；旧Mimi实现不作为回退。

用户已授权下载完整基模并继续完成全部本地实现与验证；8×A800上的真实验证单独保留。
本地验证使用原生完整权重的分模块执行和相同代码的小维度协议测试，不据此推断8卡性能。
目标实现整体移除旧mode字段与Mimi正式入口，建立`avm-model-v3`、`avm-episode-v2`与
`avm-realtime-v2`身份；未通过资格的配置保持readiness失败，不通过保留旧模型分支运行。

## 2. 系统和端到端张量

Model Core 只处理 tensor、状态和采样。Model Service 接收物理音视频并返回 PCM，
Harness 拥有设备与播放，Training 通过共享 train/recipe/evaluate 管理全部阶段。
时间戳仅用于传输、调度、审计和 time-discount GAE。

原生声学worker可在Model Service内部子进程运行，也可放置到专用GPU，通过内部Unix
socket提供同一encode/decode/reset/synchronize协议。该socket不属于Harness物理输入，
也不是第二套模型路径。握手核对codec、revision、tokenizer hash及source-lock hash，
单一stream lease拒绝并发生命期；关闭释放lease，恢复按完整已采样时间线重建缓存。
测试可运行显式synthetic身份的协议服务；正式配置仍先检查真实资产与资格，不允许
用测试实现回退。预热时ActuationSignal可无speech字段，有PCM时携带其来源unit、
计划播放时间和取消epoch；结束通过控制面flush交付剩余PCM，不伪造零音频。
PCM累计够4800 samples时立即交给Harness并携带未来的计划播放时间，Harness负责等待；
不等到deadline才开始RPC和入播放队列。生产timestamp使用同一主机单调时钟毫秒，
不能把会话相对时间直接与播放器的绝对单调时钟比较。
deadline检查包含观察传输、计算及声学RPC；PCM虽已生成但到达过晚也显式失败。
正式transport clock为monotonic。显式synthetic协议fixture使用logical_fixture时钟，
只验证时间线和排程，不把CPU测试耗时当作200 ms实时验收；原生codec或正式数据禁止
选择该fixture时钟。clock kind不进入Model Core的音视频输入。
会话结束flush在有界播放队列满时等待空位再交付尾部，普通unit提交仍不得阻塞采集；
不能通过扩大无界队列或丢弃最后几个200 ms片段来完成关闭。

| 模块 | 原生/新增规模 | 每个完整 200 ms unit 的有效接口 |
|---|---|---|
| 混合麦克风 | 16 kHz mono | `[B,3200]` |
| 流式 mel | 80 bins、10 ms hop | 20 个新有效帧，边界上下文另计 |
| 音频 encoder | 24 层、1024、16 heads、FFN4096 | 卷积后 `[B,10,1024]` |
| 音频投影/pooling | 1024→4096→4096、pool5 | `[B,2,4096]` |
| 视觉 encoder | 27 层、1152、16 heads、FFN4304、patch14 | 448×448 切片有 1024 个 patch |
| 原生视觉 resampler | 64 queries、输出4096 | S 个切片为 `[B,64S,4096]` |
| 联合 Perceiver | 2 层、内部2048、16 heads、FFN8192 | `[B,2+64S,4096]` → `[B,64,4096]` |
| Qwen3 backbone | 36 层、4096、FFN12288、32 Q/8 KV heads | `[P_t+H_(t-1),C_(t-1)]` 为 `[B,80,4096]` |
| 工作/语义状态 | H64、C16 | `[B,64,4096]` 和 `[B,16,4096]` |
| 语音条件 | 保留原生 semantic projector | 64 个4096维 H → 64 个768维连续条件 |
| 语音 decoder | 20 层、768、12 heads、FFN3072、一个声学码本 | 严格输出 `[B,5]` 有效声学 token |
| Token2Wav | 锁定原生资产 | 新 token + 固定前瞻 → 连续24 kHz PCM |
| JEPA | 2 层、内部1024、16 heads、FFN4096 | 预测200 ms后 `[B,64,4096]` |
| Value | 小型汇聚/MLP，仅训练 | `[B]` |

原生模块不缩层、不降宽。64/16 slots 与辅助模块宽度是容量候选；比较32/16与64/16
时固定训练数据、预算并报告GPU小时，不创建第二条正式规模或训练循环。

完整源checkpoint的tensor元数据计数为9,371,787,666。AVM导入其中7,983,569,810个
非文本运行参数；不挂载原生文本embedding/LM输出头、TTS文本embedding及未使用的
speaker projector（共1,388,217,856）。新增AVM模块186M左右，最终为8,169,568,767。
这是去掉文本通道后的参数组成差异，原生编码器、Qwen3层和语音decoder的层数、宽度、
head和FFN均保持原配置。完整分片来源锁SHA-256为
`e7b2896c53ca5a1350b31e15db33fc11a55891f2bf2ba60e7e8740c12b707a7a`；
来源哈希、严格导入与两个连续unit的Core张量执行均已通过，正式声学与训练门禁另计。

| AVM模块 | 参数量 |
|---|---:|
| 音频encoder与投影 | 328,196,096 |
| 视觉encoder与resampler | 506,700,016 |
| Qwen3层与final norm | 6,946,075,648 |
| 原生语音head | 202,598,050 |
| 两层Perceiver | 151,214,080 |
| 两层JEPA | 33,653,760 |
| Value（仅训练） | 1,057,281 |
| 模态embedding、初始C与SlowMemory参数 | 73,836 |

配套冻结声学栈另有307,245,895个参数：S3 tokenizer 123,714,568、flow
155,802,352、HiFT 20,821,295，以及CAM++ ONNX initializer 6,907,680。
按这一计数口径，含训练Value的完整AVM与冻结声学资产合计8,476,814,662（约8.48B）；
8.17B只指Model Core（含固定位置表和训练Value）。冻结声学资产已实际加载计数，
Core完整加载及连续两个unit的CPU BF16校准推理已经通过。

## 3. 流式输入、位置与状态

16 kHz直接进入前端，不在模型内执行24→16 kHz重采样。设备格式转换属于Harness。
首块预热、STFT右上下文和卷积重叠分别计入算法延迟；不得通过零补齐、重复token或
读取未到达音频，伪造每unit两个有效token。视觉保持原生预处理，低帧率使用因果帧。

RecentKV按30秒即150个unit有界，每层K/V各为 `[B,8,9600,128]`，只缓存H。
H/C在当前unit内部互见，但不看未来unit。GQA的query/KV映射、RoPE缓存重定位与
未旋转内容key上的SlowMemory读写必须显式验证。长期会话不能通过原生音频缓存
溢出后静默reset来处理。状态跨unit连续，只在TBPTT边界detach或会话边界reset。

为避免200 ms边界读取尚未到达的音频，流式STFT采用400点窗、160点hop、240点历史，
不执行center右填充；只有会话起点定义左边界状态。两层原生卷积各保留两个历史位置，
按因果相位推进，20个新mel严格产生10个新encoder位置，再投影并pool5为两个token。
这保留原生参数与卷积核，改变的是窗口相位：相对居中STFT/卷积存在约22.5 ms特征
滞后，须纳入适配训练和时延报告。不得在每个chunk两端重新补零。mel归一化使用会话
running maximum，不能读取未来峰值。音频每层KV有界；原生1500位置表逐值保留，
超过表长按Whisper正弦形式解析延伸，不重复最后位置、不重置缓存。原生BF16表与理想
解析正弦有实测差异（最大约0.00610），延伸属于需适配训练与长会话验证的显式变更。

视觉保留原生SigLIP2的位置bucket映射、归一化及resampler；同一batch使用相同尺寸的
切片，多个切片沿显式S轴输入，mask不混合不同样本。不强制把原生4900位置表改小。
无缓存视觉注意力的FP32计算保留原生eager的matmul→scale→FP32 softmax顺序，避免
SDPA重排导致27层累积误差超过既定FP32门槛；BF16计算仍使用SDPA。两者共享相同参数、
输入和数学目标，按计算dtype选择数值后端。测试核对FP32输出/梯度与原生顺序，且确认
BF16及流式音频缓存仍使用SDPA；真实448像素核验继续使用原来的3e-5绝对门槛。
实时视频使用原生`max_slice_nums=1`的overview预处理：保持宽高比、约448²像素面积、
按patch14取整和PIL bicubic，随后归一化；448×448只是方形帧示例，不把所有画面拉伸
成方形。Model Core同时接受显式S轴的同尺寸多切片。持久化逐帧NPZ数组支持帧尺寸
变化，不依赖object/pickle数组；源视频采集保存实际画幅比例。
Qwen3及语音Llama缓存保存RoPE前的content key，裁剪后重新赋予相对窗口位置，使有界
缓存不积累旋转误差。语音窗口按完整条件64+codes5的组裁剪，末尾第五个code也必须
消费到cache；teacher forcing与采样使用相同的因果顺序。
Qwen3 attention直接使用SDPA的GQA接口，K/V保持8个head；不得先把历史K/V物化复制
为32个query head，否则150步反向的历史激活会放大约四倍。A800需实测Flash/SDPA选择与
峰值内存，不能仅按静态KV大小估算训练显存。

语音局部历史以2秒为首轮评估点，按完整条件/声学组裁剪并保持一致的位置和mask；
不得将每unit64个条件无限追加。局部缓存裁剪不等于认知状态重置。

## 4. 固定五token与梯度

每个unit无条件生成五个真实声学token，包含讲话、停顿和静音。只能在锁定的有效
声学词表内采样，EOS/控制符排除；训练与PPO概率使用同一归一化分布。
固定步数不等于设置原生 `max_new_token=5`：必须验证返回切片、第五个token消费与
KV推进不存在off-by-one。生成API的 `inference_mode` 不可用于训练前向。

监督序列按“条件_t、五个codes、条件_(t+1)、五个codes”组织，保持时间顺序。
真实目标音频连续编码再分组，不能把每个200 ms独立编码。缺失监督仍有有效mask，
但静音不能被mask。loss只有声学token CE与既有JEPA；mode标签、mode CE和mode概率
从最终模型、数据、训练与trace移除，不保留兼容开关。

目标编码固定采用`fixed-speaker-prefix-v1`：将锁定的2秒、16 kHz音色参考放在整段目标
录音前，用未修改的原生tokenizer一起编码，核对总token数后仅移除参考对应的50个
位置。保留全部目标位置，包含静音；不替换code、不补token。参考、长度、trim数量、
tokenizer权重及编码协议均纳入身份。此固定教师上下文不进入ObservationSignal。
原生log-mel按整段峰值设下限，固定参考为全静音episode提供与解码音色一致的上下文。
早期对照曾观察到较低静音RMS，但未固定构造期的非持久化噪声缓冲，不能作为可复现
保真结论。最终完整来源锁、固定seed17的复验中全静音RMS为0.00383697，仍超过0.001；
保留固定前缀的明确目标编码语义，但不以此宣称声学资格已通过。

Token2Wav的flow推理精度显式锁为FP32、10个采样步，tokenizer与HiFT仍为原生FP32。
set_stream_cache的flow初始化与推理使用相同autocast，参考读取/tokenizer不在该上下文
内。精度、种子与worker源码hash进入报告。FP16作为显式声学实验单独保留：本机五token
切换片段出现超出归一化范围的PCM，未取得资格，不能混用该实验的性能数据或激活它。

未来speech loss沿decoder→条件适配→H→Backbone→Perceiver/Encoder及更早C/SlowMemory
回传。200 ms/声学分块不detach，30秒TBPTT为150步；重计算分段不截断计算图。
Token2Wav作为固定声学资产，策略梯度不穿过离散采样或PCM；若需声学适配训练，必须
先定义声学目标并纳入共享训练路径，不以临时head或另一个训练loop替代。

## 5. 前瞻、留尾和固定延迟

前瞻L从真实 `flow.pre_lookahead_len` 读取并锁定。前瞻来自后续unit，使用但不消费：
例如L=3时，组1的a1…a5用组2的a6…a8作前瞻；下一组正文仍从a6开始。
等待周期至少为 `ceil(L/5)*200 ms`，还要计入计算和vocoder留尾。

上游Step-Audio2代码的vocoder保留8个50 Hz mel帧，即160 ms波形尾部；首次stream
调用可能只返回40 ms，后续才稳定返回200 ms，结束flush返回留尾。具体实现以锁定
运行时代码为准。不能要求每次底层stream调用都立即返回4800 samples，更不能补零
或裁剪来伪装。Model Service须累计真实有效PCM，在完成预热后按4800 samples交付。

固定播放延迟D由L、声学留尾、实测计算与调度预算共同决定。调度器按完整unit分别
预留前瞻、留尾与计算：L=3、留尾160 ms、计算预留200 ms时，保守预算为600 ms。
不能沿用未计留尾的400 ms估算；降低预算须另有实测依据。正式报告必须给出所有分量及
首次有效语音、观察变化到可听变化、长时欠载和deadline违约率。

讲话与静音使用同一条连续token/声学时间轴；不因静默flush或reset。只有会话结束
执行last-chunk flush。外部取消清空未播放结果，增加generation epoch以拒收迟到结果；
认知记忆保留。没有独立mode事件，因此模型自主停止要经过完整声音延迟，不承诺
旧模式下的立即静默。不得添加VAD/能量阈值播放gate来补偿。

## 6. 资源、训练和身份

资源基线为8×A800 80GB，BF16。Pretrain/SFT使用8卡；Online RL候选布局为1卡认知
serving、1卡声学serving、1卡冻结reference、5卡candidate，始终只有一个active lifetime
session。首个实现采用单进程按层模型并行，保留共享训练循环与单一生命期语义；
不以DDP复制完整优化器。Pretrain/SFT按参数量把完整模块分配至8卡，candidate使用
GPU3—7，policy为GPU0、原生声学服务为GPU1、冻结reference为GPU2。边界tensor的
device copy保留autograd，各层KV和SlowMemory驻留对应层设备；公共输出回到输入设备。
AdamW状态跟随candidate参数，actor只保留CPU优化器快照；推荐主机RAM至少256GB。
actor的快照是完整AdamW序列化状态容器，不绑定CUDA参数，也不执行optimizer step；
候选在各层设备上恢复真实AdamW。接纳时原子替换policy权重及CPU快照，checkpoint仍由
共享CheckpointManager保存恢复，避免把约65GB的moments先装到actor单卡再迁出。
模型和优化器快照的复制均在候选线程进行。门禁通过后，后台先把候选迁移到actor卡并
同步，释放其梯度及设备优化器状态；actor卡短时容纳两套FP32权重（约65.4GB，加状态）。
主线程在完整unit边界交换就绪模型引用与CPU状态，不在该边界复制32GB权重。
H/C/KV/SlowMemory/audio/speech-local原样继承，不能通过reset或历史重放替代。
在线阶段的周期checkpoint在后台持久化不可变policy/CPU优化器和边界状态快照，保存期间
actor继续服务、对应窗口标记为checkpointing且不训练；下一次candidate等待保存完成。
RNG在快照边界捕获，不能在写文件线程稍后重新读取。最终保存先结束/脱离环境，再持久化。
周期快照标识其精确时间线前缀；恢复仍要求checkpoint、完整时间线与外部Harness cursor
一致，外部设备已推进或丢失观察时拒绝恢复，不承诺长时间停机后的实时无缝续接。
参数与AdamW状态保留FP32，服务和训练的CUDA计算使用BF16 autocast，避免actor上的
BF16参数把CPU优化器快照量化；actor/ref各自的FP32权重约32.7GB，仍需计KV和工作空间。
注意力Q/K/V及有界KV采用当前autocast计算dtype，初始空缓存不能将后续缓存提升到
FP32；FP32验证仍使用FP32缓存。计算精度不改变参数或AdamW精度，也不detach缓存。
测试覆盖两种Q/K归一化、音频KV、跨unit反向和序列化后继续推进，实际8卡峰值仍待测。
这是可实施的容量基线，尚无8卡吞吐或最优性证据。后续若引入FSDP必须先验证跨TBPTT
重计算、单生命期推理和checkpoint收集的collective顺序，不能只加入launch配置即视为完成。

本地测试覆盖CPU多分区与未分区数值/梯度一致、状态分配和checkpoint恢复，CUDA可用时
覆盖跨设备副本的反向传播；不把CPU分区测试当作8卡显存或通信性能的测量。
recipe在阶段完成后释放返回的模型，再加载共享evaluation模型或下一阶段，避免重叠驻留
两套完整模型；测试覆盖CPU快照的状态形状、dtype和恢复后的实际AdamW更新一致性。

正式顺序固定Pretrain→SFT→Online Recurrent PPO，共享train/recipe/evaluate。PPO的动作
概率为每unit五个声学token的条件log probability之和，静音也训练。生成、条件版本、
计划播放、实际播放、取消及声学epoch必须入trace；不能只拼接已播放codes重算策略。
reward归属沿真实ObservationSignal/播放时间线，不把固定延迟简单视为标签平移。

迁移须建立新模型、数据与物理协议身份，拒绝旧Mimi targets、mode字段、旧优化器和
会话恢复。source-lock与资格报告hash纳入manifest、checkpoint及tracking谱系。

## 7. 测试设计与执行门禁

1. 原生资产：固定revision/license、逐文件hash、索引shard完整性、原生形状、无路径
   越界、无旧codec替代；审计失败不加载模型、不启动服务、不返回passed。
2. 固定输出：五个有效codes、无EOS、最后token/KV消费、teacher forcing/采样概率一致；
   静音target保持监督且可反传，不存在mode参数或零PCM旁路。
3. 前瞻调度：L为0、3、5、6等边界，前瞻不重复消费；序号倒退/跳跃拒绝；取消后
   旧epoch结果丢弃；内存和待播长度有界；无padding、裁剪、重复token掩盖失败。
4. 真实声学资格：连续语音、至少10秒真实静音、语音/静音切换、会话结束flush；核对
   输入25 Hz codes与累计输出24 kHz样本时间关系，单列首块/留尾，检查低能量、接缝、
   延迟、吞吐和重复运行稳定性。测试fake只能验证协议，不产生正式资格报告。
5. 循环训练：H/C与音频/语音缓存连续、30秒边界、有界RoPE、GQA、SlowMemory及跨unit
   梯度；单机与分片损失/梯度等价；恢复后条件和播放时间线一致。
6. 端到端：三阶段谱系、逐阶段validation、最终test、真实视频因果性、单生命期PPO、
   含静默的token概率、外部取消与长时deadline。短跑只证明管线，不宣称质量收敛。

本地实施增加以下必测对照：原生真实权重严格按名称/形状覆盖导入，未使用的文本输入、
文本输出和文本语音分支显式列出；Qwen3/Llama/Whisper/SigLIP2各层与锁定官方实现
数值对照；整段与分块前端一致、未来扰动不影响已输出特征、超过30秒无隐式reset；
语音第五步cache推进、跨条件梯度、窗口裁剪、恢复后逐token概率一致；静音接受CE与
PPO完整五步梯度。物理协议将16 kHz输入与24 kHz输出时钟分开，并显式表达预热期间
无PCM产出、固定计划播放时间、结束flush和取消epoch，禁止零音频冒充预热输出。

完整分片下载后的CPU验证直接使用Canary原生配置和共享严格初始化器，以memory-mapped
BF16参数执行至少两个连续unit；不缩层、不缩宽、不替换编码器或语音head。输入明确为
16 kHz零PCM和有效范围内的视频校准信号，核对H64/C16、音频每步两个token、36层KV
每步增加64个位置、20层语音KV每步增加69个位置及五个合法codes。此探针仅证明完整
Core的资产导入与张量执行，不启动正式Model Service、优化器或物理会话，不证明对话
质量、端到端声学资格或实时性。报告须区分这些范围；本机无法承担的原生优化器更新
继续留给8×A800实测。新AVM模块仍为按种子初始化，不能把校准输入的输出当成已适配能力。

## 8. 来源

- [MiniCPM-o 4.5配置](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/config.json)
- [MiniCPM原生实现](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py)
- [配套声学资产](https://huggingface.co/openbmb/MiniCPM-o-4_5/tree/main/assets/token2wav)
- [Step-Audio2声学实现](https://github.com/stepfun-ai/Step-Audio2/blob/main/token2wav.py)

这些链接用于查阅，不替代执行时的不可变revision和文件hash。
