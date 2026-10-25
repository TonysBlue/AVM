# 原生连续声学资格实测：2026-10-25至26日

## 范围与结论

此实验只验证MiniCPM-o 4.5配套tokenizer/Token2Wav是否能按五token推进，包含真实语音、
10秒全静音与语音/静音切换。不加载Qwen3，不训练AVM，不验证认知实时性或8卡容量。
首轮结果和后续固定初始化复验分节保留，以末节完整来源锁复验为当前结论。

结果为 **未通过**。三段音频的token数量、累计样本数、前瞻复用与结束留尾全部对齐；
本机FP32解码慢于实时，独立全静音重建RMS超过预先设置的0.001门槛。不能据此激活
正式原生模型，也不能将旧Mimi实现当作新配置的回退。需要继续研究静音编码/声学保真，
并在目标A800上重新验证时延；本结果不说明A800同样不达标。

## 已锁定资产与环境

- 模型仓库：`openbmb/MiniCPM-o-4_5`。
- 不可变revision：`eb4141de24550a3a1a4082f718be58f46b761728`。
- 首轮仅逐文件校验官方声学权重的公布LFS SHA-256；完整基模在后续复验前下载并审计。
- 原生运行时：`stepaudio2-minicpmo==0.1.1`，wheel SHA-256为
  `7bf176cdadedabca30f4abb1fa49b7514981fbc8f763314fa811c2e4160a4523`。
- 独立环境Torch/Torchaudio 2.9.1、Transformers 4.51.3，FP32、seed17。
- 实测硬件：RTX 2080 SUPER，8 GiB；目标8×A800 80GB尚未接入。
- 固定音色从仓库system_ref_audio生成可追溯2秒参考；语音来自仓库assistant音频。
  静音为明确的数字零校准信号，均由真实原生tokenizer编码，非fixture decoder。
- source-lock SHA-256：`902239a9deb5b6c9efedaae044a81d3fd16001991b51746ca00abf91dd850ad6`。
- corpus SHA-256：`033fd77efce29201d61c9696f8ea1b743b302e10b35daac54e3c25a85fcbc392`。

原生set_stream_cache返回flow/vocoder两组缓存，调用方必须将其附加到模型，不能忽略
返回值。原生代码要求soundfile读取固定参考音频；Torchaudio 2.9忽略backend参数后会
改走未安装的TorchCodec。worker仅对这个锁定参考文件恢复soundfile读取语义，并在
成功或异常后恢复原函数，不修改原生网络、token或样本数。

## 结果

原生前瞻为3个token，vocoder留尾为3840 samples，即160 ms。保守调度预算为600 ms，
未证明该预算在本机能满足墙钟截止期。解码时间包含worker传输和同步，tokenizer离线
编码时间单列；没有把逻辑播放时间轴视为真实实时播放。

| 片段 | 时长 | codes | 输出samples/24 kHz | 解码p95 | 含flush的RTF | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 语音 | 4 s | 100 | 96,000 | 727.77 ms | 2.346 | 时间门槛失败 |
| 全静音 | 10 s | 250 | 240,000 | 807.14 ms | 2.529 | 时间与静音门槛失败 |
| 语音/静音切换 | 6 s | 150 | 144,000 | 797.07 ms | 2.058 | 时间门槛失败 |

预先固定的时间门槛为decode p95≤200 ms、RTF≤0.8；边界跳变上限0.15。三段最大
块边界差分别为0.1233、0.00215、0.0833，均未超过门槛；这不证明自然度或语义正确。

全静音重建RMS为0.0050485，去掉最初1秒后仍为0.0049203，峰值0.02364，因此不能
将失败仅归因于首块预热。切换片段注释静音区间RMS为0.000002524，表明问题与独立
静音编码或上下文相关；需要进一步隔离原因，不能直接认定是某个网络模块的问题。
不提高噪声门槛、不加入VAD、不强制补零掩盖该现象。

## 静音编码对照

在相同资产、固定音色、FP32、seed17与独立重置的解码缓存下，进一步比较两组各40个
原生token。第一组取自10秒纯零录音的4.0～5.6秒；第二组取自切换录音静音段的
2.2～3.8秒。两组都独立按五token流式解码，含正常结束flush，各输出38,400 samples，
即1.6秒。此对照用于定位原因，不替代完整资格测试。

| 来源 | 静音区域log-mel值 | 选中token分布 | 独立重建RMS |
|---|---:|---|---:|
| 整段纯零 | -1.5 | 4254×40 | 0.00407970 |
| 语音中的零样本区间 | -0.5848608 | 6486×21、6405×19 | 0.0000914162 |

原生s3tokenizer使用 `maximum(log_spec, log_spec.max() - 8.0)`，再执行
`(log_spec + 4.0) / 4.0`。因此即使局部PCM同为零，整段最大能量不同也会改变该区域的
特征下限。编码结果确实不同；将两组token放入相同解码初始状态后，重建RMS仍相差约
45倍，支持继续检查监督目标的编码上下文。此实验同时改变了归一化下限和encoder
上下文，尚不能断言归一化是唯一原因，也不能把三个观察到的code硬编码成静音协议。

下一步应先明确目标音频的连续编码、上下文和归一化契约，并分别验证独立长静音与
会话内静音；若修改该契约，必须更新身份、测试与资格报告。不得直接替换code、mask
静音loss或增加播放gate。该对照当时目标8卡服务器与主干资产尚未准备，不能推断容量和实时性。

对照报告位于 `/tmp/avm-native-qualification/silence-diagnostic/diagnostic.json`，
包含完整codes、环境身份、mel统计和两个重建WAV。诊断脚本SHA-256为
`11535ea7a95002b535fee3b5ae000d2c62bd59d49f6f19b2c28d5f3e642627f4`。

## 固定参考上下文复验与原生层数值核对（2026-10-25）

采用锁定2秒参考前缀连续编码并裁去50个参考token后，完整FP32复验得到
`acoustic_passed=true`、`timing_passed=false`。这是对目标编码上下文的显式修订；
没有替换静音code、静音mask、波形clamp或播放旁路。

| 片段 | tokens / 24 kHz samples | decode p95 | RTF | 静音 RMS |
|---|---|---|---|---|
| 讲话4秒 | 100 / 96000 | 727.26 ms | 2.29765 | 不适用 |
| 全静音10秒 | 250 / 240000 | 907.17 ms | 2.43359 | 0.000377242 |
| 讲话—静音—讲话6秒 | 150 / 144000 | 901.77 ms | 2.71607 | 0.000000298 |

三段时长精确、PCM有限且归一化，最大200 ms边界跳变分别为0.03245、0.0001412、
0.05314。静音RMS门槛仍为0.001；时延门槛仍为p95≤200 ms且RTF≤0.8，故整体未通过。
报告：`/tmp/avm-native-qualification/results-context-fp32/qualification.json`；
source-lock SHA-256为`4da726fb68dc4efcebf5df7b96f37faf8818baf4732dc5974bfcb6c718953df4`，
该次worker SHA-256为`683c8a34d4c88ff2462e0c429f7b07821ed553873e8abe4ee46a9c3781a29379`。
worker后续身份采集或代码变更都要求重新资格检查，不能复用旧hash的通过状态。

下载的真实最后一个基模分片已用于与锁定官方实现逐值核对。FP32最大绝对误差：
语音Llama层`9.53674e-7`，SigLIP层`3.81470e-6`，视觉resampler为0；
`tools/native-speech/verify_model.py`可复现。核对包含原生位置编码和真实参数，
不等于完整AVM预训练质量或8卡性能证据。

## 固定初始化后的最终复验

此前worker只在构造完成后设seed；上游flow在构造时已创建非持久化`rand_noise`缓冲。
该缓冲不在权重文件中，因此早期不同启动的声学结果不一致，前述单次通过不能视为
可复现资格。worker现于构造前设置Python/NumPy/Torch/CUDA seed17，并在stream reset时
恢复推理RNG；不覆盖上游噪声、不替换code、不clamp波形，也不放宽门槛。

最终报告`results-seeded-complete/qualification.json`为整体失败：speech峰值
1.0690228、transition峰值1.0690172，均触发归一化检查；全静音RMS为0.00383697，
decode p95为882.32 ms，RTF为2.37507。单段失败后仍检查其余片段，报告保留全部失败。
当前worker SHA-256为`029dbbca8bfd64b1b4f3e3def19b7741cd0304963fba14e931f8c693b612025a`。
固定参考前缀没有解决所有噪声初始化下的静音保真，后续声学适配必须有新的可复现证据。

真实完整原生Speech Head（202,598,050参数）另在GPU上通过前向/反向：输出
`[1,5,1,6562]`、20层KV各`[1,12,69,64]`，teacher forcing/采样最大误差约1.62e-5，
全部参数梯度有限，H条件梯度范数4.61039。Qwen3真实层误差约2.38e-6。
这些结果证明网络接口和梯度，不能抵消Token2Wav资格失败或证明对话质量。

进一步在CPU使用同一真实分片验证完整27层视觉encoder，56×56和42×70输入相对
官方实现的FP32最大误差分别为1.70e-5和1.41e-5，resampler误差为0。
小分辨率用于核对全部层、位置bucket及非方形画幅，不证明448像素下的吞吐。
报告为`/tmp/avm-native-model-parity-fullvision.json`；完整Speech Head在CPU上也通过
teacher forcing/采样一致性与全部参数梯度有限检查。

实际加载计数的冻结声学栈为307,245,895个参数，按Torch参数加CAM++ ONNX initializer
口径；与8,169,568,767个Model Core参数合计8,476,814,662。该计数不代表资格通过。

## 可复现输出

本次资产与输出位于仓库外的 `/tmp/avm-native-qualification/`：source-lock.json、
corpus.json、results/qualification.json和三段重建WAV。源码锁保存所有文件hash与
固定参考裁剪来源；语料清单保存重采样、裁剪位置与生成脚本hash。

```bash
uv run --no-sync data inspect-native-assets --acoustic-only \
  --source-lock /tmp/avm-native-qualification/source-lock.json
uv run --no-sync data qualify-native-speech \
  --source-lock /tmp/avm-native-qualification/source-lock.json \
  --corpus /tmp/avm-native-qualification/corpus.json \
  --output /tmp/avm-native-qualification/results
```

第二个命令因门槛失败返回2是本次预期结果，不能改成0或伪造通过报告。
源数据、权重、音频、缓存与报告不进入Git。

## 完整资产后的复验设计（2026-10-26）

完整四分片来源锁生成后，以同一语料、seed17、FP32/10步和原门槛重新资格验证。
另外仅比较官方system_ref_audio中0、6、10秒起点的两秒音色参考，分别建立来源锁，
固定前缀编码与解码同时使用相同候选。不得改变tokenizer、flow、HiFT、种子、静音标签
或PCM；候选结果全部保留，不能将实验来源锁或失败报告替代正式配置。
这用于判断失败是否与固定音色参考选择有关，不预先声称能解决模型的静音保真问题。
在参考10秒起点不变时，另以独立来源锁复验原生FP16 flow，同样固定构造前seed17；
不得将失败精度的速度数据与另一次精度的质量结果拼成通过报告。

## 完整来源锁、音色与精度结果（2026-10-26）

正式完整来源锁SHA-256为
`e7b2896c53ca5a1350b31e15db33fc11a55891f2bf2ba60e7e8740c12b707a7a`。
资产根目录为仓库外的`~/latentloop-data/models/minicpm-o-4_5/`
`eb4141de24550a3a1a4082f718be58f46b761728/`。以下均使用同一语料、构造前seed17及原门槛：

| 参考起点 | flow精度 | 首个越界峰值：讲话 / 切换 | 10秒静音RMS | 静音decode p95 | 静音RTF |
|---|---|---|---:|---:|---:|
| 10秒（正式） | FP32 | 1.06902 / 1.06902 | 0.00383697 | 876.25 ms | 2.43846 |
| 0秒（对照） | FP32 | 1.00234 / 1.06625 | 0.00556713 | 920.48 ms | 2.56288 |
| 6秒（对照） | FP32 | 未越界 / 1.04842 | 0.00412328 | 954.68 ms | 2.33016 |
| 10秒（对照） | FP16 | 1.06902 / 1.06902 | 0.00383686 | 1234.47 ms | 3.43587 |

所有候选的声学与时间资格均失败。越界片段在发现错误时终止，表中不是完整片段的峰值
上界；静音均完整输出250个token和240,000个samples。FP16没有修复静音，也未在本机
取得速度收益；不据此推断A800的精度性能排序。

结果分别保存于资产根目录的`qualification/`、`qualification-reference-start0/`、
`qualification-reference-start6/`、`qualification-flow-float16/`。独立来源锁hash依次为
正式上述hash、`18799c8756fba5a361624b65ee3a7c7939a9f775838be453e483908428b4709b`、
`eb1c5fe72b04e5ff7caf071e9c9cee4695b9951c97c92b83d9574a72864fd0f2`、
`947eb70ed35044585fccc8b691d60bb86995e6740e3e6b25d7f5972dca642b70`。
正式配置保留原来源锁；更换参考或开启原生FP16不是已验证的修复。后续若做声学适配，
须先定义可监督目标并纳入共享训练能力；不以局部补零、clamp、改code或修改门槛掩盖失败。
