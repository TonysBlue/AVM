# AVM 实时全双工音视频模型完整方案（基于 LatentLoop）

> 状态：项目顶层最终架构文档
> 日期：2026-10-24
> 目标：构建持续接收混合音频与通用视频、直接生成语音的实时全双工 Omni 对话模型。
> 专项协议：[直接流式语音实施说明](direct-speech.md) · [AVM 架构与验收契约](avm-contract.md)
> 训练协议：[统一三阶段训练架构](three-stage-training.md) · [Online RL：Online Recurrent PPO 与实时音视频对话环境](online-recurrent-ppo-training.md)
> 数据协议：[Harness 驱动的数据采集系统](data-collection.md) · [无模型 Harness 在环数据回放](data-replay.md)

## 1. 方案概述

### 1.0 当前架构：语义 slots 与长期关联记忆

当前正式模型使用 Backbone 内的共享语义 memory slots `C_t`，替代外部
`WorldStateUpdate/Z_t`。`H_t` 是当前工作状态，`C_t` 是跨 unit 保存的语义状态；
`C_t` 不直接连接 Speech/Value/JEPA heads，只通过下一 unit 的 Backbone
影响 `H_t`。每层 RecentKV 保留最近 375 units 的精确历史，过期 KV 经过 surprise
gate 和 Delta/KDA 更新写入固定容量 SlowMemory。

```text
O_t -> Encoders -> Perceiver -> P_t
P_t + H_(t-1) + C_(t-1) + RecentKV + SlowMemory
    -> Backbone -> H_t, C_t
H_t -> SpeechHead / JEPAHead
expired KV -> MemoryUpdater -> SlowMemory_t
```

$$
\widehat{P}_{t+1\mid t} = \mathrm{JEPAHead}(H_t)
$$

$$
P^{\mathrm{target}}_{t+1}
= \mathrm{stopgrad}\!\left(\mathrm{Perceiver}(O_{t+1})\right)
$$

$$
\mathcal{L}_{\mathrm{JEPA},t}
= \mathrm{distance}\!\left(
\widehat{P}_{t+1\mid t}, P^{\mathrm{target}}_{t+1}
\right)
$$

`RecentKV` 是短期精确上下文；`SlowMemory` 是每层固定容量的长期关联摘要；`C_t`
是共享语义/认知状态。SlowMemory 不是完整 softmax attention 的严格等价替代，375
unit 窗口内仍保持 RecentKV 精确语义。

### 1.1 系统边界

AVM 继承 LatentLoop 的循环主干与记忆设计，裁剪动作，面向通用全双工实时 Omni 对话。
Model Core 只处理 tensor；Model Service 只接收音视频物理信号并返回语音 PCM；
Harness 负责摄像头/视频流/屏幕共享、混合音频采集、播放回流与会话生命周期；
Training 负责共享三阶段训练、评估与 checkpoint；Data 是共享持久化域。

```text
Harness -> mic PCM + video pixels/revision + time -> Model Service
Harness <- speech PCM                          <- Model Service
```

模型不调用设备或工具。reward、对端身份、字幕、审计和隐藏状态只属于控制面。
运行时不设置文本输入输出头，不经过 ASR→文本模型→TTS。视频可包含文字。
完整身份、时序和验收见 [AVM 契约](avm-contract.md)。

## 2. 设计目标

1. 连续接收混合麦克风和视频输入，支持长期 always-on 运行。
2. 模型输出语音 时仍持续更新 H、C、RecentKV、SlowMemory 和局部状态。
3. 支持用户插话、补充、纠正和打断，真实回流进入后续 unit。
4. 直接从多模态主干 hidden 生成语音 codec，不经过文本或 TTS。
6. 使用固定容量 $C_t$ 保存共享语义状态，并用每层 SlowMemory 保存长期关联摘要。
7. 使用有界 KV 保存近期精确多模态历史，控制显存和延迟。
8. 让未来 Speech loss 与跨 unit 的 JEPA source 梯度通过 $C_t$ 和 SlowMemory 监督早期记忆更新。
9. 训练、验证、checkpoint 恢复和推理使用同一状态转移。
11. Pretrain、SFT、Online RL（算法为 Online Recurrent PPO）使用同一语音输出模型和状态转移完整训练全模型；
    RL Value Head 仅用于 actor-critic 估值，不跨物理信号边界。
12. 使用单参数 Perceiver 和 JEPA 单步目标，使 $H_t$ 学习可预测的观测表示，并经递归影响 $C_t$。

## 3. 输入与输出

### 3.1 单路混合麦克风输入

模型接收设备实际采集的一路混合音频：

$$
x_t^{\mathrm{mic}} = u_t + e_t + o_t + n_t
$$

其中：

- u_t：用户或现场说话人的语音；
- e_t：模型语音经过声卡、扬声器、房间和麦克风后的回流；
- o_t：其他人、电视、音乐、电脑提示音等声音；
- n_t：环境噪声、设备噪声和声学失真。

模型不接收播放参考、来源分离通道或用户/模型语音标识。

### 3.2 通用视频视觉输入

模型接收一路通用视频，可来自摄像头、视频流或屏幕共享。物理帧携带采集时间和
revision，适配器缩放为 224×224 RGB 后进入模型。每 unit 使用最新因果帧；
低帧率复用最近帧，缺失首帧等待，断连或超时显式失败，不合成黑帧。

### 3.3 直接语音输出

Speech Head 直接预测 Mimi codec token：

~~~
H_t + speech_local_(t-1)
    -> speech mode
    -> Mimi codec frame
    -> frozen causal codec decoder
    -> 1920-sample waveform
    -> playback
~~~

每 80 ms unit 输出 SILENCE 或 SPEECH。SILENCE 不生成 codec token，也不调用 codec decoder。

### 3.4 输出范围与架构来源

AVM 不包含电脑动作输出。本节保留 LatentLoop 迁移边界：旧 ActionFrame、
ControlSignal、action-local state 以及 GUI 执行环境均退出当前协议。
唯一生成输出是直接语音；静默由 Speech Head 的 SILENCE 表达。

### 3.5 文本旁路边界

文本可以作为离线审计、调试、字幕或数据准备工具，但不是运行时输出头，也不是 Speech Head 的中间目标。生产推理链路只有 Speech Head。

## 4. 核心状态

设系统按 unit t 运行：

| 符号 | 含义 |
|---|---|
| $O_t$ | 当前混合音频、视频和时间观测 |
| $P_t$ | $\mathrm{Perceiver}(O_t)$ 的 16 个多模态 slots |
| $\widehat{P}_{t+1\mid t}$ | JEPAHead 对下一时刻 Perceiver slots 的预测 |
| $\mathrm{RecentKV}_t$ | 有界逐层 Transformer Key/Value Cache |
| $\mathrm{SlowMemory}_t$ | 每层固定容量长期关联摘要 |
| $C_t$ | 固定容量共享语义 memory slots |
| $H_t$ | 当前 unit 的完整 final-normalized hidden |
| $\mathrm{speech\_local}_t$ | 语音 temporal state 和上一帧 codec |

### 4.1 KV Cache

每层 KV 保存最近进入主干的 16 个 Perceiver slots。缓存按完整 unit 追加和淘汰，
不能在 unit 中间截断。Canary 上限为 375 个 80 ms unit，即每层 6,000 tokens、30 秒。
语义 memory slots 不作为额外 token 直接写入 KV。超过精确窗口的完整 unit KV 才写入
SlowMemory。

### 4.2 共享语义 memory slots

$$
C_t\in\mathbb R^{B\times M\times d_{\mathrm{model}}}
$$

$C_t$ 由固定数量的 slots 构成，用于保存：

- 用户目标和长期约束；
- 当前任务阶段和未完成子目标；
- 应用、窗口和重要 UI 状态；
- 已经离开 KV 窗口但仍影响未来输出的信息。

$C_t$ 的容量与运行时长无关，不承诺逐 token 复制历史。

### 4.3 每层 SlowMemory

第 $l$ 层维护固定形状的关联矩阵和归一化状态：

$$
\mathrm{SlowMemory}^{(l)}_t
= \left(M^{(l)}_t,n^{(l)}_t\right)
$$

其容量不随 session 长度增长，仅接收从 RecentKV 淘汰的完整 unit KV。

### 4.4 完整 H_t

$$
H_t\in\mathbb R^{B\times 16\times d_{model}}
$$

$H_t$ 是当前 unit 的完整主干输出，而不是单个 query 或 pooled summary。它必须保存在
`RecurrentState.hidden`，并作为下一时刻 Backbone 的工作状态输入。

### 4.5 局部状态

decoder 连续性。二者不是长期认知记忆，不能替代 $C_t$ 或 SlowMemory。

### 4.6 状态初始化和边界

episode/session 开始时：

- $C_0$ 使用 learned semantic slot identity 初始化；$H_0$、SlowMemory、audio cache、
- $\mathrm{RecentKV}_0$ 为空；
- unit_index 从零开始。

正常 unit 边界不重置状态。TBPTT 边界只 detach 计算图，不清空状态数值。

## 5. 多模态时间单元

### 5.1 Unit 格式

每个 unit 的输入协议为：

~~~
timestamp_ms       [B] int64
delta_ms           [B] int64
mic_audio          [B, 1920] float32
video             [B, 3, 224, 224] float32
~~~

delta_ms 必须为正，时间戳严格递增。

### 5.2 Perceiver 输入与输出

音频、视觉和时间 stem 先组织为带模态与位置编码的输入序列：

~~~
<TIME>
<AUDIO_0> ... <AUDIO_N>
<VISION_0> ... <VISION_15>
~~~

16 个 learned Perceiver queries 通过两层 `slot-to-input cross-attention -> slot
self-attention -> FFN` 将该序列压缩为 `[B,16,model_dim]` 的 P_t。视觉 stem 仍输出
4x4 空间特征，但 Perceiver slots 是混合模态高层表示，不再与视觉位置一一对应。所有
观测必须经过 Perceiver，原始 stem token 不得绕过它进入 Backbone。静态和动态视频每个
unit 都走同一条编码路径。

### 5.3 Codec 时间对齐

~~~
sample_rate          24000 Hz
unit_ms              80
audio_samples/unit   1920
codec_frame_rate     12.5 Hz
codec_frames/unit    1
codebooks            8
vocabulary           2048
~~~

音频和 codec 时间轴必须严格对应，不能累计四舍五入。

### 5.4 多频率调度

采集器可以高频采集音频和视频，编码器可以使用内部帧率，但主干状态转移和输出协议统一在 80 ms unit 上。任何降频、合并或背压都必须显式记录 delta_ms，并保持状态顺序。

### 5.5 时间间隔编码

模型不直接接收绝对时间。`timestamp_ms` 只用于 unit 排序、计算 `delta_ms`、延迟统计和
轨迹记录。当前观察间隔通过 DeltaTimeEncoder 进入 Backbone：

$$
T_t^{\Delta} = \mathrm{DeltaTimeEncoder}(\Delta t_t)
$$

DeltaTimeEncoder 使用 log-scaled interval 和多尺度 Fourier 特征：

$$
\phi_k(\Delta t) =
\left[
\sin\left(2\pi\frac{\Delta t}{p_k}\right),
\cos\left(2\pi\frac{\Delta t}{p_k}\right)
\right]
$$

其中周期集合默认为 `80, 160, 320, 640, 1280, 2560, 5120, 10240 ms`。时间间隔通过
当前 $P_t$ 进入 Backbone，不要求 $C_t$ 或 SlowMemory 遵循物理微分方程。

## 6. 模型架构

```text
MIC_MIXED --> Streaming Audio Encoder --┐
SCREEN   --> Vision Encoder -----------+--> Perceiver --> P_t
DELTA_T  --> DeltaTimeEncoder ---------┘                 |
                                                         v
H_(t-1), C_(t-1), RecentKV_(t-1), SlowMemory_(t-1)
                    --> Backbone --> H_t, C_t, RecentKV_t, SlowMemory_t
                                      |             |
                               Speech Head
                                      |             |
                             Mimi waveform
                                      |
                                      +--> JEPAHead --> P_hat_(t+1|t)
```

### 6.1 流式音频编码器

音频编码器增量处理新增采样并保留 audio_cache。它只接收混合麦克风，不接收播放参考或来源标签。

### 6.2 视觉编码器

视频帧统一归一化为 [B,3,224,224]，视觉编码器生成 4×4 空间 token，
加入位置表示后进入 Perceiver。视觉 token 不旁路连接输出；Speech Head 读取融合后的完整 H_t。
同音频不同视频的任务用于检查模型是否使用视觉，避免语音目标与视频无关。

### 6.3 多模态主干

主干执行：

$$
\left(H_t,C_t,\mathrm{RecentKV}_t,\mathrm{SlowMemory}_t\right)
= \mathcal{F}_{\theta}\!\left(
P_t,H_{t-1},C_{t-1},
\mathrm{RecentKV}_{t-1},\mathrm{SlowMemory}_{t-1}
\right)
$$

每层把当前工作 tokens 与持久语义 tokens 拼成 $[H,C]$，共同执行 cached self-attention、
SlowMemory read 和 feed-forward，因此同一 unit 的 H/C 双向互见。只有 H 写入
RecentKV；固定数量的 C 只跨 unit 持久化，不写入 RecentKV。

### 6.4 JEPAHead 与单参数 JEPA

JEPAHead 只读取当前 $H_t$，以固定 slot 顺序预测一个 80 ms 后的 Perceiver 表示：

$$
\widehat{P}_{t+1\mid t} = \mathrm{JEPAHead}(H_t)
$$

Perceiver 只有一份参数，不维护 EMA teacher。预测输出只参与 JEPA loss，不再作为 Future
slots 回注行为主干，因此 Speech/Value loss 不训练 JEPAHead。

## 7. 语义状态与长期记忆更新

### 7.1 严格更新顺序

语义 slots 以学习到的槽位身份初始化：

$$
C_0 = E_C
$$

随后当前观测、上一工作状态和上一语义状态共同进入每一层 Backbone：

$$
X_t^{(0)} = \left[P_t+H_{t-1},C_{t-1}\right]
$$

$$
X_t^{(l+1)} = \mathrm{BackboneLayer}^{(l)}\!\left(
X_t^{(l)},\mathrm{RecentKV}_{t-1}^{(l)},\mathrm{SlowMemory}_{t-1}^{(l)}
\right)
$$

$$
(H_t,C_t)=\mathrm{split}\!\left(\mathrm{Norm}(X_t^{(L)})\right)
$$

$C_t$ 是 Backbone token 序列中专门划出的持久语义区域；$H_t$ 服务当前输出，$C_t$
保存到下一 unit。$E_C$ 只在 session 初始化时使用，不在每个 unit 重复相加。二者使用
同一 model dimension，但生命周期和职责不同；不存在 Backbone 外的独立更新网络。

### 7.2 SlowMemory 写入、覆盖和遗忘

令从 RecentKV 淘汰的第 $l$ 层 KV 为 $(K_t^{(l)},V_t^{(l)})$：

$$
\widetilde{K}_t^{(l)}=\phi\!\left(K_t^{(l)}\right)
$$

$$
\widehat{V}_t^{(l)}
= \frac{\widetilde{K}_t^{(l)}M_{t-1}^{(l)}}
{\widetilde{K}_t^{(l)}n_{t-1}^{(l)}+\varepsilon}
$$

$$
E_t^{(l)}=V_t^{(l)}-\widehat{V}_t^{(l)}
$$

$$
g_t^{(l)}=\sigma\!\left(
\mathrm{WriteController}^{(l)}\!\left(\lVert E_t^{(l)}\rVert_2\right)
\right)
$$

$$
M_t^{(l)}
= \rho_t^{(l)}M_{t-1}^{(l)}
{}+g_t^{(l)}\left(\widetilde{K}_t^{(l)}\right)^{\!\top}E_t^{(l)}
$$

$$
n_t^{(l)}
= \rho_t^{(l)}n_{t-1}^{(l)}
{}+g_t^{(l)}\sum_i\widetilde{K}_{t,i}^{(l)}
$$

Delta 写入用预测误差修正旧关联，门控决定写入强度，$\rho_t^{(l)}$ 提供有界遗忘。

### 7.3 长时监督

未来输出 loss 沿以下路径反向传播：

```text
future Speech loss
  -> future H
  -> C / SlowMemory read and write
  -> earlier H, C and expired KV
```

长期记忆是否有效，以跨窗口 Speech 行为评测以及 C/SlowMemory on/off 消融为准。

## 8. 直接语音生成

### 8.1 Speech Head

Speech Head 使用 H_t 的当前输出位置和 speech_local：

~~~
H_t + speech_local_(t-1)
    -> speech mode logits
    -> causal/factorized codec logits
    -> generated codes
~~~

Speech mode 为 SILENCE 或 SPEECH。SPEECH unit 预测一帧 8-codebook Mimi token；SILENCE unit 的 codec mask 为 false。

### 8.2 Codec 契约

| 字段 | 值 |
|---|---|
| Codec | Mimi |
| 采样率 | 24 kHz mono |
| 帧率 | 12.5 Hz |
| 帧长 | 80 ms / 1920 samples |
| Codebook | 8 |
| Vocab | 2048 |
| Revision | 配置锁定 |
| Weight SHA-256 | 配置锁定 |

Mimi decoder 是冻结的声学解码器，不是 TTS。

### 8.3 环境自听

codec 解码和播放后的声音经过真实扬声器、房间和麦克风回流：

$$
Y_t\rightarrow Playback\rightarrow Environment\rightarrow x_{t+\delta}^{mic}
$$

回流作为下一 unit 的混合音频重新进入 Perceiver，不直接把 codec token 回灌主干。

## 9. 音视频 Harness 与对话会话

### 9.1 通用音视频来源

摄像头、一般视频流和屏幕共享使用同一 VideoSignal，屏幕共享不构成 GUI 任务假设。
每帧有 captured_at_ms 和 revision；按当前 unit 选择最新因果帧，低帧率复用最近帧。
无首帧等待就绪，断连、未来帧、倒退与超过 video_max_age_ms 的帧显式失败。

### 9.2 持续采集与播放

音频与视频使用持续采集线程，不等待模型完成发声。每 80 ms 形成一条物理 observation。
播放独立消费语音 PCM，队列默认最多 2 units；静音清除待播放内容并中断播放设备。
采集 backlog 默认最多 3 units，超出预算报错，不静默丢音频或重置模型状态。

### 9.3 正式部署

FFmpeg 接入摄像头、视频文件/流和屏幕共享，PulseAudio 提供混合输入与播放。
mixed_audio_source 必须包含对端音频与模型播放回流；只读单路混合信号，不增加参考通道。
场景由 scenario_id 标识，无电脑快照、键鼠执行或 QEMU/KVM 必需依赖。

## 10. 并发输出

全双工意味着模型发声时仍更新 H、C、KV、SlowMemory 与音频状态。
用户插话后的新观察可以改变下一 unit 的语音 mode，静音取消未播放语音；
模型持续听看并决定恢复回应，不使用录完一轮再生成一轮的串行流程。
停止延迟包括采集、模型推理、codec、播放缓冲，必须测量而非仅由 80 ms 时钟推断。

## 11. 完整状态转移

### 11.1 感知与预测

$$
P_t = \mathrm{Perceiver}(O_t)
$$

$$
\widehat{P}_{t+1\mid t} = \mathrm{JEPAHead}(H_t)
$$

### 11.2 记忆

$$
\mathrm{SlowMemory}_t
= \mathrm{DeltaWrite}\!\left(
\mathrm{SlowMemory}_{t-1},\mathrm{ExpiredKV}_t
\right)
$$

### 11.3 主干

$$
\left(H_t,C_t,\mathrm{RecentKV}_t,\mathrm{SlowMemory}_t\right)
= \mathrm{Backbone}\!\left(
P_t,H_{t-1},C_{t-1},
\mathrm{RecentKV}_{t-1},\mathrm{SlowMemory}_{t-1}
\right)
$$

### 11.4 输出

$$
U_t = \mathrm{SpeechHead}(H_t, \mathrm{speech\_local}_{t-1})
$$

### 11.5 状态保存

$$
\mathrm{state}_{t+1} = \left(
C_t,H_t,\mathrm{RecentKV}_t,\mathrm{SlowMemory}_t,\mathrm{audio\_cache}_t,
\mathrm{speech\_local}_t,\mathrm{unit\_cursor}_t
\right)
$$

### 11.6 环境演化

语音播放改变真实环境；其后续麦克风、视频和时间输入构成 O_(t+1)。
模型不读取隐藏的执行成功标签，也不把 $U_t$ 作为显式 JEPAHead 条件。

## 12. 上下文管理

### 12.1 有界 KV

KV 统一保留最近配置窗口：

$$
\mathrm{RecentKV}_t
= \mathrm{CURRENT\_SLOTS}[t-\mathrm{kv\_units}+1:t]
$$

Canary 上下文为 375 units（30 秒），每个 unit 固定 16 个 slots。KV 按完整 unit 淘汰，
保留后的 token 仍按原始时间顺序参与 causal attention。

### 12.2 语义 memory 和 SlowMemory 读取

$C_t$ 作为固定数量的持久 token 在主干每层与 $H_t$ 联合 self-attention；SlowMemory 在
每层由当前 query 进行归一化关联读取。两者都不并入 RecentKV；episode/session reset 时
$C_t$ 回到 learned 初始值，SlowMemory 清零。

### 12.3 持久化

checkpoint 保存 C、H、RecentKV、SlowMemory、audio cache、speech local 和
unit cursor。会话持久化必须记录当前 model、codec 和 model identity；不完整身份拒绝恢复。

## 13. 实时运行时

### 13.1 运行线程

~~~
Audio Capture       音频环形缓冲
Screen Capture      每 unit 完整视频帧
Perceiver           音频/视觉/时间 stem 与 16-slot 感知编码
Backbone Worker     Backbone、C/RecentKV/SlowMemory 状态
Speech Worker       codec frame 和播放块
Telemetry           延迟、队列、状态和轨迹
~~~

### 13.2 单 GPU 调度

递归模型、C、RecentKV、SlowMemory、audio cache 和两个 local state 留在同一个 GPU
进程。Ray 只负责外围 CPU 任务，不逐 unit 搬运状态。

### 13.3 背压

当计算延迟超过实时 tick：

- 音频块可以合并，但必须更新 delta_ms；
- 视觉按 unit 输入当前完整帧；低帧率复用最近因果帧，超时报错并记录 telemetry；
- 超时、取消和紧急停止优先级最高；
- 任何丢帧、时间跳跃或 worker 错误都写入 telemetry。

## 14. 训练数据

### 14.1 轨迹字段

训练轨迹由以下数据构成：

~~~
mic_audio          单路混合麦克风
video_frames      每 unit 一帧完整视频
target_speech      仅用于离线 codec 编码的目标音频
timestamps         unit 时间
runtime_events     播放、执行、丢帧和延迟审计
~~~

构造过程中的来源分离、TTS 中间产物、环境参数和任务标签不能作为模型输入。

当前 WebDataset episode 由 `meta.json`、`mic.flac`、
`target_speech.flac`、`video.npz`、`timeline.npz`、`speech_codes.npy` 和
`turns.json` 组成；timeline 保存 speech mode/mask、codec mask 和时间戳。
metadata 使用 schema_id=avm-episode-v1，runtime 使用 avm-realtime-v1。
旧动作时间线、controls.npy 和 memory target 不属于 AVM；receipt/reward 仅用于原始
采集或在线训练审计，不进入监督样本输入。完整字段见
`docs/data/trajectory-schema.md`。

### 14.2 场景覆盖

最终数据集应覆盖：

- 安静单人语音；
- 用户打断、重叠语音和多人环境；
- 键盘、风扇、音乐、电视和系统声音；
- 播放延迟、混响、设备频响、回流衰减和回流缺失；
- 自然场景、共同观看、事件变化和连续运动目标；
- 视觉指代、音视频关联、插话、暂停和恢复回应；
- 用户纠正、视觉指代、环境事件和长对话；
- 长时间无语音、纯视频观察和模型静音。

### 14.3 声学域与视觉域随机化

训练可以随机化房间 impulse response、设备频响、播放延迟、时钟漂移、非线性失真、自动增益、背景噪声、回流截断和视觉变化速度，但最终输入仍必须是一路混合麦克风和视频 unit。

### 14.4 数据隔离

按 device/session 分组切分 train/validation/test。同一 session 不得跨 split。每个 manifest 锁定 source、license、content hash、codec identity 和 session hash。

## 15. 训练目标

### 15.1 Speech loss

$$
\mathcal{L}_{\mathrm{speech}}
= \mathcal{L}_{\mathrm{speech\_mode}}
{}+ \mathcal{L}_{\mathrm{speech\_codec}}
$$

mode loss 对有效 SILENCE/SPEECH 标签计算 CE；codec loss 只对 SPEECH unit 的有效 Mimi frame/codebook 计算 CE。

### 15.2 行为监督范围

AVM 删除动作监督，不计算动作 loss，也不将缺失动作标签改为占位目标。
mode CE 和有效 codec CE 构成唯一行为监督；JEPA 保持既有单参数时序。

### 15.3 并发输出

mode 和 codec 共享主干梯度。SILENCE 只监督 mode，SPEECH 监督 mode 与有效 codec。
缺少语音标签的 unit 不产生虚假静音目标，但仍按 episode 连续推进状态。

### 15.4 长期记忆监督

$C_t$ 和 SlowMemory 没有独立 memory loss、probe loss、write-budget 或 diversity loss。
JEPAHead 使用独立表示目标，但它不是记忆状态的独立 target。未来 Speech loss 仍通过：

```text
future loss -> future H -> earlier C / SlowMemory read-write -> earlier H and KV
```

监督长期信息选择。

### 15.5 JEPA 表示预测损失

同一个参数版本同时计算 source 和 target，target 侧停止梯度：

$$
\mathcal{L}_{\mathrm{pred}}
= \mathbb{E}_{t,s}\!\left[
\left\|
\mathcal{N}(\widehat{P}_{t+1\mid t,s})
{}- \mathrm{stopgrad}\!\left(\mathcal{N}(P_{t+1,s})\right)
\right\|_2^2
\right]
$$

其中
$\mathcal{N}(X)=X/\max(\lVert X\rVert_2,10^{-4})$。为防止单参数 Perceiver 表示坍塌，在有效 source
表示上按 slot、channel 跨 batch-time 计算：

$$
\sigma_{s,d}
= \sqrt{\mathrm{Var}_{b,t}(P_{t,s,d}) + 10^{-4}}
$$

$$
\mathcal{L}_{\mathrm{var}}
= \mathbb{E}_{s,d}\!\left[\max(0, 1-\sigma_{s,d})\right]
$$

$$
\mathcal{L}_{\mathrm{JEPA}}
= \mathcal{L}_{\mathrm{pred}} + \mathcal{L}_{\mathrm{var}}
$$

有效 source 少于两个时跳过 $\mathcal{L}_{\mathrm{var}}$；不得跨 episode、session 或不连续
observation 配对。

### 15.6 总损失

Pretrain、SFT 和 Online RL 分别使用 1.0、0.5 和两路 0.1 的 JEPA 系数，完整公式由
[统一三阶段训练架构](three-stage-training.md) 约束。各模块影响关系为：

| 模块 | 直接梯度 | 主要行为影响 |
|---|---|---|
| Speech Head | Speech mode/codec loss | 语音 mode、codec 准确率和局部连续性 |
| Perceiver | Speech/RL、JEPA source loss | 当前多模态观察表示 |
| JEPAHead | JEPA loss | 单步未来表示预测 |
| Backbone | Speech/RL loss | 共享多模态理解和输出条件表示 |
| Semantic slots C | 未来输出 loss 与 JEPA source loss | 长期目标、约束、计划和抽象任务状态保持 |
| SlowMemory updater | 未来输出 loss 与 JEPA source loss | 窗口外关联摘要的选择、覆盖和遗忘 |
| RecentKV state | 无参数 loss | 近期精确上下文 |
| local states | 对应 head loss | 语音跨帧连续性 |

## 16. 闭环训练

### 16.1 训练分布

训练输入中的未来麦克风可以由真实录音、目标语音经过声学环境后的回流或模型实际生成语音经过环境模拟后的回流构成；无论来源如何，模型看到的都必须是同一路混合麦克风。

### 16.2 环境反馈建模

声学环境可包含播放延迟、房间响应、设备频响、噪声、重叠、截断和回流缺失。视觉环境可包含窗口变化、连续运动、传感器延迟和对话中断。

### 16.3 梯度路径

单个 unit 的输出 loss 通过当前 Backbone 和 Perceiver 反向传播；跨 unit 的未来行为 loss
通过 C、SlowMemory 和 H 反向传播。JEPA target 侧 detach，source 侧通过 JEPAHead、H
和监督 horizon 内历史状态传播。rematerialization 只重算激活，不截断这个 horizon；
梯度只在明确的 horizon 边界 detach，不能在每个 unit 或重计算分段重置状态。

正式训练把 375-unit horizon 划分为若干 rematerialization segment。每段前向只保留段末
完整 recurrent state，以及各 unit 计算 Speech/JEPA/PPO 所必需的轻量输出张量；
段内逐步 H/C/RecentKV/SlowMemory 状态不进入持久输出列表，反向时从段首状态整段重放。
segment 边界保持 autograd 连接，因此分段大小只影响显存与重算量，不改变 375-unit 监督语义。

### 16.4 外部执行边界

物理播放、操作系统和 UI-TARS 执行本身不是可微模块。它们提供后续观察和行为 reward；训练接口只接收合法的 input/target/observation，不把隐藏执行结果作为额外模型输入。

## 17. 训练约束

1. Canary 是唯一正式规模，使用公共训练入口和状态协议；后续规模另行设计。
2. 所有训练 episode 按时间顺序处理，不能每个窗口重置状态。
3. 正式 Canary 的 memory horizon 和 TBPTT 为 375 units。
4. 所有 targets 配套 mask；缺失标签屏蔽对应 loss，不伪造 NOOP 或 SILENCE。
5. 未来输出 loss 必须能够在 TBPTT 范围内回传到早期 C/SlowMemory 更新。
6. 训练、验证、推理和恢复共享同一 forward_step 语义。
7. codec、model schema、trajectory schema、unit 时钟和 checkpoint identity 必须一致。

## 18. 推理算法

```text
state = initial_state()

for each 80 ms observation O_t:
    P_t = Perceiver(O_t)
    H_t, C_t, RecentKV_t, SlowMemory_t = Backbone(
        P_t, state.H, state.C, state.RecentKV, state.SlowMemory
    )
    P_hat = JEPAHead(H_t)
    speech_t = SpeechHead(H_t, state.speech_local)

    submit speech_t to codec/playback if mode=SPEECH
    submit speech PCM to bounded Harness playback

    state = {
        C: C_t,
        H: H_t,
        RecentKV: RecentKV_t,
        SlowMemory: SlowMemory_t,
        speech_local: speech_t.local,
    }
```

输出产生的真实音频通过播放回流进入后续 unit；视频持续反映当前情境。
Speech Head 每 unit 决定 SPEECH 或 SILENCE，不依赖外部轮次状态机。

## 19. MiniCPM 系基础实现

MiniCPM 或同类多模态主干可以提供视觉编码、音频编码、Perceiver stems、因果 Backbone
和增量 KV。项目必须在此基础上保持：

1. 固定 80 ms unit；
2. 完整 H_t 暂存；
3. Backbone 内持久化 $C_t$，每层持久化固定容量 SlowMemory；
4. 独立 Speech Head；
6. 单路混合麦克风输入；
7. 有界 RecentKV、固定 semantic slots 和固定 SlowMemory；
8. 同一 checkpoint/data/trajectory identity。

不得引入与上述状态协议不一致的 control、memory 或 action 过渡接口。

## 20. 配置约束

目标配置必须明确：

- model_dim、层数、heads 和 FFN；
- perceiver_slots、semantic_memory_slots、kv_units、kv_window_ms；
- audio sample rate、unit_ms、video shape；
- Mimi codec identity；
- model_schema_id、coordinate_grid_size、type_bytes_per_unit、hotkey_keys_per_unit；
- tbptt_units、memory_horizon_units、mixed precision；
- loss weights、checkpoint cadence、manifest 和 run identity。

Canary 的正式 horizon 为 375 units。正式 GPU capacity smoke 必须保持 Canary 的模型形状、
RecentKV、TBPTT 和 memory horizon；只有快速 CPU 单元测试 fixture 可以缩小这些数值，且不构成
Canary 容量证据。

## 21. 评测与消融

### 21.1 基线与消融

| 方案 | 用途 |
|---|---|
| 有界 RecentKV，无 C/SlowMemory | 近期上下文基线 |
| 有界 RecentKV + C + SlowMemory | 核心长期记忆方案 |
| 长 KV 对照 | 精确历史上界 |
| 核心方案去掉完整 H 暂存 | 验证完整 hidden 的必要性 |
| 核心方案去掉 Speech codec mask | 验证 SILENCE 语义 |

### 21.2 语音指标

- mode accuracy；
- speech-active fraction；
- 每个 codebook accuracy；
- SILENCE 误触发率；
- codec RTF、首块延迟和长语音连续性；
- 帧漂移、NaN、削波和播放队列积压；
- 自声回流、噪声和重叠条件下的稳定性。

### 21.3 音视频对话指标

评测视频问答、视觉指代、事件变化描述、音视频一致性及对话连贯性。
同音频不同视频应产生对应不同目标；视频替换与遮挡消融衡量视觉依赖。
重叠语音、插话后停止、恢复回应和长期静默覆盖全双工行为。

### 21.4 Long-term state 指标

- C/SlowMemory on/off 跨窗口任务差异；
- KV 窗口缩短时的目标保持曲线；
- 用户约束更新和冲突修正；
- 早期对话内容在后续的正确利用；
- H_t 完整暂存与摘要替代的对照；
- C、RecentKV、SlowMemory、H 和 local state 的容量上界。

### 21.5 实时系统指标

- unit latency、p95/p99；
- 音频、视觉、codec 和 playback 队列长度；
- GPU 显存、CPU 内存和 socket 健康；
- 长时间运行中的 NaN、丢帧、underrun 和队列增长；
- checkpoint 恢复后的下一 unit 输出与 loss 一致性。

## 22. 风险与缓解

| 风险 | 缓解策略 |
|---|---|
| C/SlowMemory 被近期 KV 忽略 | 窗口外任务、memory on/off 消融和长期行为评测 |
| C/SlowMemory 过度写入或过度保持 | 依靠未来输出任务、门值监控和状态范数监控 |
| slots 同质化 | learned slot identity、行为消融和容量监控 |
| 声学回流导致重复响应 | 回流延迟/音量/缺失随机化、重复行为评测 |
| 用户插话导致状态错乱 | 真实混合音频、严格 unit 顺序和完整 H 暂存 |
| 动态画面导致动作滞后 | 连续视觉流训练、80 ms unit 时序和运动任务评测 |
| 单卡吞吐跟不上 | 音频合并、视觉降频、activation checkpoint、显式背压 |
| 长时间队列增长 | 有界队列、超时、取消和实时 telemetry |
| 隐状态难审计 | 完整轨迹、checkpoint lineage 和紧急停止 |

## 23. 安全边界

Harness 拥有设备采集与播放生命周期，Model Service 无电脑控制或工具执行接口。
音视频数据需具备授权、来源锁与内容哈希，隐藏环境状态不进入物理输入。
正式运行设备、对端或 Judge 不可用时明确失败；进程内 fixture 仅用于协议测试。

## 24. 部署与运行时契约

- 模型、RecentKV、SlowMemory、C、H、codec local state 在同一推理进程内保持；
- W&B 只负责指标、配置和谱系，不进入模型 forward；
- Ray 只负责 CPU 数据、环境和评测，不维护 GPU recurrent state；
- codec worker 通过带身份校验的本地接口提供 encode/decode；
- runtime、checkpoint、manifest、codec 和 model schema identity 必须一致；
- 所有异常通过显式失败、隔离、恢复或安全拒绝处理。

## 25. 成功标准

1. 模型可以持续处理一路混合麦克风和视频流。
2. Speech Head 以 80 ms 对齐生成 SILENCE/SPEECH 和 Mimi codec。
3. 自声回流、用户插话、噪声和屏幕变化不会破坏状态顺序。
4. 有界 RecentKV、固定 C 和 SlowMemory 的容量不随运行时长无限增长。
5. C/SlowMemory on/off 对跨窗口目标、约束和错误恢复产生可测差异。
6. 同音频不同视频的对话任务能区分目标，证明模型使用视频。
7. Harness 持续采集、播放回流、及时停止发声并保持有界队列。
8. checkpoint 恢复后的下一 unit 输出和 loss 与连续运行一致。
9. 长时间运行中延迟、队列、显存和 socket 保持有界。
10. 训练、验证、推理和恢复均符合本顶层状态协议。

## 26. 最终架构定义

AVM = Audio-Video Model without Text，是基于 LatentLoop 裁剪动作后的通用实时
全双工 Omni 对话模型。单路混合音频、一路视频与时间经编码器、循环主干和长期记忆，
直接生成语音 codec tokens。共享训练入口执行 Pretrain → SFT → Online Recurrent PPO。
保持最终语义一致，不保留动作占位头、GUI 任务或文字中间推理通道。
