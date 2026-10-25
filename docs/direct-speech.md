# 直接流式语音（Speech Head）实施说明

> 原生接口契约（2026-10-25）：MiniCPM-o 4.5 原生规模、200 ms、16 kHz输入、
> 连续五声学token、原生Token2Wav、8×A800 80GB。细节见[原生AVM契约](minicpm-native-avm.md)。
> 正式运行仍要求真实资产、声学资格和数据readiness通过；本地测试不构成8卡性能证据。

> 状态：最终目标语音协议
> 日期：2026-08-20
> 关联顶层架构：[实时流多模态 LatentLoop 完整方案](realtime-multimodal-latent-loop.md)
> 场景契约：[AVM 全双工音视频对话](avm-contract.md)

## 1. 完成边界

Model Service 对外输出 24 kHz mono PCM。原生声学 token 只存在于 Model Core 的
Speech Head、训练 target 和内部 runtime 解码路径；Harness 只接收 PCM 并负责播放。
静音也由原生声学token解码，不生成全零PCM旁路；Harness只理解物理信号。

直接语音路径将模型时钟固定为 200 ms。每个 unit 接收一路16 kHz、3200样本的混合麦克风输入。完整状态顺序为：

$$
P_t = \mathrm{Perceiver}(O_t)
$$

$$
\left(H_t,C_t,\mathrm{RecentKV}_t,\mathrm{SlowMemory}_t\right)
= \mathrm{Backbone}\!\left(
P_t,H_{t-1},C_{t-1},
\mathrm{RecentKV}_{t-1},\mathrm{SlowMemory}_{t-1}
\right)
$$

$$
\mathrm{speech}_t
= \mathrm{SpeechHead}(H_t, \mathrm{speech\_local}_{t-1})
$$

Speech Head每unit严格生成五个原生声学token，包括静音。Token2Wav累计真实波形后按4800个24 kHz样本交付，带有未来计划播放时间。运行路径不经过文本或 TTS；播放回流在下一 unit 作为混合麦克风输入重新进入模型。

## 2. 固定 Codec

| 字段 | 值 |
|---|---|
| Codec | `minicpm-o-4_5-s3-25hz` |
| tokenizer输入 / 播放输出 | 16 kHz / 24 kHz mono |
| 帧率 | 25 Hz，40 ms/token |
| 每认知unit | 200 ms，五token；输入3200 / 输出4800 samples |
| Codebook / Vocab | 1 / 6562，EOS 6561排除 |
| 上游revision | `eb4141de24550a3a1a4082f718be58f46b761728` |
| tokenizer SHA-256 | `d43342aa12163a80bf07bffb94c9de2e120a8df2f9917cd2f642e7f4219c6f71` |
| 原生前瞻 / 留尾 | 3 tokens / 3840 samples（160 ms） |

原生Token2Wav的flow、HiFT、音色参考和推理代码与精度均由source-lock逐文件锁定。
固定2秒音色参考只作为离线目标编码前缀和冻结声学资产，绝不进入ObservationSignal。
目标整段编码后裁去50个参考token，每五token分组。

内部原生worker支持health、reset、encode、decode、synchronize和release，可放专用GPU，
单一stream lease拒绝并发生命期。完整episode连续编码，推理使用真实原生小块解码。
数据、配置、runtime、source-lock和checkpoint身份必须匹配。

## 3. Speech Head

完整H的64个4096维位置经原生semantic MLP（4096→768→768、ReLU、L2 normalize）
形成64个连续条件，交给原生20层768维Llama（12 heads、FFN3072）、声学embedding
和weight-normalized code head。所有原生层数和维度保留，不新增发声分类head。

```text
condition_t[64] -> a1 a2 a3 a4 a5 -> condition_(t+1)[64] -> a6 ... a10
```

teacher forcing监督每一有效声学token，采样严格推进五次。EOS禁止采样且从CE、
PPO old/reference分布中同样排除，控制符不占五个位置。第五个token也消费到KV。
语音局部窗口按完整69位置组裁剪，首轮10组；裁剪仅影响语音局部缓存，不reset认知状态。
未来语音loss可穿过条件及跨unit状态回传，声学离散采样/Token2Wav不接收策略梯度。

## 4. 数据契约

### 4.1 当前数据契约

每个 episode 主要包含：

~~~
meta.json
mic.flac
target_speech.flac
video.npz
timeline.npz
speech_codes.npy
turns.json
~~~

每个 unit 的语音 target 为：

~~~
speech_codes         [B, 5, 1]
speech_codec_mask    [B, 5]
~~~

`mic.flac` 是模型唯一音频输入。`target_speech.flac` 只用于离线编码、审计和评测；纯静音unit的五个codec mask均为true，缺失标签才为false。旧的 `controls.npy`、SpeechControl 状态标签不属于当前协议；历史数据已清理。

### 4.2 编码和校验

数据准备先生成 staging shard，再用锁定原生tokenizer 生成 processed shard。训练前必须验证 schema、codec revision、权重 hash、manifest/content hash 和 session split。目标音频可以由授权 TTS 在数据准备阶段生成，但运行时不出现 TTS 路径。

## 5. 训练目标

语音相关目标只有：

$$
\mathcal{L}_{\mathrm{speech}}
= \mathcal{L}_{\mathrm{speech\_codec}}
$$

$\mathcal{L}_{\mathrm{speech\_codec}}$ 对讲话、停顿和静音的五个有效声学token计算CE。
没有独立 SpeechControl、prosody、boundary、memory 或 write loss。

当前和未来 speech loss 通过 Speech Head、Backbone、Perceiver、语义 slots 和每层
SlowMemory 传播。JEPA Head 不参与行为输出路径，只由 JEPA loss 训练。长期路径为：

~~~
future speech loss
  -> future H
  -> Backbone recurrence
  -> earlier C / RecentKV / SlowMemory
~~~

监督长期记忆是否保留有用信息。

## 6. 运行时与验收

运行时必须保证：

- 每200 ms严格生成五个有效code，静音也走同一路径，无VAD、EOS停止或零PCM回退；
- 前瞻从后续unit借用且不重复消费，首次原生调用可能只返回40 ms，不能补齐或裁剪；
- 累计真实PCM后按4800 samples提前交付Harness，计划播放延迟基线600 ms；
- 会话末flush完整留尾，外部cancel增加epoch并清队列；认知状态保留；
- codec token不直接回灌认知主干，只有真实播放回流进入混合麦克风；
- worker、数据、checkpoint、音色参考及qualification身份一致；
- PCM有限且归一化、时长无漂移，静音RMS、边界跳变、p95、RTF和欠载率有实测；
- 五token联合概率、Speech/Encoder/Backbone/C/SlowMemory梯度、TBPTT及JEPA边界通过。

本机真实FP32声学资格：最终固定初始化复验的峰值、静音、p95和RTF未通过，
正式readiness继续失败；早期单次通过结果保留为历史实验，不能作为当前资格。
复现命令与原始数值见[声学资格实测](experiments/native-acoustic-qualification.md)。
本地小维度三阶段recipe证明共享协议和训练谱系可执行，不证明原生质量或8卡性能。
正式训练仍通过`scripts/run-training.sh --recipe configs/recipes/canary.yaml --run-id <new-id>`，
真实声学服务通过`uv run model-service serve-acoustics --config configs/canary.yaml --socket <path>`
启动；未通过资产/资格门禁时明确失败。
