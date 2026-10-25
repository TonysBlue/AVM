# 当前轨迹契约

> 原生接口契约（2026-10-25）：MiniCPM-o 4.5 原生规模、200 ms、16 kHz输入、
> 连续五声学token、原生Token2Wav、8×A800 80GB。细节见[原生AVM契约](../minicpm-native-avm.md)。
> 正式运行仍要求真实资产、声学资格和数据readiness通过；本地测试不构成8卡性能证据。

监督 episode、SFT 样本和 Online RL 的 Online Recurrent PPO window 统一使用当前轨迹契约。项目只维护这一份
数据契约，使用 AVM 新身份，拒绝旧动作资产。

每个 `meta.json` 和 manifest entry 必须包含 `dataset_scale`、
`sample_kind`、`stage`、`supervision_kind`、`task_id`、`environment_id/version`、
`protocol_version`、`schema_id=avm-episode-v2`、codec identity 和完整
`runtime_identity`。每个 unit 固定 200 ms、16 kHz单声道、每unit五个原生声学token。

timeline 按 unit 保存 timestamps_ms、delta_ms和
speech_codec_mask；video.npz 保存归一化视频，speech_codes.npy 保存 codec targets。
时间字段仅用于对齐、审计和训练折扣；感知编码器仅接收 AudioVideoInput 的音视频张量。
meta.json 必须包含 schema_id=avm-episode-v2，protocol_version=avm-realtime-v2。
模型不接收动作、文本或额外音频参考通道。采集 ledger 的 receipts 与在线 RewardEvent
只属于审计/训练控制面，不写入模型物理输入。processed sample 必须记录
speech_codes_encoded=true、unit 数、持续时间和 content SHA-256。

数据准备从具有真实音视频来源锁的 manifest 构建 staging/processed shards，经 原生声学 worker、
音视频审计和 readiness 后进入训练。缺失真实视频不得用示意图、黑帧或 GUI fixture 替代。

```text
unit_ms            = 200
audio_sample_rate  = 16000
audio_samples      = 3200
codec_frame_rate   = 25
codec_frames/unit  = 5
```

## 1. StreamUnit

```python
@dataclass
class StreamUnit:
    timestamp_ms: Tensor
    delta_ms: Tensor
    mic_audio: Tensor
    video: Tensor
    speech_codes: Tensor
    speech_codec_mask: Tensor
```

物理 delta_ms 固定200，时间戳严格递增；每 unit 必须有 RGB patch对齐的overview或显式多切片 视频张量。
视频 source NPZ 的 tick 0 必须存在；低帧率按时间线复用最近因果帧，不以黑帧补缺。
reader 验证语音 masks、codec 范围、时间顺序、schema 和内容哈希，拒绝 action_* 时间线。

## 2. Lifetime rollout window

每unit保存五个采样codes、joint old/reference log-prob、value、receipt 与 reward
lineage。策略log-prob为五个受约束声学token的条件log-prob之和，静音也参与。
窗口保存 policy version、lineage、观测证据链与 reward finalization watermark。
receipt/reward 不进入 ObservationSignal，JEPA lookahead 不计入 PPO units。

## 3. Checkpoint identity

checkpoint 使用 model_schema_id=avm-model-v3，并保存数据、codec、阶段与父 checkpoint
身份。完整恢复检查配置哈希和严格 state-dict。旧 LatentLoop 权重仅能显式迁移到新谱系，
不得继承动作状态、优化器或旧 rollout。

## 4. 验证设计

- 当前 episode 写入/读取逐字段往返，语音与视频保持同一时间线；
- 无标签与监督静音明确区分；首帧缺失和未来帧拒绝；
- 旧动作字段、错误 schema/codec/hash 和不完整 metadata fail closed；
- 采集审计 metadata 不进入模型输入；
- checkpoint、数据、runtime 身份一致，迁移建立新 run。
