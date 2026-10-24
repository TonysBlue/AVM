# Harness 驱动的数据采集系统

## 目标与边界

采集系统生产通用全双工音视频对话轨迹。专家扮演助手，用户与专家边听、边看、边说。
采集器不加载模型，不调用 Model Service，不读取 GUI/DOM 或隐藏 evaluator 状态。
专家只提供每 unit 的 SpeechSignal；Harness 持续采集混合音频与通用视频并播放专家语音。
摄像头、一般视频流和屏幕共享使用同一 VideoSignal，播放回流通过唯一 mic 通道进入输入。

## 端到端时序

```text
scenario_id + real audiovisual sources
  -> start_lifetime_session
  -> expert speech PCM -> bounded playback -> mixed microphone feedback
  -> every 80 ms: mixed mic + causal video + time
  -> append observation/speech/receipt hash chain
  -> seal -> Mimi encode -> AVM WebDataset
```

单一 lifetime 内 observation、speech、receipt 与下一 observation 按 unit 连续记录。
采集不等待专家语音播放结束；源断连、背压或时钟错误使会话失败。

## 原始 ledger

`CaptureLedger` 为追加写、崩溃安全的 session 目录：

```text
<session>/
  observation-000000000000.pb
  observation-000000000001.pb
  actuation-000000000000.pb
  receipt-000000000000.pb
  transitions.jsonl
  sealed.json
```

`transitions.jsonl` 保存 unit、payload SHA-256、前序/当前 hash chain和 speech 静音标记。protobuf payload 保存 canonical observation、实际 actuation 和 Harness receipt。`sealed.json` 只有在 unit 连续且存在至少一个完整 transition 时生成。

## 训练导出

episode_from_capture 转换为 Episode/StreamUnit，沿用 WebDataset、Mimi、manifest、audit 和
readiness。输入仅含 mixed mic、video 和 time；目标语音只用于编码与监督。
语音 mode/codec masks 按实际专家语音设置，receipt 和设备诊断仅属于原始 ledger 审计。
导出 identity 为 avm-episode-v1 / avm-realtime-v1，禁止动作标签。

## 接口与失败策略

- start() 创建 Harness lifetime 并写入 unit 0。
- step(SpeechSignal) 提交专家语音，写入下一 observation 和 receipt。
- finish(metadata) 验证时序、身份并封存；未封存目录不得进入 manifest。
- 设备、协议、顺序、音频或视频失败立即停止，不回退 fixture。

JSONL 每行提供 speech_pcm_b64 和 speech_silent；静音必须提供全零 80 ms PCM。

```bash
uv run data capture --config configs/canary.yaml \
  --harness-socket "$HOME/latentloop-data/runtime/canary/harness-control.sock" \
  --scenario conversation --session-id expert-001 --lineage-id capture-001 \
  --task-id dialogue-001 --split train \
  --ledger "$HOME/latentloop-data/datasets/canary/capture/expert-001" \
  --output "$HOME/latentloop-data/datasets/canary/shards/staging/sft/sft-%06d.tar" \
  --input -
```

该命令只生成 speech_codes_encoded=false 的 staging shard，正式训练仍要求完整编码与审计。

## 运行与验收

先配置真实对话设备和 scenario，再启动 harness serve。摄像头或共享视频的查看由采集端
应用负责，不依赖 QEMU/SPICE。验收 80 ms/24 kHz 对齐、连续证据链、语音/receipt identity、
播放回流延迟、视频复用与失效、插话时及时停止，以及结束后的线程与进程清理。
