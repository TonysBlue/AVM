# AVM 实时音视频对话运行手册

## 环境与场景

AVM 运行在真实音视频会话中，不要求 GUI 或虚拟机。宿主机需要 FFmpeg（摄像头使用
v4l2，屏幕共享使用 x11grab）、PulseAudio/PipeWire 的 PulseAudio 接口及 pacat。
音频输入需事先建立真实混音，包含对端语音、环境声音与模型播放回流；不能只选扬声器
monitor 而遗漏用户麦克风。模型始终只接收混合后的一路音频，不增加参考通道。

复制并填写 Harness 配置，保持 environment_id/version 与训练配置一致：

```yaml
environment_id: avm-dialogue
environment_version: "1"
protocol_version: avm-realtime-v1
control_socket: ~/latentloop-data/runtime/canary/harness-control.sock
mixed_audio_source: avm_mix.monitor
playback_sink: avm_playback
video_max_age_ms: 1000
capture_backlog_units: 3
playback_queue_units: 2
device_timeout_s: 5
scenarios:
  conversation:
    video_kind: camera
    video_source: /dev/video0
```

video_kind 可为 camera、video（文件或网络流）、screen（X11 display）。每会话一路视频。
实时对端通过实际音频设备参与对话；播放视频不能代替在线对话对端。
缺失混音、设备、首视频帧或场景时失败，不回退静音、黑帧或测试环境。

## 服务启动与时序

```bash
uv run harness serve --config /absolute/path/to/dialogue.yaml
uv run model-service serve --config configs/services/model/canary.yaml \
  --checkpoint /absolute/path/to/avm.pt \
  --codec-socket "$HOME/latentloop-data/runtime/sockets/mimi.sock"
uv run harness converse --config /absolute/path/to/dialogue.yaml \
  --model-socket "$HOME/latentloop-data/runtime/canary/model-service.sock" \
  --scenario conversation --session-id live-001
```

模型 checkpoint、Mimi worker 与服务参数沿用直接语音运行手册。converse 连接
对话环境与 Model Service，或由 Training 的 PhysicalRolloutClient
执行在线策略；二者共享相同物理协议。
模型发声时音视频线程持续采集。SILENCE 清除待播语音，下一次 SPEECH 可以恢复。
超出采集/播放预算时明确报错，不丢音频来掩盖推理速度不足。响应与停止时延需在真实设备
上测量；80 ms unit 本身不能证明端到端延迟达到 80 ms。

## 视频数据资产

准备正式 Canary 前，提供 AVM_VIDEO_SOURCES 指向的 JSONL。每行包含 episode_id、
path、source_sha256、source_license、source_version、alignment_id 与可选 start_seconds。
alignment_id 标识人工或可信工具确认的音视频/对话对齐；源视频必须覆盖完整 episode。
适配器验证哈希、解码真实视频、按 12.5 Hz 采样为 224×224，并保存来源 sidecar。
不得用原 LatentLoop 示意图适配器或未对齐的视频替代真实资产。

```bash
AVM_VIDEO_SOURCES=/absolute/path/to/avm-video-sources.jsonl scripts/prepare-data.sh canary all
uv run data check-readiness --config configs/canary.yaml
```

数据准备中的文本/TTS 仅用于离线监督。视觉指代的答案必须与实际画面一致，生成的通用
对话模板本身不能证明视觉理解。完整训练仍需来源锁、授权、Mimi 报告与 split 审计。

## 权重迁移与训练

旧 LatentLoop checkpoint 不直接续训。显式迁移校验 codec、共有参数键与形状，删除动作
权重，初始化新的训练 Value Head，不继承旧 optimizer、recurrent state 或 rollout：

```bash
uv run training migrate-latentloop --config configs/canary.yaml \
  --source /absolute/path/to/latentloop.pt --output /absolute/path/to/avm-import.pt
uv run training train --config configs/canary.yaml --init-from /absolute/path/to/avm-import.pt
```

迁移产物仅用作 init-from，parent SHA-256 记录来源；新配置、新 run、新 checkpoint 谱系。
正式 recipe 继续使用共享 Pretrain → SFT → Online Recurrent PPO，逐阶段 validation，最终 test。
RL 的 sessions.jsonl 每行提供 session_id、scenario_id、seed；Judge、reward 规范与 rubric
必须锁定且与配置一致。设备、Judge 或真实数据不可用时不能宣称正式训练验收完成。
