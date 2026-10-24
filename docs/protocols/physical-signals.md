# AVM 物理信号协议

Model Service 数据平面使用 avm-realtime-v1 Protobuf over Unix socket。
ObservationSignal 包含 session、unit、时间、24 kHz mono mic PCM 与 VideoSignal；
ActuationSignal 只包含 session、unit 和 24 kHz speech PCM。每音频帧恰好 80 ms。

VideoSignal 使用 raw RGB/RGBA pixels、尺寸、captured_at_ms 和 revision。视频时间不得晚于
observation；重复帧保持原 timestamp/revision。输入适配器缩放到 224×224，低帧率可复用。
物理协议使用显式 protocol_version 字段，旧 LatentLoop payload 即使字段编号相近也会被拒绝。

reward、receipt、对端身份、任务元数据、字幕和设备来源标签仅属于控制面。
无 ControlSignal 或文本字段，无黑帧回退。详见 [AVM 契约](../avm-contract.md)。
