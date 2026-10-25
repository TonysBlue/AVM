# Model Service

Model Service 是 AVM 的物理信号推理边界，不采集设备、不执行电脑动作或工具。
输入为每 200 ms 的单路混合麦克风 PCM 和通用视频帧，输出为 24 kHz 语音 PCM。
传输时间字段用于时效与顺序检查，不进入模型感知编码器。
VideoSignal 携带采集时间与 revision，服务验证时效、因果顺序与重复 revision 的一致性，
将原始 RGB/RGBA 图像缩放为 224×224。缺失或失效显式失败，不生成替代画面。

服务维护每会话的 H/C/RecentKV/SlowMemory/audio cache/speech-local state 和 unit cursor，
会话边界 reset，单个 unit 成功生成和解码后才推进 cursor。原生声学 tokens 只存在内部 runtime。
checkpoint 必须使用 avm-model-v3，物理消息必须使用 avm-realtime-v2；旧动作模型不直接兼容。

语音播放期间继续接收新观察；每 unit 的静音输出由 Harness 用于取消未播放语音。
全双工时序、回流与播放有界性见 [AVM 契约](../avm-contract.md)。
