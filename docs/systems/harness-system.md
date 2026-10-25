# Harness System

Harness 是 AVM 的音视频物理对端，负责混合音频、通用视频、时钟、扬声器和会话生命周期。
Model Service 只接收 ObservationSignal 并返回 speech-only ActuationSignal。
raw tokens、文字转录、设备来源标签、receipt、reward 和隐藏环境状态不进入模型输入。
独立冻结 Judge 只消费 canonical ObservationSignal 字节与证据链。

## 生产部署模块

正式启动使用 harness.deployment.dialogue，通过持续 FFmpeg 进程接入摄像头、视频流或
屏幕共享，PulseAudio 接入包含对端与播放回流的混合输入，pacat 播放模型语音。
音视频采集线程独立于推理与播放；低帧率复用最新因果帧，首帧等待、超时或断连显式失败。
默认采集 backlog 上限 3 units、播放队列上限 2 units，超出预算失败，不静默丢音频。
外部cancel增加epoch并清除旧队列；静音仍按连续声学token播放。

配置声明 mixed_audio_source、playback_sink、video_max_age_ms 和 scenarios。每个 scenario
包含 video_kind（camera/video/screen）及 video_source；scenario_id 是场景身份，不是电脑快照。
正式配置默认不填写设备与场景，以确保未部署时 fail closed。缺失 FFmpeg、pacat、真实来源
或独立 Judge 不得回退 fixture。environment_id/version/protocol_version 参与控制握手。
关闭或启动失败时回收全部设备进程、线程与队列，不包含 QEMU、SPICE 或键鼠后端。

进程内 backend 仅用于明确标记的协议/集成测试；真实设备时延和对话质量需要独立验收。
配置示例和源资产要求见 [运行手册](../avm-runtime.md)。
