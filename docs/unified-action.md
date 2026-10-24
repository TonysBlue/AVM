# LatentLoop 动作协议迁移说明

原 LatentLoop 的 Unified Action Head、ActionFrame 与 ControlSignal 已从 AVM 删除。
AVM 面向通用全双工实时音视频对话，不包含电脑动作或 GUI 任务环境。
本文件仅保留来源说明，不再是有效专项协议；完整架构见
[AVM 顶层架构](realtime-multimodal-latent-loop.md) 和 [AVM 契约](avm-contract.md)。
历史实现可通过 Git 历史追溯；旧动作数据和 checkpoint 不直接兼容 AVM。
