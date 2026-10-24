# Training System

Training System 统一执行 `Pretrain -> SFT -> Online RL`。Online RL 的当前算法为 Online Recurrent PPO。
Pretrain/SFT 通过 Data replay 训练 Model Core；Online RL 由 Training Policy Runner 通过 Harness lifetime timeline
client 获取 ObservationSignal，并提交由 token decoder 还原的 ActuationSignal。训练进程
重新计算 log-prob 并更新完整模型。

Reward Event、receipt 和 old/reference log-prob 只存在训练 trace，不进入 ObservationSignal。
Reward Judge 也只接收 canonical ObservationSignal bytes，不接收采样 token、receipt 或隐藏任务字段。

训练系统只依赖正式的 `model`、`data`、`media`、`runtime`、`contracts` 包以及 Harness
control client。模型、数据、checkpoint、PPO、evaluation 和 tracking 的实现分别由这些正式
workspace 边界提供；旧单体包不再作为 Training System 的实现来源或反向依赖。

Online RL 的 Online Recurrent PPO 运行遇到 Harness/音视频设备/codec 的基础设施失败时封存窗口并记录失败类别，但不把
失败伪装成低 task reward。只有 identity、session、unit、observation hash chain 和 reward
finalization watermark 均通过校验的窗口才能进入 actor 更新。
