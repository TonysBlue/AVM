# 物理 Rollout 闭环

Online Recurrent PPO 的正式 rollout 使用 Harness 作为唯一环境边界。训练进程在 GPU 上运行
Model Core，Harness 进程负责采集 ObservationSignal、播放 ActuationSignal，并返回
receipt；独立 Reward Judge 产生只基于观察的 Reward Event。训练进程不再向环境发送 speech
token。

```text
Harness.start_lifetime_session -> ObservationSignal
Model Core.forward -> Speech mode + codec tokens
Training codec decoder -> ActuationSignal
Harness.apply(ActuationSignal) -> ObservationSignal + EnvironmentReceipt
Observation timeline -> frozen perceptual Reward Judge -> RewardEvent
```

模型内部的 speech token、old/reference log-prob 和 advantage 只保存在训练 trace；物理 socket
只传输 protobuf ObservationSignal、ActuationSignal 和控制面的 receipt。Reward Judge socket
只接收 canonical ObservationSignal bytes 与其 hash-chain identity，不接收 receipt、action、
task manifest、DOM 或隐藏 evaluator 字段。

正式 Harness control socket 必须校验 environment identity、session 和 unit。连接失败、设备失败、设备断连或 receipt 不完整时，rollout 标记为
infrastructure failure，不作为低 reward 样本更新策略。

## 会话和时钟约束

`start_lifetime_session(scenario_id, seed, session_id)` 为生命期会话创建操作；普通任务完成不会 reset。
训练暂停后的 `resume_lifetime_session(session_id, expected_next_unit)` 只能返回该 session 当前
未消费 observation，不创建 backend、不恢复电脑快照，也不改变 unit cursor；session 不存在或
cursor 不一致时 fail closed。
空 scenario/session、错误 session、重复或跳跃 unit 必须 fail closed。
视频按采集时间和 revision 选择最新因果帧，超时与断连明确失败。

生命期创建返回 unit 0；unit N 的 ActuationSignal 只能作用于同一 session 的 unit N
ObservationSignal，并返回 unit N receipt 和 unit N+1 ObservationSignal。80 ms 音频单元固定为
24 kHz 单声道，即 float32 PCM 为 7,680 bytes，int16 PCM 为 3,840 bytes。Harness 不接受
任意长度 PCM 来掩盖音频设备或 codec 错误。

EnvironmentReceipt 使用与物理 signal 相同版本的 protobuf schema，外层 control request
只负责 operation、session identity 和 length framing。普通任务完成不终止 lifetime session；
`infrastructure_failure` 表示样本不可用于策略更新，且不进入下一轮 ObservationSignal。

## 生产 backend

正式 backend 是持续音视频对话 session，部署模块为 harness.deployment.dialogue。
FFmpeg 采集摄像头/视频/屏幕共享和 PulseAudio 混合音频，pacat 在独立队列播放语音。
必须配置真实输入、播放回流与 scenario_id，不回退静音、空画面或 fixture reward。
close 与启动失败清理设备进程/线程，采集和播放队列保持有界。

## 契约测试设计

契约测试必须覆盖 protobuf round-trip、模型输出张量不跨物理边界、80 ms PCM 校验、identity 查询不
泄漏 backend、重复 reset 关闭旧 backend、未知 session、错序 unit、错 session、
backend 返回错误 identity、server shutdown 清理全部 session，以及 receipt protobuf
round-trip。设备生命周期测试验证视频因果时序、播放打断、持续采集、队列上限和资源回收。

端到端 Online Recurrent PPO 测试使用 fake codec worker 与实现同一严格协议的进程内 Harness，
调用正式 `train_online_ppo` 和 `PhysicalRolloutClient`。测试使用一个 lifetime session、多个
连续 RewardEvent，并至少完成一次 optimizer update。它必须断言：

1. speech token 经正式 worker 解码为恰好一个 80 ms PCM，Harness 收不到 raw token tensor。
2. reset、apply、receipt 和下一轮 observation 的 session/unit 严格递增，
   speech PCM 经过 protobuf round-trip。
3. receipt、任务成功、DOM 和 evaluator 字段不能发送给 Reward Judge，也不能出现在下一轮
   `ObservationSignal` 或模型输入张量；Reward Event 单独保存在训练元数据。
4. `infrastructure_failure`、设备断连、错序 unit 或 播放拒绝 不产生 advantage，也不
   参加策略梯度；只有 finalization watermark 覆盖且 identity 正确的 sealed window 才更新。
5. 更新后的 checkpoint 记录 schema/codec/environment/reward identity、reward 分量、
   有效 unit 数和实际 optimizer update。

真实部署验收还要在摄像头/视频、混合音频、实际播放回流和独立 Reward Judge endpoint
上运行同一矩阵；本机缺少外部设备时必须明确标记未执行，不得用 fixture 结果声称闭环通过。
