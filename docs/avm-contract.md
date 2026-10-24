# AVM 架构与验收契约

## 定义与来源

AVM（Audio-Video Model without Text）是基于 LatentLoop 裁剪动作输出后的通用全双工
实时 Omni 对话模型。输入为单路混合音频、一路视频及时间，输出为语音 PCM。
不设置文本输入输出头，不以 ASR 或 TTS 串联推理；视频中的文字和离线文本工具不受此限制。
不包含电脑操控、工具调用、视频生成或以 GUI 为前提的任务环境。

## 状态与输出

80 ms、24 kHz mono、1920 samples/unit，Mimi 8×2048。保持 LatentLoop 的 InputEncoder、
Backbone、C/H、逐层 RecentKV/SlowMemory、audio cache 与 speech-local state。
Canary KV 上限为 375 units。跨 chunk 连续、TBPTT 边界 detach，会话边界 reset。
仅保留 Speech Head；SILENCE 输出零 PCM，SPEECH 直接生成 codec tokens。
JEPA 是训练分支，Value Head 仅在 Training 使用，不作为服务输出。

## 物理协议与实时运行

ObservationSignal = session/unit/time + mic PCM + VideoSignal；ActuationSignal = speech PCM。
VideoSignal 包含图像、尺寸、格式、采集时间、revision。每 unit 使用不晚于当前时刻的
最新帧，低帧率复用 revision。首帧未到等待；断连、未来帧、时间倒退和超时显式失败。
摄像头、视频流和屏幕共享使用同一接口。模型发声期间持续采集与推理，播放队列有界，
静音决策清除未播放语音。播放回流进入后续混合麦克风，不另设参考通道。
reward、环境状态、字幕和设备来源元数据不输入模型。

## 数据与训练

统一 train dispatcher 与 recipe 执行 Pretrain → SFT → Online Recurrent PPO。
监督损失为 speech mode CE + masked codec CE + 既有 JEPA；无动作 loss 或独立记忆 loss。
数据覆盖自然对话、静默、插话、重叠、视觉指代、音视频事件和跨时间记忆。
缺失标签不同于监督静音。视频与语音资产、codec 和内容哈希必须锁定。

RL 保持单生命期实时音视频对话环境、冻结 SFT reference、time-discount GAE、窗口封存、
全模型策略梯度与 SFT replay。概率只含 mode 和 SPEECH 时的 codec 联合概率。
奖励规范 avm-dialogue-v1：task + 0.2 × (0.5 × speech_quality + 0.5 × latency_quality)
+ safety_quality。Judge 只读可感知音视频与时间；模型语音通过播放回流进入证据。
正式环境及 Judge 未配置即失败，不以离线 rollout 或 fixture 替代。

## 测试设计与验收

1. 模型参数、状态、输出、loss、配置和 metrics 无动作字段；梯度到达编码器、主干和记忆。
2. 保留状态连续性、KV 上限、SlowMemory、TBPTT detach、JEPA 边界及 checkpoint 原子性测试。
3. 视频协议往返，低帧率复用、未来帧、倒退、失效和缺失首帧；屏幕与摄像头同一契约。
4. 全双工验证同时采集播放、静默、重叠、被打断停止、恢复与有界背压。
5. 新 schema/codec/hash readiness；旧动作 schema 与旧 checkpoint fail closed。
6. 语音 mode/codec joint probability、静音不含 codec、PPO clipping/KL/GAE 和全模型梯度。
7. 同音频不同视频的目标区分与记忆消融；不以 fixture 或短跑宣称质量收敛。
8. 完整 pytest、workspace Ruff/compileall、shell 语法、diff 检查、inspect-model 和 readiness。

建立 AVM 新身份，不直接续训 LatentLoop checkpoint。显式迁移仅提取共有模型权重，
记录源 SHA-256 并建立新 run；不迁移优化器、动作状态、旧 rollout 或会话。
实际 Canary 验收需新 run、完整三阶段、逐阶段 validation 和最终 test；报告更新数、units、
监督密度、时间、显存与 W&B 模式。资产缺失保留失败，代码验收与真实训练验收分开记录。
