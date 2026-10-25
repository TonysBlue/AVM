# 原生迁移本地验证：2026-10-25至26日

## 范围

本记录区分真实原生模块验证、显式小维度协议测试和尚未完成的门禁。
目标配置见[原生架构契约](../minicpm-native-avm.md)，声学测量见
[声学资格实验](native-acoustic-qualification.md)。本机为RTX 2080 SUPER 8 GiB，
不能据此证明8×A800的容量、吞吐或质量。

## 已有证据

- 四个官方权重分片均已下载，完成LFS SHA-256、普通源码Git blob hash和完整来源锁审计。
  严格初始化导入1,406个tensor、7,983,569,810个原生参数，Core合计8,169,568,767。
  完整来源锁hash为`e7b2896c53ca5a1350b31e15db33fc11a55891f2bf2ba60e7e8740c12b707a7a`。
- 通过共享Core执行两个连续CPU BF16校准unit，完整模型不缩层、不缩宽。每步音频输出
  `[1,2,4096]`，H/C为`[1,64,4096]`/`[1,16,4096]`，语音logits为`[1,5,1,6562]`；
  主干KV长度64→128、音频10→20、语音69→138，五个输出codes均合法。
  总耗时155.39秒，两个推理步分别47.90/48.95秒；不包含优化器更新、PCM或物理会话，
  不构成实时性与对话质量证据。新AVM模块仍为seed17初始化。
- 最终加入Core源码、探针源码及模型配置指纹后复验通过，总耗时120.57秒，两步分别
  35.26/29.20秒，峰值进程RSS为17,077,215,232 bytes（约15.90 GiB）。该次与首次的
  页面缓存、系统负载不同，不将耗时变化解释为模型加速；完整规模参数始终为BF16。
  最终整模和模块报告还保存于正式来源锁同目录的`verification/core-probe.json`和
  `verification/native-module-parity.json`，由`verification/reports.json`记录报告hash。
- 使用真实权重对照官方Qwen3/Llama/SigLIP层、完整27层视觉encoder和视觉resampler；
  完整原生Speech Head已在CPU和GPU验证前向、反向与teacher forcing/采样缓存一致。
- 共享train入口在CPU/GPU分区的小维度显式fixture上完成一次更新和checkpoint：
  8 units、40个监督帧、监督密度1、4.848秒，峰值分配27,195,904 bytes。
  tracking disabled；这些数值不代表原生模型容量。
- KV精度修正后，通过同一入口在单GPU执行BF16 fixture更新与checkpoint：8 units、
  40帧、监督密度1、5.493秒，峰值分配34,612,224 bytes、保留56,623,104 bytes，
  tracking disabled。参数和AdamW仍为FP32；缓存dtype、跨unit梯度及保存恢复有专项测试。
- 实际FFmpeg/PIL视频适配器将一秒320×180测试视频转换为5帧，输出每帧
  `[3,336,602]`，保留画幅比例且值有限。测试图源不作为正式视频资产。
- 严格导入测试验证完整materialization与梯度，并拒绝缺失或未归类的参数；下载测试
  验证LFS SHA-256和普通源码Git blob hash，拒绝同长度修改。

## 本轮运行限制

本轮沙箱内完整pytest触及12项Unix socket相关测试，初次因系统禁止创建或连接socket
而失败。两次自动审批超时后，后续完整回归获批并实际运行；先前的权限限制已经解除。
新增两次PPO更新用例发现一处测试假设错误：没有新奖励事件的后续窗口不应被要求正奖励。
完整资产工具与FP32视觉顺序修正后，完整回归为259 passed、1 skipped，耗时284.78秒，包含异步checkpoint期间actor
继续推进、窗口归属和边界RNG检查。Ruff、compileall、Shell语法和diff检查均通过。

完整视觉encoder在小尺寸上通过，但448×448随机输入的GPU FP32对照最大误差约0.00154，
超过3e-5门槛。独立448像素对照中embedding误差为0，SDPA输出RMSE约8.23e-6；
改用相同eager运算顺序的控制对照误差为0，定位为注意力浮点运算顺序差异。
随后将FP32视觉注意力对齐原生eager运算顺序，BF16保留SDPA；完整448×448与336×602
编码器和resampler误差均为0，在原3e-5门槛下通过。音频24层stack误差7.63e-6，
Qwen3层约2.62e-6，Speech Head仍通过完整梯度与采样缓存检查。
报告为`/tmp/avm-final-native-eager-parity.json`；原SDPA失败报告保留，未放宽门槛。

原生声学资格仍失败：全静音RMS约0.00384，讲话峰值约1.069；本机解码p95约882 ms，
RTF约2.38。正式readiness继续拒绝运行，且缺少新协议的真实Canary数据及审计。
不得降低门槛、补零、clamp波形或复用旧Mimi资产来改变结论。
完整来源锁下复验及另外两个官方音色片段、FP16 flow对照均失败，详见声学资格实验；
正式来源锁没有被候选覆盖。259项回归通过不代表这些真实资格报告通过。

## 后续门禁

1. 完整下载、来源锁、全资产审计和严格原生初始化已完成；来源锁变化后不能复用旧资格报告。
2. 在允许Unix socket和GPU访问的本机环境执行完整pytest，重点检查单生命期PPO的
   周期checkpoint、采集连续性、边界RNG和恢复游标。本轮完整回归已通过，后续改动仍须复验。
3. 解决声学保真失败并重新资格验证，准备真实Canary数据，保持失败显式。
4. 最后在8×A800验证原生规模更新、峰值显存、持续200 ms截止期和三阶段谱系。
