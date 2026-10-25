# AVM — Audio-Video Model without Text

原生架构：[MiniCPM-o 4.5 AVM](docs/minicpm-native-avm.md)，200 ms、16 kHz输入、
连续五声学token、无发声分类头，Model Core约8.17B，含冻结声学栈约8.48B。
正式资源为8×A800 80GB；本地持续验证
真实原生模块和完整协议，未通过声学时延与数据readiness时正式运行显式失败。

AVM 是面向通用全双工实时 Omni 对话交互的音视频模型研究与训练项目。模型架构基于
LatentLoop，移除电脑动作能力，保留连续音视频感知、循环状态、长期记忆与直接语音生成。
模型按固定 200 ms 节拍持续接收混合音频和视频，边听、边看、边说，自主决定回应、静默或停止发声。
运行时不以文本为输入输出通道，也不经过 ASR→文本模型→TTS 链路。

当前 Omni 的范围是音视频输入、语音输出，不包含电脑操控、工具调用或视频生成。
视频可来自摄像头、一般视频流或屏幕共享；GUI 不构成模型、数据或训练环境的前提。
底层保留有界 RecentKV、语义 slots `C_t`、SlowMemory、直接 codec Speech Head、
递归重计算、精确恢复、WebDataset 和 W&B Local。

完整架构见 `docs/realtime-multimodal-latent-loop.md`，本地训练设计见
`docs/local-training-platform.md`。直接流式语音的实现与运行命令见
`docs/direct-speech.md`。原 LatentLoop 动作协议已退出 AVM。

实时摄像头/视频对话、播放回流、真实视频来源锁与旧模型权重迁移的命令见
[`AVM 实时运行手册`](docs/avm-runtime.md)。正式数据与设备缺失时显式失败，不回退测试实现。

## 环境初始化

```bash
scripts/bootstrap.sh
uv run data inspect-model --config configs/smoke.yaml
uv run pytest
```

`data inspect-model` 会验证配置并给出模型参数和 KV 规模。快速 CPU fixture 与正式 Canary
容量配置分别用以下命令确认：

```bash
uv run data inspect-model --config configs/smoke.yaml
uv run data inspect-model --config configs/canary-gpu-smoke.yaml
uv run data inspect-model --config configs/canary.yaml
```

Canary与GPU capacity smoke使用完全相同的8,169,568,767参数和150-unit horizon。
容量smoke需要目标8卡，执行一次完整forward、backward、AdamW update与checkpoint；
本机8GB GPU只验证原生模块和显式小维度协议，不能证明完整模型训练容量：

```bash
uv run training train --config configs/canary-gpu-smoke.yaml
```

`configs/smoke.yaml` 只是不代表容量的快速 CPU/unit-test fixture。

## 数据与训练

生成并校验合成 WebDataset：

```bash
uv run data generate-data \
  --config configs/smoke.yaml \
  --output "$HOME/latentloop-data/datasets/generated/train-%06d.tar"

uv run data validate-data \
  --config configs/smoke.yaml \
  --shards "$HOME/latentloop-data/datasets/generated/train-*.tar"
```

原生语音过拟合门禁使用 `configs/direct-speech-overfit.yaml`，完整命令和本机
验收结果见 [`docs/direct-speech.md`](docs/direct-speech.md)。

使用 Ray 的 CPU worker 生成数据：

```bash
uv run data generate-data \
  --config configs/smoke.yaml \
  --output "$HOME/latentloop-data/datasets/generated/train-%06d.tar" \
  --ray
```

运行 smoke 训练或恢复当前 checkpoint：

```bash
uv run training train --config configs/smoke.yaml
scripts/run-training.sh \
  --recipe configs/recipes/smoke.yaml \
  --run-id smoke-tracking-001 \
  --resume "$HOME/latentloop-data/checkpoints/smoke/step-00000002.pt"
```

训练数据默认写入 `~/latentloop-data/datasets`；当前实验、checkpoint、W&B 和运行时文件分别
写入 `experiments/`、`checkpoints/`、`tracking/` 和 `runtime/`，不保留历史运行产物。

## W&B Local

```bash
scripts/wandb-local.sh up
scripts/wandb-local.sh status
```

服务只监听 `http://127.0.0.1:8080`。首次启动需要完成数据库迁移；脚本以 `/ready`
为就绪条件。打开 UI 获取本地 API key 后执行：

```bash
uv run wandb login --host http://127.0.0.1:8080
uv run training train \
  --config configs/smoke.yaml \
  --set tracking.enabled=true \
  --set tracking.mode=online
```

服务不可用或未登录时，Tracker 自动写入本地 offline run。服务管理命令：

```bash
scripts/wandb-local.sh logs
scripts/wandb-local.sh down
```
直接流式语音使用200 ms主时钟、原生S3 25 Hz声学token和24 kHz输出。实现和运行命令见
[`docs/direct-speech.md`](docs/direct-speech.md)。

Canary 数据准备、外部语料许可证锁、离线 CosyVoice/ASR、真实视频适配器及审计门禁见
[`docs/canary-data.md`](docs/canary-data.md)。本地 fixture 可以在不下载语料和模型的
情况下跑通准备链：

真实一小时Canary的固定数据下载、TTS/ASR、原生S3编码、训练和validation/test评测
使用统一入口，完整说明见 [`docs/canary-runbook.md`](docs/canary-runbook.md)：

```bash
scripts/prepare-data.sh canary all
scripts/run-training.sh --recipe configs/recipes/canary.yaml --run-id canary-001
```

```bash
uv run data fetch-canary-data --config configs/local-dev.yaml --fixture
uv run data select-canary-voices --config configs/local-dev.yaml --fixture
uv run data build-canary-text --config configs/local-dev.yaml --fixture
uv run data synthesize-canary --config configs/local-dev.yaml --fixture
uv run data build-canary-manifest --config configs/local-dev.yaml --fixture
uv run data audit-canary-data --config configs/local-dev.yaml --fixture
```
