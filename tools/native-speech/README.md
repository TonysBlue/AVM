# 原生资产与声学资格工具

该子项目下载并审计真实MiniCPM-o资产，核对原生模块数值与梯度，并运行配套
Step-Audio2 tokenizer/Token2Wav，验证200 ms五token连续解码、静音、边界与留尾。
没有fixture或Mimi回退，也不承担正式训练循环。
Torch/Torchaudio固定同一版本，与主workspace环境隔离；现有栈不含这套配套声学实现，
因此在此单独锁定 `stepaudio2-minicpmo` 及其依赖，不将它们混入主训练环境。

安装命令：

```bash
uv sync --project tools/native-speech --locked
```

`huggingface-hub`在此锁定，用于不可变revision下载与LFS SHA-256核验。资产目录必须在
仓库外；完整下载和来源锁生成命令为：

```bash
tools/native-speech/.venv/bin/python tools/native-speech/fetch_assets.py --help
tools/native-speech/.venv/bin/python tools/native-speech/verify_model.py --help
```

`verify_model.py`读取实际原生权重，对照官方模块并执行完整Speech Head反向验证；
分模块通过不代表完整主干加载、对话质量或8卡训练已经通过。
选择CUDA时还核对448×448与非方形完整视觉encoder，并记录eager运算顺序控制对照。
FP32视觉保留原生eager顺序，BF16仍用SDPA；任一严格FP32绝对误差不达标时保留
JSON报告并返回2，不能用控制对照替代生产结果。完整来源锁下还检查真实音频24层stack。

完整Core的CPU memory-mapped BF16校准探针使用主workspace环境，保持Canary原生维度：

```bash
uv run --no-sync python tools/native-speech/verify_core.py \
  --config configs/canary.yaml --output /tmp/avm-complete-native-core-probe.json
```

至少连续执行两个unit并验证所有层的缓存推进；不启动优化器或正式物理会话，
`core_probe_passed`不代表声学资格、训练容量或实时性。未适配的新AVM模块仍为种子初始化。

`worker.py`由共享runtime客户端启动，不直接接收设备；固定音色是锁定资产，不是额外的
麦克风输入。运行时必须有CUDA GPU、真实模型资产和源码锁。不要把它替换成上游同名但
不同包的Token2Wav，也不要用缺失依赖时的空音频替代错误。

## 源码锁

`data inspect-native-assets --source-lock /absolute/path/source-lock.json`使用共享Data
审计模块；路径指向仓库外的真实来源锁，不会下载或伪造权重。JSON根字段为：

声学资格验证只审计真实声学资产与原生config，不要求先下载9B主干；可使用
`inspect-native-assets --acoustic-only`检查这一范围。结果明确标记
model_weights_audited=false，不计主干参数、不证明完整模型可用。默认检查仍要求全部shards。

| 字段 | 要求 |
|---|---|
| schema_id | `avm-native-source-lock-v1` |
| repository | `openbmb/MiniCPM-o-4_5` |
| revision | 真实40位model commit，不接受main |
| license | 来源许可证 |
| root | 模型本地目录，或相对来源锁的目录 |
| files | 相对文件路径到真实SHA-256的映射 |
| speaker_reference | files中锁定的音色参考音频路径 |
| acoustic_runtime | 下述独立声学实现身份 |

模型files必须覆盖config、权重索引、全部索引shards、tokenizer、flow、vocoder、说话人
模型和固定音色。额外配置/源码也应锁入，不能仅锁大权重而放任配置变化。

acoustic_runtime字段为root、license、revision、python、entrypoint和files。python是
本子项目虚拟环境Python的绝对路径，entrypoint是实际Token2wav类定义文件相对root的
路径；所有该root下Python源码必须在files中锁定。revision可为真实40位源码commit；
也可为 `stepaudio2-minicpmo==0.1.1`，此时package_archive必须指向files中锁定的原始
wheel，SHA-256必须与发布件一致。审计和worker都会检查实际导入的类文件身份。

## 真实资格语料

语料清单schema_id为 `avm-native-speech-corpus-v1`。clips至少包含speech、silence和
transition三种kind；各项记录path、sha256、license，transition还记录
silence_intervals_ms区间。路径相对语料清单，音频为16 kHz mono且包含完整200 ms单位。
speech至少2秒、silence至少10秒、transition至少4秒；编码连续音频后再按五token分组。

清单thresholds必须明确silence_rms_max、boundary_jump_max、decode_p95_ms_max及rtf_max。
200 ms解码截止期和RTF<1不能放宽。报告只证明该声学资产的形状、静音、连续性及吞吐
资格，不证明AVM对话自然度，也不替代完整端到端或8卡训练验收。

```bash
uv run --no-sync data qualify-native-speech \
  --source-lock /absolute/path/source-lock.json \
  --corpus /absolute/path/corpus.json \
  --output /absolute/path/qualification
```

输出目录必须在Git仓库外。报告保存来源锁hash、语料hash、原生前瞻、实际vocoder留尾、
解码时间、样本数、边界差值及重建音频hash。失败返回非零，不生成通过声明。
首次底层调用返回不足200 ms是可能的；只有真实PCM累计足够后才交付完整块，不能补零。
