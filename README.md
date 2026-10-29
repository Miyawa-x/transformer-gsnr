# Transformer GSNR

可以在视觉 Transformer 上用 GSNR 调节训练。GSNR 衡量的是参数梯度的信噪比，卷积层、注意力层、MLP 都可以测。它在这里负责决定动量，不替代分类或蒸馏损失。

这是独立的研究原型仓库。已有小型 CPU 正确性测试；尚无 CIFAR-100 正式训练结果，不能声称比 AdamW 更准或收敛更快。当前模型是自定义 pre-LayerNorm ViT，不是官方 DeiT 复现。

## 已实现

- 小型 ViT：32×32 图像、4×4 patch、宽度 192、6 个 block、3 个 head、100 类；共 2,710,948 个学生参数。
- 按 embedding、每个 block 的 attention / MLP / 两个 LayerNorm、最终 LayerNorm、分类头划分参数，特征 adapter 单独成组。
- 纯分类 CE、原 DeepKD 三路 TOG/TCG/NCG 蒸馏、附加 CLS 特征蒸馏三种目标。Feature KD 用显式 Linear adapter 对齐通道，不宣称 GSNR 自动完成异构对齐。
- 每 10 个 epoch 校准一次分层 GSNR，同时保存绝对值、原始/去偏/截断后的信号、噪声、下降率、动量和平均增益代理。
- 固定动量、GSNR 自适应 Heavy-Ball、AdamW 三种入口；9 份配置只是起始模板，没有启动 sweep。
- 模型、adapter、独立动量缓冲器、随机状态、历史统计的保存与恢复；输入/代码/教师校验；同目录互斥锁与状态记录。

原版损失、优化器和状态隔离代码以未修改快照复用，来源和 SHA256 在 [UPSTREAMS.json](UPSTREAMS.json)。旧项目源码未改变。

## GSNR 怎样影响 Transformer

对一个固定参数状态，取 K 个相同大小的小批次，各自计算某层某条梯度流 `g_b`：

```text
noise = sum_b ||g_b - mean(g)||² / (K - 1)
signal_raw = ||mean(g)||² - noise / K
signal = max(signal_raw, 0)
gsnr = signal / (noise + 1e-12)
mu = mu_min + (mu_max - mu_min) * gsnr / (gsnr + kappa)
v_t = mu_t * v_(t-1) + g_t
theta_(t+1) = theta_t - lr * sum_streams(v_t)
```

分子去偏需要独立、同分布的梯度观测假设；比值本身不是无偏估计。参数在整个校准窗口不更新，图像从校准集有放回独立抽取并固定复用。不能把不同训练步的梯度直接当作同一个固定分布，也不能把同图的 patch/token 当独立图像样本。

校准时关闭 dropout，恢复原模式、已有 `.grad`、模型状态和 RNG；因此测的是固定参数下的数据噪声。它不测 dropout 噪声。LayerNorm 没有 BN 那样的跨样本统计，这使固定参数的小批次比较更容易解释，但不保证高 GSNR 就更有用。

GSNR 对整条梯度的统一缩放近似不变，但对各参数不同的重缩放/预条件一般不保持不变。因此不同层的数值不能直接当作“哪层含有更多语义信息”的排行榜；AdamW 更新空间中的 GSNR 也不能与原始梯度 GSNR 混报。

Feature 流不经过分类头，分类头的 feature-GSNR 标记为未激活，保持原动量，不拿缺失梯度伪造高可靠性。校准不包含 L2 和梯度裁剪。当前训练固定 FP32、无 AMP、无 clipping、默认 dropout=0，尚未验证 AMP/DDP。

## 本地验证

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python scripts/frozen_diagnostic.py
.venv/bin/python scripts/check_momentum_theory.py
```

测试使用微型随机张量，验证代码正确性；不训练真实数据。`frozen_diagnostic.py` 不执行任何优化器更新，教师、图片和标签都是合成的。报告见 [reports](reports)。正式数据训练入口拒绝 CPU，遵守原项目的服务器训练约定。

## 服务器运行入口（未执行）

本仓库在 `transformer-gsnr/`，不属于原 `reproduction/` 的 Mutagen Alpha，不能认为已经同步。恢复服务器连接后，显式传输本仓库到获准项目目录内的新子目录，排除 `.git`、`.venv`、数据、权重和输出，核对源文件哈希，先做 GPU 冒烟，再提交可靠后台任务。不要把 Mac 的 `.venv` 复制过去。

```bash
# 在分配到的 GPU 环境，使用该环境的 Python；先准备 CIFAR-100 数据。
python -m transformer_gsnr.train \
  --config configs/ce_fixed.json --data /approved/project/data \
  --output /approved/project/outputs/vit_ce_fixed_seed0 --device cuda:0
```

CUDA 设备必须由调度器或 `CUDA_VISIBLE_DEVICES` 限定为已分配的卡。上面的 `/approved/project` 是待按已核实目录替换的占位符。首次数据下载只能在允许的准备阶段显式加 `--download`；默认不会下载。

纯 CE 先验证优化器能稳定训练。蒸馏入口还必须提供一个本仓库格式、真实训练过的 ViT 教师导出文件：

```bash
python -m transformer_gsnr.train \
  --config configs/kd_adaptive.json --data /approved/project/data \
  --output /approved/project/outputs/vit_kd_adaptive_seed0 \
  --teacher /approved/project/checkpoints/teacher_export.pt \
  --teacher-sha256 VERIFIED_SHA256 --expected-teacher-top1 MEASURED_FRACTION \
  --device cuda:0
```

`VERIFIED_SHA256` 和 `MEASURED_FRACTION` 必须替换为真实记录。入口核对类别顺序、数据划分和教师权重，实际评估教师，允许误差 0.2 个百分点；不接受随机教师冒充正式蒸馏。当前没有可用且已验收的本任务 ViT 教师。旧 ResNet32×4 教师不能直接塞入这个 ViT 专用导出接口。

数据划分固定为 CIFAR-100 官方训练集中的 40,000 张训练、5,000 张验证、5,000 张校准；官方测试集不参与调参，当前入口也不报告测试精度。教师和学生使用相同划分。此口径不同于旧 DeepKD 的 50,000 张训练，不能直接对比旧论文表格。

`last.pt` 保存完整恢复状态，`teacher_export.pt` 是最佳验证模型的纯权重导出；只对自己生成的 `last.pt` 使用 `--resume`。代码、配置、划分或环境变化会拒绝恢复。

## 目前最重要的限制和下一步

1. **不能直接把 AdamW 的 beta1 改成动态数值。** 其偏差修正、二阶预条件和有效更新空间都要重新定义。本版保留原未归一化 Heavy-Ball；AdamW 作为独立对照，不称作“GSNR-AdamW”。
2. **高动量也会增加更新幅度。** 固定均值、固定 μ 时平均增益是 `1/(1-mu)`。默认 μ 在 0.5–0.95 之间，增益为 2–20；固定 μ=0.9 的增益为 10。训练改善可能部分来自步长变化。完整论文实验必须加匹配增益/匹配更新尺度对照；当前代码未实现自适应多流的精确增益补偿。
3. **时间结构不能忽略。** 合成梯度回放已展示：输入 GSNR 相同，改变出现顺序，缓冲器 GSNR 仍可差很多。不能把 iid 动量去噪公式无条件用到真实 Transformer 梯度。
4. **当前超参数没有调优。** AdamW 模板 LR=3e-4，SGD 模板 LR=1e-2，weight decay 统一为 0，避免先混入 AdamW 解耦 decay 与 SGD L2 的差别。这不是充分调优的强基线，也不能拿模板直接宣称方法优越。
5. **校准成本须实测。** 三路反传比单次总损失反传贵，统计/隔离还占用内存。正式比较需同时报告训练时间、校准时间、峰值显存、3 seeds，以及各层/各流曲线；不能只报最终准确率。

建议先完成同一 CE 任务的固定动量与自适应动量短跑，确认稳定性和成本；再验收 ViT 教师，做同架构 logit KD；最后打开 feature adapter。全部阶段固定数据划分、种子、batch、模型和增强。此仓库探索不替代主项目 CIFAR-100 ResNet32×4→ResNet8×4 的官方复现主线。
