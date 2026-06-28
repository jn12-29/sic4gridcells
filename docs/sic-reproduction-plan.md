# SIC Grid Cells 复现计划

## 范围

本计划面向 Schaeffer et al. 2023, "Self-Supervised Learning of Representations for Space Generates Multi-Modular Grid Cells" 的从论文材料复现。依据材料为 `/mnt/sata2/xh/ai4neuron/gridCells/docs/sic` 下的 PDF 和 arXiv LaTeX 源码包。

arXiv 源码已解压到 `.work/sic-reproduction/source/` 供检查；该目录是工作材料，不是项目源码。

## 材料依据

- 论文把 SIC 定义为由训练数据、数据增强、损失函数和 RNN 架构组成的自监督学习框架（`04_ssl.tex:5-6`）。
- 每个梯度步采样一条长度为 `T` 的速度序列，再用 `B` 个随机置换构造 batch，并从共享初始状态送入所有置换轨迹（`04_ssl.tex:9-18`）。
- RNN 转移为 `W(v_t) = MLP(v_t)`，其中 `W: R^2 -> R^(N x N)`，状态更新为 `g_t = Norm(ReLU(W(v_t) g_{t-1}))`（`04_ssl.tex:21-37`）。
- 损失项包括 separation、path invariance、capacity 和 conformal isometry（`04_ssl.tex:54-96`）。
- 附录超参包括 batch size 130、trajectory length 60、速度 `Uniform^2(-0.15, 0.15)` meters、128 个 RNN units、3 层 MLP、`sigma_x=0.05`、`sigma_g=0.4`、`lambda_sep=1.0`、`lambda_inv=0.1`、`lambda_cap=0.5`、AdamW、ReduceLROnPlateau、learning rate `2e-5`、gradient clip `0.1`、gradient accumulation 2、`2e6` 个梯度步（`07_appendix.tex:8-33`）。
- 论文报告的目标现象包括多个离散 grid modules、对 2 m/3 m/4 m arena 的泛化、Fourier/period/orientation 结构、phase tiling、toroidal state-space structure，以及对 loss、置换增强和 `sigma_g` 的消融（`05_results.tex:13-60`，`07_appendix.tex:44-87`）。

## 假设与未知项

- 当前项目中未发现 SIC 官方训练代码；本地材料只包含论文源码和图。
- 附录表没有列出 `lambda_coniso`。先把它作为实验参数处理，默认候选可设为 `1.0`，再通过 sweep 或 ablation 校准。
- MLP hidden width、初始化、scheduler patience/factor、`Norm(ReLU)` 遇到 dead unit/zero vector 的精确处理、checkpoint 频率、evaluation trajectory generator 都没有在材料里明确给出。实现时需要在 config 中显式记录默认值。
- 论文使用 PyTorch 和 PyTorch Lightning；复现项目先用 plain PyTorch 更便于审计 loss、batch 构造和显存控制。除非训练编排变复杂，否则不引入 Lightning。
- 论文规模训练很重：`B*T = 7800` 时，全 pairwise loss mask 约 6000 万对。实现必须支持 chunking 或 pair sampling；小型 smoke run 只能证明代码链路可运行，不能证明论文级复现。

## 环境计划

在当前仓库内使用独立 uv 环境：

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python torch numpy scipy matplotlib pyyaml tqdm tensorboard pytest pillow scikit-learn
```

推荐项目设置：

- `pyproject.toml` 设置 `requires-python = ">=3.12,<3.13"`。
- 源码包放在 `src/sic4gridcells`。
- 生成数据、checkpoint 和图放入已忽略的 `data/`、`results/` 或 `outputs/`。
- 长跑前用 `nvidia-smi` 检查 GPU，并通过 `CUDA_VISIBLE_DEVICES=<id>` 显式选择可用 CUDA GPU。

## 项目骨架

```text
sic4gridcells/
├── pyproject.toml
├── configs/
│   ├── smoke.yaml
│   ├── sic_paper.yaml
│   └── ablations.yaml
├── src/sic4gridcells/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── analysis.py
│   └── plotting.py
├── scripts/
│   ├── train_sic.py
│   ├── eval_checkpoint.py
│   └── run_ablations.py
└── tests/
    ├── test_data.py
    ├── test_model.py
    ├── test_losses.py
    └── test_analysis.py
```

## 实施计划

### 阶段 1：可测试核心

目标：先让 SIC batch、模型和 loss contract 在不长时间训练的情况下可测试。

1. 实现 config 加载。
   - 验证：`pytest tests/test_config.py` 检查默认值和 paper config。
2. 实现 SIC velocity batch generation。
   - 每步采样形状为 `(T, 2)` 的 base velocities，分布为 `Uniform(-0.15, 0.15)`。
   - 生成 `B` 个随机置换和对应的 permuted velocity sequences。
   - 从同一原点计算每条置换轨迹的 cumulative positions。
   - 验证：置换保持速度 multiset；所有轨迹终点在浮点误差内一致。
3. 实现 `NormReLU`。
   - 使用带 epsilon 的安全归一化，并记录 zero vector 行为。
   - 验证：非零输入输出非负且 unit norm。
4. 实现 `MLP(v) -> W(v)` 和 recurrent rollout。
   - 尽量避免无必要地保存每个 `W(v)`，但行为要忠实于公式。
   - 验证：输出形状为 `(B, T, N)`，梯度可回传，smoke 设置下无 NaN。
5. 实现 losses。
   - Separation：空间远距离 pair 惩罚神经表征过近。
   - Invariance：空间近距离 pair 惩罚神经表征距离。
   - Capacity：mean representation 的 squared norm 取负。
   - Conformal isometry：对小的非零速度计算 `||g_t-g_{t-1}|| / ||v_t||` 的方差。
   - 验证：用确定性 toy cases 检查 mask、符号、有限值，以及 tiny tensor 上 chunked 结果与 naive all-pairs 一致。

### 阶段 2：训练循环

目标：先跑可控 smoke 和中等规模训练，再进入论文规模。

1. 构建 plain PyTorch training loop：AdamW、gradient clipping、gradient accumulation、ReduceLROnPlateau、checkpoint、TensorBoard logging、config snapshot。
   - 验证：CPU 和 GPU 上 10-step smoke run 都能完成。
2. 增加显存安全的 pairwise loss 计算。
   - exact 小/中规模设置先用 blockwise pair computation。
   - pair sampling 只作为 quick experiments 的明确近似选项。
   - 验证：tiny examples 上 exact blockwise loss 与 naive all-pairs loss 一致。
3. 跑中等规模 sanity experiment。
   - 示例：`B=16`、`T=30`、`N=64`、数千步。
   - 验证：loss terms 有限、checkpoint 可 reload、ratemaps 可生成。
4. 跑 paper config。
   - `B=130`、`T=60`、`N=128`、论文 loss weights、gradient accumulation 2。
   - 验证：记录 GPU memory、throughput、coverage/zero/invalid/active response counts 和周期性 checkpoint。

### 阶段 3：评估与图目标

目标：判断学到的表征是否对应论文主张。

1. Ratemap generation。
   - 在 2 m、3 m、4 m box 里评估 hidden states。
   - 评估轨迹使用更平滑的 bounded trajectories，与训练用 i.i.d. uniform velocities 分开。
   - 验证：导出全部 128 units 的 ratemap PDFs。
2. Gridness 和 module detection。
   - 复用或 vendor `/mnt/sata2/xh/ai4neuron/gridCells/grid-cells-torch/grid_cells/analysis/scores.py` 的 scoring 逻辑，不临时手写 grid score。
   - 计算 spatial autocorrelograms、grid scores、grid scales、orientations 和 scale clustering。
   - 验证：成功 run 的 histograms 显示离散 periods。
3. Generalization。
   - 输出大 arena 下的 ratemaps 和 pairwise neural-distance plots。
   - 验证：2 m、3 m、4 m evaluation 中 periodicity 保持稳定。
4. State-space analysis。
   - 对有足够多 co-modular units 的 module 计算 Fourier peaks、沿 lattice vectors 的 phase distributions，PCA 到 6D 后用 spectral embedding 或 Isomap 到 3D。
   - 若标准 SIC run 中同模块 units 不足，使用多数 units 同周期的对照 run 做 Fig. 3 风格分析，并在结果中标明来源。
   - 验证：phase tiling 和 toroidal/ring projections 定性匹配论文。

### 阶段 4：消融

目标：验证论文中的因果主张。

用匹配 seeds 和训练预算运行：

- No capacity loss：预期转向单一 grid module。
- Reduced `sigma_g`：预期出现 place-cell-like responses，field size 随 `sigma_g` 改变。
- No separation loss。
- No invariance loss。
- No conformal isometry loss：附录报告仍有多个 modules，但 lattice 更 sheared、更接近 square。
- No permutation augmentation：预期失去 spatial tuning。

验证：

- 所有消融使用同一 evaluation pipeline 和 plots。
- 保存 summary table：grid-score distribution、detected scale clusters 数量、coverage/zero/invalid/active response counts、代表性 ratemaps。

## 邻近代码复用计划

`/mnt/sata2/xh/ai4neuron/gridCells/grid-cells-torch` 只能作为参考和工具来源，不应当直接当作 SIC 实现。

适合复用或 vendor 的部分：

- `grid_cells/analysis/scores.py`：rate maps、SAC、grid scores。
- `grid_cells/analysis/spatial_stats.py`：split-half reliability、shuffle significance、scale summaries。
- 评估输出中的 plotting conventions。

不要复用它的 supervised Banino training loop 作为 SIC training loop。该项目学习 place-cell/head-direction targets，而 SIC 必须避免 supervised position targets。

## 验收标准

最小复现：

- uv 管理的 package 可以本地安装。
- data generation、model rollout 和所有 loss terms 的单元测试通过。
- smoke training run 输出有限 losses、checkpoints 和 ratemaps。
- 中等规模 run 至少在部分 units 中出现 periodic spatial tuning。

论文主张复现：

- paper-config run 产生多数 periodic units，并出现论文主图级别的离散 grid-scale modules；若先用定性验收，必须同时保存 grid-scale histogram 和 scale-clustering 判据。
- 至少用多个 seeds 和小范围 hyperparameter sweep 检查关键结果是否稳健，覆盖 `sigma_x`、`sigma_g`、`lambda_sep`、`lambda_inv`、`lambda_cap` 的论文报告范围。
- 2 m、3 m、4 m evaluation 的 ratemaps 保持稳定 periodicity，并输出 pairwise neural-distance vs spatial/temporal separation；空间 decorrelation length 应接近 `sigma_x`，zero spatial separation 检查用于评估 path integration。
- Fourier/phase/state-space analyses 显示与 Fig. 3 可比的 module-level structure；该分析需要足够多 co-modular units，必要时使用多数 units 同周期的对照 run。
- Ablations 复现 Fig. 4 和 appendix ratemaps 描述的定性变化。

## 当前下一步

1. 运行中等规模 sanity experiment，并用 `scripts/eval_checkpoint.py` 生成 ratemaps、SAC、grid stats 和 figures。
2. 检查 evaluation 输出的 `module_summary.csv/json`、`grid_metrics.npz`、`pairwise_distance_stats.csv/json`、`pairwise_distance.png`、`fourier_stats.csv/json`、`phase_summary.csv/json`、`state_space_summary.csv/json`、`trajectory_stats.json`、grid-score histogram 和 scale histogram。
3. 用 `configs/ablations.yaml` 和 `scripts/run_ablations.py` 执行 matched ablations，包括 no permutation augmentation。
4. 按 `docs/runbook.md` 记录 paper-scale GPU memory、throughput、checkpoint 和 evaluation cadence。
