# SIC Grid Cells 实现计划

## 目标与环境约定

目标是在当前仓库实现一个 plain PyTorch 版 SIC 复现项目，先完成可测试的核心训练链路，再扩展到论文级训练、评估和消融。依据文档是 `docs/sic-reproduction-plan.md` 和 `.work/sic-reproduction/source/` 中的论文 LaTeX 源码。

环境约定：

- 使用 `uv run python` 运行 Python 命令；uv 会从仓库根目录发现本地 `.venv`。
- 项目依赖以 `pyproject.toml` 为权威，安装或刷新依赖时在仓库根目录运行 `uv pip install -e .`。

## 非目标

- 不实现 Banino supervised place-cell/head-direction target 训练。
- 不引入 PyTorch Lightning；除非 plain PyTorch 编排已经成为实际阻塞。
- 不把 `.work/sic-reproduction/source/` 当成项目源码；它只作为论文材料证据。
- 第一阶段不追求论文图完全复刻，只追求核心数据、模型、loss、训练和 smoke evaluation 可验证。

## 项目文件状态

核心训练源码和配置：

```text
pyproject.toml
configs/smoke.yaml
configs/sic_paper.yaml
src/sic4gridcells/__init__.py
src/sic4gridcells/config.py
src/sic4gridcells/data.py
src/sic4gridcells/model.py
src/sic4gridcells/losses.py
src/sic4gridcells/logging_utils.py
src/sic4gridcells/train.py
scripts/train_sic.py
tests/test_config.py
tests/test_data.py
tests/test_model.py
tests/test_losses.py
tests/test_train_step.py
```

评估文件：

```text
src/sic4gridcells/evaluate.py
src/sic4gridcells/validation.py
src/sic4gridcells/analysis.py
src/sic4gridcells/plotting.py
scripts/eval_checkpoint.py
scripts/validate_eval.py
tests/test_analysis.py
tests/test_evaluate.py
tests/test_validation.py
```

实验编排文件：

```text
configs/ablations.yaml
scripts/run_ablations.py
docs/runbook.md
```

## 核心接口与默认决策

### Config

使用 YAML 配置，`config.py` 负责加载、合并默认值、基础校验和保存 effective config。

关键字段：

- `seed`: int。
- `device`: `"cpu"`、`"cuda"` 或 `"auto"`，默认 `"auto"`。
- `output_dir`: 默认 `results/smoke` 或 `results/<timestamp>`。
- `data.batch_size`: paper 为 `130`。
- `data.trajectory_length`: paper 为 `60`。
- `data.velocity_low`: `-0.15`。
- `data.velocity_high`: `0.15`。
- `data.augmentation_mode`: `"permutation"` 或 `"identity"`；默认 `"permutation"`，`"identity"` 仅用于 no-permutation augmentation 消融。
- `data.initial_position_mode`: `"zero"` 或 `"uniform_box"`；默认 `"zero"` 保持共享原点 baseline。
- `data.initial_position_low/high`: 仅在 `"uniform_box"` 时用于采样共享 batch 初始位置。
- `model.n_units`: paper 为 `128`。
- `model.mlp_layers`: paper 为 `3`。
- `model.mlp_hidden_width`: 默认 `256`，论文未指定，必须写入 effective config。
- `model.trainable_initial_state`: 默认 `true`，论文只说明 shared `g0`，未说明是否 trainable。
- `model.initial_position_encoding`: `"none"` 或 `"additive_mlp"`；默认 `"none"` 保持 shared `g0` baseline。
- `model.initial_position_hidden_width`: 默认 `64`。
- `loss.sigma_x`: paper 为 `0.05`。
- `loss.sigma_g`: paper 为 `0.4`。
- `loss.lambda_sep`: `1.0`。
- `loss.lambda_inv`: `0.1`。
- `loss.lambda_cap`: `0.5`。
- `loss.lambda_coniso`: 默认 `1.0`，论文未在附录表列出，必须标为 assumption。
- `loss.pairwise_reduction`: `"sum"` 或 `"mean"`；`sic_paper.yaml` 用 `"sum"` 贴近论文公式，`smoke.yaml` 可用 `"mean"` 保持数值温和。
- `loss.chunk_size`: 默认 `512`。
- `train.optimizer`: `"adamw"`。
- `train.scheduler`: `"reduce_on_plateau"`。
- `train.scheduler_monitor`: `"loss/total"`。
- `train.scheduler_factor`: 默认 `0.5`，论文未指定，必须写入 effective config。
- `train.scheduler_patience`: 默认 `1000` optimizer steps，论文未指定，必须写入 effective config。
- `train.lr`: `2e-5`。
- `train.weight_decay`: `0.0`。
- `train.grad_clip_norm`: `0.1`。
- `train.accumulate_grad_batches`: `2`。
- `train.max_optimizer_steps`: smoke 为 `10`，paper 为 `2000000`；该字段对应论文 “gradient descent steps”，不计入梯度累积内部的 microbatch 数。
- `train.checkpoint_every`: 默认 `1000`，paper 长跑可改大。
- `train.log_every`: 默认 `10`。

### Data

`data.py` 提供：

- `sample_base_velocities(cfg, generator, device) -> Tensor[T, 2]`。
- `sample_permutations(batch_size, trajectory_length, generator, device) -> LongTensor[B, T]`。
- `sample_velocity_orders(cfg, generator, device) -> LongTensor[B, T]`，根据 `data.augmentation_mode` 生成随机置换或 identity 顺序。
- `make_sic_batch(cfg, generator, device) -> SicBatch`。

`SicBatch` 字段：

- `base_velocities`: `(T, 2)`。
- `permutations`: `(B, T)`。
- `initial_positions`: `(B, 2)`；`zero` 模式为全零，`uniform_box` 模式从配置区间采样一个 shared 起点并扩展到 batch。
- `velocities`: `(B, T, 2)`，由 `base_velocities[permutations]` 得到。
- `positions`: `(B, T, 2)`，等于 `initial_positions[:, None, :] + velocities.cumsum(dim=1)`。

必须满足：

- `permutation` 模式下每条轨迹包含同一组 base velocities 的随机顺序；`identity` 模式下每条轨迹保留同一 base velocity 顺序。
- 所有轨迹终点一致。
- `positions[:, t]` 表示执行第 `t` 个 velocity 后的位置。
- 默认 `data.initial_position_mode: "zero"` 保持共享原点 baseline；`"uniform_box"` 只在配合初始位置编码的实验中使用。

### Model

`model.py` 提供：

- `norm_relu(x, eps=1e-8)`：`relu(x)` 后按 L2 norm 归一化；全零向量输出零向量，并在训练 metrics 里记录 zero-norm fraction。
- `VelocityConditionedRNN(cfg)`：
  - `transition_mlp(v_t)` 输出 `(B, N*N)`，reshape 为 `(B, N, N)`。
  - 每个 time step 只 materialize 当前 `W(v_t)`，用 `torch.bmm(W_t, g_prev[..., None])` 更新，避免保存完整 `(B,T,N,N)`。
  - `g0_raw` 为 trainable parameter；默认 `model.initial_position_encoding: "none"` 时使用 `norm_relu(g0_raw)` 后扩展到 `(B, N)`。
  - `model.initial_position_encoding: "additive_mlp"` 时要求传入 `initial_positions`，并使用 `norm_relu(g0_raw + position_encoder(initial_positions))` 作为 batch 初始状态。
  - `forward(velocities, initial_positions=None) -> RNNRollout`，其中 `initial_state` 为 `(B, N)`，`hidden_states` 为 `(B, T, N)`；`"none"` 模式拒绝非空 `initial_positions`。
  - `Norm(ReLU)` 遇到全零向量时输出零向量是实现假设；论文只给出除以范数公式，因此 metrics 必须记录 zero-norm fraction。

### Losses

`losses.py` 提供：

- `sic_losses(batch, rollout, cfg) -> dict[str, Tensor]`，其中 `rollout.initial_state` 用于 ConIso 的 `t=0`，`rollout.hidden_states` 用于 pairwise losses 和 capacity。
- `pairwise_sic_losses(positions, hidden_states, cfg)` 使用 blockwise exact pair computation。
- `conformal_isometry_loss(velocities, initial_state, hidden_states, sigma_x)`。

实现细节：

- Flatten 后参与 pairwise 的总点数为 `P = B*T`。
- 位置距离用 `torch.cdist(pos_chunk, pos_all)`。
- 神经距离平方用范数和矩阵乘法计算，避免构造 `(chunk, P, N)`。
- Separation mask: `||x_i - x_j||_2 > sigma_x`。
- Invariance mask: `||x_i - x_j||_2 < sigma_x`。
- Separation term: `exp(-||g_i-g_j||^2 / (2*sigma_g^2))`。
- Invariance term: `||g_i-g_j||^2`。
- Capacity term: `-||mean(g)||^2`。
- Conformal isometry: 对 `0 < ||v_t|| < sigma_x` 的 step 计算 `||g_t-g_{t-1}|| / ||v_t||` 的方差；`t=0` 使用 rollout 的 `initial_state` 作为 `g_{t-1}`；没有有效 step 时返回 0 并记录 count 0。
- `total = lambda_sep*sep + lambda_inv*inv + lambda_cap*cap + lambda_coniso*coniso`。

必须提供 tiny tensor naive implementation 只用于测试，验证 chunked exact 与 naive 一致。

### Training

`train.py` 提供：

- `train(config_path: str | Path, resume_checkpoint: str | Path | None = None, *, overwrite_output: bool = False) -> RunResult`。
- 输出目录包含：
  - `run.log`：训练生命周期和进度日志。
  - `train_events.jsonl`：严格 JSONL 的结构化事件流。
  - `config.yaml`：effective config。
  - `metrics.jsonl`：每步或每 `log_every` 的 loss、loss components、lr、grad norm、zero-norm fraction、pair counts、step timing、throughput、disk free space 和 CUDA memory diagnostics。
  - `tensorboard/`。
  - `checkpoints/step_<step>.pt`。
  - `checkpoints/latest.pt`。
  - `checkpoints/checkpoint_manifest.json`。

训练流程：

1. 设置随机种子。
2. 构建 device、model、optimizer、scheduler。
3. 每 step 调 `make_sic_batch`，forward 得到 `RNNRollout`，再用 `sic_losses(batch, rollout, cfg)` 计算 loss。
4. 按 `accumulate_grad_batches` 做 microbatch 梯度累积；`max_optimizer_steps` 只统计 optimizer step，不统计 microbatch。
5. optimizer step 前做 gradient clipping，并在 optimizer step 后根据 `scheduler_monitor` 更新 ReduceLROnPlateau。
6. 定期保存 checkpoint 和 metrics。

`scripts/train_sic.py` 只做 CLI 参数解析并调用 `sic4gridcells.train.train`。

### Logging

- `scripts/train_sic.py`、`scripts/eval_checkpoint.py` 和 `scripts/run_ablations.py` 都接受 `--log-level`，用于 stderr 端的标准 logging。
- 运行时的 human-readable 日志写入各自的 `run.log`。
- 结构化事件写入各自的 `*_events.jsonl`，行级记录只包含可 JSON 序列化字段，非有限数值会写成 `null`。
- resume 时，训练的事件日志会按 checkpoint step 裁剪，和 `metrics.jsonl` 的裁剪逻辑保持一致。

### Runtime safety and recovery

- Fresh training、evaluation 和 ablation runs 默认拒绝复用非空 output directory；显式 resume 或 `--overwrite-output` 才能复用。
- Training resume 仍只允许 checkpoint config 与当前 config 在 `train.max_optimizer_steps` 上不同。
- Checkpoint 写入使用 atomic replace；每次 checkpoint 保存同时写 `checkpoints/step_<step>.pt`、`checkpoints/latest.pt` 和 `checkpoints/checkpoint_manifest.json`。
- Checkpoint 文件必须继续能用默认 `torch.load(path, map_location="cpu")` 加载。
- Training 在 backward 前检查 floating loss tensors 是否 finite，并在 gradient clipping 时启用 non-finite gradient error。
- `scripts/run_ablations.py --resume-existing` 从每个 variant 的 latest checkpoint 继续；`--skip-completed` 跳过 latest checkpoint 已达到 `train.max_optimizer_steps` 的 variant。

### Evaluation validation

- `src/sic4gridcells/validation.py` 读取既有 evaluation output directory，并返回 JSON-serializable `ValidationReport`。
- `scripts/validate_eval.py` 是薄 CLI；它不重跑 evaluation，只检查 `summary.json`、per-arena required artifacts、coverage、active units、invalid response units 和 module evidence。
- Validation blocker 表示 artifact 不完整或证据不足；不是 training/evaluation 崩溃信号。
- CLI 默认在存在 blockers 时返回非零；`--allow-fail` 只用于保存诊断报告且不阻断 shell workflow。
- Smoke validation 可以放宽 quality thresholds；paper-claim validation 应显式要求 `--arena-sizes 2.0,3.0,4.0` 并保存 `validation.json`。

## 分阶段执行计划

### 阶段 0：项目启动

变更：

- 创建 `pyproject.toml`，包名 `sic4gridcells`，Python 约束 `>=3.12,<3.13`。
- 创建 package skeleton、configs 和 scripts。

验证：

```bash
uv run python -m pytest
uv run python scripts/train_sic.py --help
```

### 阶段 1：核心单元测试

变更：

- 实现 config、data、model、losses。
- 添加 `tests/test_config.py`、`tests/test_data.py`、`tests/test_model.py`、`tests/test_losses.py`。

验证：

```bash
uv run python -m pytest tests/test_config.py tests/test_data.py tests/test_model.py tests/test_losses.py
```

验收：

- Data tests 证明 permutation batch 终点一致。
- Model tests 证明输出 shape、非负性、非零状态 unit norm、梯度可回传。
- Loss tests 证明 masks、loss sign、finite value、chunked-vs-naive 一致。

### 阶段 2：smoke training

变更：

- 实现训练循环、checkpoint、metrics 和 TensorBoard。
- 添加 `tests/test_train_step.py`。

验证：

```bash
uv run python -m pytest tests/test_train_step.py
uv run python scripts/train_sic.py --config configs/smoke.yaml
```

验收：

- 10-step smoke run 在 CPU 或 GPU 上完成。
- `results/smoke/` 下有 effective config、metrics 和 checkpoint。
- loss components 全部 finite。

### 阶段 3：评估链路

当前实现：

- `src/sic4gridcells/evaluate.py` 实现 bounded random-walk evaluation trajectory generator、checkpoint reload 和 artifact 写出。
- `src/sic4gridcells/analysis.py` 最小移植 GridScorer 风格的 ratemap、SAC、grid score 和 grid scale 逻辑。
- `src/sic4gridcells/plotting.py` 输出 `summary.png`、`ratemaps.pdf`、`sacs.pdf` 和指标直方图。
- `scripts/eval_checkpoint.py` 是薄 CLI；训练 config 从 checkpoint 中读取，不另传 `--config`。
- `scripts/validate_eval.py` 是薄 CLI；它读取 evaluation output 并输出 artifact/quality blockers。
- 评估输出包含 `run.log` 和 `eval_events.jsonl`，并继续写出 `summary.json`、`config.yaml` 和 per-arena artifact。
- `ratemaps.npz` 将未访问空间 bin 保存为 `NaN`；访问过但响应为零的 bin 保持 `0.0`。
- `occupancy.npz` 保存 `occupancy_counts`，是 coverage 指标的 source of truth；访问过的 bin 若出现非有限响应，归类为 invalid response，而不是 coverage gap。
- `summary.json` 输出 `visited_bins`、`unvisited_bins`、`total_bins`、`coverage_fraction`、`units_without_coverage`、`zero_response_units`、`invalid_response_units` 和 `active_units`。
- `grid_stats.json/csv` 对每个 unit 输出 `response_status`、`max_abs_response`、`zero_response`、`invalid_response`、`scale_pixels`、`scale_meters`、`orientation_degrees` 和 `module_id`，不再用 `dead_units` 混合 coverage 和 zero response。
- `grid_metrics.npz` 保存 per-unit grid score、scale、orientation、peak count 和 module id。
- `module_summary.csv/json` 汇总 scale-based module clustering；`pairwise_distance_stats.csv/json` 和 `pairwise_distance.png` 汇总 neural distance vs spatial/temporal separation；`fourier_stats.csv/json`、`phase_summary.csv/json`、`state_space_summary.csv/json` 和 `state_space_modules.npz` 输出 preliminary Fourier、phase proxy 和 state-space PCA summaries；`trajectory_stats.json` 记录 evaluation trajectory statistics。
- SAC/grid score 计算使用 finite ratemap bins 作为 overlap mask，未访问 bin 只在 FFT 数值计算中填 0，不作为 measured zero response。
- bounded random-walk 的步长按 arena 尺度设定，不随 `--steps` 增加而缩小；evaluation seed 默认来自 checkpoint config，也可通过 CLI 指定。

验证：

```bash
uv run python -m pytest tests/test_analysis.py tests/test_evaluate.py
uv run python scripts/eval_checkpoint.py --checkpoint results/smoke/checkpoints/step_10.pt --output-dir results/smoke/eval --device cpu --arena-sizes 1.0 --nbins 8 --trajectories 2 --steps 16
uv run python -m pytest tests/test_validation.py
uv run python scripts/validate_eval.py --output-dir results/smoke/eval --arena-sizes 1.0 --min-coverage 0.0 --min-active-units 0 --min-module-count 0
```

验收：

- smoke checkpoint 可 reload。
- eval 输出 ratemap arrays、occupancy counts、grid stats JSON/CSV、module summary、pairwise neural-distance stats、trajectory stats、至少一个 PDF 或 PNG。
- validation CLI 可区分 artifact/quality blockers 和通过的 relaxed smoke artifact check。
- 评估代码不读取或生成 supervised target。

### 阶段 4：中等规模 sanity run

当前实现：

- `configs/medium.yaml` 已提供 `B=16`、`T=30`、`N=64`、`max_optimizer_steps=5000` 的 sanity-run profile。
- 训练和评估仍按独立命令运行；中等规模长跑尚未在本仓库完成。

验证：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_sic.py --config configs/medium.yaml
uv run python scripts/eval_checkpoint.py --checkpoint results/medium/checkpoints/step_5000.pt --output-dir results/medium/eval --arena-sizes 2.0,3.0,4.0 --nbins 32 --trajectories 32 --steps 256 --seed 0
uv run python scripts/validate_eval.py --output-dir results/medium/eval --arena-sizes 2.0,3.0,4.0 --json-output results/medium/eval/validation.json --allow-fail
```

验收：

- 中等规模 run 不 OOM。
- 至少部分 units 出现空间调谐或周期趋势。
- metrics 中记录 pair counts、zero-norm fraction、throughput、disk free space 和 CUDA memory diagnostics；coverage/zero/invalid/active response counts 由 evaluation summary 记录。

### 阶段 5：paper config 和消融

当前实现：

- `configs/sic_paper.yaml` 保留 paper-scale 训练参数。
- `configs/ablations.yaml` 和 `scripts/run_ablations.py` 提供 config-driven ablation orchestration、可选训练后评估和 aggregate summary。
- `data.augmentation_mode: identity` 支持 no-permutation augmentation 消融。
- ablation 根输出目录写 `run.log` 和 `ablation_events.jsonl`；每个 variant 的训练和评估继续复用各自的 run/event 日志契约。

验证：

```bash
CUDA_VISIBLE_DEVICES=<id> uv run python scripts/train_sic.py --config configs/sic_paper.yaml
uv run python scripts/eval_checkpoint.py --checkpoint results/sic_paper/checkpoints/step_2000000.pt --output-dir results/sic_paper/eval --arena-sizes 2.0,3.0,4.0 --nbins 32 --trajectories 32 --steps 256 --seed 0
uv run python scripts/validate_eval.py --output-dir results/sic_paper/eval --arena-sizes 2.0,3.0,4.0 --json-output results/sic_paper/eval/validation.json
uv run python scripts/run_ablations.py --config configs/ablations.yaml
```

验收：

- paper config 使用 `B=130`、`T=60`、`N=128`、paper loss weights、`max_optimizer_steps=2000000`。
- 至少多个 seeds 和小范围 hyperparameter sweep 被记录。
- 输出 grid-scale histogram、2 m/3 m/4 m ratemaps、pairwise neural-distance summaries、state-space analysis。
- 消融覆盖 no capacity、reduced `sigma_g`、no separation、no invariance、no coniso、no permutation augmentation。

## Review 与质量门

每个阶段合并前执行：

```bash
uv run python -m pytest
```

阶段 1 和阶段 2 修改核心接口后，使用一个只读 reviewer 检查：

- 数据 shape 和论文 batch 定义是否一致。
- `W(v)`、`Norm(ReLU)`、loss 符号是否与论文公式一致。
- 是否引入了 supervised position targets。
- 是否有未记录的假设。

阶段 3 以后增加第二个 reviewer，分别检查：

- 训练/配置/运行时合同。
- 评估指标和论文图目标。

## 当前后续顺序

1. 跑中等规模 sanity training，并用 `scripts/eval_checkpoint.py` 评估生成的 checkpoint，再用 `scripts/validate_eval.py` 保存 validation report。
2. 用 `configs/ablations.yaml` 和 `scripts/run_ablations.py` 执行 matched ablations。
3. 按 `docs/runbook.md` 记录 paper-scale throughput、GPU memory、checkpoint 和 evaluation cadence。
4. 基于 paper-scale 结果完成 toroidal manifold confirmation 和论文图级 figure selection。
