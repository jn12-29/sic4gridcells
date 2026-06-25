# SIC Grid Cells 实现计划

## 目标与当前环境

目标是在当前仓库实现一个 plain PyTorch 版 SIC 复现项目，先完成可测试的核心训练链路，再扩展到论文级训练、评估和消融。依据文档是 `docs/sic-reproduction-plan.md` 和 `.work/sic-reproduction/source/` 中的论文 LaTeX 源码。

当前环境状态：

- 已创建 `.venv/`。
- `.venv` 中已安装 `torch 2.12.1+cu130`、`numpy 2.5.0`、`scipy 1.18.0`、`matplotlib 3.11.0`、`pyyaml 6.0.3`、`pytest 9.1.1`、`tensorboard 2.20.0`、`scikit-learn 1.9.0` 等依赖。
- `.venv/bin/python` 能导入上述依赖，`torch.cuda.is_available()` 为 `True`，可见 8 张 CUDA GPU。
- 注意：不要裸跑 `uv pip install ...`；本机上它会使用当前 conda 环境。后续安装必须使用 `uv pip install --python .venv/bin/python ...`。

## 非目标

- 不实现 Banino supervised place-cell/head-direction target 训练。
- 不引入 PyTorch Lightning；除非 plain PyTorch 编排已经成为实际阻塞。
- 不把 `.work/sic-reproduction/source/` 当成项目源码；它只作为论文材料证据。
- 第一阶段不追求论文图完全复刻，只追求核心数据、模型、loss、训练和 smoke evaluation 可验证。

## 项目文件计划

第一批要创建的源码和配置：

```text
pyproject.toml
configs/smoke.yaml
configs/sic_paper.yaml
src/sic4gridcells/__init__.py
src/sic4gridcells/config.py
src/sic4gridcells/data.py
src/sic4gridcells/model.py
src/sic4gridcells/losses.py
src/sic4gridcells/train.py
scripts/train_sic.py
tests/test_config.py
tests/test_data.py
tests/test_model.py
tests/test_losses.py
tests/test_train_step.py
```

第二批评估文件：

```text
src/sic4gridcells/evaluate.py
src/sic4gridcells/analysis.py
src/sic4gridcells/plotting.py
scripts/eval_checkpoint.py
tests/test_analysis.py
tests/test_evaluate.py
```

第三批实验编排：

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
- `model.n_units`: paper 为 `128`。
- `model.mlp_layers`: paper 为 `3`。
- `model.mlp_hidden_width`: 默认 `256`，论文未指定，必须写入 effective config。
- `model.trainable_initial_state`: 默认 `true`，论文只说明 shared `g0`，未说明是否 trainable。
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
- `make_sic_batch(cfg, generator, device) -> SicBatch`。

`SicBatch` 字段：

- `base_velocities`: `(T, 2)`。
- `permutations`: `(B, T)`。
- `velocities`: `(B, T, 2)`，由 `base_velocities[permutations]` 得到。
- `positions`: `(B, T, 2)`，从同一原点 cumulative sum。

必须满足：

- 每条置换轨迹包含同一组 base velocities。
- 所有轨迹终点一致。
- `positions[:, t]` 表示执行第 `t` 个 velocity 后的位置。

### Model

`model.py` 提供：

- `norm_relu(x, eps=1e-8)`：`relu(x)` 后按 L2 norm 归一化；全零向量输出零向量，并在训练 metrics 里记录 zero-norm fraction。
- `VelocityConditionedRNN(cfg)`：
  - `transition_mlp(v_t)` 输出 `(B, N*N)`，reshape 为 `(B, N, N)`。
  - 每个 time step 只 materialize 当前 `W(v_t)`，用 `torch.bmm(W_t, g_prev[..., None])` 更新，避免保存完整 `(B,T,N,N)`。
  - `g0_raw` 为 trainable parameter；每个 batch 使用 `norm_relu(g0_raw)` 后扩展到 `(B, N)`。
  - `forward(velocities) -> RNNRollout`，其中 `initial_state` 为 `(B, N)`，`hidden_states` 为 `(B, T, N)`。
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

- `train(config_path: str) -> RunResult`。
- 输出目录包含：
  - `config.yaml`：effective config。
  - `metrics.jsonl`：每步或每 `log_every` 的 loss、loss components、lr、grad norm、zero-norm fraction、pair counts。
  - `tensorboard/`。
  - `checkpoints/step_<step>.pt`。

训练流程：

1. 设置随机种子。
2. 构建 device、model、optimizer、scheduler。
3. 每 step 调 `make_sic_batch`，forward 得到 `RNNRollout`，再用 `sic_losses(batch, rollout, cfg)` 计算 loss。
4. 按 `accumulate_grad_batches` 做 microbatch 梯度累积；`max_optimizer_steps` 只统计 optimizer step，不统计 microbatch。
5. optimizer step 前做 gradient clipping，并在 optimizer step 后根据 `scheduler_monitor` 更新 ReduceLROnPlateau。
6. 定期保存 checkpoint 和 metrics。

`scripts/train_sic.py` 只做 CLI 参数解析并调用 `sic4gridcells.train.train`。

## 分阶段执行计划

### 阶段 0：项目启动

变更：

- 创建 `pyproject.toml`，包名 `sic4gridcells`，Python 约束 `>=3.12,<3.13`。
- 创建 package skeleton、configs 和 scripts。

验证：

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/train_sic.py --help
```

### 阶段 1：核心单元测试

变更：

- 实现 config、data、model、losses。
- 添加 `tests/test_config.py`、`tests/test_data.py`、`tests/test_model.py`、`tests/test_losses.py`。

验证：

```bash
.venv/bin/python -m pytest tests/test_config.py tests/test_data.py tests/test_model.py tests/test_losses.py
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
.venv/bin/python -m pytest tests/test_train_step.py
.venv/bin/python scripts/train_sic.py --config configs/smoke.yaml
```

验收：

- 10-step smoke run 在 CPU 或 GPU 上完成。
- `results/smoke/` 下有 effective config、metrics 和 checkpoint。
- loss components 全部 finite。

### 阶段 3：评估链路

变更：

- 实现 bounded random-walk evaluation trajectory generator。
- Vendor 或最小移植 `grid-cells-torch/grid_cells/analysis/scores.py` 的 GridScorer 逻辑，并保留来源说明。
- 实现 ratemap、SAC、grid score、grid scale histogram、all-unit PDF。

验证：

```bash
.venv/bin/python -m pytest tests/test_analysis.py tests/test_evaluate.py
.venv/bin/python scripts/eval_checkpoint.py --checkpoint results/smoke/checkpoints/step_10.pt --config configs/smoke.yaml
```

验收：

- smoke checkpoint 可 reload。
- eval 输出 ratemap arrays、grid stats JSON/CSV、至少一个 PDF 或 PNG。
- 评估代码不读取或生成 supervised target。

### 阶段 4：中等规模 sanity run

变更：

- 增加 `configs/medium.yaml`，建议 `B=16`、`T=30`、`N=64`、`max_optimizer_steps=5000`。
- 增加 runbook 中的命令和资源建议。

验证：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_sic.py --config configs/medium.yaml
.venv/bin/python scripts/eval_checkpoint.py --checkpoint <medium-checkpoint> --config configs/medium.yaml
```

验收：

- 中等规模 run 不 OOM。
- 至少部分 units 出现空间调谐或周期趋势。
- metrics 中记录 pair counts、throughput、GPU memory 和 dead/zero-norm fraction。

### 阶段 5：paper config 和消融

变更：

- 完成 `configs/sic_paper.yaml`。
- 完成 `configs/ablations.yaml` 和 `scripts/run_ablations.py`。

验证：

```bash
CUDA_VISIBLE_DEVICES=<id> .venv/bin/python scripts/train_sic.py --config configs/sic_paper.yaml
.venv/bin/python scripts/run_ablations.py --config configs/ablations.yaml
```

验收：

- paper config 使用 `B=130`、`T=60`、`N=128`、paper loss weights、`max_optimizer_steps=2000000`。
- 至少多个 seeds 和小范围 hyperparameter sweep 被记录。
- 输出 grid-scale histogram、2 m/3 m/4 m ratemaps、pairwise neural-distance plots、state-space analysis。
- 消融覆盖 no capacity、reduced `sigma_g`、no separation、no invariance、no coniso、no permutation augmentation。

## Review 与质量门

每个阶段合并前执行：

```bash
.venv/bin/python -m pytest
```

阶段 1 和阶段 2 修改核心接口后，使用一个只读 reviewer 检查：

- 数据 shape 和论文 batch 定义是否一致。
- `W(v)`、`Norm(ReLU)`、loss 符号是否与论文公式一致。
- 是否引入了 supervised position targets。
- 是否有未记录的假设。

阶段 3 以后增加第二个 reviewer，分别检查：

- 训练/配置/运行时合同。
- 评估指标和论文图目标。

## 立即开工顺序

1. 创建 `pyproject.toml`、package skeleton、`configs/smoke.yaml` 和 `configs/sic_paper.yaml`。
2. 实现 `config.py`、`data.py`、`model.py`、`losses.py` 及对应单元测试。
3. 跑 `.venv/bin/python -m pytest`。
4. 实现训练循环和 `scripts/train_sic.py`。
5. 跑 10-step smoke training，并把结果路径记录到后续 runbook。
