# Experiment 1 — Missing Views

This experiment evaluates NeuS reconstruction quality when only a subset of training views is available.

## Structure

| File | Description |
|---|---|
| `dataset_subsample.py` | Dataset wrapper that deterministically subsamples views by a given ratio |
| `exp_runner_exp1.py` | Runner that wires the subsampled dataset into the base training loop |
| `conf_100.conf` | Config using 100 % of views (baseline) |
| `conf_50.conf` | Config using 50 % of views |
| `conf_25.conf` | Config using 25 % of views |

## Usage

```bash
python experiments/exp1_missing_views/exp_runner_exp1.py \
    --conf experiments/exp1_missing_views/conf_100.conf \
    --case <CASE_NAME> \
    --view_ratio 1.0 \
    --mode train
```

Replace `conf_100.conf` / `--view_ratio 1.0` with `conf_50.conf` / `0.5` or `conf_25.conf` / `0.25` to run the sparse-view variants.
