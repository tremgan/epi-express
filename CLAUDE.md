# Experiment log (local, gitignored)

All runs: `uv run model_infrastruture/train.py`, ConvMaxPoolFeedForward (ConvTower defaults: kernel 5, 3 layers, stride 3, filter_factor 3, hidden 64), MSE on log1p(gex), log1p inputs, train = X1+X2 train, val = X1+X2 val (chr14/19), MPS.

## Baseline (2026-10-15)
- Window 1024 up / 512 down, BIN_SIZE 16 (96 bins)
- SGD lr 1e-2, momentum 0.9, no weight decay, 28 epochs
- Best val spearman 0.829 (epoch 8), best val loss 1.205 (epoch 12); overfits after ~epoch 15 (train loss 0.49 vs val 1.46 at epoch 28)

## Run 2 (2026-10-15): WORSE than baseline
- Window 4096 up / 2048 down, BIN_SIZE 32 (192 bins)
- SGD lr 1e-4, momentum 0.9, weight_decay 1e-4, 50 epochs
- Result: worse than baseline (exact numbers not recorded)
- Confounded: window, bin size, lr and weight decay all changed at once, so the cause is unknown. lr 1e-4 is 100x lower and may simply be undertrained.

## Sweep (2026-10-15): lr x window, ConvMaxPoolFeedForward
- SGD momentum 0.9, weight_decay 1e-4, BIN_SIZE 16, 30 epochs, no fixed seed (~±0.01 noise)
- Window = factor x (1024 up / 512 down). Best val spearman (epoch of best):

| window | 1e-3 | 3e-3 | 1e-2 | 3e-2 |
|---|---|---|---|---|
| 1x | 0.824 (5) | 0.826 (3) | 0.824 (5) | 0.827 (11) |
| 2x | 0.801 (8) | 0.810 (8) | 0.812 (15) | 0.807 (4) |
| 4x | 0.785 (9) | 0.790 (8) | 0.794 (15) | 0.753 (3), diverged ep 4 |
| 8x | 0.756 (8) | 0.760 (7) | 0.757 (14) | 0.309, never trained |

- Wider window is monotonically worse (~-0.02 per doubling); lr barely matters in 1e-3..1e-2.
- lr 3e-2 diverges for wide windows (train loss 20846 at w4 ep 4, then constant predictions -> spearman nan).
- All runs peak early (ep 3-15) then overfit; final-epoch spearman 0.03-0.08 below best.
- Hypothesis: global max pool can't localize, so distant peaks add noise. Next: attention pooling w/ positional bias, then dilated tower.
