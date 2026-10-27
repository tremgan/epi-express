# Experiment log

Every call of `python -m model_infrastruture.train "optional note on what changed"` appends a
section below automatically: what changed compared to the previous run, the full configuration,
one line per epoch, and a final status (completed / interrupted / failed / killed). Lines are
written as the run goes, so interrupted runs are kept. The machine-readable copy is
`runs/runs.jsonl`.

Goal: Spearman correlation on the X3 test genes (Kaggle). Train on X1 + X2, validate on the
X1 + X2 val splits.

## Baseline (configuration in the code before the first logged run)

- Model: `ConvResNet(in_channels=7, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.1, hidden=128)`
  — 757,793 parameters (hand count; the first run logs the exact number)
- Optimizer: `AdamW(lr=0.001, weight_decay=0.01)` with `CosineAnnealingLR(T_max=60)` (stepped per
  epoch, lr 1e-3 → 0), no gradient clipping | loss `HuberLoss(delta=1.0)`
  | batch size 64 | 60 epochs | no seed set
- Input: 7 tracks (DNase, H3K4me3, H3K4me1, H3K27ac, H3K36me3, H3K27me3, H3K9me3),
  window 1024 bp upstream / 512 bp downstream of the TSS, 16 bp bins (96 bins), `log1p`,
  then z-score per cell line and track (mean/std from the cell line's train genes; X3 from its
  own test genes); target `log1p(gex)`
- Changed before the first run: input normalization was added (z-score per cell line and
  track), the loss went from `MSELoss` to `HuberLoss`, and the optimizer from
  `SGD(lr=0.01, momentum=0.9)` to AdamW + cosine schedule. The first logged run already includes
  all of these. To get
  an un-normalized reference, set `NORMALIZE = False` in `train.py` for one run.
- Result: **not run yet.** The first logged run below is the baseline, and its numbers are the
  reference for everything after it.

## Runs

## Run 20261027-133337
- Changes vs previous run:
  - first logged run (baseline)
- Model: `ConvResNet(in_channels=5, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.1, hidden=128)` — 757,473 parameters
- Optimizer: `SGD(lr=0.01, momentum=0.9, weight_decay=0, nesterov=False)` | loss `MSELoss` | batch size 64 | 60 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3'], upstream=1024, downstream=512, bin_size=16
- Epochs:
  - epoch 1: train loss 1.7050 | val loss 1.4670 | val Spearman 0.7837
  - epoch 2: train loss 1.2202 | val loss 1.2921 | val Spearman 0.8214
  - epoch 3: train loss 1.1358 | val loss 1.1882 | val Spearman 0.8241
  - epoch 4: train loss 1.0951 | val loss 1.1766 | val Spearman 0.8152
  - epoch 5: train loss 1.0826 | val loss 1.2528 | val Spearman 0.8205
  - epoch 6: train loss 1.0513 | val loss 1.1849 | val Spearman 0.8305
  - epoch 7: train loss 1.0233 | val loss 1.2033 | val Spearman 0.8208
  - epoch 8: train loss 1.0155 | val loss 1.2019 | val Spearman 0.8207
  - epoch 9: train loss 0.9985 | val loss 1.1726 | val Spearman 0.8300
  - epoch 10: train loss 0.9794 | val loss 1.1673 | val Spearman 0.8139
  - epoch 11: train loss 0.9581 | val loss 1.2220 | val Spearman 0.8207
  - epoch 12: train loss 0.9426 | val loss 1.2519 | val Spearman 0.8181
  - epoch 13: train loss 0.9309 | val loss 1.2262 | val Spearman 0.7994
  - epoch 14: train loss 0.9076 | val loss 1.2338 | val Spearman 0.8082
  - epoch 15: train loss 0.8942 | val loss 1.2678 | val Spearman 0.7943
  - epoch 16: train loss 0.8828 | val loss 1.1891 | val Spearman 0.8148
- **Result: interrupted** — 16 epoch(s) finished, 4.1 min; best val Spearman 0.8305 (epoch 6), last 0.8148, last val loss 1.1891

## Run 20261027-134315
- Changes vs previous run:
  - `data.normalize`: — → True
- Model: `ConvResNet(in_channels=5, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.1, hidden=128)` — 757,473 parameters
- Optimizer: `SGD(lr=0.01, momentum=0.9, weight_decay=0, nesterov=False)` | loss `MSELoss` | batch size 64 | 60 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3'], upstream=1024, downstream=512, bin_size=16, normalize=True
- Epochs:
  - epoch 1: train loss 1.7359 | val loss 2.8919 | val Spearman 0.8070
  - epoch 2: train loss 1.2460 | val loss 1.3243 | val Spearman 0.8134
  - epoch 3: train loss 1.1583 | val loss 1.2274 | val Spearman 0.8087
  - epoch 4: train loss 1.0960 | val loss 1.1883 | val Spearman 0.8269
  - epoch 5: train loss 1.0827 | val loss 1.2244 | val Spearman 0.8006
  - epoch 6: train loss 1.0547 | val loss 1.2057 | val Spearman 0.8217
  - epoch 7: train loss 1.0258 | val loss 1.1817 | val Spearman 0.8245
  - epoch 8: train loss 1.0162 | val loss 1.2027 | val Spearman 0.8249
  - epoch 9: train loss 1.0046 | val loss 1.1560 | val Spearman 0.8281
  - epoch 10: train loss 0.9727 | val loss 1.1817 | val Spearman 0.8216
  - epoch 11: train loss 0.9593 | val loss 1.1763 | val Spearman 0.8311
  - epoch 12: train loss 0.9381 | val loss 1.1865 | val Spearman 0.8203
  - epoch 13: train loss 0.9191 | val loss 1.2372 | val Spearman 0.8114
  - epoch 14: train loss 0.9095 | val loss 1.2206 | val Spearman 0.8157
  - epoch 15: train loss 0.9016 | val loss 1.2310 | val Spearman 0.8134
  - epoch 16: train loss 0.8724 | val loss 1.2303 | val Spearman 0.8251
  - epoch 17: train loss 0.8712 | val loss 1.1956 | val Spearman 0.8288
  - epoch 18: train loss 0.8396 | val loss 1.2068 | val Spearman 0.8256
  - epoch 19: train loss 0.8245 | val loss 1.1846 | val Spearman 0.8162
  - epoch 20: train loss 0.7973 | val loss 1.2373 | val Spearman 0.8134
- **Result: interrupted** — 20 epoch(s) finished, 6.0 min; best val Spearman 0.8311 (epoch 11), last 0.8134, last val loss 1.2373

## Run 20261027-135044
- Changes vs previous run:
  - `loss`: MSELoss → HuberLoss
- Model: `ConvResNet(in_channels=5, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.1, hidden=128)` — 757,473 parameters
- Optimizer: `SGD(lr=0.01, momentum=0.9, weight_decay=0, nesterov=False)` | loss `HuberLoss` | batch size 64 | 60 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3'], upstream=1024, downstream=512, bin_size=16, normalize=True
- Epochs:
  - epoch 1: train loss 0.4750 | val loss 0.4349 | val Spearman 0.8141
  - epoch 2: train loss 0.3870 | val loss 0.4128 | val Spearman 0.8244
  - epoch 3: train loss 0.3675 | val loss 0.4319 | val Spearman 0.8144
  - epoch 4: train loss 0.3544 | val loss 0.4074 | val Spearman 0.8169
  - epoch 5: train loss 0.3518 | val loss 0.4143 | val Spearman 0.8152
  - epoch 6: train loss 0.3485 | val loss 0.4033 | val Spearman 0.7775
  - epoch 7: train loss 0.3438 | val loss 0.4155 | val Spearman 0.8140
- **Result: interrupted** — 7 epoch(s) finished, 1.8 min; best val Spearman 0.8244 (epoch 2), last 0.8140, last val loss 0.4155

## Run 20261027-135410
- Changes vs previous run:
  - `extra.scheduler`: — → CosineAnnealingLR(T_max=60)
  - `optimizer.betas`: — → [0.9, 0.999]
  - `optimizer.lr`: 0.01 → 0.001
  - `optimizer.momentum`: 0.9 → —
  - `optimizer.name`: SGD → AdamW
  - `optimizer.nesterov`: False → —
  - `optimizer.weight_decay`: 0 → 0.01
- Model: `ConvResNet(in_channels=5, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.1, hidden=128)` — 757,473 parameters
- Optimizer: `AdamW(lr=0.001, betas=[0.9, 0.999], weight_decay=0.01)` | loss `HuberLoss` | batch size 64 | 60 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3'], upstream=1024, downstream=512, bin_size=16, normalize=True
- Extra: scheduler=CosineAnnealingLR(T_max=60)
- Epochs:
  - epoch 1: train loss 0.4339 | val loss 0.4197 | val Spearman 0.8261
  - epoch 2: train loss 0.3719 | val loss 0.4207 | val Spearman 0.8263
  - epoch 3: train loss 0.3608 | val loss 0.4101 | val Spearman 0.8270
  - epoch 4: train loss 0.3573 | val loss 0.4100 | val Spearman 0.8237
  - epoch 5: train loss 0.3517 | val loss 0.4053 | val Spearman 0.8241
  - epoch 6: train loss 0.3433 | val loss 0.4179 | val Spearman 0.8284
  - epoch 7: train loss 0.3376 | val loss 0.4079 | val Spearman 0.8192
  - epoch 8: train loss 0.3342 | val loss 0.4011 | val Spearman 0.8298
  - epoch 9: train loss 0.3263 | val loss 0.4143 | val Spearman 0.8288
  - epoch 10: train loss 0.3255 | val loss 0.4087 | val Spearman 0.8273
  - epoch 11: train loss 0.3164 | val loss 0.4116 | val Spearman 0.8278
  - epoch 12: train loss 0.3105 | val loss 0.3967 | val Spearman 0.8314
  - epoch 13: train loss 0.3005 | val loss 0.4159 | val Spearman 0.8233
  - epoch 14: train loss 0.2927 | val loss 0.4216 | val Spearman 0.8163
  - epoch 15: train loss 0.2794 | val loss 0.4396 | val Spearman 0.8231
  - epoch 16: train loss 0.2715 | val loss 0.4197 | val Spearman 0.8216
  - epoch 17: train loss 0.2560 | val loss 0.4311 | val Spearman 0.8217
  - epoch 18: train loss 0.2354 | val loss 0.4282 | val Spearman 0.8139
- **Result: interrupted** — 18 epoch(s) finished, 6.3 min; best val Spearman 0.8314 (epoch 12), last 0.8139, last val loss 0.4282

## Run 20261027-140119
- Changes vs previous run:
  - `optimizer.lr`: 0.001 → 0.0001
- Model: `ConvResNet(in_channels=5, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.1, hidden=128)` — 757,473 parameters
- Optimizer: `AdamW(lr=0.0001, betas=[0.9, 0.999], weight_decay=0.01)` | loss `HuberLoss` | batch size 64 | 60 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3'], upstream=1024, downstream=512, bin_size=16, normalize=True
- Extra: scheduler=CosineAnnealingLR(T_max=60)
- Epochs:
  - epoch 1: train loss 0.4572 | val loss 0.4622 | val Spearman 0.8118
  - epoch 2: train loss 0.3698 | val loss 0.4211 | val Spearman 0.7980
  - epoch 3: train loss 0.3509 | val loss 0.4031 | val Spearman 0.8198
  - epoch 4: train loss 0.3398 | val loss 0.4043 | val Spearman 0.8245
  - epoch 5: train loss 0.3307 | val loss 0.4008 | val Spearman 0.8213
  - epoch 6: train loss 0.3205 | val loss 0.4254 | val Spearman 0.8291
  - epoch 7: train loss 0.3118 | val loss 0.4095 | val Spearman 0.7983
  - epoch 8: train loss 0.3042 | val loss 0.4012 | val Spearman 0.8168
  - epoch 9: train loss 0.2869 | val loss 0.4072 | val Spearman 0.8101
  - epoch 10: train loss 0.2724 | val loss 0.4182 | val Spearman 0.8125
  - epoch 11: train loss 0.2591 | val loss 0.4350 | val Spearman 0.8108
  - epoch 12: train loss 0.2419 | val loss 0.4288 | val Spearman 0.8166
- **Result: interrupted** — 12 epoch(s) finished, 3.3 min; best val Spearman 0.8291 (epoch 6), last 0.8166, last val loss 0.4288

## Run 20261027-140551
- Changes vs previous run:
  - `batch_size`: 64 → 128
  - `optimizer.lr`: 0.0001 → 0.001
- Model: `ConvResNet(in_channels=5, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.1, hidden=128)` — 757,473 parameters
- Optimizer: `AdamW(lr=0.001, betas=[0.9, 0.999], weight_decay=0.01)` | loss `HuberLoss` | batch size 128 | 60 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3'], upstream=1024, downstream=512, bin_size=16, normalize=True
- Extra: scheduler=CosineAnnealingLR(T_max=60)
- Epochs:
  - epoch 1: train loss 0.4353 | val loss 0.4134 | val Spearman 0.8236
  - epoch 2: train loss 0.3683 | val loss 0.4157 | val Spearman 0.8261
  - epoch 3: train loss 0.3577 | val loss 0.4050 | val Spearman 0.8291
  - epoch 4: train loss 0.3509 | val loss 0.3991 | val Spearman 0.8320
  - epoch 5: train loss 0.3475 | val loss 0.4223 | val Spearman 0.8287
  - epoch 6: train loss 0.3441 | val loss 0.3979 | val Spearman 0.8294
  - epoch 7: train loss 0.3355 | val loss 0.4125 | val Spearman 0.8261
  - epoch 8: train loss 0.3293 | val loss 0.4087 | val Spearman 0.8275
  - epoch 9: train loss 0.3282 | val loss 0.4039 | val Spearman 0.8286
  - epoch 10: train loss 0.3172 | val loss 0.4060 | val Spearman 0.8297
  - epoch 11: train loss 0.3169 | val loss 0.4023 | val Spearman 0.8266
  - epoch 12: train loss 0.3036 | val loss 0.4104 | val Spearman 0.8266
  - epoch 13: train loss 0.2950 | val loss 0.4331 | val Spearman 0.8214
  - epoch 14: train loss 0.2851 | val loss 0.4144 | val Spearman 0.8234
  - epoch 15: train loss 0.2731 | val loss 0.4132 | val Spearman 0.8106
- **Result: interrupted** — 15 epoch(s) finished, 3.5 min; best val Spearman 0.8320 (epoch 4), last 0.8106, last val loss 0.4132

## Run 20261027-142240
- Changes vs previous run:
  - `data.tracks`: ['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3'] → ['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'CpG']
  - `model.n_params`: 757473 → 757633
  - `model.params.in_channels`: 5 → 6
- Model: `ConvResNet(in_channels=6, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.1, hidden=128)` — 757,633 parameters
- Optimizer: `AdamW(lr=0.001, betas=[0.9, 0.999], weight_decay=0.01)` | loss `HuberLoss` | batch size 128 | 60 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'CpG'], upstream=1024, downstream=512, bin_size=16, normalize=True
- Extra: scheduler=CosineAnnealingLR(T_max=60)
- Epochs:
  - epoch 1: train loss 0.4293 | val loss 0.4274 | val Spearman 0.8243
  - epoch 2: train loss 0.3725 | val loss 0.4184 | val Spearman 0.8243
  - epoch 3: train loss 0.3580 | val loss 0.4000 | val Spearman 0.8304
  - epoch 4: train loss 0.3477 | val loss 0.4139 | val Spearman 0.8272
  - epoch 5: train loss 0.3435 | val loss 0.4003 | val Spearman 0.8277
  - epoch 6: train loss 0.3431 | val loss 0.4180 | val Spearman 0.8177
  - epoch 7: train loss 0.3347 | val loss 0.4043 | val Spearman 0.8283
  - epoch 8: train loss 0.3302 | val loss 0.4095 | val Spearman 0.8324
  - epoch 9: train loss 0.3239 | val loss 0.3934 | val Spearman 0.8226
  - epoch 10: train loss 0.3145 | val loss 0.3978 | val Spearman 0.8295
  - epoch 11: train loss 0.3077 | val loss 0.3945 | val Spearman 0.8291
  - epoch 12: train loss 0.2978 | val loss 0.4074 | val Spearman 0.8250
  - epoch 13: train loss 0.2902 | val loss 0.3994 | val Spearman 0.8273
  - epoch 14: train loss 0.2751 | val loss 0.4200 | val Spearman 0.8229
  - epoch 15: train loss 0.2674 | val loss 0.4119 | val Spearman 0.8207
  - epoch 16: train loss 0.2477 | val loss 0.4193 | val Spearman 0.8178
  - epoch 17: train loss 0.2314 | val loss 0.4328 | val Spearman 0.8209
  - epoch 18: train loss 0.2134 | val loss 0.4304 | val Spearman 0.8149
  - epoch 19: train loss 0.1979 | val loss 0.4372 | val Spearman 0.8195
  - epoch 20: train loss 0.1800 | val loss 0.4468 | val Spearman 0.8040
  - epoch 21: train loss 0.1634 | val loss 0.4411 | val Spearman 0.8123
  - epoch 22: train loss 0.1466 | val loss 0.4561 | val Spearman 0.8081
  - epoch 23: train loss 0.1357 | val loss 0.4509 | val Spearman 0.7990
  - epoch 24: train loss 0.1275 | val loss 0.4454 | val Spearman 0.7944
  - epoch 25: train loss 0.1141 | val loss 0.4523 | val Spearman 0.8049
  - epoch 26: train loss 0.1080 | val loss 0.4626 | val Spearman 0.7989
  - epoch 27: train loss 0.0966 | val loss 0.4489 | val Spearman 0.7952
  - epoch 28: train loss 0.0966 | val loss 0.4541 | val Spearman 0.7943
  - epoch 29: train loss 0.0880 | val loss 0.4457 | val Spearman 0.7960
  - epoch 30: train loss 0.0819 | val loss 0.4470 | val Spearman 0.8026
- **Result: interrupted** — 30 epoch(s) finished, 6.1 min; best val Spearman 0.8324 (epoch 8), last 0.8026, last val loss 0.4470

## Run 20261027-143159
- Changes vs previous run:
  - `model.params.dropout`: 0.1 → 0.3
  - `optimizer.weight_decay`: 0.01 → 0.05
- Model: `ConvResNet(in_channels=6, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.3, hidden=128)` — 757,633 parameters
- Optimizer: `AdamW(lr=0.001, betas=[0.9, 0.999], weight_decay=0.05)` | loss `HuberLoss` | batch size 128 | 60 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'CpG'], upstream=1024, downstream=512, bin_size=16, normalize=True
- Extra: scheduler=CosineAnnealingLR(T_max=60)
- Epochs:
  - epoch 1: train loss 0.4621 | val loss 0.4356 | val Spearman 0.8194
  - epoch 2: train loss 0.3921 | val loss 0.4027 | val Spearman 0.8184
  - epoch 3: train loss 0.3747 | val loss 0.4133 | val Spearman 0.8278
  - epoch 4: train loss 0.3701 | val loss 0.3971 | val Spearman 0.8309
  - epoch 5: train loss 0.3615 | val loss 0.4087 | val Spearman 0.8243
  - epoch 6: train loss 0.3557 | val loss 0.4101 | val Spearman 0.8288
  - epoch 7: train loss 0.3513 | val loss 0.4140 | val Spearman 0.8292
  - epoch 8: train loss 0.3472 | val loss 0.4160 | val Spearman 0.8275
  - epoch 9: train loss 0.3427 | val loss 0.4036 | val Spearman 0.8300
  - epoch 10: train loss 0.3378 | val loss 0.3930 | val Spearman 0.8318
  - epoch 11: train loss 0.3317 | val loss 0.3982 | val Spearman 0.8280
  - epoch 12: train loss 0.3291 | val loss 0.4083 | val Spearman 0.8280
  - epoch 13: train loss 0.3222 | val loss 0.4084 | val Spearman 0.8274
  - epoch 14: train loss 0.3200 | val loss 0.4034 | val Spearman 0.8275
  - epoch 15: train loss 0.3152 | val loss 0.4022 | val Spearman 0.8249
  - epoch 16: train loss 0.3057 | val loss 0.4085 | val Spearman 0.8260
  - epoch 17: train loss 0.3010 | val loss 0.4284 | val Spearman 0.8258
  - epoch 18: train loss 0.2883 | val loss 0.4046 | val Spearman 0.8241
- **Result: interrupted** — 18 epoch(s) finished, 3.8 min; best val Spearman 0.8318 (epoch 10), last 0.8241, last val loss 0.4046

## Run 20261027-144454
- Changes vs previous run:
  - `data.tracks`: ['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'CpG'] → ['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'CpG', 'DNase@10kb', 'H3K4me3@10kb', 'H3K4me1@10kb', 'H3K27ac@10kb', 'H3K36me3@10kb']
  - `model.n_params`: 757633 → 758433
  - `model.params.in_channels`: 6 → 11
- Model: `ConvResNet(in_channels=11, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.3, hidden=128)` — 758,433 parameters
- Optimizer: `AdamW(lr=0.001, betas=[0.9, 0.999], weight_decay=0.05)` | loss `HuberLoss` | batch size 128 | 60 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'CpG', 'DNase@10kb', 'H3K4me3@10kb', 'H3K4me1@10kb', 'H3K27ac@10kb', 'H3K36me3@10kb'], upstream=1024, downstream=512, bin_size=16, normalize=True
- Extra: scheduler=CosineAnnealingLR(T_max=60)
- Epochs:
  - epoch 1: train loss 0.4433 | val loss 0.4205 | val Spearman 0.8131
  - epoch 2: train loss 0.3917 | val loss 0.4352 | val Spearman 0.8202
  - epoch 3: train loss 0.3696 | val loss 0.3992 | val Spearman 0.8270
  - epoch 4: train loss 0.3550 | val loss 0.3982 | val Spearman 0.8208
  - epoch 5: train loss 0.3498 | val loss 0.4031 | val Spearman 0.8245
  - epoch 6: train loss 0.3397 | val loss 0.3990 | val Spearman 0.8282
  - epoch 7: train loss 0.3347 | val loss 0.3940 | val Spearman 0.8214
  - epoch 8: train loss 0.3264 | val loss 0.3990 | val Spearman 0.8303
  - epoch 9: train loss 0.3173 | val loss 0.3996 | val Spearman 0.8271
  - epoch 10: train loss 0.3148 | val loss 0.3828 | val Spearman 0.8287
  - epoch 11: train loss 0.3068 | val loss 0.3873 | val Spearman 0.8250
  - epoch 12: train loss 0.2973 | val loss 0.3913 | val Spearman 0.8302
  - epoch 13: train loss 0.2868 | val loss 0.3950 | val Spearman 0.8217
  - epoch 14: train loss 0.2754 | val loss 0.3960 | val Spearman 0.8233
  - epoch 15: train loss 0.2660 | val loss 0.3982 | val Spearman 0.8204
  - epoch 16: train loss 0.2543 | val loss 0.4009 | val Spearman 0.8238
- **Result: interrupted** — 16 epoch(s) finished, 3.3 min; best val Spearman 0.8303 (epoch 8), last 0.8238, last val loss 0.4009

## Run 20261027-144934
- Changes vs previous run:
  - `data.downstream`: 512 → 1024
  - `data.upstream`: 1024 → 2048
- Model: `ConvResNet(in_channels=11, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.3, hidden=128)` — 758,433 parameters
- Optimizer: `AdamW(lr=0.001, betas=[0.9, 0.999], weight_decay=0.05)` | loss `HuberLoss` | batch size 128 | 60 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'CpG', 'DNase@10kb', 'H3K4me3@10kb', 'H3K4me1@10kb', 'H3K27ac@10kb', 'H3K36me3@10kb'], upstream=2048, downstream=1024, bin_size=16, normalize=True
- Extra: scheduler=CosineAnnealingLR(T_max=60)
- Epochs:
  - epoch 1: train loss 0.4925 | val loss 0.4361 | val Spearman 0.8151
  - epoch 2: train loss 0.3982 | val loss 0.4154 | val Spearman 0.8201
  - epoch 3: train loss 0.3757 | val loss 0.4105 | val Spearman 0.8132
  - epoch 4: train loss 0.3650 | val loss 0.4183 | val Spearman 0.8198
  - epoch 5: train loss 0.3578 | val loss 0.3999 | val Spearman 0.8176
  - epoch 6: train loss 0.3455 | val loss 0.3973 | val Spearman 0.8218
  - epoch 7: train loss 0.3414 | val loss 0.4037 | val Spearman 0.8170
  - epoch 8: train loss 0.3371 | val loss 0.4023 | val Spearman 0.8265
  - epoch 9: train loss 0.3298 | val loss 0.4162 | val Spearman 0.8243
  - epoch 10: train loss 0.3188 | val loss 0.4097 | val Spearman 0.8243
  - epoch 11: train loss 0.3153 | val loss 0.3920 | val Spearman 0.8266
  - epoch 12: train loss 0.3019 | val loss 0.3941 | val Spearman 0.8223
  - epoch 13: train loss 0.2904 | val loss 0.3961 | val Spearman 0.8230
  - epoch 14: train loss 0.2829 | val loss 0.4505 | val Spearman 0.8230
  - epoch 15: train loss 0.2759 | val loss 0.4255 | val Spearman 0.8131
  - epoch 16: train loss 0.2617 | val loss 0.4032 | val Spearman 0.8159
  - epoch 17: train loss 0.2511 | val loss 0.4179 | val Spearman 0.8164
  - epoch 18: train loss 0.2409 | val loss 0.4162 | val Spearman 0.8200
  - epoch 19: train loss 0.2276 | val loss 0.4142 | val Spearman 0.8178
  - epoch 20: train loss 0.2167 | val loss 0.4113 | val Spearman 0.8153
  - epoch 21: train loss 0.2069 | val loss 0.4258 | val Spearman 0.8128
  - epoch 22: train loss 0.1894 | val loss 0.4422 | val Spearman 0.8158
  - epoch 23: train loss 0.1824 | val loss 0.4275 | val Spearman 0.8084
- **Result: interrupted** — 23 epoch(s) finished, 8.4 min; best val Spearman 0.8266 (epoch 11), last 0.8084, last val loss 0.4275

## Run 20261027-145852
- Changes vs previous run:
  - `optimizer.lr`: 0.001 → 1e-05
- Model: `ConvResNet(in_channels=11, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.3, hidden=128)` — 758,433 parameters
- Optimizer: `AdamW(lr=1e-05, betas=[0.9, 0.999], weight_decay=0.05)` | loss `HuberLoss` | batch size 128 | 60 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'CpG', 'DNase@10kb', 'H3K4me3@10kb', 'H3K4me1@10kb', 'H3K27ac@10kb', 'H3K36me3@10kb'], upstream=2048, downstream=1024, bin_size=16, normalize=True
- Extra: scheduler=CosineAnnealingLR(T_max=60)
- Epochs:
  - epoch 1: train loss 0.7242 | val loss 0.7092 | val Spearman 0.7112
  - epoch 2: train loss 0.5751 | val loss 0.6443 | val Spearman 0.7085
  - epoch 3: train loss 0.5430 | val loss 0.6021 | val Spearman 0.7254
  - epoch 4: train loss 0.5188 | val loss 0.5836 | val Spearman 0.7337
  - epoch 5: train loss 0.5014 | val loss 0.5659 | val Spearman 0.7419
  - epoch 6: train loss 0.4899 | val loss 0.5422 | val Spearman 0.7516
  - epoch 7: train loss 0.4782 | val loss 0.5283 | val Spearman 0.7595
  - epoch 8: train loss 0.4652 | val loss 0.5253 | val Spearman 0.7565
  - epoch 9: train loss 0.4552 | val loss 0.5028 | val Spearman 0.7709
  - epoch 10: train loss 0.4449 | val loss 0.4975 | val Spearman 0.7753
  - epoch 11: train loss 0.4431 | val loss 0.4857 | val Spearman 0.7810
  - epoch 12: train loss 0.4326 | val loss 0.4859 | val Spearman 0.7761
  - epoch 13: train loss 0.4278 | val loss 0.4760 | val Spearman 0.7864
- **Result: interrupted** — 13 epoch(s) finished, 7.3 min; best val Spearman 0.7864 (epoch 13), last 0.7864, last val loss 0.4760

## Run 20261027-151150
- Changes vs previous run:
  - `model.n_params`: 758433 → 1479329
  - `model.params.n_bins`: — → 192
  - `model.params.pooling`: — → flatten
  - `optimizer.lr`: 1e-05 → 0.001
- Model: `ConvResNet(in_channels=11, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.3, hidden=128, pooling=flatten, n_bins=192)` — 1,479,329 parameters
- Optimizer: `AdamW(lr=0.001, betas=[0.9, 0.999], weight_decay=0.05)` | loss `HuberLoss` | batch size 128 | 60 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'CpG', 'DNase@10kb', 'H3K4me3@10kb', 'H3K4me1@10kb', 'H3K27ac@10kb', 'H3K36me3@10kb'], upstream=2048, downstream=1024, bin_size=16, normalize=True
- Extra: scheduler=CosineAnnealingLR(T_max=60)
- Epochs:
  - epoch 1: train loss 0.4777 | val loss 0.4315 | val Spearman 0.7998
  - epoch 2: train loss 0.3837 | val loss 0.3970 | val Spearman 0.8226
  - epoch 3: train loss 0.3697 | val loss 0.3953 | val Spearman 0.8244
  - epoch 4: train loss 0.3591 | val loss 0.4185 | val Spearman 0.8227
  - epoch 5: train loss 0.3507 | val loss 0.4027 | val Spearman 0.8222
  - epoch 6: train loss 0.3444 | val loss 0.4098 | val Spearman 0.8177
  - epoch 7: train loss 0.3383 | val loss 0.3931 | val Spearman 0.8217
  - epoch 8: train loss 0.3288 | val loss 0.3964 | val Spearman 0.8260
  - epoch 9: train loss 0.3210 | val loss 0.3924 | val Spearman 0.8231
  - epoch 10: train loss 0.3090 | val loss 0.4005 | val Spearman 0.8229
  - epoch 11: train loss 0.3023 | val loss 0.4189 | val Spearman 0.8196
- **Result: interrupted** — 11 epoch(s) finished, 5.2 min; best val Spearman 0.8260 (epoch 8), last 0.8196, last val loss 0.4189

## Run 20261027-152500
- Changes vs previous run:
  - `model.n_params`: 1479329 → 7051681
  - `model.params.hidden`: 128 → —
  - `model.params.hidden_dims`: — → [512, 128]
  - `model.params.strides`: — → [1, 2, 2]
- Model: `ConvResNet(in_channels=11, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.3, hidden_dims=[512, 128], pooling=flatten, n_bins=192, strides=[1, 2, 2])` — 7,051,681 parameters
- Optimizer: `AdamW(lr=0.001, betas=[0.9, 0.999], weight_decay=0.05)` | loss `HuberLoss` | batch size 128 | 60 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'CpG', 'DNase@10kb', 'H3K4me3@10kb', 'H3K4me1@10kb', 'H3K27ac@10kb', 'H3K36me3@10kb'], upstream=2048, downstream=1024, bin_size=16, normalize=True
- Extra: scheduler=CosineAnnealingLR(T_max=60)
- Epochs:
  - epoch 1: train loss 0.4225 | val loss 0.4208 | val Spearman 0.8114
  - epoch 2: train loss 0.3648 | val loss 0.3840 | val Spearman 0.8053
  - epoch 3: train loss 0.3478 | val loss 0.4008 | val Spearman 0.8243
  - epoch 4: train loss 0.3345 | val loss 0.3997 | val Spearman 0.8238
  - epoch 5: train loss 0.3244 | val loss 0.4016 | val Spearman 0.8144
  - epoch 6: train loss 0.3127 | val loss 0.3998 | val Spearman 0.8234
  - epoch 7: train loss 0.2970 | val loss 0.4014 | val Spearman 0.8103
  - epoch 8: train loss 0.2788 | val loss 0.3966 | val Spearman 0.8129
  - epoch 9: train loss 0.2534 | val loss 0.4041 | val Spearman 0.8092
  - epoch 10: train loss 0.2362 | val loss 0.4139 | val Spearman 0.7945
  - epoch 11: train loss 0.2104 | val loss 0.4111 | val Spearman 0.8117
  - epoch 12: train loss 0.1875 | val loss 0.4303 | val Spearman 0.7854
  - epoch 13: train loss 0.1663 | val loss 0.4205 | val Spearman 0.7997
  - epoch 14: train loss 0.1509 | val loss 0.4260 | val Spearman 0.7968
  - epoch 15: train loss 0.1319 | val loss 0.4293 | val Spearman 0.7952
- **Result: interrupted** — 15 epoch(s) finished, 9.6 min; best val Spearman 0.8243 (epoch 3), last 0.7952, last val loss 0.4293

## Run 20261027-153845
- Changes vs previous run:
  - `extra.scheduler`: CosineAnnealingLR(T_max=60) → CosineAnnealingLR(T_max=5)
  - `model.params.dropout`: 0.3 → 0.5
  - `n_epochs`: 60 → 5
- Model: `ConvResNet(in_channels=11, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.5, hidden_dims=[512, 128], pooling=flatten, n_bins=192, strides=[1, 2, 2])` — 7,051,681 parameters
- Optimizer: `AdamW(lr=0.001, betas=[0.9, 0.999], weight_decay=0.05)` | loss `HuberLoss` | batch size 128 | 5 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'CpG', 'DNase@10kb', 'H3K4me3@10kb', 'H3K4me1@10kb', 'H3K27ac@10kb', 'H3K36me3@10kb'], upstream=2048, downstream=1024, bin_size=16, normalize=True
- Extra: scheduler=CosineAnnealingLR(T_max=5)
- Epochs:
- **Result: interrupted** — no epoch finished

## Run 20261027-153929
- Changes vs previous run:
  - none (identical config — differences are run-to-run noise)
- Model: `ConvResNet(in_channels=11, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.5, hidden_dims=[512, 128], pooling=flatten, n_bins=192, strides=[1, 2, 2])` — 7,051,681 parameters
- Optimizer: `AdamW(lr=0.001, betas=[0.9, 0.999], weight_decay=0.05)` | loss `HuberLoss` | batch size 128 | 5 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'CpG', 'DNase@10kb', 'H3K4me3@10kb', 'H3K4me1@10kb', 'H3K27ac@10kb', 'H3K36me3@10kb'], upstream=2048, downstream=1024, bin_size=16, normalize=True
- Extra: scheduler=CosineAnnealingLR(T_max=5)
- Epochs:
  - epoch 1: train loss 0.4568 | val loss 0.4329 | val Spearman 0.8025
  - epoch 2: train loss 0.3751 | val loss 0.3987 | val Spearman 0.8213
  - epoch 3: train loss 0.3510 | val loss 0.3959 | val Spearman 0.8268
  - epoch 4: train loss 0.3286 | val loss 0.3916 | val Spearman 0.8256
  - epoch 5: train loss 0.3026 | val loss 0.3805 | val Spearman 0.8204
- **Result: completed** — 5 epoch(s) finished, 2.8 min; best val Spearman 0.8268 (epoch 3), last 0.8204, last val loss 0.3805

## Run 20261027-154419
- Changes vs previous run:
  - `extra.scheduler`: CosineAnnealingLR(T_max=5) → CosineAnnealingLR(T_max=20)
  - `n_epochs`: 5 → 20
- Model: `ConvResNet(in_channels=11, base_channels=32, n_blocks=3, kernel_size=5, dropout=0.5, hidden_dims=[512, 128], pooling=flatten, n_bins=192, strides=[1, 2, 2])` — 7,051,681 parameters
- Optimizer: `AdamW(lr=0.001, betas=[0.9, 0.999], weight_decay=0.05)` | loss `HuberLoss` | batch size 128 | 20 epochs | 28620 training genes
- Data: tracks=['DNase', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K36me3', 'CpG', 'DNase@10kb', 'H3K4me3@10kb', 'H3K4me1@10kb', 'H3K27ac@10kb', 'H3K36me3@10kb'], upstream=2048, downstream=1024, bin_size=16, normalize=True
- Extra: scheduler=CosineAnnealingLR(T_max=20)
- Epochs:
  - epoch 1: train loss 0.4336 | val loss 0.4167 | val Spearman 0.7999
  - epoch 2: train loss 0.3745 | val loss 0.3988 | val Spearman 0.8295
  - epoch 3: train loss 0.3572 | val loss 0.4067 | val Spearman 0.8234
  - epoch 4: train loss 0.3453 | val loss 0.4240 | val Spearman 0.8067
  - epoch 5: train loss 0.3338 | val loss 0.3864 | val Spearman 0.8307
  - epoch 6: train loss 0.3244 | val loss 0.3866 | val Spearman 0.8247
  - epoch 7: train loss 0.3107 | val loss 0.3829 | val Spearman 0.8199
  - epoch 8: train loss 0.3005 | val loss 0.3826 | val Spearman 0.8266
  - epoch 9: train loss 0.2846 | val loss 0.3955 | val Spearman 0.8117
  - epoch 10: train loss 0.2645 | val loss 0.4021 | val Spearman 0.8111
  - epoch 11: train loss 0.2390 | val loss 0.3894 | val Spearman 0.8084
  - epoch 12: train loss 0.2107 | val loss 0.4085 | val Spearman 0.8067
  - epoch 13: train loss 0.1874 | val loss 0.4108 | val Spearman 0.7899
- **Result: interrupted** — 13 epoch(s) finished, 17.2 min; best val Spearman 0.8307 (epoch 5), last 0.7899, last val loss 0.4108
