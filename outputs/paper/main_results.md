| Model                        | Names    |   Seeds | Accuracy     | Macro-F1     | Acc. 95% CI (seed 42)   | Trainable params   |
|:-----------------------------|:---------|--------:|:-------------|:-------------|:------------------------|:-------------------|
| Baseline text                | original |       1 | 99.31        | 99.19        | [98.98, 99.60]          | —                  |
| LoRA multimodal              | original |       3 | 99.25 ± 0.14 | 99.12 ± 0.16 | [98.72, 99.42]          | 1,894,162          |
| LoRA text                    | original |       3 | 98.99 ± 0.13 | 98.83 ± 0.14 | [98.76, 99.45]          | 678,162            |
| Baseline multimodal          | original |       1 | 97.63        | 97.47        | [97.04, 98.21]          | —                  |
| LoRA image                   | original |       3 | 96.45 ± 0.11 | 95.91 ± 0.07 | [95.69, 97.12]          | 825,618            |
| Baseline image               | original |       1 | 83.86        | 83.03        | [82.44, 85.29]          | —                  |
| LoRA multimodal (masked)     | masked   |       3 | 98.27 ± 0.09 | 98.14 ± 0.12 | [97.74, 98.76]          | 1,894,162          |
| Baseline multimodal (masked) | masked   |       1 | 93.65        | 93.34        | [92.73, 94.52]          | —                  |
| Baseline text (masked)       | masked   |       1 | 83.10        | 83.53        | [81.64, 84.45]          | —                  |
| LoRA text (masked)           | masked   |       3 | 81.95 ± 0.48 | 82.37 ± 0.47 | [81.05, 83.90]          | 678,162            |