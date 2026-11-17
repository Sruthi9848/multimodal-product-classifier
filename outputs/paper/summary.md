# Paper results summary

## Main results (test set)

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


## Significance (McNemar exact, Holm-corrected)

| Model A                | Model B              |   Acc A |   Acc B |   A right, B wrong |   A wrong, B right |          p |   p (Holm) | Significant (α=0.05)   |
|:-----------------------|:---------------------|--------:|--------:|-------------------:|-------------------:|-----------:|-----------:|:-----------------------|
| lora_multimodal        | lora_text            |   99.09 |   99.12 |                  6 |                  7 | 1          | 1          | no                     |
| lora_multimodal        | baseline_text        |   99.09 |   99.31 |                  5 |                 11 | 0.2101     | 0.8405     | no                     |
| lora_text              | baseline_text        |   99.12 |   99.31 |                  3 |                  8 | 0.2266     | 0.8405     | no                     |
| lora_multimodal        | baseline_multimodal  |   99.09 |   97.63 |                 46 |                  6 | 1.033e-08  | 5.163e-08  | yes                    |
| lora_image             | baseline_image       |   96.42 |   83.86 |                380 |                 36 | 1.429e-73  | 1e-72      | yes                    |
| lora_multimodal_masked | lora_text_masked     |   98.28 |   82.51 |                445 |                 13 | 1.46e-113  | 1.314e-112 | yes                    |
| lora_multimodal_masked | baseline_text_masked |   98.28 |   83.1  |                433 |                 17 | 1.886e-105 | 1.509e-104 | yes                    |
| lora_text_masked       | baseline_text_masked |   82.51 |   83.1  |                115 |                131 | 0.3389     | 0.8405     | no                     |
| lora_multimodal_masked | lora_image           |   98.28 |   96.42 |                 61 |                 10 | 4.645e-10  | 2.787e-09  | yes                    |


## E1 masking statistics (test names)

- names_changed_pct: 92.33
- words_removed_pct: 19.53
- names_empty_pct: 0.00
- avg_words_before: 5.67
- avg_words_after: 4.56


## Per-class gain from the photo (masked names, seed 42)

| Category                 |   F1 text only |   F1 text + photo |   Gain from photo |   Test items |
|:-------------------------|---------------:|------------------:|------------------:|-------------:|
| Belts                    |          0.632 |             1.000 |             0.368 |          122 |
| Wallets                  |          0.635 |             0.979 |             0.344 |          139 |
| Dress                    |          0.678 |             0.966 |             0.288 |           72 |
| Bottomwear               |          0.721 |             0.984 |             0.263 |          225 |
| Shoes                    |          0.742 |             0.978 |             0.236 |          225 |
| Topwear                  |          0.753 |             0.987 |             0.233 |          225 |
| Flip Flops               |          0.738 |             0.953 |             0.215 |          137 |
| Bags                     |          0.797 |             0.986 |             0.189 |          225 |
| Sandal                   |          0.758 |             0.924 |             0.166 |          144 |
| Socks                    |          0.838 |             0.995 |             0.157 |          105 |
| Loungewear and Nightwear |          0.846 |             0.942 |             0.097 |           69 |
| Eyewear                  |          0.928 |             1.000 |             0.072 |          161 |
| Innerwear                |          0.924 |             0.991 |             0.067 |          225 |
| Watches                  |          0.969 |             0.998 |             0.029 |          225 |
| Saree                    |          0.985 |             1.000 |             0.015 |           64 |
| Jewellery                |          0.985 |             0.997 |             0.012 |          162 |
| Fragrance                |          0.993 |             0.993 |             0.000 |          150 |
| Lips                     |          1.000 |             0.984 |            -0.016 |           64 |


## E2 robustness

Test accuracy under each condition (— = not applicable)

| model                      |   clean |   no_photo |   no_name |   words_dropped_50 |   masked_at_test |
|:---------------------------|--------:|-----------:|----------:|-------------------:|-----------------:|
| baseline_multimodal        |  0.9763 |     0.6816 |    0.8039 |             0.9193 |           0.8799 |
| baseline_multimodal_masked |  0.9365 |     0.3870 |    0.8288 |             0.8974 |         nan      |
| baseline_text              |  0.9931 |   nan      |  nan      |             0.7200 |           0.6371 |
| baseline_text_masked       |  0.8310 |   nan      |  nan      |             0.5557 |         nan      |
| lora_image                 |  0.9642 |   nan      |  nan      |           nan      |         nan      |
| lora_multimodal            |  0.9909 |     0.9843 |    0.9438 |             0.9460 |           0.9222 |
| lora_multimodal_masked     |  0.9828 |     0.7156 |    0.9587 |             0.9704 |         nan      |
| lora_text                  |  0.9912 |   nan      |  nan      |             0.5984 |           0.3593 |
| lora_text_masked           |  0.8251 |   nan      |  nan      |             0.4958 |         nan      |