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