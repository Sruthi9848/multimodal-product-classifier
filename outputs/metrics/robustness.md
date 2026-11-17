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