# Experiment Scripts

## CLIP experiments

Evaluate individual models

```bash
# CLIP-ViT-B/32
python scripts/clip_individual.py model=ViT-B-32
# CLIP-ViT-L/14
python scripts/clip_individual.py model=ViT-L-14
```

Multi-task model fusion experiments

- Task Arithmetic
```bash
# CLIP-ViT-B/32
python scripts/clip_task_arithmetic.py model=ViT-B-32
# CLIP-ViT-L/14
python scripts/clip_task_arithmetic.py model=ViT-L-14
```
- Ties-Merging
```bash
# CLIP-ViT-B/32
python scripts/clip_ties_merging.py model=ViT-B-32
# CLIP-ViT-L/14
python scripts/clip_ties_merging.py model=ViT-L-14
```
- Task-wise AdaMerging
```bash
# CLIP-ViT-B/32
python scripts/clip_task_wise_adamerging.py model=ViT-B-32
# CLIP-ViT-L/14
python scripts/clip_task_wise_adamerging.py model=ViT-L-14
```
- Layer-wise AdaMerging
```bash
# CLIP-ViT-B/32
python scripts/clip_layer_wise_adamerging.py model=ViT-B-32
# CLIP-ViT-L/14
python scripts/clip_layer_wise_adamerging.py model=ViT-L-14
```
- **Concrete Task Arithmetic**
```bash
# CLIP-ViT-B/32
python scripts/clip_concrete_task_arithmetic.py model=ViT-B-32
# CLIP-ViT-L/14
python scripts/clip_concrete_task_arithmetic.py \
    model=ViT-L-14 batch_size=4
```
- **Concrete Task-wise AdaMerging**
```bash
# CLIP-ViT-B/32
python scripts/clip_task_wise_concrete_adamerging.py model=ViT-B-32
```
- **Concrete Layer-wise AdaMerging**
```bash
# CLIP-ViT-B/32
python scripts/clip_layer_wise_concrete_adamerging.py model=ViT-B-32
# CLIP-ViT-L/14
python scripts/clip_layer_wise_concrete_adamerging.py \
    model=ViT-L-14 batch_size=4
```

Generalization experiments

- Task Arithmetic
```bash
# CLIP-ViT-B/32 generation_exp1
python scripts/clip_generalization_task_arithmetic.py \
    --config-name clip_generalization_exp1 \
    exp_name=generalization_exp1 \
    model=ViT-B-32
# CLIP-ViT-B/32 generation_exp2
python scripts/clip_generalization_task_arithmetic.py \
    --config-name clip_generalization_exp2 \
    exp_name=generalization_exp2 \
    model=ViT-B-32
```
- Ties-Merging
```bash
# CLIP-ViT-B/32 generation_exp1
python scripts/clip_generalization_ties_merging.py \
    --config-name clip_generalization_exp1 \
    exp_name=generalization_exp1 \
    model=ViT-B-32
# CLIP-ViT-B/32 generation_exp2
python scripts/clip_generalization_ties_merging.py \
    --config-name clip_generalization_exp2 \
    exp_name=generalization_exp2 \
    model=ViT-B-32
```
- **Concrete Task Arithmetic**
```bash
# CLIP-ViT-B/32 generation_exp1
python scripts/clip_generalization_concrete_task_arithmetic.py \
    --config-name clip_generalization_exp1 \
    exp_name=generalization_exp1 \
    model=ViT-B-32
# CLIP-ViT-B/32 generation_exp2
python scripts/clip_generalization_concrete_task_arithmetic.py \
    --config-name clip_generalization_exp2 \
    exp_name=generalization_exp2 \
    model=ViT-B-32
```
- **Task-wise Concrete AdaMerging**
```bash
# CLIP-ViT-B/32 generation_exp1
python scripts/clip_generalization_task_wise_concrete_adamerging.py \
    --config-name clip_generalization_exp1 \
    exp_name=generalization_exp1 \
    model=ViT-B-32
# CLIP-ViT-B/32 generation_exp2
python scripts/clip_generalization_task_wise_concrete_adamerging.py \
    --config-name clip_generalization_exp2 \
    exp_name=generalization_exp2 \
    model=ViT-B-32
```
- **Layer-wise Concrete AdaMerging**
```bash
# CLIP-ViT-B/32 generation_exp1
python scripts/clip_generalization_layer_wise_concrete_adamerging.py \
    --config-name clip_generalization_exp1 \
    exp_name=generalization_exp1 \
    model=ViT-B-32
# CLIP-ViT-B/32 generation_exp2
python scripts/clip_generalization_layer_wise_concrete_adamerging.py \
    --config-name clip_generalization_exp2 \
    exp_name=generalization_exp2 \
    model=ViT-B-32
```

## Flan-T5 Experiments

Evaluate individual models

```bash
# flan-t5-base
python scripts/flan_t5_individuals.py models=flan-t5-base
python scripts/flan_t5_individuals.py models=flan-t5-base peft=lora-16

# flan-t5-large
python scripts/flan_t5_individuals.py models=flan-t5-large peft=lora-16
```

Multi-task model fusion experiments

- Simple Averaging
```bash
# flan-t5-base
python scripts/flan_t5_averaging.py models=flan-t5-base
python scripts/flan_t5_averaging.py models=flan-t5-base peft=lora-16

# flan-t5-large
python scripts/flan_t5_averaging.py models=flan-t5-large peft=lora-16
```
- Task Arithmetic
```bash
# flan-t5-base
python scripts/flan_t5_task_arithmetic.py models=flan-t5-base
python scripts/flan_t5_task_arithmetic.py models=flan-t5-base peft=lora-16

# flan-t5-large
python scripts/flan_t5_task_arithmetic.py models=flan-t5-large peft=lora-16
```
- Ties-Merging
```bash
# flan-t5-base
python scripts/flan_t5_ties_merging.py models=flan-t5-base
python scripts/flan_t5_ties_merging.py models=flan-t5-base peft=lora-16

# flan-t5-large
python scripts/flan_t5_ties_merging.py models=flan-t5-large peft=lora-16
```
- Task-wise AdaMerging
```bash
# flan-t5-base
python scripts/flan_t5_task_wise_adamerging.py models=flan-t5-base peft=lora-16

# flan-t5-large
python scripts/flan_t5_task_wise_adamerging.py models=flan-t5-large peft=lora-16
```
- **Concrete Task Arithmetic**
```bash
# flan-t5-base
python scripts/flan_t5_concrete_task_arithmetic.py models=flan-t5-base peft=lora-16

# flan-t5-large
python scripts/flan_t5_concrete_task_arithmetic.py models=flan-t5-large peft=lora-16
```
- **Layer-wise AdaMerging**
```bash
# flan-t5-base
python scripts/flan_t5_layer_wise_arithmetic.py \
    models=flan-t5-base peft=lora-16

# flan-t5-large
python scripts/flan_t5_layer_wise_adamerging.py \
    models=flan-t5-large peft=lora-16 fast_dev_run=false batch_size=4
```
- **Task-wise Concrete AdaMerging**
```bash
# flan-t5-base
python scripts/flan_t5_task_wise_concrete_adamerging.py \
    models=flan-t5-base peft=lora-16

# flan-t5-large
python scripts/flan_t5_task_wise_concrete_adamerging.py \
    models=flan-t5-large peft=lora-16 batch_size=4
```
- **Layer-wise Concrete AdaMerging**
```bash
# flan-t5-base
python scripts/flan_t5_layer_wise_concrete_adamerging.py \
    models=flan-t5-base peft=lora-16

# flan-t5-large
python scripts/flan_t5_layer_wise_concrete_adamerging.py \
    models=flan-t5-large peft=lora-16 batch_size=4
```

# Prerun results

## CLIP-ViT-B/32 Multi-task Model Fusion

| **Method**                            | **SUN397** | **Cars** | **RESISC45** | **EuroSAT** | **SVHN** | **GRSRB** | **MNIST** | **DTD**  | **Avg.** | source                                                                                                                      |
| ------------------------------------- | ---------- | -------- | ------------ | ----------- | -------- | --------- | --------- | -------- | -------- | --------------------------------------------------------------------------------------------------------------------------- |
| Individual                            | 75.3       | 77.7     | 96.1         | 99.9        | 97.5     | 98.7      | 99.7      | 79.4     | 90.5     | [file](ViT-B-32/individuals.csv)                                                                                            |
| Traditional MTL                       | 73.9       | 74.4     | 93.9         | 98.2        | 95.8     | 98.9      | 99.5      | 77.9     | 88.9     | AdaMerging paper                                                                                                            |
|                                       |            |          |              |             |          |           |           |          |          |                                                                                                                             |
| Weight Averaging                      | 65.3       | 63.3     | 71.4         | 73.6        | 64.2     | 52.8      | 87.5      | 50.1     | 66.0     | [file](ViT-B-32/averaging.csv)                                                                                              |
| Fisher Merging                        | 68.6       | 69.2     | 70.7         | 66.4        | 72.9     | 51.1      | 87.9      | 59.9     | 68.3     | AdaMerging paper                                                                                                            |
| RegMean                               | 65.3       | 63.5     | 75.6         | 78.6        | 78.1     | 67.4      | 93.7      | 52.0     | 71.8     | AdaMerging paper                                                                                                            |
|                                       |            |          |              |             |          |           |           |          |          |                                                                                                                             |
| *Task Arithmetic (TA)-Based*          |            |          |              |             |          |           |           |          |          |                                                                                                                             |
| Task Arithmetic                       | 55.3       | 54.9     | 66.7         | 77.4        | 80.2     | 69.7      | 97.3      | 50.1     | 69.0     | [file](ViT-B-32/task_arithmetic.csv)                                                                                        |
| Ties-Merging                          | **65.0**   | **64.3** | 74.7         | 76.8        | 81.3     | 69.4      | 96.5      | **54.3** | 72.8     | [file](ViT-B-32/ties_merging.csv)                                                                                           |
| **Concrete TA**                       | 62.5       | 61.1     | **76.0**     | **95.7**    | **91.0** | **81.9**  | **98.5**  | 51.9     | **77.3** | [meta-learn](ViT-B-32/clip_concrete_task_arithmetic.csv)                                                                    |
|                                       |            |          |              |             |          |           |           |          |          |                                                                                                                             |
| *Task-wise AdaMerging (TW AM)-Based*  |            |          |              |             |          |           |           |          |          |                                                                                                                             |
| TW AM                                 | 58.3       | 53.2     | 71.8         | 80.1        | 81.6     | 84.4      | 93.4      | 42.7     | 70.7     | [file](ViT-B-32/task_wise_adamerging.csv)                                                                                   |
| TW AM++                               | 60.8       | 56.9     | 73.1         | 83.4        | 87.3     | 82.4      | 95.7      | **50.1** | 73.7     | AdaMerging paper                                                                                                            |
| **TW Concrete AM**                    | **62.7**   | **58.9** | **74.5**     | **94.8**    | **91.1** | **95.0**  | **98.1**  | 34.6     | **76.2** | [TTA](ViT-B-32/clip_task_wise_concrete_adamerging_tta.csv)([meta-learn](ViT-B-32/clip_task_wise_concrete_adamerging.csv))   |
|                                       |            |          |              |             |          |           |           |          |          |                                                                                                                             |
| *Layer-wise AdaMerging (LW AM)-Based* |            |          |              |             |          |           |           |          |          |                                                                                                                             |
| LW AM                                 | 64.2       | 69.5     | 82.4         | 92.5        | 86.5     | 93.7      | 97.7      | 61.1     | 80.9     | [file](ViT-B-32/task_wise_adamerging.csv)                                                                                   |
| LW AM++                               | 66.6       | 68.3     | 82.2         | 94.2        | 89.6     | 89.0      | 98.3      | 60.6     | 81.1     | AdaMerging paper                                                                                                            |
| **LW Concrete AM**                    | **67.8**   | **70.0** | **87.5**     | **96.0**    | **91.6** | **96.7**  | **98.7**  | **63.8** | **84.0** | [TTA](ViT-B-32/clip_layer_wise_concrete_adamerging.csv)([meta-learn](ViT-B-32/clip_layer_wise_concrete_adamerging_tta.csv)) |

## CLIP-ViT-L/14 Multi-task Model Fusion

| **Method**                            | **SUN397** | **Cars** | **RESISC45** | **EuroSAT** | **SVHN** | **GRSRB** | **MNIST** | **DTD**  | **Avg.** | source                                                                                                                       |
| ------------------------------------- | ---------- | -------- | ------------ | ----------- | -------- | --------- | --------- | -------- | -------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Individual                            | 82.3       | 92.4     | 97.4         | 99.9        | 98.1     | 99.2      | 99.7      | 84.1     | 94.1     | [file](ViT-L-14/individuals.csv)                                                                                             |
| Traditional MTL                       | 80.8       | 90.6     | 96.3         | 96.3        | 97.6     | 99.1      | 99.6      | 84.4     | 93.5     | AdaMerging paper                                                                                                             |
|                                       |            |          |              |             |          |           |           |          |          |                                                                                                                              |
| Weight Averaging                      | 72.1       | 81.6     | 82.6         | 91.4        | 78.2     | 70.6      | 97.0      | 62.8     | 79.5     | [file](ViT-L-14/averaging.csv)                                                                                               |
| Fisher Merging                        | 69.2       | 88.6     | 87.5         | 93.5        | 80.6     | 74.8      | 93.3      | 70.0     | 82.2     | AdaMerging paper                                                                                                             |
| RegMean                               | 73.3       | 81.8     | 86.1         | 97.0        | 88.0     | 84.2      | 98.5      | 60.8     | 83.7     | AdaMerging paper                                                                                                             |
|                                       |            |          |              |             |          |           |           |          |          |                                                                                                                              |
| *Task Arithmetic (TA)-Based*          |            |          |              |             |          |           |           |          |          |                                                                                                                              |
| Task Arithmetic                       | 82.1       | 65.6     | 92.6         | 86.8        | 98.9     | 86.7      | 74.1      | 87.9     | 84.4     | [file](ViT-L-14/task_arithmetic.csv)                                                                                         |
| Ties-Merging                          | 84.5       | **67.7** | 94.3         | 82.1        | 98.7     | 88.0      | **75.0**  | 85.7     | 84.5     | [file](ViT-L-14/ties_merging.csv)                                                                                            |
| **Concrete TA**                       | **86.2**   | 66.9     | **96.7**     | **93.4**    | **99.1** | **89.0**  | 74.6      | **93.6** | **87.4** | [file](ViT-L-14/clip_concrete_task_arithmetic.csv)                                                                           |
|                                       |            |          |              |             |          |           |           |          |          |                                                                                                                              |
| *Layer-wise AdaMerging (LW AM)-Based* |            |          |              |             |          |           |           |          |          |                                                                                                                              |
| LW AM                                 | 79.0       | 90.3     | 90.8         | 96.2        | 93.4     | **98.0**  | 99.0      | **79.9** | 90.8     | AdaMerging paper                                                                                                             |
| LW AM++                               | **79.4**   | 90.3     | 91.6         | **97.4**    | 93.4     | 97.5      | 99.0      | 79.2     | 91.0     | AdaMerging paper                                                                                                             |
| **LW Concrete AM**                    | 77.8       | **91.2** | **92.1**     | 97.0        | **94.4** | 97.9      | 99.0      | 79.5     | **91.1** | [file](ViT-L-14/clip_layer_wise_concrete_adamerging_tta.csv)([meta-learn](ViT-L-14/clip_layer_wise_concrete_adamerging.csv)) |


## CLIP-ViT-B/32 Generalization Experiments

setting 1

| **Method**      | **SUN397** | **Cars** | **RESISC45** | **DTD**  | **SVHN** | **GTSRB** | **Avg.** | **MNIST** | **EuroSAT** | **Avg.** | source                                                                                                                                                        |
| --------------- | ---------- | -------- | ------------ | -------- | -------- | --------- | -------- | --------- | ----------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Task Arithmetic | 63.4       | 62.3     | 75.3         | 57.8     | 84.7     | 80.4      | 70.7     | 77.3      | 45.6        | 61.4     | [file](ViT-B-32/generalization_exp1/task_arithmetic.csv)                                                                                                      |
| Ties-Merging    | **67.8**   | 66.2     | 77.0         | 56.2     | 77.2     | 71.0      | 69.2     | 75.9      | 43.1        | 59.5     | [file](ViT-B-32/generalization_exp1/ties_merging.csv)                                                                                                         |
| **Concrete TA** | 66.2       | **66.4** | **82.0**     | **58.3** | **91.4** | **92.7**  | **72.9** | **80.7**  | **52.9**    | **66.8** | [meta-learn](ViT-B-32/generalization_exp1/concrete_task_arithmetic.csv)                                                                                       |
|                 |            |          |              |          |          |           |          |           |             |          |                                                                                                                                                               |
| AdaMerging      | 65.2       | 65.9     | 88.5         | 61.1     | 92.2     | 91.5      | 77.4     | **84.0**  | **56.1**    | **70.0** | AdaMerging paper                                                                                                                                              |
| AdaMerging++    | 68.2       | 67.6     | 86.3         | 63.6     | 92.6     | 89.8      | 78.0     | 83.9      | 53.5        | 68.7     | AdaMerging paper                                                                                                                                              |
| **Concrete AM** | **68.9**   | **71.7** | **91.2**     | **66.9** | **94.1** | **97.5**  | **81.7** | 83.6      | 53.9        | 69.7     | [TTA](ViT-B-32/generalization_exp1/layer_wise_concrete_adamerging_tta.csv)([meta-learn](ViT-B-32/generalization_exp1/layer_wise_concrete_adamerging_tta.csv)) |

setting 2

| **Method**      | **SUN397** | **Cars** | **GTSRB** | **EuroSAT** | **DTD**  | **MNIST** | **Avg.** | **RESISC45** | **SVHN** | **Avg.** | source                                                                                                                                                    |
| --------------- | ---------- | -------- | --------- | ----------- | -------- | --------- | -------- | ------------ | -------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Task Arithmetic | 63.8       | 63.9     | 75.2      | 87.3        | 56.6     | 95.7      | 73.8     | 52.5         | 49.9     | 51.2     | [file](ViT-B-32/generalization_exp2/task_arithmetic.csv)                                                                                                  |
| Ties-Merging    | **67.8**   | **67.2** | 67.8      | 78.9        | 56.2     | 92.8      | 71.8     | **58.4**     | 49.3     | 53.9     | [file](ViT-B-32/generalization_exp2/ties_merging.csv)                                                                                                     |
| **Concrete TA** | 66.4       | 65.7     | **90.0**  | **96.4**    | **57.2** | **98.1**  | **79.0** | 54.3         | **58.9** | **56.6** | [meta-learn](ViT-B-32/generalization_exp2/concrete_task_arithmetic.csv)                                                                                   |
|                 |            |          |           |             |          |           |          |              |          |          |                                                                                                                                                           |
| AdaMerging      | 67.1       | 67.8     | 94.8      | 94.4        | 59.6     | 98.2      | 80.3     | 50.2         | 60.9     | 55.5     | AdaMerging paper                                                                                                                                          |
| AdaMerging++    | 68.9       | 69.6     | 91.6      | 94.3        | 61.9     | 98.7      | 80.8     | **52.0**     | **64.9** | **58.5** | AdaMerging paper                                                                                                                                          |
| **Concrete AM** | **69.6**   | **71.0** | **97.6**  | **97.3**    | **68.7** | **99.0**  | **83.9** | 48.1         | 62.3     | 55.2     | [TTA](ViT-B-32/generalization_exp2/layer_wise_concrete_adamerging_tta.csv)([meta-learn](ViT-B-32/generalization_exp1/layer_wise_concrete_adamerging.csv)) |

#### Table: Multi-task performance when merging Flan-T5-base (LoRA fine-tuned) models on all eight tasks.

| **Method**                           | **CoLA** | **MNLI** | **MRPC** | **QNLI** | **QQP**  | **RTE** | **SST2** | **STSB** | **Avg.** | source                                                                                                                                    |
| ------------------------------------ | -------- | -------- | -------- | -------- | -------- | ------- | -------- | -------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Individual                           | 69.1     | 82.7     | 85.5     | 90.9     | 84.0     | 84.4    | 92.9     | 87.4     | 84.6     | [file](flan-t5-base_lora-16/individuals.csv)                                                                                              |
| Weight Averaging                     | 69.7     | 59.7     | 78.9     | 90.1     | 83.8     | 80.5    | 91.2     | 72.0     | 78.2     | [file](flan-t5-base_lora-16/averaging.csv)                                                                                                |
|                                      |          |          |          |          |          |         |          |          |          |                                                                                                                                           |
| *Task Arithmetic (TA)-Based*         |          |          |          |          |          |         |          |          |          |                                                                                                                                           |
| Task Arithmetic                      | 68.8     | 55.2     | 78.7     | 89.8     | 83.7     | 79.1    | 91.5     | 72.4     | 77.4     | [file](flan-t5-base_lora-16/task_arithmetic.csv)                                                                                          |
| Ties-Merging                         | 68.3     | 56.3     | **79.4** | 89.8     | 83.7     | 79.4    | 91.6     | 71.2     | 77.5     | [file](flan-t5-base_lora-16/ties_merging.csv)                                                                                             |
| **Concrete TA**                      | **69.1** | **58.1** | 78.4     | **89.9** | 83.5     | 79.4    | 91.6     | **73.4** | **78.0** | [meta-learn](flan-t5-base_lora-16/concrete_task_arithmetic.csv)                                                                           |
|                                      |          |          |          |          |          |         |          |          |          |                                                                                                                                           |
| *Layer-wise AdaMergin (LW AM)-Based* |          |          |          |          |          |         |          |          |          |                                                                                                                                           |
| LW AM                                | **69.1** | **60.3** | 78.4     | **90.0** | **83.6** | 79.1    | 91.6     | 74.1     | 78.3     | [TTA](flan-t5-base_lora-16/layer_wise_adamerging.csv)                                                                                     |
| **LW Concrete AM**                   | 69.0     | 59.4     | **80.1** | 89.9     | 82.9     | 79.1    | **91.7** | **75.4** | **78.5** | [TTA](flan-t5-base_lora-16/layer_wise_concrete_adamerging_tta.csv)([meta-learn](flan-t5-base_lora-16/layer_wise_concrete_adamerging.csv)) |

#### Table: Multi-task performance when merging Flan-T5-large (LoRA fine-tuned) models on all eight tasks.

| **Method**                           | **CoLA** | **MNLI** | **MRPC** | **QNLI** | **QQP**  | **RTE**  | **SST2** | **STSB** | **Avg.** | source                                                                                                                                      |
| ------------------------------------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Individual                           | 80.2     | 88.5     | 89.2     | 94.4     | 87.2     | 91.7     | 95.2     | 90.9     | 89.6     | [file](flan-t5-large_lora-16/individuals.csv)                                                                                               |
| Weight Averaging                     | 74.6     | 84.3     | 84.1     | 92.8     | 86.3     | 87.4     | 94.8     | 88.0     | 86.5     | [file](flan-t5-large_lora-16/averaging.csv)                                                                                                 |
|                                      |          |          |          |          |          |          |          |          |          |                                                                                                                                             |
| *Task Arithmetic (TA)-Based*         |          |          |          |          |          |          |          |          |          |                                                                                                                                             |
| Task Arithmetic                      | 76.9     | 85.4     | 85.3     | 93.9     | 85.8     | 88.1     | 95.2     | 87.8     | 87.3     | [file](flan-t5-large_lora-16/task_arithmetic.csv)                                                                                           |
| Ties-Merging                         | **77.1** | 85.1     | **86.3** | 93.9     | **86.0** | 87.7     | 95.1     | **88.0** | 87.4     | [file](flan-t5-large_lora-16/ties_merging.csv)                                                                                              |
| **Concrete TA**                      | 76.6     | **86.4** | 86.0     | 93.9     | 85.9     | **88.4** | 95.2     | 87.9     | **87.5** | [meta-learn](flan-t5-large_lora-16/concrete_task_arithmetic.csv)                                                                            |
|                                      |          |          |          |          |          |          |          |          |          |                                                                                                                                             |
| *Layer-wise AdaMergin (LW AM)-Based* |          |          |          |          |          |          |          |          |          |                                                                                                                                             |
| LW AM                                | **76.7** | 87.6     | 84.8     | 93.8     | 85.9     | 88.1     | 95.2     | **88.6** | **87.6** | [TTA](flan-t5-large_lora-16/layer_wise_adamerging.csv)                                                                                      |
| **Concrete LW AM**                   | 76.1     | **87.7** | **85.5** | **93.8** | 85.9     | 88.1     | **95.4** | 87.1     | 87.5     | [TTA](flan-t5-large_lora-16/layer_wise_concrete_adamerging_tta.csv)([meta-learn](flan-t5-large_lora-16/layer_wise_concrete_adamerging.csv)) |


## OOD Experiments

```bash
source offline_mode.sh
# model=ViT-B-16 
model=ViT-B-32
function ood_exp(){
    version=$1
    corruption=$2
    python scripts/clip_concrete_task_arithmetic.py \
        lr=3e-4 \
        model=$model test_datasets="[Cars,EuroSAT,RESISC45,GTSRB]" \
        corruption=$corruption version=$version
}

CUDA_VISIBLE_DEVICES=0 ood_exp 2 null &
CUDA_VISIBLE_DEVICES=1 ood_exp 3 motion_blur &
CUDA_VISIBLE_DEVICES=2 ood_exp 4 impulse_noise &
CUDA_VISIBLE_DEVICES=3 ood_exp 5 gaussian_noise &
CUDA_VISIBLE_DEVICES=4 ood_exp 6 pixelate &
CUDA_VISIBLE_DEVICES=5 ood_exp 7 spatter &
CUDA_VISIBLE_DEVICES=6 ood_exp 8 contrast &
CUDA_VISIBLE_DEVICES=7 ood_exp 9 jpeg_compression &
```