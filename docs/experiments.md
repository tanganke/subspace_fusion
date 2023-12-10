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