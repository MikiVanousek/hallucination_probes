# Hallucination Detection for Meditron3-8B

This repository is a fork of [obalcells/hallucination_probes](https://github.com/obalcells/hallucination_probes) developed at EPFL's LiGHT laboratory under the supervision of Annie Hartley and Fay Elhassan.

## Setup

### Hardware Requirements
This project was developed and tested on an Nvidia H100 GPU with 80GB of memory.

### Installation

All dependencies are pinned using `uv` for reproducibility. Ensure you have [`uv` installed]( https://docs.astral.sh/uv/getting-started/installation/ ), then run:

```bash
uv sync
```

### Environment Configuration

Set the required environment variables in your shell:

```bash
export WANDB_API_KEY=...
```

If you plan to upload models to HuggingFace, also add:

```bash
export HF_WRITE_TOKEN=...
```

### Meditron3-8B Access
Before you can pull the Meditron3-8B model from HuggingFace, you need to accept the model's terms of use. Visit the [Meditron3-8B model page](https://huggingface.co/OpenMeditron/Meditron3-8B) and accept the terms to gain access.

## Reproducing Results

### Creating the Datasets

#### Step 1: Generate Meditron Answers

To create the dataset with annotated hallucinations, first generate answers to PubMedQA questions using Meditron3-8B:

```bash
uv run python annotation_pipeline/meditron_dataset_creation.py \
    --max-samples 200 \
    --dataset-name pubmedqa-meditron-conversations \
    --username YOUR_HF_USERNAME
```

The generated answers are available on [HuggingFace](https://huggingface.co/datasets/MikiV/pubmedqa-meditron-conversations-labeled) at `MikiV/pubmedqa-meditron-conversations-labeled`.

#### Step 2: Annotate Hallucinations

First, ensure you have set the Anthropic API key:

```bash
export ANTHROPIC_API_KEY=...
```

Next, annotate the Meditron answers with hallucination labels:

```bash
uv run python -m annotation_pipeline.run \
    --hf_dataset_name MikiV/pubmedqa-meditron-conversations-labeled \
    --hf_dataset_split "train[0:200]" \
    --hf_dataset_subset "" \
    --output_hf_dataset_name "MikiV/pubmedqa-meditron-conversations-annotated-claude" \
    --output_hf_dataset_split "test" \
    --push_intermediate_every 10 \
    --parallel False \
    --model_id "claude-sonnet-4-20250514"
```

The annotated dataset is available on [HuggingFace](https://huggingface.co/datasets/MikiV/pubmedqa-meditron-conversations-annotated-claude) at `MikiV/pubmedqa-meditron-conversations-annotated-claude`.

#### Step 3: Translate to Czech

To translate the annotated dataset to Czech, run:

```bash
uv run python translate_with_deepl_preserve_spans.py \
    --dataset MikiV/pubmedqa-meditron-conversations-annotated-claude \
    --split test \
    --target-lang de \
    --no-push
```

The translated dataset is available on [HuggingFace](https://huggingface.co/datasets/MikiV/pubmedqa-meditron-conversations-annotated-claude-cz) at `MikiV/pubmedqa-meditron-conversations-annotated-claude-cz`.

### Training Probes

To train a probe using a specified configuration file, run:

```bash
uv run python -m probe.train --config CONFIG_PATH
```

Each probe is automatically evaluated immediately after training, with results logged to Weights and Biases for tracking and analysis.

#### Running Experimental Sweeps

**LoRA Rank Sweep:** To reproduce the LoRA rank sweep experiments, train all configurations in `configs/lora_rank_sweep/`.

**Multi-Layer Residual Probing:** To reproduce the multi-layer residual probing experiments, train all configurations in `configs/tapped_layer_index/`.