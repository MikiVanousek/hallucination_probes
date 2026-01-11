# Hallucination Detection for Meditron3-8B
This repository is a fork of obalcells/hallucination_probes

 for the project at EPFL's LiGHT laboratory, under the supervision of Annie Hartley and Fay Elhassan.

## Setup
We used an Nvidia H100 GPU with 80GB of memory to run the code in this repository.

All dependecies pinned with uv. Make sure you have uv installed and then run
```
uv sync
```
Set the following environment variables in your shell:
```
export WANDB_API_KEY=...
```

If you want to upload your models to HuggingFace, also set
```
export HF_WRITE_TOKEN=...
```


## Reproducing Results
### Creating the Datasets
#### Generate Meditron Answers
To create the dataset with annotated hallucinations, first generated answers to PubMedQA questions with Meditron3-8B. This can be done with
```
uv run python annotation_pipeline/meditron_dataset_creation.py \
    --max-samples 200 \
    --dataset-name pubmedqa-meditron-conversations \
    --username YOUR_HF_USERNAME \
```
The generated answers are on HuggingFace at `MikiV/pubmedqa-meditron-conversations-labeled`.

#### Annotating Hallucinations
Make sure you have set the `ANTHROPIC_API_KEY` environment variable:
```
export ANTHROPIC_API_KEY=...
```
To annotate the Meditron answers with hallucination labels, run
```
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

The annotated dataset is on HuggingFace at `MikiV/pubmedqa-meditron-conversations-annotated-claude`

#### Translate To Czech
To translate the annotated dtaset to Czech, run

```
uv run python translate_with_deepl_preserve_spans.py \
      --dataset MikiV/pubmedqa-meditron-conversations-annotated-claude \
      --split test \
      --target-lang de \
      --no-push
```

The translated dataset is on Hugging face at `MikiV/pubmedqa-meditron-conversations-annotated-claude-cz`

### Training Probes
For a given config file at `CONFIG_PATH`, run
```
uv run python -m probe.train --config CONFIG_PATH
``

Each probe is evaluated immidiately after training. The results are logged to Weights and Biases.


To reproduce the LoRA rank sweep, run training for all configs in `configs/lora_rank_sweep/`.

To reproduce the Multi-Layer residual probing, run training for all configs in `configs/tapped_layer_index/`.