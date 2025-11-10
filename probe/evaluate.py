"""Evaluation script for hallucination detection probes."""

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.file_utils import load_yaml, save_jsonl
from utils.metrics import compute_clf_metrics, plot_roc_curves, print_eval_metrics
from utils.model_utils import load_model_and_tokenizer

from .config import EvaluationConfig
from .dataset import (
    TokenizedProbingDataset,
    create_probing_dataset,
    tokenized_probing_collate_fn,
)
from .loss import compute_probe_bce_loss
from .value_head_probe import ValueHeadProbe, setup_probe


@torch.no_grad()
def evaluate_probe(
    probe: ValueHeadProbe,
    eval_dataloader: DataLoader,
    threshold: float = 0.5,
    metric_key_prefix: Optional[str] = None,
    verbose: bool = True,
    save_roc_curves: bool = True,
    save_dir: Optional[Path] = None,
    dump_raw_results: bool = False,
    compute_ci: bool = True,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Evaluate a probe on a dataset.

    Args:
        probe: The probe to evaluate
        eval_dataloader: DataLoader for evaluation data
        threshold: Classification threshold
        metric_key_prefix: Prefix for metric keys
        verbose: Whether to print metrics
        save_roc_curves: Whether to save ROC curve plots
        save_dir: Directory to save results
        dump_raw_results: Whether to save raw predictions
        compute_ci: Whether to compute confidence intervals
        n_bootstrap: Number of bootstrap samples for CI
        confidence_level: Confidence level for CI (e.g., 0.95 for 95% CI)
        random_state: Random seed for bootstrap sampling

    Returns:
        Dictionary of evaluation metrics
    """
    # Force garbage collection before evaluation
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize metric collections for different aggregation levels
    all_probs = {"all": [], "span": [], "span_max": []}
    all_preds = {"all": [], "span": [], "span_max": []}
    all_labels = {"all": [], "span": [], "span_max": []}

    total_lm_loss = 0
    total_probe_loss = 0
    total_sparsity = 0
    num_batches = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(probe.device)
        attention_mask = batch["attention_mask"].to(probe.device)
        classification_labels = batch["classification_labels"].to(probe.device)
        classification_weights = batch["classification_weights"].to(probe.device)
        lm_labels = batch["lm_labels"].to(probe.device)
        pos_spans = batch["pos_spans"]
        neg_spans = batch["neg_spans"]

        # Forward pass
        outputs = probe(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )

        probe_logits = outputs["probe_logits"].squeeze(-1)
        probe_probs = torch.sigmoid(probe_logits).float()
        probe_preds = (probe_probs > threshold).float()

        probe_loss = compute_probe_bce_loss(
            probe_logits=probe_logits,
            classification_labels=classification_labels,
            classification_weights=classification_weights,
        )

        # 1. All-token metrics (excluding padding and ignored tokens)
        valid_mask = (attention_mask == 1) & (classification_labels != -100.0)
        all_probs["all"].extend(probe_probs[valid_mask].cpu().numpy())
        all_preds["all"].extend(probe_preds[valid_mask].cpu().numpy())
        all_labels["all"].extend(classification_labels[valid_mask].cpu().numpy())

        # 2. Span-level metrics
        annotated_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(len(input_ids)):
            span_list = pos_spans[i] + neg_spans[i]
            for span_range in span_list:
                start, end = span_range[0], span_range[1]
                assert start <= end, f"Invalid span range: {span_range}"
                annotated_tokens_mask[i, start : end + 1] = True

        # Filter out ignored tokens
        annotated_tokens_mask = annotated_tokens_mask & (
            classification_labels != -100.0
        )

        all_probs["span"].extend(probe_probs[annotated_tokens_mask].cpu().numpy())
        all_preds["span"].extend(probe_preds[annotated_tokens_mask].cpu().numpy())
        all_labels["span"].extend(
            classification_labels[annotated_tokens_mask].cpu().numpy()
        )

        # 3. Span-level metrics (with max aggregation)
        for i in range(len(input_ids)):
            all_spans_max = pos_spans[i] + neg_spans[i]
            if len(all_spans_max) == 0:
                continue

            span_labels = [1.0] * len(pos_spans[i]) + [0.0] * len(neg_spans[i])

            for label, (start, end) in zip(span_labels, all_spans_max):
                max_prob = probe_probs[i, start : end + 1].max().cpu().item()
                max_pred = probe_preds[i, start : end + 1].max().cpu().item()

                all_probs["span_max"].append(max_prob)
                all_preds["span_max"].append(max_pred)
                all_labels["span_max"].append(label)

        # Update running metrics
        total_lm_loss += outputs["lm_loss"].item()
        total_probe_loss += probe_loss.item()
        total_sparsity += probe_probs[attention_mask == 1].mean().item()
        num_batches += 1

    # Compute average metrics
    metrics = {
        "lm_loss": total_lm_loss / num_batches,
        "probe_loss": total_probe_loss / num_batches,
        "sparsity": total_sparsity / num_batches,
        "probe_threshold": threshold,
    }

    # Convert lists to numpy arrays
    all_probs = {k: np.array(v) for k, v in all_probs.items()}
    all_preds = {k: np.array(v) for k, v in all_preds.items()}
    all_labels = {k: np.array(v) for k, v in all_labels.items()}

    # Compute classification metrics for each aggregation level
    for agg_level in ["all", "span", "span_max"]:
        if len(all_labels[agg_level]) == 0:
            continue

        clf_metrics = compute_clf_metrics(
            all_preds[agg_level],
            all_labels[agg_level],
            all_probs[agg_level],
            compute_ci=compute_ci,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        )

        for metric_name, metric_value in clf_metrics.items():
            metrics[f"{agg_level}_{metric_name}"] = metric_value

    # Add prefix if specified
    if metric_key_prefix:
        metrics = {f"{metric_key_prefix}/{k}": v for k, v in metrics.items()}

    # Print metrics if verbose
    if verbose:
        print_eval_metrics(
            metrics,
            metric_key_prefix=metric_key_prefix or "",
            all_labels=all_labels,
            include_random_baseline=True,
        )

    # Save ROC curves
    if save_roc_curves and save_dir:
        plot_roc_curves(
            all_preds,
            all_labels,
            all_probs,
            save_dir=str(save_dir),
            prefix=metric_key_prefix,
        )

    # Save raw results if requested
    if dump_raw_results and save_dir:
        results_dir = Path(save_dir) / "eval_results"
        if metric_key_prefix:
            results_dir = results_dir / metric_key_prefix
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Save raw predictions
        for agg_level in ["span_max"]:
            if len(all_labels[agg_level]) > 0:
                np.save(results_dir / f"{agg_level}_probs.npy", all_probs[agg_level])
                np.save(results_dir / f"{agg_level}_preds.npy", all_preds[agg_level])
                np.save(results_dir / f"{agg_level}_labels.npy", all_labels[agg_level])

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def evaluate_on_multiple_datasets(
    probe: ValueHeadProbe,
    eval_config: EvaluationConfig,
    tokenizer: AutoTokenizer,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate probe on multiple datasets.

    Args:
        probe: The probe to evaluate
        eval_config: Evaluation configuration
        tokenizer: Tokenizer for the model

    Returns:
        Dictionary mapping dataset IDs to their metrics
    """
    all_metrics = {}

    # Create output directory
    output_dir = (
        Path(eval_config.output_dir)
        if eval_config.output_dir
        else Path("evaluation_results")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate on each dataset
    for dataset_config in eval_config.dataset_configs:
        print(f"\nEvaluating on {dataset_config.dataset_id}...")

        # Create dataset
        dataset = create_probing_dataset(dataset_config, tokenizer)
        print(f"  Dataset size: {len(dataset)} samples")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=eval_config.per_device_eval_batch_size,
            collate_fn=tokenized_probing_collate_fn,
            shuffle=False,
        )

        # Evaluate
        metrics = evaluate_probe(
            probe=probe,
            eval_dataloader=dataloader,
            threshold=eval_config.probe_config.threshold,
            metric_key_prefix=dataset_config.dataset_id,
            verbose=True,
            save_roc_curves=eval_config.save_roc_curves,
            save_dir=output_dir if eval_config.save_roc_curves else None,
            dump_raw_results=eval_config.save_raw_results,
            compute_ci=eval_config.compute_ci,
            n_bootstrap=eval_config.n_bootstrap,
            confidence_level=eval_config.confidence_level,
            random_state=eval_config.ci_random_state,
        )

        # Save metrics
        all_metrics[dataset_config.dataset_id] = metrics
        # Create a copy without the dataset_id for saving
        save_metrics = {k: v for k, v in metrics.items()}
        save_metrics["dataset_id"] = dataset_config.dataset_id
        save_jsonl([save_metrics], str(output_dir / "eval_metrics.jsonl"), append=True)

    return all_metrics


def main(eval_config: EvaluationConfig):
    """Main evaluation function."""
    # Load environment variables from .env if present
    load_dotenv()
    print("Evaluation Configuration:")
    for key, value in eval_config.__dict__.items():
        print(f"  {key}: {value}")

    # Load model and tokenizer
    print(f"\nLoading model: {eval_config.probe_config.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        eval_config.probe_config.model_name,
    )

    tokenizer.padding_side = "right"

    # Load probe using the config
    print(f"\nLoading probe: {eval_config.probe_config.probe_id}")
    model, probe = setup_probe(model, eval_config.probe_config)

    print(f"Device: {next(probe.parameters()).device}")

    # Run evaluation
    print(f"\nEvaluating on {len(eval_config.dataset_configs)} datasets...")
    results = evaluate_on_multiple_datasets(probe, eval_config, tokenizer)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for dataset_id, metrics in results.items():
        print(f"\n{dataset_id}:")
        key_metrics = ["span_max_accuracy", "span_max_f1", "span_max_auc"]
        for metric in key_metrics:
            full_key = f"{dataset_id}/{metric}"
            if full_key in metrics:
                value = metrics[full_key]
                ci_lower_key = f"{dataset_id}/{metric}_ci_lower"
                ci_upper_key = f"{dataset_id}/{metric}_ci_upper"

                if ci_lower_key in metrics and ci_upper_key in metrics:
                    ci_lower = metrics[ci_lower_key]
                    ci_upper = metrics[ci_upper_key]
                    if not (np.isnan(ci_lower) or np.isnan(ci_upper)):
                        print(
                            f"  {metric}: {value:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})"
                        )
                    else:
                        print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a probe")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to evaluation configuration file",
    )

    args = parser.parse_args()

    # Load config from YAML
    config_dict = load_yaml(args.config)
    eval_config = EvaluationConfig(**config_dict)

    main(eval_config)
