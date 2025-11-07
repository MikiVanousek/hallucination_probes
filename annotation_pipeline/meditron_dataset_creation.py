#!/usr/bin/env python3
"""
Script to create a conversation dataset using PubMedQA and OpenMeditron/Meditron3-8B model.

This script:
1. Loads the PubMedQA dataset
2. Uses OpenMeditron/Meditron3-8B to generate responses
3. Formats conversations according to the annotation pipeline requirements
4. Pushes the dataset to HuggingFace under username MikiV
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
import tqdm
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeditronConversationGenerator:
    """Generator for creating conversations using Meditron model."""

    def __init__(self, model_name: str = "OpenMeditron/Meditron3-8B"):
        """Initialize the conversation generator.

        Args:
            model_name: Name of the Meditron model to use
        """
        self.model_name = model_name
        logger.info(f"Loading model: {model_name}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model loaded successfully")

    def generate_response(
        self, question: str, context: str = None, max_length: int = 512
    ) -> str:
        """Generate a response using the Meditron model.

        Args:
            question: The medical question to answer
            context: Optional context/abstract to base the answer on
            max_length: Maximum length of generated response

        Returns:
            Generated response text
        """
        # Construct the prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        # Tokenize input
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and clean up response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt) :].strip()

        return response


def load_pubmedqa_dataset(subset: str = "pqa_labeled") -> Dataset:
    """Load the PubMedQA dataset.

    Args:
        subset: Which subset of PubMedQA to load

    Returns:
        Loaded dataset
    """
    logger.info(f"Loading PubMedQA dataset: {subset}")
    dataset = load_dataset("pubmed_qa", subset)

    # Use train split, or first available split
    if "train" in dataset:
        return dataset["train"]
    else:
        available_splits = list(dataset.keys())
        logger.warning(f"No train split found. Using {available_splits[0]}")
        return dataset[available_splits[0]]


def create_conversation_dataset(
    pubmedqa_data: Dataset,
    generator: MeditronConversationGenerator,
    max_samples: int = None,
) -> List[Dict[str, Any]]:
    """Create conversation dataset from PubMedQA using Meditron.

    Args:
        pubmedqa_data: PubMedQA dataset
        generator: Meditron conversation generator
        max_samples: Maximum number of samples to process (None for all)

    Returns:
        List of conversation dictionaries
    """
    conversations = []

    # Limit samples if specified
    if max_samples:
        pubmedqa_data = pubmedqa_data.select(
            range(min(max_samples, len(pubmedqa_data)))
        )

    logger.info(f"Processing {len(pubmedqa_data)} samples")

    for i, item in enumerate(tqdm.tqdm(pubmedqa_data, desc="Generating conversations")):
        try:
            # Extract information from PubMedQA item
            question = item["question"]
            context = item.get("context", {})

            # Combine context information if available
            context_text = ""
            if isinstance(context, dict):
                if "contexts" in context:
                    context_text = " ".join(context["contexts"])
                elif "abstract" in context:
                    context_text = context["abstract"]
            elif isinstance(context, str):
                context_text = context

            # Generate response using Meditron
            response = generator.generate_response(question, context_text)

            # Format as conversation
            conversation_item = {
                "conversation": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response},
                ],
                "pubmedqa_id": item.get("pubid", f"item_{i}"),
                "original_answer": item.get("final_decision", ""),
                "context": context_text[:1000]
                if context_text
                else "",  # Truncate context
                "source": "pubmedqa_meditron",
            }

            conversations.append(conversation_item)

            # Log progress every 100 items
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1} items")

        except Exception as e:
            logger.error(f"Error processing item {i}: {str(e)}")
            continue

    logger.info(f"Successfully generated {len(conversations)} conversations")
    return conversations


def save_dataset_locally(conversations: List[Dict[str, Any]], output_path: str):
    """Save the conversation dataset locally.

    Args:
        conversations: List of conversation dictionaries
        output_path: Path to save the dataset
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            json.dump(conv, f, ensure_ascii=False)
            f.write("\n")

    logger.info(f"Dataset saved locally to: {output_path}")


def push_to_huggingface(
    conversations: List[Dict[str, Any]], dataset_name: str, username: str = "MikiV"
):
    """Push the dataset to HuggingFace Hub.

    Args:
        conversations: List of conversation dictionaries
        dataset_name: Name for the dataset on HF
        username: HuggingFace username
    """
    logger.info(f"Creating HuggingFace dataset: {username}/{dataset_name}")

    # Create HF dataset
    hf_dataset = Dataset.from_list(conversations)

    # Push to hub
    repo_id = f"{username}/{dataset_name}"
    hf_dataset.push_to_hub(repo_id, private=False)

    logger.info(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}")


def main():
    """Main function to run the dataset creation pipeline."""
    parser = argparse.ArgumentParser(
        description="Create Meditron conversation dataset from PubMedQA"
    )
    parser.add_argument(
        "--model", default="OpenMeditron/Meditron3-8B", help="Meditron model to use"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--output-dir",
        default="./meditron_conversations",
        help="Output directory for local files",
    )
    parser.add_argument(
        "--dataset-name",
        default="pubmedqa-meditron-conversations",
        help="Name for the HuggingFace dataset",
    )
    parser.add_argument("--username", default="MikiV", help="HuggingFace username")
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push to HuggingFace, only save locally",
    )
    parser.add_argument(
        "--pubmedqa-subset", default="pqa_labeled", help="PubMedQA subset to use"
    )

    args = parser.parse_args()

    try:
        # Load PubMedQA dataset
        pubmedqa_data = load_pubmedqa_dataset(args.pubmedqa_subset)

        # Initialize Meditron generator
        generator = MeditronConversationGenerator(args.model)

        # Generate conversations
        conversations = create_conversation_dataset(
            pubmedqa_data, generator, max_samples=args.max_samples
        )

        if not conversations:
            logger.error("No conversations generated. Exiting.")
            return

        # Save locally
        output_file = Path(args.output_dir) / f"{args.dataset_name}.jsonl"
        save_dataset_locally(conversations, output_file)

        # Push to HuggingFace if requested
        if not args.no_push:
            push_to_huggingface(conversations, args.dataset_name, args.username)

        logger.info("Dataset creation completed successfully!")

    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
