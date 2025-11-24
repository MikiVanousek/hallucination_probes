uv run python -m annotation_pipeline.run \
    --hf_dataset_name MikiV/pubmedqa-meditron-conversations-labeled \
    --hf_dataset_split "train[100:102]" \
    --hf_dataset_subset "" \
    --output_hf_dataset_name "MikiV/pubmedqa-meditron-conversations-annotated2" \
    --output_hf_dataset_split "test" \
    --model_id "claude-sonnet-4-5-20250929"
