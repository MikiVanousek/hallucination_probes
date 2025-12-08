uv run python -m annotation_pipeline.run \
    --hf_dataset_name MikiV/pubmedqa-meditron-conversations-labeled \
    --hf_dataset_split "train[0:200]" \
    --hf_dataset_subset "" \
    --output_hf_dataset_name "MikiV/pubmedqa-meditron-conversations-annotated-claude" \
    --output_hf_dataset_split "test" \
    --push_intermediate_every 10 \
    --parallel False \
    --model_id "claude-sonnet-4-20250514"
    # --model_id "gpt-5-search-api"
