uv run python -m annotation_pipeline.run \
    --hf_dataset_name MikiV/pubmedqa-meditron-conversations-labeled \
    --hf_dataset_split "train[0:25]" \
    --hf_dataset_subset "" \
    --output_hf_dataset_name "MikiV/pubmedqa-meditron-conversations-annotated-gpt" \
    --output_hf_dataset_split "test" \
    --model_id "gpt-5-search-api"
