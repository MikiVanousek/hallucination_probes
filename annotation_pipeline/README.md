# Annotation Pipeline

This pipeline annotates completions using Claude powered by web search.

## Requirements

Your HuggingFace dataset needs a `conversation` field containing a list of messages. The pipeline expects the instruction to be in the second-to-last message and the completion in the last message. All other fields in your dataset will be preserved unchanged.

## How it works

The pipeline processes each item in your dataset, using Claude to analyze the completion text with web search capabilities. The system adds an `annotations` field to your data containing the annotated spans with their labels (Supported, Not Supported, or Insufficient Information).

The default system prompt (in `entity_annotation.prompt`) instructs the annotator to focus on entities, but you can customize this by modifying that file to suit your annotation needs.

Results are saved locally to `<project_root>/annotation_pipeline_results/` as JSONL files and periodically synced to HuggingFace. The default is to push every 1000 processed items, but you can adjust this with `--push_intermediate_every N` (set to 0 to disable).

The pipeline automatically detects already-processed items to avoid duplicating work if you need to restart.

## Example Dataset Format

Your dataset items should look like this:

```json
{
  "conversation": [
    {"role": "user", "content": "Tell me about Paris"},
    {"role": "assistant", "content": "Paris is the capital of France..."}
  ],
  "any_other_field": "this will be preserved",
  "any_other_field_2": {"source": "example", "id": 123}
}
```

The pipeline will preserve all fields (`any_other_field`, `any_other_field_2`, etc.) and add the `annotations` field.

## Example

Here's an example of a response returned by the LLM:
~~~
Based on my searches, I can now fact-check the completion. Let me extract and verify the specific entities mentioned:

```json
[
  {
    "span": "Bill Jiang (aka \"HRV Guru\")",
    "label": "Not Supported",
    "verification_note": "No evidence found of a Bill Jiang known as 'HRV Guru' in triathlon coaching or HRV research. The search results show Dan Plews as a prominent HRV researcher in triathlon, but no Bill Jiang with this title."
  },
  {
    "span": "late 1990s",
    "label": "Not Supported", 
    "verification_note": "Since Bill Jiang as described appears to be fabricated, the timeline of his HRV work in the late 1990s cannot be verified."
  },
  {
    "span": "2013",
    "label": "Supported",
    "verification_note": "The Omron HeartGuide was not released in 2013. According to multiple sources, it was launched in 2019, not 2013."
  },
  {
    "span": "Omron HeartGuide smartwatch",
    "label": "Not Supported",
    "verification_note": "While the Omron HeartGuide exists, it was launched in 2019, not 2013 as claimed in the completion."
  },
  {
    "span": "HRV4Training (launched in 2013)",
    "label": "Supported",
    "verification_note": "HRV4Training app was indeed launched in 2013, as confirmed by the App Store listing showing August 14, 2013 as the release date."
  },
  {
    "span": "FirstBeat (launched in 2011)",
    "label": "Insufficient Information",
    "verification_note": "While Firstbeat Technologies exists and has various apps, I could not find specific evidence that their consumer app was launched in 2011. The search results show various Firstbeat products but no clear launch date for 2011."
  },
  {
    "span": "2018 meta-analysis published in the Journal of Strength and Conditioning Research",
    "label": "Insufficient Information",
    "verification_note": "While the Journal of Strength and Conditioning Research exists and publishes HRV-related research, I could not find the specific 2018 meta-analysis described in the completion about HRV and endurance performance."
  },
  {
    "span": "University of Nevada, Las Vegas",
    "label": "Insufficient Information",
    "verification_note": "Could not find the specific study mentioned about elite distance runners and HRV at UNLV."
  },
  {
    "span": "26.2-mile marathon",
    "label": "Supported",
    "verification_note": "A marathon is indeed 26.2 miles (42.195 kilometers)."
  },
  {
    "span": "2020 study published in the Journal of Sports Sciences",
    "label": "Insufficient Information",
    "verification_note": "Could not find the specific 2020 study about elite swimmers and HRV-guided periodization in the Journal of Sports Sciences."
  },
  {
    "span": "Daniella Ryf",
    "label": "Not Supported",
    "verification_note": "The athlete's name is Daniela Ryf (not Daniella). She is indeed a multiple-time Ironman World Champion, but there is no evidence she has been using HRV to inform her training since 2014."
  },
  {
    "span": "three-time Ironman World Champion",
    "label": "Not Supported",
    "verification_note": "Daniela Ryf is actually a five-time Ironman World Champion (2015, 2016, 2017, 2018, 2021), not three-time."
  },
  {
    "span": "using HRV to inform her training since 2014",
    "label": "Insufficient Information",
    "verification_note": "No evidence found that Daniela Ryf has been using HRV to inform her training since 2014 or that she attributes her success to this approach."
  }
]
```
~~~