from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, model_validator


class AnnotatedSpan(BaseModel):
    """Represents an entity with its verification status"""
    span: str = Field(description="The minimal span containing just the entity (e.g., 'Sarah Chen', not 'Dr. Sarah Chen from MIT')")
    label: Optional[Literal["Supported", "Not Supported", "Insufficient Information"]] = Field(
        description="Whether the entity/fact is verified as real, fabricated, or unverifiable. Downstream we treat 'Insufficient Information' in the same way as 'Not Supported'."
    )
    verification_note: str = Field(description="Brief explanation of the verification result")
    index: Optional[int] = Field(default=None, description="The index of the span within the completion/corpus of text")

    @model_validator(mode='before')
    @classmethod
    def validate_label(cls, values):
        """Validate and normalize the label field"""
        if isinstance(values, dict) and 'label' in values:
            valid_labels = ["Supported", "Not Supported", "Insufficient Information"]
            if values['label'] not in valid_labels:
                # Set to a default value instead of throwing an error
                values['label'] = None
        return values 


class DatasetItem(BaseModel):
    """Minimal dataset item for one-shot labeling pipeline.
    
    Only requires 'conversation' field. All other fields 
    from the original dataset will be preserved automatically.
    """
    model_config = {"extra": "allow"}  # Preserve all extra fields from the dataset
    
    # Required fields - only these are necessary for the pipeline
    conversation: List[Dict[str, Any]] = Field(description="The conversation between the user and the model")

    # Field added by the pipeline
    annotations: Optional[List[AnnotatedSpan]] = Field(default=None, description="List of annotated spans with their labels and verification notes")

