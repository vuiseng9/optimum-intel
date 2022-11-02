from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class OVTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to OpenVINO/NNCF-enabled training flow
    """

    nncf_compression_config: str = field(default=None,
        metadata={"help": "NNCF configuration .json file for compression-enabled training"}
    )