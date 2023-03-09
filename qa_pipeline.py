import os
from transformers import AutoTokenizer, pipeline
from optimum.intel.openvino import OVModelForQuestionAnswering
import torch
import numpy as np

# Set this serialize intermediate FM for comparison;
os.environ["OV_DUMP_N_OFM"]="30"
os.environ["OV_DUMP_PATH"]="/tmp/dump_delta"

model_dir="/home/vchua/jpqd-mobilebert-9500/checkpoint-9500"

# FP32 Inference  ------------------------------------------------------------------------------------------
os.environ["OV_INFER_BF16"]="f32"
ov_model_fp32 = OVModelForQuestionAnswering.from_pretrained(model_dir, file_name="openvino_model.xml")
tokenizer_fp32 = AutoTokenizer.from_pretrained(model_dir)
question_answerer_fp32 = pipeline("question-answering", model=ov_model_fp32, tokenizer=tokenizer_fp32)

preds_fp32 = question_answerer_fp32(
    question="What is the name of the repository?",
    context="The name of the repository is huggingface/transformers",
)
fp32_dump_path = os.environ["OV_DUMP_PATH"] + "_" + os.environ.get("OV_INFER_BF16", "f32")+ ".pth"

# BF16 Inference ------------------------------------------------------------------------------------------
os.environ["OV_INFER_BF16"]="bf16"
ov_model_bf16 = OVModelForQuestionAnswering.from_pretrained(model_dir, file_name="openvino_model.xml")
tokenizer_bf16 = AutoTokenizer.from_pretrained(model_dir)
question_answerer_bf16 = pipeline("question-answering", model=ov_model_bf16, tokenizer=tokenizer_bf16)

preds_bf16 = question_answerer_bf16(
    question="What is the name of the repository?",
    context="The name of the repository is huggingface/transformers",
)
bf16_dump_path = os.environ["OV_DUMP_PATH"] + "_" + os.environ.get("OV_INFER_BF16", "f32") + ".pth"

# compare
fp32_tensors = torch.load(fp32_dump_path)
bf16_tensors = torch.load(bf16_dump_path)

assert len(set(fp32_tensors.keys()) - set(bf16_tensors.keys())) == 0, "bug, keys should match "

for k, fp32_t in fp32_tensors.items():
    bf16_t = bf16_tensors[k]
    delta = bf16_t - fp32_t
    a_delta = np.absolute(delta)
    r_delta = np.absolute(a_delta/fp32_t)*100
    print("max relative: {:12.1f}% | max absolute: {:12.5f} | {}".format(r_delta.max(), a_delta.max(), k))

# model_dir="helenai/bert-base-uncased-squad-v1-jpqd-ov-int8"
# ov_model = OVModelForQuestionAnswering.from_pretrained(model_dir)
