from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM, OVQuantizer
import logging
import nncf
import gc

nncf.set_log_level(logging.ERROR)

# model_id = 'EleutherAI/gpt-j-6b'
# model_id = 'meta-llama/Llama-2-7b-chat-hf'
# model_id='facebook/opt-125m'
model_id = 'facebook/opt-2.7b'
# model_id='facebook/opt-6.7b'
# model_id='facebook/opt-13b'

model_root = "/data1/vchua/ov-llm"
pt_model_id = model_id

CONVERT_INT8 = True
EXPORT_FP32 = True

# rarely enable the following
CONVERT_FP16 = False
QUANTIZE_INT8 = False

tokenizer=AutoTokenizer.from_pretrained(pt_model_id)

model_dir = Path(model_root, model_id) / "INT8_weights"
if CONVERT_INT8 and not model_dir.exists():
    pt_model = AutoModelForCausalLM.from_pretrained(pt_model_id)
    quantizer = OVQuantizer.from_pretrained(pt_model)
    quantizer.quantize(save_directory=model_dir, weights_only=True)    
    tokenizer.save_pretrained(model_dir)

    del quantizer
    del pt_model
gc.collect()

if False:
    #need calibration data
    model_dir = Path(model_root, model_id) / "INT8"
    if CONVERT_INT8 and not model_dir.exists():
        pt_model = AutoModelForCausalLM.from_pretrained(pt_model_id)
        quantizer = OVQuantizer.from_pretrained(pt_model)
        quantizer.quantize(save_directory=model_dir, weights_only=False)    
        tokenizer.save_pretrained(model_dir)

        del quantizer
        del pt_model
    gc.collect()

    model_dir = Path(model_root, model_id) / "FP16"
    if CONVERT_FP16 and not model_dir.exists():
        ov_model = OVModelForCausalLM.from_pretrained(pt_model_id, export=True, compile=False)
        ov_model.half()
        ov_model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        del ov_model

model_dir = Path(model_root, model_id) / "FP32"
if EXPORT_FP32 and not model_dir.exists():
    ov_model = OVModelForCausalLM.from_pretrained(pt_model_id, export=True, compile=False)
    ov_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    del ov_model
gc.collect()

print("done.")