# script to generate command to benchmark 2nd token latency of set of sparse quantized models

import os
import json
from collections import OrderedDict

def generate_input_shape(ctxlen, bs=1):
    # hardcoding
    n_kvlayer = 24
    n_head = 16
    head_dim = 64
    shape_str = ""
    for l in range(n_kvlayer):
        shape_str += f"past_key_values.{l}.key[{bs},{n_head},{ctxlen+1},{head_dim}],past_key_values.{l}.value[{bs},{n_head},{ctxlen+1},{head_dim}],"

    return shape_str + f"input_ids[{bs},1],attention_mask[{bs},{ctxlen+2}]"

MODELROOT="/data1/vchua/temp/synthetic-sparse-8bit-opt-350m"

sweep_length = [16, 32, 64, 256, 512, 1024, 2016]
sweep_beam = [1, 4]

sparsity_model_map = {
    "0.1": "hand-zeroed-10pc-ov-opt-350m-8bit-kv-cache",
    "0.2": "hand-zeroed-20pc-ov-opt-350m-8bit-kv-cache",
    "0.3": "hand-zeroed-30pc-ov-opt-350m-8bit-kv-cache",
    "0.4": "hand-zeroed-40pc-ov-opt-350m-8bit-kv-cache",
    "0.5": "hand-zeroed-50pc-ov-opt-350m-8bit-kv-cache",
    "0.6": "hand-zeroed-60pc-ov-opt-350m-8bit-kv-cache",
    "0.7": "hand-zeroed-70pc-ov-opt-350m-8bit-kv-cache",
    "0.8": "hand-zeroed-80pc-ov-opt-350m-8bit-kv-cache",
    "0.9": "hand-zeroed-90pc-ov-opt-350m-8bit-kv-cache",
}

benchhdl = open("bench_cmds.txt", "w")
benchhdl.write("MODELROOT=\n")
benchhdl.write("OUTLOGROOT=\n")

nloop = 0
for sparsity, model_folder in sparsity_model_map.items():
    
    sparsity_threshold = float(sparsity) - 0.05 # need to be lower so that they trigger TLD operator

    rtcfg_filename = f"ov_rt_cfg_{sparsity}.json"

    cfg = dict(CPU=dict(CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE=sparsity_threshold))

    with open(f"{MODELROOT}/{model_folder}/{rtcfg_filename}", "w") as outfile:
        json.dump(cfg, outfile, indent=4)


    for beam in sweep_beam:
        for ctxlen in sweep_length:
            input_shape = generate_input_shape(ctxlen=ctxlen,bs=beam)

            cmd_goto = f"cd $MODELROOT/{model_folder}"
            cmd = f"benchmark_app -m openvino_model.xml -load_config {rtcfg_filename} -shape {input_shape} 2>&1 | tee $OUTLOGROOT/log.$HOSTNAME.2nd_token_tput__ctxlen-{ctxlen}__beam-{beam}__sparsity-{sparsity}_"

            benchhdl.write(f"\n\n# Run {nloop} {'-'*50}\n")
            benchhdl.write(f"echo Info: Run {nloop} ...\n")
            benchhdl.write(f"sleep 1m\n")
            benchhdl.write(f"{cmd_goto}\n")
            benchhdl.write(f"{cmd}\n")
            
            print(f"\n\n# Run {nloop}")
            print(cmd_goto)
            print(cmd)
            nloop +=1

# non-tld
for beam in sweep_beam:
    for ctxlen in sweep_length:
        input_shape = generate_input_shape(ctxlen=ctxlen,bs=beam)

        cmd_goto = f"cd $MODELROOT/{model_folder}"
        cmd = f"benchmark_app -m openvino_model.xml -shape {input_shape} 2>&1 | tee $OUTLOGROOT/log.$HOSTNAME.NoTLD_2nd_token_tput__ctxlen-{ctxlen}__beam-{beam}__sparsity-0.0_"

        benchhdl.write(f"\n\n# Run {nloop} - NoTLD {'-'*50}\n")
        benchhdl.write(f"echo Info: Run {nloop} ...\n")
        benchhdl.write(f"sleep 1m\n")
        benchhdl.write(f"{cmd_goto}\n")
        benchhdl.write(f"{cmd}\n")
            
        print(f"\n\n# Run {nloop} - NoTLD")
        print(cmd_goto)
        print(cmd)
        nloop +=1

    cmd_goto = f"cd $MODELROOT/{model_folder}"
    
benchhdl.close()
print(nloop)
