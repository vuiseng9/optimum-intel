#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=
export WANDB_API_KEY=

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="optimum-ov (sixer)"
export CUDA_VISIBLE_DEVICES=0

NEPOCH=8
RUNID=image/p4-jpqd-dev--mvmt-ft-swin-b-food101-${NEPOCH}eph
NNCFCFG=nncf_config/swin-base-movement-sparsity.json

OUTROOT=/data5/yourid/run/optimum-intel/
WORKDIR=/data5/yourid/dev/optimum-intel/optimum-intel/examples/openvino/image-classification

CONDAROOT=/data5/yourid/miniconda3
CONDAENV=optimum-intel
# ---------------------------------------------------------------------------------------------

OUTDIR=$OUTROOT/$RUNID

# override label if in dryrun mode
if [[ $1 == "dryrun" ]]; then
    OUTDIR=$OUTROOT/dryrun-${RUNID}
    RUNID=dryrun-${RUNID}
fi

mkdir -p $OUTDIR
cd $WORKDIR


cmd="
python run_image_classification.py \
    --model_name_or_path microsoft/swin-base-patch4-window7-224 \
    --ignore_mismatched_sizes \
    --dataset_name food101 \
    --remove_unused_columns False \
    --dataloader_num_workers 8 \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs $NEPOCH \
    --logging_steps 1 \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_steps 1000 \
    --seed 42 \
    --overwrite_output_dir \
    --run_name $RUNID \
    --output_dir $OUTDIR \
    --nncf_compression_config $NNCFCFG \
"


if [[ $1 == "local" ]]; then
    echo "${cmd}" > $OUTDIR/run.log
    echo "### End of CMD ---" >> $OUTDIR/run.log
    cmd="nohup ${cmd}"
    eval $cmd >> $OUTDIR/run.log 2>&1 &
    echo "logpath: $OUTDIR/run.log"
elif [[ $1 == "dryrun" ]]; then
    echo "[INFO: dryrun, add --max_steps 25 to cli"
    cmd="${cmd} --max_steps 25"
    echo "${cmd}" > $OUTDIR/dryrun.log
    echo "### End of CMD ---" >> $OUTDIR/dryrun.log
    eval $cmd >> $OUTDIR/dryrun.log 2>&1 &
    echo "logpath: $OUTDIR/dryrun.log"
else
    source $CONDAROOT/etc/profile.d/conda.sh
    conda activate ${CONDAENV}
    eval $cmd
fi

