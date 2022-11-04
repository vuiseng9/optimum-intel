#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=
export WANDB_API_KEY=

# ---------------------------------------------------------------------------------------------
# export WANDB_PROJECT="optimum-ov (sixer)"
export WANDB_PROJECT="p4-jpqd (sixer)"
export CUDA_VISIBLE_DEVICES=0

NEPOCH=20
RUNID=p4-jpqd-dev-optimum-mvmt-bert-${NEPOCH}eph
NNCFCFG=nncf_config/bert-base-movement-sparsity.json

OUTROOT=/data5/yourid/run/optimum-intel/
WORKDIR=/data5/yourid/dev/optimum-intel/optimum-intel/examples/openvino/question-answering

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
python run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad \
    --do_eval \
    --do_train \
    --learning_rate 3e-5 \
    --num_train_epochs $NEPOCH \
    --per_device_eval_batch_size 128 \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --logging_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --save_steps 2500 \
    --overwrite_output_dir \
    --run_name $RUNID \
    --output_dir $OUTDIR \
    --nncf_compression_config $NNCFCFG \
"
    # --nncf_config $NNCFCFG \
    # --teacher bert-large-uncased-whole-word-masking-finetuned-squad \
    # --teacher_ratio 0.9 \

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

# --teacher vuiseng9/bert-base-uncased-squad \
# --teacher_ratio 0.9 \
# --lr_scheduler_type cosine_with_restarts \
# --warmup_ratio 0.05 \

    # --optimize_model_before_eval  \
    # --optimized_checkpoint /data2/yourid/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled  \