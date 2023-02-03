#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="p4-jpqd (mobilebert)"
export WANDB_RUN_GROUP="optimum-env"
export CUDA_VISIBLE_DEVICES=0

NEPOCH=16
NTXBLK=15
T=2
ALPHA=0.9
RUNID=optimum-jpqd-mobilebert-${NTXBLK}-txblk-squad-${NEPOCH}eph-T${T}-W${ALPHA}-r0.020-s3e10-fp32
NNCFCFG=/data5/vchua/dev/jpqd-mobilebert/optimum-intel/examples/openvino/question-answering/configs/mobilebert-base-jpqd.json

OUTROOT=/data3/vchua/run/jpqd-mobilebert/squad/mobilebert
WORKDIR=/data5/vchua/dev/jpqd-mobilebert/optimum-intel/examples/openvino/question-answering

CONDAROOT=/data5/vchua/miniconda3
CONDAENV=jpqd-mobilebert
# ---------------------------------------------------------------------------------------------

OUTDIR=$OUTROOT/$RUNID

# override label if in dryrun mode
if [[ $1 == "dryrun" ]]; then
    OUTDIR=$OUTROOT/dryrun-${RUNID}
    RUNID=dryrun-${RUNID}
fi

mkdir -p $OUTDIR
cd $WORKDIR
# --fp16
cmd="
python run_qa.py \
    --dataset_name squad \
    --model_name_or_path google/mobilebert-uncased \
    --num_tx_block $NTXBLK \
    --teacher_model_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --distillation_weight $ALPHA \
    --distillation_temperature $T \
    --do_eval \
    --do_train \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --optim adamw_torch \
    --num_train_epochs $NEPOCH \
    --per_device_eval_batch_size 128 \
    --per_device_train_batch_size 32 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --save_steps 500 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --nncf_compression_config $NNCFCFG \
    --run_name $RUNID \
    --output_dir $OUTDIR \
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

# --teacher vuiseng9/bert-base-uncased-squad \
# --teacher_ratio 0.9 \
# --lr_scheduler_type cosine_with_restarts \
# --warmup_ratio 0.05 \

    # --optimize_model_before_eval  \
    # --optimized_checkpoint /data2/vchua/tld-poc/bert-base-squadv1-local-hybrid-filled-lt-compiled  \