export CODE_ROOT="/home/gkf/project/CausalSTDiT"
export PYTHONPATH=$PYTHONPATH:$CODE_ROOT

cd $CODE_ROOT
pwd


CFG_PATH=${1}
# EXP_NAME=${2}
MASTER_PORT=${2}
export CUDA_VISIBLE_DEVICES=${3}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

export IS_DEBUG=1
export DEBUG_COND_LEN=1
export DEBUG_WITHOUT_LOAD_PRETRAINED=0
export TOKENIZERS_PARALLELISM=false
torchrun \
    --nnodes=1 \
    --master-port=$MASTER_PORT \
    --nproc-per-node=$NUM_GPUS \
    scripts/train.py \
    --config $CODE_ROOT/$CFG_PATH

<<comment

# debug overfit

    bash scripts/train.sh \
    configs/causal_stdit/overfit_beach_25x256x256_ar8.py \
    9686 0

# train skytimelapse

    bash scripts/train.sh \
    configs/causal_stdit/train_SkyTimelapse_33x256x256_TPE33.py \
    9686 0

# train baseline:

    # full-attn fixed tpe
    bash /home/gkf/project/CausalSTDiT/scripts/train.sh \
    configs/baselines/full_attn_fixed_tpe.py \
    9686 0

    # causla attn fixed tpe
    bash /home/gkf/project/CausalSTDiT/scripts/train.sh \
    configs/baselines/exp2_causal_attn_fixed_tpe.py \
    9686 0

    # full attn cyclic tpe
    bash /home/gkf/project/CausalSTDiT/scripts/train.sh \
    configs/baselines/exp3_full_attn_cyclic_tpe64.py \
    9686 0

# train partial-causal:

    # overfit-beach
        bash /home/gkf/project/CausalSTDiT/scripts/train.sh \
        configs/causal_stdit/overfit_beach_ParitalCausal_CyclicTpe33.py \
        9686 0

    train skyline timelapse
        bash /home/gkf/project/CausalSTDiT/scripts/train.sh \
        configs/baselines/exp4_partialcausal_attn_cyclic_tpe33.py \
        9686 0
    


comment