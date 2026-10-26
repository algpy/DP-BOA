#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/run.sh <pretrain|full|fd> <dataset> [arguments]

Datasets: aircraft, Animalia, Arachnida, scars, cifar100, cub,
          domainnet, Fungi, herbarium_19, imagenet_100, Mollusca, pets
Aliases: cars -> scars; imagenet100 -> imagenet_100;
         animalia, arachnida, fungi, mollusca -> capitalized names

Set EXP_ROOT, WARMUP_MODEL_DIR and CUDA_VISIBLE_DEVICES as needed.
EOF
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if (( $# < 2 )); then
    usage >&2
    exit 2
fi

stage=$1
dataset=$2
shift 2

case "$stage" in
    pretrain|full|fd) ;;
    *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac
case "$dataset" in
    cars|scars) dataset=scars ;;
    imagenet100|imagenet_100) dataset=imagenet_100 ;;
    animalia|Animalia) dataset=Animalia ;;
    arachnida|Arachnida) dataset=Arachnida ;;
    fungi|Fungi) dataset=Fungi ;;
    mollusca|Mollusca) dataset=Mollusca ;;
    aircraft|cifar100|cub|domainnet|herbarium_19|pets) ;;
    *) printf 'Unknown dataset: %s\n' "$dataset" >&2; exit 2 ;;
esac
if [[ $stage == fd && ( $dataset == aircraft || $dataset == domainnet ) ]]; then
    printf 'No FD preset is provided for %s.\n' "$dataset" >&2
    exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_root"

seed=${1:-1}
epoch=100
batch_size=256
grad_from_block=12
memax_weight=1
entry=main.py
prefix=$dataset
domain_suffix=
extra_args=()

if [[ $dataset == cifar100 ]]; then
    prefix=cifar100_80
    # The existing FD preset uses its Python default for this option.
    if [[ $stage != fd ]]; then
        extra_args+=(--num_labeled_classes 80)
    fi
fi
if [[ $dataset == domainnet ]]; then
    src=${3:-real}
    aux=${4:-clipart}
    domain_suffix="_src_env${src}._aux_env${aux}."
    extra_args+=(--src_env "$src" --aux_env "$aux")
else
    extra_args+=(--use_ssb_splits)
fi

if [[ $stage == pretrain ]]; then
    entry=supervised_pretrain.py
    epoch=${2:-100}
    grad_from_block=11
    if [[ $dataset != domainnet ]]; then
        batch_size=${3:-256}
    fi
    case "$dataset" in
        cub|pets|Animalia|Arachnida|Fungi|Mollusca) memax_weight=2 ;;
    esac
    exp_name="${prefix}_labeled_data_supervised_pretrain_epoch${epoch}.${domain_suffix}_seed${seed}"
    extra_args+=(--pretrain)
else
    alpha=${2:-1e-9}
    cov_suffix=_covModefull.
    if [[ $stage == fd ]]; then
        entry=FD_memory.py
        default_rank=32
        case "$dataset" in
            cub|scars|Animalia) default_rank=16 ;;
        esac
        fd_rank=${2:-$default_rank}
        fd_ell=${3:-32}
        alpha=${4:-1e-9}
        extra_args+=(--fd_rank "$fd_rank" --fd_ell "$fd_ell")
    elif [[ $dataset == aircraft || $dataset == Animalia ]]; then
        cov_suffix=
    fi
    exp_name="${prefix}_OCD_n0auto_alpha${alpha}.${cov_suffix}${domain_suffix}_seed${seed}"
    if [[ $stage == fd ]]; then
        exp_name+="._rank${fd_rank}._ell${fd_ell}._memory"
    fi
    exp_root=$(python -c "from config import exp_root; print(exp_root)")
    checkpoint_name="${prefix}_labeled_data_supervised_pretrain_epoch100.${domain_suffix}_seed${seed}"
    checkpoint="${WARMUP_MODEL_DIR:-$exp_root/simgcd/log/$checkpoint_name/checkpoints/model.pt}"
    extra_args+=(--tau_new 1.0 --alpha "$alpha" --cov_mode full
                 --warmup_model_dir "$checkpoint")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" python "$entry" \
    --dataset_name "$dataset" \
    --batch_size "$batch_size" \
    --grad_from_block "$grad_from_block" \
    --epochs "$epoch" \
    --num_workers 8 \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform imagenet \
    --lr 0.1 \
    --eval_funcs v2 \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight "$memax_weight" \
    --exp_name "$exp_name" \
    --seed "$seed" \
    --fp16 \
    "${extra_args[@]}"
