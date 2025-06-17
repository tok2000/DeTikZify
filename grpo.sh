#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --partition=gpu_h100
#SBATCH --mail-user=tim.krause@students.uni-mannheim.de
#SBATCH --mail-typ=BEGIN,END,FAIL
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%j-grpo.out

module load devel/cuda/12.8
source /pfs/work9/workspace/scratch/ma_tikrause-detikzify/myvenv/bin/activate
echo "Job is running on: $(hostname)"

export LD_LIBRARY_PATH=$HOME/poppler_install/lib64:$LD_LIBRARY_PATH
export PATH=$HOME/texlive/2023/bin/x86_64-linux:$HOME/poppler_install/bin:$PATH
export MANPATH=$HOME/texlive/2023/texmf-dist/doc/man:$MANPATH
export INFOPATH=$HOME/texlive/2023/texmf-dist/doc/info:$INFOPATH

export GS=$(which gs)
export PDFTOPPM=$(which pdftoppm)
export PDFLATEX=$(which pdflatex)
export PDFINFO=$(which pdfinfo)

echo "Using pdflatex: $PDFLATEX"
echo "Using pdftoppm: $PDFTOPPM"
echo "Using pdfinfo: $PDFINFO"
echo "Using gs: $GS"

# Increase timeout for Hugging Face downloads (avoiding timeouts)
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$SCRATCH/hf_cache"
export USE_GRADIENT_CHECKPOINTING=1
#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
mkdir -p $HF_DATASETS_CACHE $TRANSFORMERS_CACHE

# Define paths and parameters
BASE_MODEL="detikzify-1B-trained_2048/detikzify-1B"
DATASET_PATH="datasets/arxivcap-sampled2k"
OUTPUT_DIR="detikzify-1B-grpo_2048-arxivcap2k_mgen16"
DEEPSPEED_CONFIG="deepspeed_config2.json"

# Run your fine-tuning Python script
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun --nproc_per_node=2 examples/rl.py \
    --base_model "$BASE_MODEL" \
    --datikz "$DATASET_PATH" \
    --output "$OUTPUT_DIR" \
    --gradient_checkpointing \
    --freeze_vision_encoder \
    --batch_size 16 \
    --num_completions 16
