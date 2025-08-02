#!/bin/bash
#SBATCH --job-name=cambrian-train
#SBATCH --partition=gpu_h100
#SBATCH --mail-user=tim.krause@students.uni-mannheim.de
#SBATCH --mail-typ=BEGIN,END,FAIL
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%j-cambrian.out

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

# Define paths and parameters
BASE_MODEL="detikzify/cambrian/model/cambrian_c3/detikzify-cambrian-concat-1B-clip_siglip_dino_convnext-XXL"
DATASET_PATH="datasets/datikz-filtered_0_2048"
OUTPUT_DIR="detikzify-cambrian-concat-1B-clip_siglip_dino_convnext-XXL--trained"
PROJECTOR_PATH="projector.pth"
DEEPSPEED_CONFIG="deepspeed_config2.json"

# Run your fine-tuning Python script
torchrun --nproc_per_node=4 examples/cambrian_train.py \
    --base_model "$BASE_MODEL" \
    --datikz "$DATASET_PATH" \
    --output "$OUTPUT_DIR" \
    --gradient_checkpointing \
    --sketch_ratio 0 \
    --batch_size 256 \
    --micro_batch_size 1
