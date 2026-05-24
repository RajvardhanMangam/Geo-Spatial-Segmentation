#!/bin/bash
#SBATCH --job-name=segformer_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --mem=16G
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err

# =========================
# SETUP
# =========================
mkdir -p /scratch/$USER/logs

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================"

# =========================
# LOAD MODULE
# =========================
. /etc/profile.d/modules.sh
module load anaconda/2023.03-1

# =========================
# USE EXISTING ENV (NO ACTIVATE NEEDED)
# =========================
PYTHON_BIN="/scratch/$USER/geo_env/bin/python"

# Debug
echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN -c "import torch; print('CUDA:', torch.cuda.is_available())"

nvidia-smi

# =========================
# WORK DIR (SCRATCH)
# =========================
WORK_DIR="/scratch/$USER/train_$SLURM_JOB_ID"
mkdir -p $WORK_DIR

PROJECT_DIR="$HOME/workspace/Geo-Spatial-Segmentation"

echo "Copying project..."
cp -r $PROJECT_DIR/* $WORK_DIR/

cd $WORK_DIR

# =========================
# DATA CHECK
# =========================
echo "Images count:"
ls data/images | wc -l

echo "Masks count:"
ls data/masks | wc -l

# =========================
# OUTPUT DIR
# =========================
SAVE_DIR="$HOME/workspace/Geo-Spatial-Segmentation/models"
mkdir -p $SAVE_DIR

echo "Saving model to: $SAVE_DIR"

# =========================
# TRAIN
# =========================
echo "Starting training..."

$PYTHON_BIN train.py \
    --img_dir data/images \
    --mask_dir data/masks \
    --epochs 35 \
    --batch_size 2 \
    --model_size b1 \
    --img_size 512 \
    --save_dir $SAVE_DIR

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "❌ Training failed"
    exit 1
fi

# =========================
# SAVE BEST MODEL SAFELY
# =========================
echo "Moving model to final location..."

mv -v $SAVE_DIR/*.pth /home1/cs25m107/workspace/Geo-Spatial-Segmentation/models/ 2>/dev/null

# =========================
# CLEANUP
# =========================
# rm -rf $WORK_DIR

echo "======================================"
echo "Training completed at: $(date)"
echo "======================================"