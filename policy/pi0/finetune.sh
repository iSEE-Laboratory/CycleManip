train_config_name=$1
model_name=$2
gpu_use=$3

export CUDA_VISIBLE_DEVICES=$gpu_use
echo $CUDA_VISIBLE_DEVICES
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $train_config_name --exp-name=$model_name --overwrite

#  sh finetune.sh pi0_base_aloha_robotwin_lora shake_bottle_loop-loop1-8 0