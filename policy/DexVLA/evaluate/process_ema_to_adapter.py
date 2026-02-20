import os

import torch
import shutil
from safetensors.torch import save_file

path = "/media/eai/MAD-1/wjj/qwen2_vla_aloha/qwen2_vl_only_fold_shirt_lora_combine_substep_pretrain_DIT_H_align_finetune_2w_steps_freeze_VLM_EMA_norm_stats2/checkpoint-20000"

ema_path = os.path.join(path, 'ema_weights_trainable.pth')

output_path = os.path.join(path, 'ema_adapter')
os.makedirs(output_path, exist_ok=True)
ema_state_dict = torch.load(ema_path, map_location=torch.device('cpu'))

# non_lora = torch.load(os.path.join(path, 'non_lora_trainables.bin'), map_location=torch.device('cpu'))

lora = False
if os.path.exists(os.path.join(path, 'adapter_config.json')):
    shutil.copyfile(os.path.join(path, 'adapter_config.json'), os.path.join(output_path, 'adapter_config.json'))
    lora = True

lora_state_dict = {}
non_lora_state_dict = {}
for k, v in ema_state_dict.items():
    if 'lora' in k:
        lora_state_dict[k] = v
    else:
        non_lora_state_dict[k] = v

output_file = os.path.join(output_path, 'adapter_model.safetensors')
if lora:
    save_file(lora_state_dict, output_file)
torch.save(non_lora_state_dict, os.path.join(output_path, 'ema_non_lora_trainables.bin'))



