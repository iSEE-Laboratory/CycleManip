#!/bin/bash

config_name=${1}
task_name=${2}
task_config=${3}
expert_data_num=${4}
exp_tag=${5}
seed=${6}
gpu_id=${7}

# echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    echo "dataset not exist!!"
fi

dir1=experiments/${task_name}/${task_name}-${task_config}-${exp_tag}/seed_${seed}/wo_color_pc/1500.ckpt
dir2=experiments/${task_name}/${task_name}-${task_config}-${exp_tag}/seed_${seed}/wo_color_pc/10.ckpt
dir3=experiments/${task_name}/${task_name}-${task_config}-${exp_tag}/seed_${seed}/wo_color_pc/300.ckpt
dir4=experiments/${task_name}/${task_name}-${task_config}-${exp_tag}/seed_${seed}/wo_color_pc/500.ckpt

if [ -f "$dir1" ] || [ -f "$dir2" ] || [ -f "$dir3" ] || [ -f "$dir4" ] ; then
    echo "ckpt found, just doing evaluation"
    bash scripts_sh/eval.sh ${config_name} ${task_name} ${task_config} ${expert_data_num} ${exp_tag} ${seed} ${gpu_id}
else
    echo "ckpt not found, train first "
    bash scripts/train_policy_binary.sh ${config_name} ${task_name} ${task_config} ${expert_data_num} ${exp_tag} ${seed} ${gpu_id}
    bash scripts_sh/eval.sh ${config_name} ${task_name} ${task_config} ${expert_data_num} ${exp_tag} ${seed} ${gpu_id}
fi

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention_newselftask place_cans_plasticbox demo_clean 50 0125_binary_robot_dp3_obs6_nolan_attention_newselftask 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention_newselftask turn_switch demo_clean 50 0125_binary_robot_dp3_obs6_nolan_attention_newselftask 0 3


# new task
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_newselftask beat_block_hammer_loop loop1-8-all 200 0125_loop1-8_onehot_attention_binary_newselftask 0 7
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_newselftask shake_bottle_loop loop1-8-all 200 0125_loop1-8_onehot_attention_binary_newselftask 0 0

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_countertask beat_block_hammer_loop loop1-8-all 200 0125_loop1-8_onehot_attention_binary_countertask 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_countertask shake_bottle_loop loop1-8-all 200 0125_loop1-8_onehot_attention_binary_countertask 0 6



# step
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_newselftask_stepsample grab_roller_loop loop1-8-all 200 0125_loop1-8_onehot_attention_binary_newselftask_step 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_newselftask_stepsample beat_block_hammer_loop loop1-8-all 200 0125_loop1-8_onehot_attention_binary_newselftask_step 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_newselftask_stepsample shake_bottle_loop loop1-8-all 200 0125_loop1-8_onehot_attention_binary_newselftask_step 0 6

# uniform
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_newselftask_uniformsample grab_roller_loop loop1-8-all 200 0125_loop1-8_onehot_attention_binary_newselftask_uniform 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_newselftask_uniformsample beat_block_hammer_loop loop1-8-all 200 0125_loop1-8_onehot_attention_binary_newselftask_uniform 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_newselftask_uniformsample shake_bottle_loop loop1-8-all 200 0125_loop1-8_onehot_attention_binary_newselftask_uniform 0 5

# no 4 no 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_newselftask shake_bottle_loop loop1-8-no-4 200 0124_loop1-8-no-4_onehot_attention_binary_newselftask 0 2
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_newselftask shake_bottle_loop loop1-8-no-6 200 0124_loop1-8-no-6_onehot_attention_binary_newselftask 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_newselftask double_knife_chop loop1-8-no-4 200 0124_loop1-8-no-4_onehot_attention_binary_newselftask 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention_newselftask double_knife_chop loop1-8-no-6 200 0124_loop1-8-no-6_onehot_attention_binary_newselftask 0 6



# binary_robot_dp3_obs6_action18_onehotlan_attention

# beat_block_hammer_loop_real-move121_s2s-10.zarr

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_onehotlan_attention beat_block_hammer_loop_real move121_s2s 10 1119_onehot_attention_binary_10hz 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_onehotlan_attention_newselftask beat_block_hammer_loop_real move121_s2s 10 1119_onehot_attention_binary_selftask_10hz 0 7

#



# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_tasttast cut_carrot_knife loop1-8-all 200 testest 0 7

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_basebaseline cut_carrot_knife loop1-8-all 200 testest2 0 7






# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action18_onehotlan_attention pump s2c-10hz 20 1114_onehot_attention_binary_10hz 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action18_onehotlan_attention_newselfask pump s2c-10hz 20 1114_onehot_attention_binary_selftask_10hz 0 3





# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action14_intlan_ori sweep loop1-5-all_s2s 50 1113_ori 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_onehotlan_attention sweep loop1-5-all_s2s 50 1114_onehot_attention_binary_10hz 0 0


# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action24_onehotlan_attention cut_carrot_knife_real 10hz_s2s 32 1114_onehot_attention_binary_10hz 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action24_onehotlan_attention_newselftask cut_carrot_knife_real 10hz_s2s 32 1114_onehot_attention_binary_newselftask_10hz 0 0



# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention_newselftask bbhlr_fixed_pos_DATA_Augmentation 10hz_36_s2s 36 119_attention_binary_newselftask_10hz 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention_newselftask test_bbhlr_enhance hands_10hz_s2s 8 119_attention_binary_newselftask_10hz 0 6


# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_trueonehotlan_attention_newselftask test_bbhlr_enhance hands_10hz_s2s 8 1114_trueonehot_attention_binary_newselftask_10hz 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_trueonehotlan_attention_newselftask bbhlr_fixed_pos_DATA_Augmentation 10hz_36_s2s 36 1114_trueonehot_attention_binary_newselftask_10hz 0 2
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_trueonehotlan_attention_newselftask beat_block_hammer_loop_real loop1-8-all-10hz 100 1114_trueonehot_attention_binary_newselftask_10hz 0 2
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_trueonehotlan_attention_newselftask shake_bottle_loop_real loop1-8-all-10hz 150 1114_trueonehot_attention_binary_newselftask_10hz 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_trueonehotlan_attention_newselftask sweep loop1-5-all_s2s 50 1114_trueonehot_attention_binary_newselftask_10hz 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_trueonehotlan_attention_newselftask beat_drum loop1-8-all-10hz 100 1114_trueonehot_attention_binary_newselftask_10hz 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_trueonehotlan_attention_newselftask bbhlr fixed_block_pos_10hz_s2s 16 1114_trueonehot_attention_binary_newselftask_10hz 0 6


# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_trueonehotlan_attention test_bbhlr_enhance hands_10hz_s2s 8 1114_trueonehot_attention_binary_10hz 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_trueonehotlan_attention bbhlr_fixed_pos_DATA_Augmentation 10hz_36_s2s 36 1114_trueonehot_attention_binary_10hz 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_trueonehotlan_attention beat_block_hammer_loop_real loop1-8-all-10hz 100 1114_trueonehot_attention_binary_10hz 0 7
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_trueonehotlan_attention shake_bottle_loop_real loop1-8-all-10hz 150 1114_trueonehot_attention_binary_10hz 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_trueonehotlan_attention sweep loop1-5-all_s2s 50 1114_trueonehot_attention_binary_10hz 0 7
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action14_trueonehotlan_attention beat_drum loop1-8-all-10hz 100 1114_trueonehot_attention_binary_10hz 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_trueonehotlan_attention bbhlr fixed_block_pos_10hz_s2s 16 1114_trueonehot_attention_binary_newselftask_10hz 0 6





# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_ee_newself bbhlr_fixed_pos_DATA_Augmentation 10hz_36_s2s 36 newdata2_ee_newself 0 0
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_eediff_newself bbhlr_fixed_pos_DATA_Augmentation 10hz_36_s2s 36 newdata2_eediff_newself 0 2
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_joint_newself bbhlr_fixed_pos_DATA_Augmentation 10hz_36_s2s 36 newdata2_joint_newself 0 3
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_jointdiff_newself bbhlr_fixed_pos_DATA_Augmentation 10hz_36_s2s 36 newdata2_jointdiff_newself 0 4





# wo selftask
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_eediff cut_carrot_knife_real 10hz_s2s 32 1113_eediff 0 1
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action14_intlan_attention_eediff beat_drum loop1-8-all-10hz 100 1113_eediff 0 4
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_eediff bbhlr fixed_block_pos_10hz_s2s 16 1113_eediff 0 5
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_eediff shake_bottle_loop_real loop1-8-all-10hz 150 1113_eediff 0 7


# CAT KINFE wo selftask
# PAOLE bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_ee cut_carrot_knife_real 10hz_s2s 32 1112_ee 0 0
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_eediff cut_carrot_knife_real 10hz_s2s 32 1112_eediff 0 1
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_joint cut_carrot_knife_real 10hz_s2s 32 1112_joint 0 2
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_jointdiff cut_carrot_knife_real 10hz_s2s 32 1112_jointdiff 0 3



# Beat 
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action14_intlan_ori beat_drum loop1-8-all-10hz 100 1113_ori 0 4
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action14_intlan_attention_joint_newself beat_drum loop1-8-all-10hz 100 1113_joint_newself 0 6
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action14_intlan_attention_jointdiff_newself beat_drum loop1-8-all-10hz 100 1113_jointdiff_newself 0 5

# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action14_intlan_attention_ee_newself beat_drum loop1-8-all-10hz 100 1113_ee_newself 0 6
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action14_intlan_attention_eediff_newself beat_drum loop1-8-all-10hz 100 1113_eediff_newself 0 5



# WITHOUT BICLS and FLIP
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_ee_newself_1cls cut_carrot_knife_real 10hz_s2s 32 1113_ee_newself_1clsnoflip 0 1
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_eediff_newself_1cls cut_carrot_knife_real 10hz_s2s 32 1113_eediff_newself_1clsnoflip 0 2
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_joint_newself_1cls cut_carrot_knife_real 10hz_s2s 32 1113_joint_newself_1clsnoflip 0 3
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_jointdiff_newself_1cls cut_carrot_knife_real 10hz_s2s 32 1113_jointdiff_newself_1clsnoflip 0 5


# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action18_intlan_attention_joint_newself_1cls pump s2c-10hz 20 1113_joint_newself_1clsnoflip 0 0
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action18_intlan_attention_jointdiff_newself_1cls pump s2c-10hz 20 1113_jointdiff_newself_1clsnoflip 0 3

# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_ee_newself_1cls bbhlr fixed_block_pos_10hz_s2s 16 1113_ee_newself_1clsnoflip 0 6
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_eediff_newself_1cls bbhlr fixed_block_pos_10hz_s2s 16 1113_eediff_newself_1clsnoflip 0 7
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_joint_newself_1cls bbhlr fixed_block_pos_10hz_s2s 16 1113_joint_newself_1clsnoflip 0 4
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_jointdiff_newself_1cls bbhlr fixed_block_pos_10hz_s2s 16 1113_jointdiff_newself_1clsnoflip 0 7

# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_ee_newself_1cls shake_bottle_loop_real loop1-8-all-10hz 150 1113_ee_newself_1clsnoflip 0 0
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_eediff_newself_1cls shake_bottle_loop_real loop1-8-all-10hz 150 1113_eediff_newself_1clsnoflip 0 1
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_joint_newself_1cls shake_bottle_loop_real loop1-8-all-10hz 150 1113_joint_newself_1clsnoflip 0 2
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_jointdiff_newself_1cls shake_bottle_loop_real loop1-8-all-10hz 150 1113_jointdiff_newself_1clsnoflip 0 3






# ORI
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_ori bbhlr fixed_block_pos_10hz_s2s 16 1112_ori 0 2
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action18_intlan_ori pump s2c-10hz 20 1113_ori 0 0
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_ori beat_block_hammer_loop_real loop1-8-all-10hz 100 1112_ori 0 4
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_ori cut_carrot_knife_real 10hz_s2s 32 1113_ori 0 4
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_ori shake_bottle_loop_real loop1-8-all-10hz 150 1113_ori 0 4




# ORI_Task + Ours
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention handover_block demo_clean 50 attention_binary 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention open_microwave demo_clean 50 attention_binary 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention pick_diverse_bottles demo_clean 50 attention_binary 0 2
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention place_bread_basket demo_clean 50 attention_binary 0 3

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention place_phone_stand demo_clean 50 attention_binary 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention place_shoe demo_clean 50 attention_binary 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention stamp_seal demo_clean 50 attention_binary 0 2
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention turn_switch demo_clean 50 attention_binary 0 3


# NEW_KNIFE
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_ee_newself cut_carrot_knife_real 10hz_s2s 32 1113_ee_newself 0 0
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_eediff_newself cut_carrot_knife_real 10hz_s2s 32 1113_eediff_newself 0 3
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_joint_newself cut_carrot_knife_real 10hz_s2s 32 1113_joint_newself 0 4
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_jointdiff_newself cut_carrot_knife_real 10hz_s2s 32 1113_jointdiff_newself 0 6

# NEWNEW_HUMNAOID
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action18_intlan_attention_joint pump s2c-10hz 20 1113_joint 0 0
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action18_intlan_attention_joint_newself pump s2c-10hz 20 1113_joint_newself 0 0
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action18_intlan_attention_jointdiff pump s2c-10hz 20 1113_jointdiff 0 0
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action18_intlan_attention_jointdiff_newself pump s2c-10hz 20 1113_jointdiff_newself 0 1



# NEW_HAMMER
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_ee_newself bbhlr fixed_block_pos_10hz_s2s 16 1112_ee_newself 0 2
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_eediff_newself bbhlr fixed_block_pos_10hz_s2s 16 1112_eediff_newself 0 3
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_joint_newself bbhlr fixed_block_pos_10hz_s2s 16 1112_joint_newself 0 4
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_jointdiff_newself bbhlr fixed_block_pos_10hz_s2s 16 1112_jointdiff_newself 0 0


# NEW_HUMANIOID
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action18_intlan_attention_joint_newself pump s2c-10hz 20 1112_joint_newself 0 3
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action18_intlan_attention_jointdiff_newself pump s2c-10hz 20 1112_jointdiff_newself 0 3
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action18_intlan_attention_joint pump s2c-10hz 20 1112_joint 0 3
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action18_intlan_attention_jointdiff pump s2c-10hz 20 1112_jointdiff 0 3

# NEW_KNIFE
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_ee_newself cut_carrot_knife_real 10hz_s2s 32 1112_ee_newself 0 2
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_eediff_newself cut_carrot_knife_real 10hz_s2s 32 1112_eediff_newself 0 3
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_joint_newself cut_carrot_knife_real 10hz_s2s 32 1112_joint_newself 0 4
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action24_intlan_attention_jointdiff_newself cut_carrot_knife_real 10hz_s2s 32 1112_jointdiff_newself 0 0



# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_ee_newself shake_bottle_loop_real loop1-8-all-10hz 150 1111_ee_newself 0 4
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_eediff_newself shake_bottle_loop_real loop1-8-all-10hz 150 1111_eediff_newself 0 2
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_joint_newself shake_bottle_loop_real loop1-8-all-10hz 150 1111_joint_newself 0 3
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_jointdiff_newself shake_bottle_loop_real loop1-8-all-10hz 150 1111_jointdiff_newself 0 3




# 
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_ee_newself beat_block_hammer_loop_real loop1-8-all-10hz 100 1111_ee_newself 0 4
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_ee beat_block_hammer_loop_real loop1-8-all-10hz 100 1111_ee 0 5
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_eediff_newself beat_block_hammer_loop_real loop1-8-all-10hz 100 1111_eediff_newself 0 6
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_eediff beat_block_hammer_loop_real loop1-8-all-10hz 100 1111_eediff 0 7
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_joint_newself beat_block_hammer_loop_real loop1-8-all-10hz 100 1111_joint_newself 0 0
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_joint beat_block_hammer_loop_real loop1-8-all-10hz 100 1111_joint 0 3
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_jointdiff_newself beat_block_hammer_loop_real loop1-8-all-10hz 100 1111_jointdiff_newself 0 2
# bash scripts_sh/train_eval.sh cycleplus_robot_dp3_obs6_action7_intlan_attention_jointdiff beat_block_hammer_loop_real loop1-8-all-10hz 100 1111_jointdiff 0 3

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention beat_drum loop1-8-all-10hz 100 new_intlan_attention 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention_newselftask beat_drum loop1-8-all-10hz 100 119_attention_binary_newselftask 0 6

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_actionx_intlan_attention_nopc pump s2c-25hz 20 new_attention_25hz 0 1

# new multi-task
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention_newselftask beat_block_hammer_loop_real loop1-8-all-10hz 100 119_attention_binary_newselftask_10hz 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention_newselftask shake_bottle_loop_real loop1-8-all-10hz 150 119_attention_binary_newselftask_10hz 0 6

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention_newprogresstask beat_block_hammer_loop_real loop1-8-all-10hz 100 119_attention_binary_newprogresstask_10hz 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention_newprogresstask shake_bottle_loop_real loop1-8-all-10hz 150 119_attention_binary_newprogresstask_10hz 0 6


# multi-selftask
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention_multiselftask beat_block_hammer_loop_real loop1-8-all-10hz 100 119_attention_binary_multiselftask_10hz 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention_multiselftask shake_bottle_loop_real loop1-8-all-10hz 150 119_attention_binary_multiselftask_10hz 0 5

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention_multiactiontask beat_block_hammer_loop_real loop1-8-all-10hz 100 119_attention_binary_multiactiontask_10hz 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention_multiactiontask shake_bottle_loop_real loop1-8-all-10hz 150 119_attention_binary_multiactiontask_10hz 0 6



# REAL WORLD + OURS NEW

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention shake_bottle_loop_real loop1-8-all-10hz 150 119_attention_binary_10hz 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_ah16_intlan_attention shake_bottle_loop_real loop1-8-all-10hz 150 119_attention_binary_ah16_10hz 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_mamba shake_bottle_loop_real loop1-8-all-10hz 150 mamba_binary_10hz 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention_eepose shake_bottle_loop_real loop1-8-all-10hz 150 119_attention_ee_10hz 0 4


# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention beat_block_hammer_loop_real loop1-8-all-10hz 100 119_attention_binary_10hz 0 7
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_ah16_intlan_attention beat_block_hammer_loop_real loop1-8-all-10hz 100 119_attention_binary_ah16_10hz 0 7
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention_eepose beat_block_hammer_loop_real loop1-8-all-10hz 100 119_attention_ee_10hz 0 4


# SIMULATION + OURS 
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention grab_roller_loop loop1-8-all 200 attention_binary 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention cut_carrot_knife loop1-8-all 200 attention_binary 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention beat_block_hammer_loop loop1-8-all 200 attention_binary 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention shake_bottle_loop loop1-8-all 200 attention_binary 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention double_knife_chop loop1-8-all 200 attention_binary 0 6

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention beat_egg_loop loop1-8-all 200 attention_binary 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention shake_flask_dropper_loop loop1-8-all-sfdl 200 attention_binary 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_simlan_attention morse_sos loop1-8-all 200 attention_binary 0 6

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention blocks_ranking_size demo_clean 50 attention_binary 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention place_cans_plasticbox demo_clean 50 attention_binary 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention scan_object demo_clean 50 attention_binary 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_attention stack_bowls_three demo_clean 50 attention_binary 0 4



# SIMULATION + ORI
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_ori grab_roller_loop loop1-8-all 200 new_ori_dp3 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_ori cut_carrot_knife loop1-8-all 200 new_ori_dp3 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_ori beat_block_hammer_loop loop1-8-all 200 new_ori_dp3 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_ori shake_bottle_loop loop1-8-all 200 new_ori_dp3 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_ori double_knife_chop loop1-8-all 200 new_ori_dp3 0 5

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_ori beat_egg_loop loop1-8-all 200 new_ori_dp3 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_ori shake_flask_dropper_loop loop1-8-all-sfdl 200 new_ori_dp3 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_simlan_ori morse_sos loop1-8-all 200 new_ori_dp3 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_ori blocks_ranking_size demo_clean 50 new_ori_dp3 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_ori place_cans_plasticbox demo_clean 50 new_ori_dp3 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_ori scan_object demo_clean 50 new_ori_dp3 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_nolan_ori stack_bowls_three demo_clean 50 new_ori_dp3 0 6




# HUMANOID 
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_actionx_intlan_attention_nopc pump s2c-25hz 20 new_mamba_10hz 0 1pump
# # pump-s2c-25hz-20.zarr


# REAL 10HZ DATA + CATMAMBA

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_mamba beat_block_hammer_loop_real loop1-8-all-10hz 100 new_mamba_10hz 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_mamba_pastall beat_block_hammer_loop_real loop1-8-all-10hz 100 new_mamba_pastall_10hz 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_ah16_intlan_mamba beat_block_hammer_loop_real loop1-8-all-10hz 100 new_mamba_ah16_10hz 0 2
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_mamba_binaryadptive beat_block_hammer_loop_real loop1-8-all-10hz 100 new_mamba_binaryadptive_10hz 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention beat_block_hammer_loop_real loop1-8-all-10hz 100 new_attention_10hz 0 7
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_ah16_intlan_attention beat_block_hammer_loop_real loop1-8-all-10hz 100 new_attention_ah16_10hz 0 7
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_attention_pastall beat_block_hammer_loop_real loop1-8-all-10hz 100 new_attention_pastall_10hz 0 7





# simulation + test

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba grab_roller_loop loop1-8-all 200 mamba 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba_pastall grab_roller_loop loop1-8-all 200 mamba_pastall 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_ah16_intlan_mamba grab_roller_loop loop1-8-all 200 mamba_ah16 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba_binaryadptive grab_roller_loop loop1-8-all 200 mamba_binaryadptive 0 5

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention grab_roller_loop loop1-8-all 200 attention_binary 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_ah16_intlan_attention grab_roller_loop loop1-8-all 200 attention_ah16 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention_binaryadptive grab_roller_loop loop1-8-all 200 attention_binaryadptive 0 5




# LOOP DATA + OURS_A & OURS_B
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_ah16_intlan_pcattnetion_pastall cut_carrot_knife loop1-8-all 200 cycle_ah16_pastall 0 1
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_ah16_intlan_pcattnetion_pastall grab_roller_loop loop1-8-all 200 cycle_ah16_pastall 0 5
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_ah16_intlan_pcattnetion_pastall beat_block_hammer_loop loop1-8-all 200 cycle_ah16_pastall 0 4
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_ah16_intlan_pcattnetion_pastall shake_bottle_loop loop1-8-all 200 cycle_ah16_pastall 0 5
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_ah16_intlan_pcattnetion_pastall double_knife_chop loop1-8-all 200 cycle_ah16_pastall 0 6

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_ah16_intlan_mamba_pastall cut_carrot_knife loop1-8-all 200 mamba_ah16_pastall 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_ah16_intlan_mamba_pastall grab_roller_loop loop1-8-all 200 mamba_ah16_pastall 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_ah16_intlan_mamba_pastall beat_block_hammer_loop loop1-8-all 200 mamba_ah16_pastall 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_ah16_intlan_mamba_pastall shake_bottle_loop loop1-8-all 200 mamba_ah16_pastall 0 7
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_ah16_intlan_mamba_pastall double_knife_chop loop1-8-all 200 mamba_ah16_pastall 0 7


# NEW LOOP DATA + OURS-
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba beat_egg_loop loop1-8-all 200 new_intlan_mamba 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba shake_flask_dropper_loop loop1-8-all-sfdl 200 new_intlan_mamba 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_simlan_mamba morse_sos loop1-8-all 200 new_intlan_mamba 0 6

# HAMMER DATA + OURS NEWNEW
# bash scripts_sh/train_eval.sh cycle_robot_dp3_action7_obs6_ah16_intlan_pcattnetion beat_block_hammer_loop_real loop1-8-all_s2s 100 cycle_ah16 0 5
# bash scripts_sh/train_eval.sh cycle_robot_dp3_action7_obs6_ah16_intlan_pcattnetion_pastall beat_block_hammer_loop_real loop1-8-all_s2s 100 cycle_ah16_pastall 0 5
# bash scripts_sh/train_eval.sh cycle_robot_dp3_action7_obs6_ah16_intlan_pcattnetion_stateattention_pastall beat_block_hammer_loop_real loop1-8-all_s2s 100 cycle_ah16_pastall_jatten 0 5
# bash scripts_sh/train_eval.sh cycle_robot_dp3_action7_obs6_ah16_intlan_pcattnetion_adapast beat_block_hammer_loop_real loop1-8-all 100 cycle_ah16_adapast 0 5
# bash scripts_sh/train_eval.sh cycle_robot_dp3_action7_obs6_ah16_intlan_pcattnetion beat_block_hammer_loop_real loop1-8-all-10hz 100 cycle_ah16_10hz 0 0

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_ah16_intlan_attnetion_pastall beat_block_hammer_loop_real loop1-8-all-10hz 100 attention_ah16_pastall_10hz 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_ah16_intlan_attnetion_pastall beat_block_hammer_loop_real loop1-8-all_s2s 100 attention_ah16_pastall 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_ah16_intlan_attnetion_adapast beat_block_hammer_loop_real loop1-8-all 100 attention_ah16_adapast 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_ah16_intlan_mamba_adapast beat_block_hammer_loop_real loop1-8-all 100 mamba_ah16_adapast 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_ah16_intlan_mamba beat_block_hammer_loop_real loop1-8-all-10hz 100 mamba_ah16_10hz 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_ah16_intlan_mamba_pastall beat_block_hammer_loop_real loop1-8-all 100 mamba_ah16_pastall 0 5

# Action horizen 16 + LOOP DATA / REAL DATA + OURS-
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_ah16_intlan_mamba double_knife_chop loop1-8-all 200 new_intlan_mamba_ah_16 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_ah16_intlan_mamba beat_block_hammer_loop_real loop1-8-all_s2s 100 new_intlan_mamba_ah16 0 5

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs1_intlan_mamba double_knife_chop loop1-8-all 200 new_intlan_mamba_obs1 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs1_action7_intlan_mamba beat_block_hammer_loop_real loop1-8-all_s2s 100 new_intlan_mamba_obs1 0 5

# /home/liaohaoran/code/RoboTwin/policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/binary_robot_dp3_obs1_action7_intlan_mamba.yaml

# LOOP_DATA + OURS NEW
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba_pastall cut_carrot_knife loop1-8-all 200 new_intlan_mamba_pastall 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba_pastall double_knife_chop loop1-8-all 200 new_intlan_mamba_pastall 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba_joint_adaptive_past cut_carrot_knife loop1-8-all 200 new_intlan_mamba_adaptive_past 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba_joint_adaptive_past double_knife_chop loop1-8-all 200 new_intlan_mamba_adaptive_past 0 7

# REAL_LOOP_DATA + OURS NEW
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_mamba_pastall beat_block_hammer_loop_real loop1-8-all_s2s 100 new_intlan_mamba_pastall 0 2
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_mamba_pastall shake_bottle_loop_real loop1-8-all_s2s 150 new_intlan_mamba_adaptive_past 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_mamba_joint_adaptive_past beat_block_hammer_loop_real loop1-8-all_s2s 100 new_intlan_mamba_pastall 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_mamba_joint_adaptive_past shake_bottle_loop_real loop1-8-all_s2s 150 new_intlan_mamba_adaptive_past 0 3


# LOOP DATA + OURS MAMBA ()binary_robot_dp3_obs6_intlan_mamba_counter
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba_counter cut_carrot_knife loop1-8-all 200 new_intlan_mamba_counter 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba_counter grab_roller_loop loop1-8-all 200 new_intlan_mamba_counter 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba_counter beat_block_hammer_loop loop1-8-all 200 new_intlan_mamba_counter 0 2
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba_counter shake_bottle_loop loop1-8-all 200 new_intlan_mamba_counter 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba_counter double_knife_chop loop1-8-all 200 new_intlan_mamba_counter 0 4


# REAL DATA 2 + OURS
# bash scripts_sh/train_eval.sh cycle_robot_dp3_action7_obs6_intlan_pcattnetion beat_block_hammer_loop_real loop1-8-all_s2s 100 new_intlan_mamba_3 0 2

# ORI DATA FINAL + OURS
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_nolan_pcattnetion blocks_ranking_size demo_clean 50 cycle_114 0 0
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_nolan_pcattnetion place_cans_plasticbox demo_clean 50 cycle_114 0 1
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_nolan_pcattnetion scan_object demo_clean 50 cycle_114 0 2
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_nolan_pcattnetion stack_bowls_three demo_clean 50 cycle_114 0 3

# NEW_REAL_DATA + OURS
# bash scripts_sh/train_eval.sh cycle_robot_dp3_action7_obs6_intlan_pcattnetion beat_block_hammer_loop_real loop1-8-all_s2s 100 new_intlan_mamba_3 0 2
# bash scripts_sh/train_eval.sh cycle_robot_dp3_action7_obs6_intlan_pcattnetion shake_bottle_loop_real loop1-8-all_s2s 150 new_intlan_mamba_3 0 3
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_intlan_pcattnetion beat_drum loop1-8-all_s2s 100 new_intlan_mamba_3 0 4

# NEW_REAL_DATA + OURS-
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_mamba beat_block_hammer_loop_real loop1-8-all_s2s 100 new_intlan_mamba_2 0 2
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_action7_intlan_mamba shake_bottle_loop_real loop1-8-all_s2s 150 new_intlan_mamba_2 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba beat_drum loop1-8-all_s2s 100 new_intlan_mamba_2 0 4

# CYCLE FINAL
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_intlan_pcattnetion_counter cut_carrot_knife loop1-8-all 200 cycle_114 0 0  
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_intlan_pcattnetion_counter grab_roller_loop loop1-8-all 200 cycle_114 0 0   
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_intlan_pcattnetion_counter beat_block_hammer_loop loop1-8-all 200 cycle_114 0 0  
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_intlan_pcattnetion_counter shake_bottle_loop loop1-8-all 200 cycle_114 0 1  
# bash scripts_sh/train_eval.sh cycle_robot_dp3_obs6_intlan_pcattnetion_counter double_knife_chop loop1-8-all 200 cycle_114 0 1   

# ORI DATA + OURS-
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_lan_mamba blocks_ranking_size loop1-8-all 200 new_lan_mamba 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_lan_mamba place_cans_plasticbox loop1-8-all 200 new_lan_mamba 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_lan_mamba scan_object loop1-8-all 200 new_lan_mamba 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_lan_mamba stack_bowls_three loop1-8-all 200 new_lan_mamba 0 6

# LOOP DATA + OURS-
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba cut_carrot_knife loop1-8-all 200 new_intlan_mamba 0 2
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba grab_roller_loop loop1-8-all 200 new_intlan_mamba 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba beat_block_hammer_loop loop1-8-all 200 new_intlan_mamba 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba shake_bottle_loop loop1-8-all 200 new_JIHUN_intlan_mamba_JIHUN 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba double_knife_chop loop1-8-all 200 new_JIHUN_intlan_mamba_JIHUN 0 1

# LOOP DATA + ORI
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_ori cut_carrot_knife loop1-8-all 200 new_ori_dp3 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_ori grab_roller_loop loop1-8-all 200 new_ori_dp3 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_ori shake_bottle_loop loop1-8-all 200 new_ori_dp3 0 4
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_ori double_knife_chop loop1-8-all 200 new_ori_dp3 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_ori beat_block_hammer_loop loop1-8-all 200 new_ori_dp3 0 7





# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba_counter double_knife_chop loop1-8-all 200 new_intlan_mamba_counter 0 2 
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention shake_bottle_loop loop1-8-all 200 new_intlan_attention 0 2 
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention double_knife_chop loop1-8-all 200 new_intlan_attention 0 3 
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allpadap_posi shake_bottle_loop loop1-8-all 200 new_mlan_allpadap_posi 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allpadap_posi double_knife_chop loop1-8-all 200 new_mlan_allpadap_posi 0 7


# bash scripts_sh/train_eval.sh test shake_bottle_loop loop1-8-all 200 test 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention shake_bottle_loop loop1-8-all 200 intlan_attention 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention cut_carrot_knife loop1-8-all 200 intlan_attention 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention double_knife_chop loop1-8-all 200 intlan_attention 0 0

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention_counter double_knife_chop loop1-8-all 200 intlan_attention_counter 0 0



# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_mamba shake_bottle_loop loop1-8-all 200 JIHUN_intlan_mamba_JIHUN 0 0

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_intlan_attention shake_bottle_loop loop1-8-all 200 intlan_attention 0 0


# deploy_policy_binary_robot_dp3_obs6_intlan_mamba.yml
# experiments/shake_bottle_loop/shake_bottle_loop-loop1-8-counter-mlan_allada/seed_0/wo_color_pc/1000.ckpt
# experiments/shake_bottle_loop/shake_bottle_loop-loop1-8-counter-mlan_allada/seed_0/wo_color_pc/1500.ckpt

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allpadap_posi shake_bottle_loop loop1-8-all 200 mlan_allpadap_posi 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allpadap_posi grab_roller_loop loop1-8-all 200 mlan_allpadap_posi 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allpadap_posi cut_carrot_knife loop1-8-all 200 mlan_allpadap_posi 0 2
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allpadap_posi double_knife_chop loop1-8-all 200 mlan_allpadap_posi 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allpadap_posi beat_block_hammer_loop loop1-8-all 200 mlan_allpadap_posi 0 4

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allada_counter grab_roller_loop loop1-8-counter 200 mlan_allada_counter 0 5
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allada_counter cut_carrot_knife loop1-8-counter 200 mlan_allada_counter 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allada_counter double_knife_chop loop1-8-counter 200 mlan_allada_counter 0 7

# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allada shake_bottle_loop loop1-8-counter 200 mlan_allada 0 0
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allada grab_roller_loop loop1-8-counter 200 mlan_allada 0 1
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allada cut_carrot_knife loop1-8-counter 200 mlan_allada 0 2
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allada double_knife_chop loop1-8-counter 200 mlan_allada 0 3
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_mlan_allada beat_block_hammer_loop loop1-8-counter 200 mlan_allada 0 4


# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_ori shake_bottle_loop loop1-8-counter 200 ori_dp3 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_ori grab_roller_loop loop1-8-counter 200 ori_dp3 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_ori cut_carrot_knife loop1-8-counter 200 ori_dp3 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_ori double_knife_chop loop1-8-counter 200 ori_dp3 0 6
# bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_ori beat_block_hammer_loop loop1-8-counter 200 ori_dp3 0 6
