

LEROBOT_TASK_CONFIGS = {
    'folding_blue_shirt': {
        'dataset_dir': [
            'folding_blue_tshirt_yichen_0103',
            'folding_blue_tshirt_yichen_0102',
        ],
    'episode_len': 2000,  # 1000,
    'camera_names': ['observation.images.cam_high',
                     "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    },
    'aloha_folding_shirt_lerobot_1_25': {
        'dataset_dir': [
            'fold_shirt_lxy1213',
            'fold_shirt_lxy1214',
            'fold_shirt_zmj1212',
            'fold_shirt_zmj1213',
            'fold_shirt_zzy1213',
            'folding_junjie_1224',
            'folding_zhongyi_1224',
            'fold_shirt_wjj1213_meeting_room',
            'folding_shirt_12_30_wjj_weiqing_recover',
            'folding_shirt_12_31_wjj_lab_marble_recover',
            'folding_shirt_12_31_zhouzy_lab_marble',
            "folding_blue_tshirt_yichen_0103",
            "folding_blue_tshirt_xiaoyu_0103",
            "folding_blue_tshirt_yichen_0102",
            "folding_shirt_12_28_zzy_right_first",
            "folding_shirt_12_27_office",
            "0107_wjj_folding_blue_shirt",
            'folding_second_tshirt_yichen_0108',
            'folding_second_tshirt_wjj_0108',
            'folding_random_yichen_0109',
            'folding_random_table_right_wjj_0109',
            'folding_basket_two_tshirt_yichen_0109',
            'folding_basket_second_tshirt_yichen_0110',
            'folding_basket_second_tshirt_yichen_0109',
            'folding_basket_second_tshirt_wjj_0110',
            'folding_basket_second_tshirt_yichen_0111',
            'folding_basket_second_tshirt_wjj_0113',
            'folding_basket_second_tshirt_wjj_0111',
            'folding_basket_second_tshirt_yichen_0114',
            # 1.17 2025 new add
            "weiqing_folding_basket_first_tshirt_dark_blue_yichen_0116",
            "weiqing_folding_basket_first_tshirt_pink_wjj_0115",
            # "weiqing_folding_basket_second_tshirt_blue_yichen_0115",
            "weiqing_folding_basket_second_tshirt_dark_blue_yichen_0116",
            "weiqing_folding_basket_second_tshirt_red_lxy_0116",
            "weiqing_folding_basket_second_tshirt_red_wjj_0116",
            "weiqing_folding_basket_second_tshirt_shu_red_yellow_wjj_0116",
            "weiqing_folding_basket_second_tshirt_yellow_shu_red_wjj_0116",

            # 1.21 added
            "unloading_dryer_yichen_0120",
            "unloading_dryer_yichen_0119",

            # 1.22
            "folding_random_short_first_wjj_0121",
            "folding_random_short_second_wjj_0121",

            # 1.23
            "folding_random_short_second_wjj_0122",
            "folding_random_short_first_wjj_0122",

            # 1.25
            "folding_random_tshirt_first_wjj_0124",
            "folding_random_tshirt_second_wjj_0124",

        ],
        # 'sample_weights': [1],
        'episode_len': 2000,  # 1000,
        'camera_names': ['observation.images.cam_high', "observation.images.cam_left_wrist",
                         "observation.images.cam_right_wrist"]
    },
'aloha_folding_shirt_lerobot_3_26': {
        'dataset_dir': [
            'fold_shirt_lxy1213',
            'fold_shirt_lxy1214',
            'fold_shirt_zmj1212',
            'fold_shirt_zmj1213',
            'fold_shirt_zzy1213',
            'folding_junjie_1224',
            'folding_zhongyi_1224',
            'fold_shirt_wjj1213_meeting_room',
            'folding_shirt_12_30_wjj_weiqing_recover',
            'folding_shirt_12_31_wjj_lab_marble_recover',
            'folding_shirt_12_31_zhouzy_lab_marble',
            "folding_blue_tshirt_yichen_0103",
            "folding_blue_tshirt_xiaoyu_0103",
            "folding_blue_tshirt_yichen_0102",
            "folding_shirt_12_28_zzy_right_first",
            "folding_shirt_12_27_office",
            "0107_wjj_folding_blue_shirt",
            'folding_second_tshirt_yichen_0108',
            'folding_second_tshirt_wjj_0108',
            'folding_random_yichen_0109',
            'folding_random_table_right_wjj_0109',
            'folding_basket_two_tshirt_yichen_0109',
            'folding_basket_second_tshirt_yichen_0110',
            'folding_basket_second_tshirt_yichen_0109',
            'folding_basket_second_tshirt_wjj_0110',
            'folding_basket_second_tshirt_yichen_0111',
            'folding_basket_second_tshirt_wjj_0113',
            'folding_basket_second_tshirt_wjj_0111',
            'folding_basket_second_tshirt_yichen_0114',
            # 1.17 2025 new add
            "weiqing_folding_basket_first_tshirt_dark_blue_yichen_0116",
            "weiqing_folding_basket_first_tshirt_pink_wjj_0115",
            # "weiqing_folding_basket_second_tshirt_blue_yichen_0115",
            "weiqing_folding_basket_second_tshirt_dark_blue_yichen_0116",
            "weiqing_folding_basket_second_tshirt_red_lxy_0116",
            "weiqing_folding_basket_second_tshirt_red_wjj_0116",
            "weiqing_folding_basket_second_tshirt_shu_red_yellow_wjj_0116",
            "weiqing_folding_basket_second_tshirt_yellow_shu_red_wjj_0116",

            # 1.21 added
            "unloading_dryer_yichen_0120",
            "unloading_dryer_yichen_0119",

            # 1.22
            "folding_random_short_first_wjj_0121",
            "folding_random_short_second_wjj_0121",

            # 1.23
            "folding_random_short_second_wjj_0122",
            "folding_random_short_first_wjj_0122",

            # 1.25
            "folding_random_tshirt_first_wjj_0124",
            "folding_random_tshirt_second_wjj_0124",

            # 3.26
            "fold_two_shirts_zmj_03_26_lerobot",
            "fold_two_shirts_zmj_03_21_lerobot",
            "fold_two_shirts_wjj_03_21",
            "fold_two_shirts_zmj_03_24_lerobot"

        ],
        # 'sample_weights': [1],
        'episode_len': 2000,  # 1000,
        'camera_names': ['observation.images.cam_high', "observation.images.cam_left_wrist",
                         "observation.images.cam_right_wrist"]
    },
'3_cameras_all_data_1_17': {
        'dataset_dir': [
            'fold_shirt_lxy1213',
            'fold_shirt_lxy1214',
            'fold_shirt_zmj1212',
            'fold_shirt_zmj1213',
            'fold_shirt_zzy1213',
            'folding_junjie_1224',
            'folding_zhongyi_1224',
            'fold_shirt_wjj1213_meeting_room',
            'folding_shirt_12_30_wjj_weiqing_recover',
            'folding_shirt_12_31_wjj_lab_marble_recover',
            'folding_shirt_12_31_zhouzy_lab_marble',
            "folding_blue_tshirt_yichen_0103",
            "folding_blue_tshirt_xiaoyu_0103",
            "folding_blue_tshirt_yichen_0102",
            "folding_shirt_12_28_zzy_right_first",
            "folding_shirt_12_27_office",
            "0107_wjj_folding_blue_shirt",
            'folding_second_tshirt_yichen_0108',
            'folding_second_tshirt_wjj_0108',
            'folding_random_yichen_0109',
            'folding_random_table_right_wjj_0109',
            'folding_basket_two_tshirt_yichen_0109',
            'folding_basket_second_tshirt_yichen_0110',
            'folding_basket_second_tshirt_yichen_0109',
            'folding_basket_second_tshirt_wjj_0110',
            'folding_basket_second_tshirt_yichen_0111',
            'folding_basket_second_tshirt_wjj_0113',
            'folding_basket_second_tshirt_wjj_0111',
            'folding_basket_second_tshirt_yichen_0114',
            # 1.17 2025 new add
            "weiqing_folding_basket_first_tshirt_dark_blue_yichen_0116",
            "weiqing_folding_basket_first_tshirt_pink_wjj_0115",
            # "weiqing_folding_basket_second_tshirt_blue_yichen_0115",
            "weiqing_folding_basket_second_tshirt_dark_blue_yichen_0116",
            "weiqing_folding_basket_second_tshirt_red_lxy_0116",
            "weiqing_folding_basket_second_tshirt_red_wjj_0116",
            "weiqing_folding_basket_second_tshirt_shu_red_yellow_wjj_0116",
            "weiqing_folding_basket_second_tshirt_yellow_shu_red_wjj_0116",

            # "truncate_push_basket_to_left_1_24",

            'clean_table_ljm_1217',
            'clean_table_zmj_1217_green_plate_coke_can_brown_mug_bottle',
            'clean_table_lxy_1220_blue_plate_pink_paper_cup_plastic_bag_knife',
            'clean_table_zzy_1220_green_paper_cup_wulong_bottle_pink_bowl_brown_spoon',
            'clean_table_zmj_1220_green_cup_blue_paper_ball_pink_plate_sprite',

            'clean_table_lxy_1222_pick_place_water_left_arm',

            'pick_cup_and_pour_water_wjj_weiqing_coke',
            'pick_cars_from_moving_belt_waibao_1227',
            'pick_cup_and_pour_water_wjj_weiqing_coffee',
            'pick_cars_from_moving_belt_zhumj_1227',
            'hang_cups_waibao',
            'storage_bottle_green_tea_oolong_mineral_water_ljm_weiqing_1225_right_hand',
            'storage_bottle_green_tea_oolong_mineral_water_lxy_weiqing_1225',
            'get_papercup_yichen_1223',
            'pour_coffee_zhaopeiting_1224',
            'get_papercup_and_pour_coke_yichen_1224',
            'pick_up_coke_in_refrigerator_yichen_1223',
            'pour_rice_yichen_0102',

        ],
        # 'sample_weights': [1],
        'episode_len': 2000,  # 1000,
        'camera_names': ['observation.images.cam_high', "observation.images.cam_left_wrist",
                         "observation.images.cam_right_wrist"]
    },
"folding_two_shirts_by_drag": {
        'dataset_dir': [
            "fold_two_shirts_zmj_03_26_lerobot",
            "fold_two_shirts_zmj_03_21_lerobot",
            "fold_two_shirts_wjj_03_21",
            "fold_two_shirts_zmj_03_24_lerobot"
        ],
    # 'sample_weights': [1],
    'episode_len': 2000,  # 1000,
    'camera_names': ['observation.images.cam_high', "observation.images.cam_left_wrist",
                     "observation.images.cam_right_wrist"]
},
}

### ALOHA fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
FPS = 50
# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
