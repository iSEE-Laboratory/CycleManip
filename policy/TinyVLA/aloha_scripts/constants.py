
# DATA_DIR = './datasets'
# DATA_DIR = "/home/jovyan/tzb/h5py_data/"
DATA_DIR = "/data/private/liuza/robotiwin/policy/TinyVLA/data"
# DATA_DIR = '/home/jovyan/tzb/h5py_data/'
PRETRAIN_DIR = '/data/team/xuzy/nfs/eai_data/data_WJJ/droid_1dot7t_h5py2'
LOCAL_DATA_DIR = '/home/jz08/zhumj/data'

TASK_CONFIGS = {
    "local_debug_data": {
        'dataset_dir': [
            LOCAL_DATA_DIR + '/franka/4_types_pikachu_blue_van_hex_key_glove_480_640',
            LOCAL_DATA_DIR + '/franka/t2',
        ],
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
        "sample_weights": [1, 1]
    },
    "place_object_scale": {
        'dataset_dir': [DATA_DIR + "/sim-place_object_scale/aloha-agilex-1-m1_b1_l1_h0.03_c0_D435-100"],
        'episode_len': 500, # 这里我看ACT的设置是500，我也先设置为500
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        "sample_weights": [1, 1]
    },
    "dual_shoes_place": {
        'dataset_dir': [DATA_DIR + "/sim-place_object_scale/aloha-agilex-1-m1_b1_l1_h0.03_c0_D435-100"],
        'episode_len': 500, # 这里我看ACT的设置是500，我也先设置为500
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        "sample_weights": [1, 1]
    },
    "mobile_franka_bin_picking": {
        'dataset_dir': [
            DATA_DIR + '/ume/0102_green_paper_cup_yellow_bus_hex_key_gloves_480_640/0102_green_paper_cup_yellow_bus_hex_key_gloves_480_640_succ_t0001_s-0-0',
            DATA_DIR + '/ume/0102_toy_blue_van_pear_tape_480_640/0102_toy_blue_van_pear_tape_480_640_succ_t0001_s-0-0',
            DATA_DIR + '/ume/0103_brown_mug_cutter_knife_bread_banana_480_640/0103_brown_mug_cutter_knife_bread_banana_480_640_succ_t0001_s-0-0',
            DATA_DIR + '/ume/0103_green_can_tennis_ball_sponge_brown_plate_480_640/0103_green_can_tennis_ball_sponge_brown_plate_480_640_succ_t0001_s-0-0',
            DATA_DIR + '/ume/0103_pink_penguin_lemon_cyan_trunk_gray_shovel_480_640/0103_pink_penguin_lemon_cyan_trunk_gray_shovel_480_640_succ_t0001_s-0-0',
            DATA_DIR + '/ume/0103_rubik_cube_apple_pink_cube_whiteboard_marker_480_640/0103_rubik_cube_apple_pink_cube_whiteboard_marker_480_640_succ_t0001_s-0-0',
            DATA_DIR + '/ume/0104_rubik_cube_cyan_trunk_tape_hex_key_480_640/0104_rubik_cube_cyan_trunk_tape_hex_key_480_640_succ_t0001_s-0-0',
            DATA_DIR + '/ume/0105_apple_pear_lemon_tennis_ball_480_640/0105_apple_pear_lemon_tennis_ball_480_640_succ_t0001_s-0-0',
            DATA_DIR + '/ume/0105_brown_mug_toy_tennis_ball_sponge_480_640/0105_brown_mug_toy_tennis_ball_sponge_480_640_succ_t0001_s-0-0',
            DATA_DIR + '/ume/0105_green_paper_cup_cutter_knife_whiteboard_marker_brown_plate_480_640/0105_green_paper_cup_cutter_knife_whiteboard_marker_brown_plate_480_640_succ_t0001_s-0-0',
            DATA_DIR + '/ume/0105_pink_penguin_shovel_bananan_golves_480_640/0105_pink_penguin_shovel_bananan_golves_480_640_succ_t0001_s-0-0',
        ],
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
        "sample_weights": [1, 1]
    },
    'folding_blue_shirt': { # for local debug
        'dataset_dir': [
            "/media/rl/HDD/data/data/aloha_data/4_cameras_aloha/folding_shirt",
            # "/media/rl/HDD/data/data/aloha_data/4_cameras_aloha/fold_shirt_wjj1213_meeting_room",
            # "/media/rl/HDD/data/data/aloha_data/4_cameras_aloha/fold_tshirts_129",
            # "/media/rl/HDD/data/data/aloha_data/4_cameras_aloha/fold_tshirts_zzy_1209"

        ],
        'episode_len': 1000,  # 1000,
        # 'camera_names': ['cam_front', 'cam_high', 'cam_left_wrist', 'cam_right_wrist']
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    '3_cameras_random_folding_1_25': {
        'dataset_dir': [
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_second_tshirt_yichen_0108',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_second_tshirt_wjj_0108',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_random_yichen_0109',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_random_table_right_wjj_0109',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_two_tshirt_yichen_0109',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_second_tshirt_yichen_0110',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_second_tshirt_yichen_0109',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_second_tshirt_wjj_0110',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_yichen_0111',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_wjj_0113',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_wjj_0111',

            # 1.17 2025 new add
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_first_tshirt_dark_blue_yichen_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_first_tshirt_pink_wjj_0115",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_blue_yichen_0115",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_dark_blue_yichen_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_red_lxy_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_red_wjj_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_shu_red_yellow_wjj_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_yellow_shu_red_wjj_0116",

            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_14_data_move_add_folding_shirt/move_data/folding_basket_second_tshirt_yichen_0114",

            # 1.19 2025 new add
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_18_extract/weiqing_folding_basket_second_dark_blue_shirt_to_polo_lxy_0118",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_17_folding_basket_extract/weiqing_folding_basket_first_yellow_blue_wjj_0117",
            # 3 camera views
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_17_folding_basket_extract/weiqing_folding_basket_second_dark_blue_polo_to_blue_shirt_lxy_0117",
            # 3 camera views
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_17_folding_basket_extract/weiqing_folding_basket_second_yellow_blue_wjj_0117",
            # 3 camera views

            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_21_7z_extract/folding_random_short_first_wjj_0121",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_21_7z_extract/folding_random_short_second_wjj_0121",

            # 1.23
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_22_7z_extract/folding_random_short_second_wjj_0122",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_22_7z_extract/folding_random_short_first_wjj_0122",
            # 1.25 add
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_24_folding_7z_extract/folding_random_tshirt_first_wjj_0124",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_24_folding_7z_extract/folding_random_tshirt_second_wjj_0124",
        ],
        'episode_len': 1000,  # 1000,
        # 'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    '3_cameras_all_data_1_17': {
        'dataset_dir': [

            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_lxy1213',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_lxy1214',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_zmj1212',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_zmj1213',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_zzy1213',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_junjie_1224',  # 50
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_zhongyi_1224',  # 42
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_wjj1213_meeting_room',  # 42
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_30_wjj_weiqing_recover',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_31_wjj_lab_marble_recover',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_31_zhouzy_lab_marble',
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_blue_tshirt_yichen_0103",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_blue_tshirt_xiaoyu_0103",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_blue_tshirt_yichen_0102",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_28_zzy_right_first",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_27_office",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/0107_wjj_folding_blue_shirt",
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_second_tshirt_yichen_0108',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_second_tshirt_wjj_0108',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_random_yichen_0109',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_random_table_right_wjj_0109',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_two_tshirt_yichen_0109',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_second_tshirt_yichen_0110',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_second_tshirt_yichen_0109',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_second_tshirt_wjj_0110',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_yichen_0111',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_wjj_0113',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_wjj_0111',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_14_data_move_add_folding_shirt/move_data/folding_basket_second_tshirt_yichen_0114',
            # 1.17 2025 new add
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_first_tshirt_dark_blue_yichen_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_first_tshirt_pink_wjj_0115",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_blue_yichen_0115",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_dark_blue_yichen_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_red_lxy_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_red_wjj_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_shu_red_yellow_wjj_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_yellow_shu_red_wjj_0116",

            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_ljm_1217',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_zmj_1217_green_plate_coke_can_brown_mug_bottle',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_lxy_1220_blue_plate_pink_paper_cup_plastic_bag_knife',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_zzy_1220_green_paper_cup_wulong_bottle_pink_bowl_brown_spoon',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_zmj_1220_green_cup_blue_paper_ball_pink_plate_sprite',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_zmj_1217_green_plate_coke_can_brown_mug_bottle',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_lxy_1222_pick_place_water_left_arm',

            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/pick_cup_and_pour_water_wjj_weiqing_coke',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/pick_cars_from_moving_belt_waibao_1227',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/pick_cup_and_pour_water_wjj_weiqing_coffee',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/pick_cars_from_moving_belt_zhumj_1227',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/hang_cups_waibao',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/storage_bottle_green_tea_oolong_mineral_water_ljm_weiqing_1225_right_hand',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/storage_bottle_green_tea_oolong_mineral_water_lxy_weiqing_1225',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/get_papercup_yichen_1223',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/pour_coffee_zhaopeiting_1224',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/get_papercup_and_pour_coke_yichen_1224',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/pick_up_coke_in_refrigerator_yichen_1223',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/pour_rice_yichen_0102',

            # from Shanghai University
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/pick_paper_ball_from_bike',

        ],
        'episode_len': 1000,  # 1000,
        # 'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    '3_cameras_all_data_1_17_compressed': {
    'dataset_dir': [

        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/fold_shirt_lxy1213',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/fold_shirt_lxy1214',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/fold_shirt_zmj1212',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/fold_shirt_zmj1213',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/fold_shirt_zzy1213',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_junjie_1224',  # 50
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_zhongyi_1224',  # 42
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/fold_shirt_wjj1213_meeting_room',  # 42
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_30_wjj_weiqing_recover',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_31_wjj_lab_marble_recover',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_31_zhouzy_lab_marble',
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_blue_tshirt_yichen_0103",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_blue_tshirt_xiaoyu_0103",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_blue_tshirt_yichen_0102",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_shirt_12_28_zzy_right_first",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_shirt_12_27_office",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/0107_wjj_folding_blue_shirt",
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_second_tshirt_yichen_0108',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_second_tshirt_wjj_0108',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_random_yichen_0109',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_random_table_right_wjj_0109',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_two_tshirt_yichen_0109',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_second_tshirt_yichen_0110',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_second_tshirt_yichen_0109',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_second_tshirt_wjj_0110',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_yichen_0111',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_wjj_0113',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_wjj_0111',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/1_14_data_move_add_folding_shirt/move_data/folding_basket_second_tshirt_yichen_0114',

        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_second_tshirt_yichen_0108',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_second_tshirt_wjj_0108',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_random_yichen_0109',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_random_table_right_wjj_0109',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_two_tshirt_yichen_0109',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_second_tshirt_yichen_0110',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_second_tshirt_yichen_0109',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_10_extract/folding_basket_second_tshirt_wjj_0110',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_yichen_0111',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_wjj_0113',
        '/home/jovyan/tzb/h5py_data/aloha_compressed_70/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_wjj_0111',

        # 1.17 2025 new add
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_15_16_data_extract/weiqing_folding_basket_first_tshirt_dark_blue_yichen_0116",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_15_16_data_extract/weiqing_folding_basket_first_tshirt_pink_wjj_0115",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_blue_yichen_0115",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_dark_blue_yichen_0116",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_red_lxy_0116",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_red_wjj_0116",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_shu_red_yellow_wjj_0116",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_yellow_shu_red_wjj_0116",

        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/1_14_data_move_add_folding_shirt/move_data/folding_basket_second_tshirt_yichen_0114",

        # 1.19 2025 new add
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_18_extract/weiqing_folding_basket_second_dark_blue_shirt_to_polo_lxy_0118",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_17_folding_basket_extract/weiqing_folding_basket_first_yellow_blue_wjj_0117",
        # 3 camera views
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_17_folding_basket_extract/weiqing_folding_basket_second_dark_blue_polo_to_blue_shirt_lxy_0117",
        # 3 camera views
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/7z_1_17_folding_basket_extract/weiqing_folding_basket_second_yellow_blue_wjj_0117",
        # 3 camera views

        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/1_21_7z_extract/folding_random_short_first_wjj_0121",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/1_21_7z_extract/folding_random_short_second_wjj_0121",

        # 1.23
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/1_22_7z_extract/folding_random_short_second_wjj_0122",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/1_22_7z_extract/folding_random_short_first_wjj_0122",
        # 1.25 add
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/1_24_folding_7z_extract/folding_random_tshirt_first_wjj_0124",
        "/home/jovyan/tzb/h5py_data/aloha_compressed_70/1_24_folding_7z_extract/folding_random_tshirt_second_wjj_0124",
    ],
    'episode_len': 1000,  # 1000,
    # 'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
},

    '3_cameras_1_17_standard_folding': {
        'dataset_dir': [
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_lxy1213',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_lxy1214',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_zmj1212',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_zmj1213',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_zzy1213',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_junjie_1224',  # 50
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_zhongyi_1224',  # 42
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_wjj1213_meeting_room',  # 42
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_30_wjj_weiqing_recover',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_31_wjj_lab_marble_recover',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_31_zhouzy_lab_marble',
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_blue_tshirt_yichen_0103",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_blue_tshirt_xiaoyu_0103",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_blue_tshirt_yichen_0102",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_28_zzy_right_first",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_27_office",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/0107_wjj_folding_blue_shirt",
        ],
        'episode_len': 1000,  # 1000,
        # 'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    '3_cameras_1_17_standard_folding_compress': {
        'dataset_dir': [
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/fold_shirt_lxy1213',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/fold_shirt_lxy1214',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/fold_shirt_zmj1212',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/fold_shirt_zmj1213',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/fold_shirt_zzy1213',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_junjie_1224',  # 50
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_zhongyi_1224',  # 42
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/fold_shirt_wjj1213_meeting_room',  # 42
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_30_wjj_weiqing_recover',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_31_wjj_lab_marble_recover',
            '/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_31_zhouzy_lab_marble',
            "/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_blue_tshirt_yichen_0103",
            "/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_blue_tshirt_xiaoyu_0103",
            "/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_blue_tshirt_yichen_0102",
            "/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_shirt_12_28_zzy_right_first",
            "/home/jovyan/tzb/h5py_data/aloha_compressed_70/folding_shirt_12_27_office",
            "/home/jovyan/tzb/h5py_data/aloha_compressed_70/0107_wjj_folding_blue_shirt",
        ],
        'episode_len': 1000,  # 1000,
        # 'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    '3_cameras_all_data_1_25': {
        'dataset_dir': [

            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_lxy1213',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_lxy1214',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_zmj1212',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_zmj1213',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_zzy1213',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_junjie_1224',  # 50
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_zhongyi_1224',  # 42
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/fold_shirt_wjj1213_meeting_room',  # 42
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_30_wjj_weiqing_recover',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_31_wjj_lab_marble_recover',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_30_12_31_extract/folding_shirt_12_30_12_31/folding_shirt_12_31_zhouzy_lab_marble',
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_blue_tshirt_yichen_0103",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_blue_tshirt_xiaoyu_0103",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_blue_tshirt_yichen_0102",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_28_zzy_right_first",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/folding_shirt_12_27_office",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/0107_wjj_folding_blue_shirt",
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_10_extract/folding_second_tshirt_yichen_0108',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_10_extract/folding_second_tshirt_wjj_0108',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_10_extract/folding_random_yichen_0109',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_10_extract/folding_random_table_right_wjj_0109',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_10_extract/folding_basket_two_tshirt_yichen_0109',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_10_extract/folding_basket_second_tshirt_yichen_0110',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_10_extract/folding_basket_second_tshirt_yichen_0109',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_10_extract/folding_basket_second_tshirt_wjj_0110',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_yichen_0111',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_wjj_0113',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/data_01_11_13_7z_exact/data_01_11_13/folding_basket_second_tshirt_wjj_0111',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_14_data_move_add_folding_shirt/move_data/folding_basket_second_tshirt_yichen_0114',
            # 1.17 2025 new add
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_first_tshirt_dark_blue_yichen_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_first_tshirt_pink_wjj_0115",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_blue_yichen_0115",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_dark_blue_yichen_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_red_lxy_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_red_wjj_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_shu_red_yellow_wjj_0116",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/weiqing_folding_basket_second_tshirt_yellow_shu_red_wjj_0116",

            # 1.21 added
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_20_data_extract/unloading_dryer_yichen_0120",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_20_data_extract/unloading_dryer_yichen_0119",
            #
            # 1.22
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_21_7z_extract/folding_random_short_first_wjj_0121",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_21_7z_extract/folding_random_short_second_wjj_0121",

            # 1.23
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_22_7z_extract/folding_random_short_second_wjj_0122",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_22_7z_extract/folding_random_short_first_wjj_0122",

            # 1.25
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_24_folding_7z_extract/folding_random_tshirt_first_wjj_0124",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_24_folding_7z_extract/folding_random_tshirt_second_wjj_0124",

            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/1_24_7z_extract/truncate_push_basket_to_left_1_24/",

            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_ljm_1217',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_zmj_1217_green_plate_coke_can_brown_mug_bottle',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_lxy_1220_blue_plate_pink_paper_cup_plastic_bag_knife',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_zzy_1220_green_paper_cup_wulong_bottle_pink_bowl_brown_spoon',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_zmj_1220_green_cup_blue_paper_ball_pink_plate_sprite',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_zmj_1217_green_plate_coke_can_brown_mug_bottle',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/clean_table_lxy_1222_pick_place_water_left_arm',

            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/pick_cup_and_pour_water_wjj_weiqing_coke',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/pick_cars_from_moving_belt_waibao_1227',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/pick_cup_and_pour_water_wjj_weiqing_coffee',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/pick_cars_from_moving_belt_zhumj_1227',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/hang_cups_waibao',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/storage_bottle_green_tea_oolong_mineral_water_ljm_weiqing_1225_right_hand',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/storage_bottle_green_tea_oolong_mineral_water_lxy_weiqing_1225',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/get_papercup_yichen_1223',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/pour_coffee_zhaopeiting_1224',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/get_papercup_and_pour_coke_yichen_1224',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/pick_up_coke_in_refrigerator_yichen_1223',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/pour_rice_yichen_0102',

            # from Shanghai University
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/pick_paper_ball_from_bike',

        ],
        'episode_len': 1000,  # 1000,
        # 'camera_names': ['cam_front', 'cam_high', 'cam_left_wrist', 'cam_right_wrist']
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    '3_cameras_only_unloading_dryer': {
        'dataset_dir': [
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_20_data_extract/unloading_dryer_yichen_0120",
            "/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/7z_1_20_data_extract/unloading_dryer_yichen_0119",
        ],
        'episode_len': 1000,  # 1000,
        # 'camera_names': ['cam_front', 'cam_high', 'cam_left_wrist', 'cam_right_wrist']
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
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

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / \
            (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
            PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (
            MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (
            PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (
            MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (
            PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (
            MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (
            PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (
            MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (
            PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2
