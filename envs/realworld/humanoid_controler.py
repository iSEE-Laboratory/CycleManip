# 录制完数据到replay要更改两个地方；
# 1. lcm_agent.py里面的zmq队列的地址改成127.0.0.1
# 2. cheetah_state_estimator.py 里面的_rc_command_cb函数注释掉摇杆赋值的两行代码

import json
import time
import numpy as np
import zmq
import threading
from utils.cheetah_state_estimator import StateEstimator
import lcm
from lcm_types.rc_command_lcmt import rc_command_lcmt

from lcm_types.command_lcmt import command_lcmt
import os
import mmap

latest_obs_file = "/tmp/latest_obs.dat"

# 全局变量存储最新的上肢动作和相机数据
latest_sol_q = None
latest_sol_cam = None
data_lock = threading.Lock()

# 全局变量存储最新的控制指令
latest_upper_ctrl = np.zeros(14)  # 上半身控制指令
latest_command_ctrl = np.zeros(4)  # 命令控制指令
upper_ctrl_lock = threading.Lock()  # 保护上半身控制变量的锁
command_ctrl_lock = threading.Lock()  # 保护命令控制变量的锁

SHM_SIZE = 1024 * 1024  # 1MB

def get_latest_observation():
    """
    从共享内存获取最新的观测数据{"step", ""timestamp", "joints_obs", "command_ctrl", "upper_ctrl"}
    返回: 包含观测数据的字典，如果无数据则返回None
    """
    try:
        with open(latest_obs_file, "rb") as f:
            shm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            shm.seek(0)
            obs_json = shm.read(SHM_SIZE).split(b'\x00')[0]  # 读取直到null字符
            if obs_json:
                obs_data = json.loads(obs_json.decode('utf-8'))
                shm.close()
                return obs_data
            shm.close()
    except FileNotFoundError:
        # 共享内存文件不存在
        pass
    except Exception as e:
        print(f"读取观测数据出错: {e}")
    return None

def load_log_data(log_file_path):
    """加载日志数据, 用于debug"""
    log_data = []
    with open(log_file_path, 'r') as f:
        for line in f:
            log_data.append(json.loads(line))
    return log_data

def zmq_publisher():
    """50Hz发布最新的上肢动作数据到ZMQ"""
    global latest_upper_ctrl
    
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5556")  # 使用5556端口发布数据
    
    print("开始发布上肢动作数据...")
    
    dt = 1/50  # 50Hz
    start_time = time.time()
    idx = 0
    
    while True:
        # 获取最新的上肢控制指令
        with upper_ctrl_lock:
            sol_q = latest_upper_ctrl.copy()
        
        # 发布数据
        sol_q_json = json.dumps(sol_q.tolist())
        zmq_msg = f"avp_arm_data {sol_q_json}"
        socket.send_string(zmq_msg)
        
        # 控制发送频率
        elapsed_time = time.time() - start_time
        sleep_time = max(dt - (elapsed_time - idx * dt), 0)
        time.sleep(sleep_time)
        idx += 1

def lcm_command_publisher():
    """50Hz发布最新的遥控器指令数据到LCM"""
    global latest_command_ctrl
    
    # 设置LCM
    os.environ["LCM_DEFAULT_URL"] = "eth0"
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    
    print("开始发布遥控器指令数据...")
    
    dt = 1/50  # 50Hz
    start_time = time.time()
    idx = 0

    rc_msg = rc_command_lcmt()

    while True:
        # 获取最新的命令控制指令
        with command_ctrl_lock:
            commands_scaled = latest_command_ctrl.copy()
        
        # 还原缩放因子 [2.0, 2.0, 0.25, 1.0]
        commands = np.array([
            commands_scaled[0] / 2.0,  # cmd_x
            commands_scaled[1] / 2.0,  # cmd_y
            commands_scaled[2] / 0.25, # cmd_yaw
            commands_scaled[3] / 1.0   # cmd_height
        ])

        # 发布rc_command数据
        rc_msg.left_stick = [
            -commands[1] / 0.5,  # left_stick[0]
            commands[0] / 0.6    # left_stick[1]
        ]  # 转换回摇杆值
        rc_msg.right_stick = [
            -commands[2] / 0.8,              # right_stick[0]
            (0.74 - commands[3]) / 0.46      # right_stick[1]
        ]

        lc.publish("rc_command_dummy", rc_msg.encode())

        # 控制发送频率
        elapsed_time = time.time() - start_time
        sleep_time = max(dt - (elapsed_time - idx * dt), 0)
        time.sleep(sleep_time)
        idx += 1

def update_control_variables(log_data):
    """主进程更新控制变量"""
    global latest_upper_ctrl, latest_command_ctrl
    
    print("开始更新控制变量...")
    
    # 用于每隔20条记录更新一次的计数器
    update_counter = 0
    dt = 1/50  # 50Hz
    start_time = time.time()
    
    # start for 当前采用log数据间隔20条模仿低频的DP策略；
    # TODO：
    # 改成使用while循环
    # 先通过 latest_obs = get_latest_observation()获取观测
    # 然后提供给command_ctrl, upper_ctrl = DP(latest_obs)
    # 更新全局ctrl变量
    while True:
        pass
        
    
    # for idx in range(len(log_data)):
    #     # 每隔20条记录才更新一次控制指令
    #     if update_counter % 20 == 0:
    #         data = log_data[idx]
            
    #         # 更新上半身控制指令
    #         with upper_ctrl_lock:
    #             latest_upper_ctrl = np.array(data['upper_ctrl'])
                
    #         # 更新命令控制指令
    #         with command_ctrl_lock:
    #             latest_command_ctrl = np.array(data['command_ctrl'])
            
    #         # 获取最新观测状态
    #         latest_obs = get_latest_observation()
    #         if latest_obs is not None:
    #             # 第0次: obs时间戳: 1762792516.0187137 当前时间戳: 1762792516.0439577
    #             print(f"第{idx}次: obs时间戳: {latest_obs['timestamp']} 当前时间戳: {time.time()}")
    #             # print(f"{latest_obs['step']}")
    #             # print(f"{latest_obs['joints_obs']}")
    #             # print(f"{latest_obs['command_ctrl']}")
    #             # print(f"{latest_obs['upper_ctrl']}")
                

    #         else:
    #             print(f"第{idx}次更新控制变量时未获取到观测状态")
        
    #     update_counter += 1
        
    #     # 精确控制更新频率（考虑代码执行时间）
    #     elapsed_time = time.time() - start_time
    #     sleep_time = max(dt - (elapsed_time - idx * dt), 0)
    #     time.sleep(sleep_time)
    # # end for
    
    # print("控制变量更新完成")

def main():
    # 加载日志数据
    log_file_path = "./record/5pump_3.json"
    time.sleep(2)
    try:
        log_data = load_log_data(log_file_path)
        print(f"成功加载 {len(log_data)} 条日志记录")
    except FileNotFoundError:
        print(f"未找到日志文件: {log_file_path}，将使用默认数据")
        log_data = []
    
    # 启动ZMQ发布线程
    zmq_thread = threading.Thread(target=zmq_publisher, daemon=True)
    zmq_thread.start()
    
    # 启动LCM命令发布线程
    lcm_thread = threading.Thread(target=lcm_command_publisher, daemon=True)
    lcm_thread.start()
    
    # 启动主进程更新控制变量
    update_control_variables(log_data)
    
    # 等待线程结束
    zmq_thread.join()
    lcm_thread.join()
    
    print("所有数据infer完成")

if __name__ == '__main__':
    main()