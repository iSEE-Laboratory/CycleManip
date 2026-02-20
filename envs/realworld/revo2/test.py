from revo2Controler import Revo2HandController, logging
import time


if __name__ == "__main__":
    # 简单测试Revo2手控制器
    hand_controller = Revo2HandController(port="/dev/ttyUSB0", slave_id=0x7f, logger_level=logging.DEBUG)
    # hand_controller = Revo2HandController(port="/dev/ttyUSB1", slave_id=0x7e, logger_level=logging.DEBUG)
    
    try:
        info = hand_controller.get_device_info()
        print("Device Info:", info)
        
        positions = hand_controller.get_joint_positions()
        print("Current Joint Positions:", positions)
        
        # 设置所有关节到中间位置
        target_positions = [500, 500, 500, 500, 500, 500]
        hand_controller.set_joint_positions(target_positions)
        print("Set joint positions to:", target_positions)
        
        time.sleep(2)  # 等待运动完成
        
        new_positions = hand_controller.get_joint_positions()
        print("New Joint Positions:", new_positions)
        
    finally:
        hand_controller.close()