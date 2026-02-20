import sys
import time
import asyncio
import threading
import logging
from typing import List, Optional
from .revo2_utils import *


class Revo2HandController:
    """
    Revo2灵巧手控制器 - 提供同步接口的异步手控制封装
    """
    
    def __init__(self, port: str = "/dev/ttyUSB0", slave_id: int = 0x7f, 
                 baudrate: int = libstark.Baudrate.Baud460800, 
                 logger_level: int = logging.INFO):
        """
        初始化Revo2灵巧手控制器
        
        Args:
            port: 串口设备路径
            slave_id: 设备从站ID (右手: 0x7f, 左手: 0x7e)
            baudrate: 波特率
            logger_level: 日志级别
        """
        self.port = port
        self.slave_id = slave_id
        self.baudrate = baudrate
        
        # 设置日志
        self.logger = getLogger(logger_level)
        
        # 异步相关变量
        self.client = None
        self.hand_loop = None
        self.hand_thread = None
        self._initialized = False
        self._lock = threading.Lock()
        
        # 启动手控制线程
        self._start_hand_control()
    
    def _start_hand_control(self):
        """启动手控制的异步事件循环"""
        self.hand_loop = asyncio.new_event_loop()
        self.hand_thread = threading.Thread(
            target=self._run_hand_loop, 
            daemon=True,
            name="Revo2HandThread"
        )
        self.hand_thread.start()
        
        # 等待初始化完成
        self._initialize_hand()
    
    def _run_hand_loop(self):
        """在独立线程中运行手的事件循环"""
        asyncio.set_event_loop(self.hand_loop)
        try:
            self.hand_loop.run_forever()
        except Exception as e:
            self.logger.error(f"Hand loop error: {e}")
        finally:
            if self.hand_loop.is_running():
                self.hand_loop.close()
    
    def _run_async_in_thread(self, coro, timeout: float = 10.0):
        """在异步线程中运行协程并等待结果"""
        if not self._initialized:
            raise RuntimeError("Hand controller not initialized")
        
        future = asyncio.run_coroutine_threadsafe(coro, self.hand_loop)
        return future.result(timeout=timeout)
    
    def _initialize_hand(self):
        """异步初始化手"""
        async def init_async():
            try:
                self.logger.info(f"Initializing Revo2 hand on {self.port}, slave_id: {self.slave_id}")
                
                self.client = await libstark.modbus_open(self.port, self.baudrate)
                if not self.client:
                    raise RuntimeError(f"Failed to open serial port: {self.port}")
                
                info = await self.client.get_device_info(self.slave_id)
                if not info:
                    raise RuntimeError(f"Failed to get device info for hand. Id: {self.slave_id}")
                
                self.logger.info(f"Hand initialized: {info.description}")
                
                # 设置手指单位模式为归一化模式 (0-1000)
                await self.client.set_finger_unit_mode(self.slave_id, libstark.FingerUnitMode.Normalized)
                
                if info.is_revo2():
                    if info.is_revo2_touch():
                        self.logger.info("触觉版设备")
                    else:
                        self.logger.info("标准版设备")
                
                self._initialized = True

                self.client.set_turbo_mode_enabled(self.slave_id, True)

                self.logger.info("Revo2 hand initialization completed successfully")
                return True
                
            except Exception as e:
                self.logger.critical(f"Hand initialization failed: {e}")
                self._initialized = False
                return False
        
        # 在hand_loop中运行异步初始化并等待结果
        future = asyncio.run_coroutine_threadsafe(init_async(), self.hand_loop)
        result = future.result(timeout=15.0)  # 15秒超时
        
        if not result:
            raise RuntimeError("Failed to initialize Revo2 hand")
    
    # ================================
    # 公共接口 - 同步调用
    # ================================
    
    def get_device_info(self) -> dict:
        """
        获取设备信息
        
        Returns:
            dict: 包含设备信息的字典
        """
        async def get_info_async():
            info = await self.client.get_device_info(self.slave_id)
            return {
                'description': info.description,
                'serial_number': info.serial_number,
                'firmware_version': info.firmware_version,
                'is_revo2': info.is_revo2(),
                'is_revo2_touch': info.is_revo2_touch(),
                'is_touch': info.is_touch()
            }
        
        return self._run_async_in_thread(get_info_async())
    
    def get_joint_positions(self) -> List[int]:
        """
        获取所有关节的当前位置
        
        Returns:
            List[int]: 6个关节的位置值 (0-1000)
        """
        async def get_positions_async():
            return await self.client.get_finger_positions(self.slave_id)
        
        return self._run_async_in_thread(get_positions_async())
    
    def get_joint_speeds(self) -> List[int]:
        """
        获取所有关节的当前速度
        
        Returns:
            List[int]: 6个关节的速度值
        """
        async def get_speeds_async():
            return await self.client.get_finger_speeds(self.slave_id)
        
        return self._run_async_in_thread(get_speeds_async())
    
    def get_joint_currents(self) -> List[int]:
        """
        获取所有关节的当前电流
        
        Returns:
            List[int]: 6个关节的电流值
        """
        async def get_currents_async():
            return await self.client.get_finger_currents(self.slave_id)
        
        return self._run_async_in_thread(get_currents_async())
    
    def get_motor_status(self) -> dict:
        """
        获取完整的电机状态信息
        
        Returns:
            dict: 包含位置、速度、电流和状态的字典
        """
        async def get_status_async():
            status = await self.client.get_motor_status(self.slave_id)
            return {
                'positions': status.positions,
                'speeds': status.speeds,
                'currents': status.currents,
                'states': [state.value for state in status.states],
                'description': status.description
            }
        
        return self._run_async_in_thread(get_status_async())
    
    def set_joint_positions(self, positions: List[int], speeds: Optional[List[int]] = None):
        """
        设置所有关节的目标位置
        
        Args:
            positions: 6个关节的目标位置 (0-1000)
            speeds: 6个关节的运动速度，默认为[1000, 1000, 1000, 1000, 1000, 1000]
        """
        print(f"[hand] {positions}")
        if speeds is None:
            speeds = [1000] * 6
        
        if len(positions) != 6:
            raise ValueError("Positions must contain exactly 6 values")
        if len(speeds) != 6:
            raise ValueError("Speeds must contain exactly 6 values")
        
        async def set_positions_async():
            await self.client.set_finger_positions_and_speeds(
                self.slave_id, positions, speeds
            )
        
        self._run_async_in_thread(set_positions_async(), timeout=5.0)
        self.logger.debug(f"Set joint positions: {positions}")
    
    def set_single_joint_position(self, joint_index: int, position: int, speed: int = 1000):
        """
        设置单个关节的目标位置
        
        Args:
            joint_index: 关节索引 (0-5)
            position: 目标位置 (0-1000)
            speed: 运动速度
        """
        if not 0 <= joint_index <= 5:
            raise ValueError("Joint index must be between 0 and 5")
        
        # 获取当前所有关节位置
        current_positions = self.get_joint_positions()
        # 更新指定关节的位置
        current_positions[joint_index] = position
        
        # 设置所有关节位置
        self.set_joint_positions(current_positions, [speed] * 6)
        self.logger.debug(f"Set joint {joint_index} position: {position}")
    
    def get_touch_sensor_status(self) -> Optional[dict]:
        """
        获取触觉传感器状态 (仅触觉版设备支持)
        
        Returns:
            dict: 触觉传感器状态信息，如果不支持则返回None
        """
        async def get_touch_async():
            try:
                if not self.client:
                    return None
                
                # 检查设备是否支持触觉
                info = await self.client.get_device_info(self.slave_id)
                if not info.is_touch():
                    return None
                
                touch_status = await self.client.get_touch_sensor_status(self.slave_id)
                result = {}
                
                for i, finger in enumerate(touch_status):
                    finger_name = ['thumb', 'index', 'middle', 'ring', 'pinky'][i]
                    result[finger_name] = {
                        'status': finger.status,
                        'normal_force': finger.normal_force1,
                        'tangential_force': finger.tangential_force1,
                        'tangential_direction': finger.tangential_direction1,
                        'proximity': finger.self_proximity1
                    }
                
                return result
            except Exception as e:
                self.logger.warning(f"Failed to get touch sensor status: {e}")
                return None
        
        try:
            return self._run_async_in_thread(get_touch_async(), timeout=3.0)
        except:
            return None
    
    def is_initialized(self) -> bool:
        """
        检查手控制器是否已初始化
        
        Returns:
            bool: 初始化状态
        """
        return self._initialized
    
    def wait_until_ready(self, timeout: float = 30.0):
        """
        等待手控制器准备就绪
        
        Args:
            timeout: 超时时间(秒)
        """
        start_time = time.time()
        while not self._initialized:
            if time.time() - start_time > timeout:
                raise TimeoutError("Hand controller not ready within timeout")
            time.sleep(0.1)
    
    def close(self):
        """
        关闭手控制器，释放资源
        """
        self.logger.info("Closing Revo2 hand controller...")
        
        if self.hand_loop and self.hand_loop.is_running():
            # 优雅关闭
            async def close_async():
                if self.client:
                    try:
                        # 先停止所有运动
                        await self.client.set_finger_positions_and_speeds(
                            self.slave_id, [0, 0, 0, 0, 0, 0], [1000] * 6
                        )
                        await asyncio.sleep(0.1)
                    except:
                        pass
                    
                    try:
                        await libstark.modbus_close(self.client)
                    except:
                        pass
                    finally:
                        self.client = None
            
            try:
                future = asyncio.run_coroutine_threadsafe(close_async(), self.hand_loop)
                future.result(timeout=5.0)
            except Exception as e:
                self.logger.warning(f"Error during hand close: {e}")
            
            # 停止事件循环
            self.hand_loop.call_soon_threadsafe(self.hand_loop.stop)
        
        if self.hand_thread:
            self.hand_thread.join(timeout=2.0)
        
        self._initialized = False
        self.logger.info("Revo2 hand controller closed")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

