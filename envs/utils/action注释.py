from typing import Literal
from .transforms import _tolist
import numpy as np
import sapien

class ArmTag:
    """手臂标签类，用于标识左右手臂。

    该类实现了单例模式，确保"left"和"right"只有唯一实例。

    Attributes:
        arm (str): 手臂标签，仅能为"left"或"right"。
    """

    _instances = {}  # 存储已创建的实例，保证单例

    def __new__(cls, value):
        """创建新的ArmTag实例或返回已存在的实例。

        Args:
            value (str|ArmTag): 手臂标签，"left"或"right"。

        Returns:
            ArmTag: 对应的ArmTag实例。

        Raises:
            ValueError: 如果value不是"left"或"right"。
        """
        if isinstance(value, ArmTag):  # 如果已经是ArmTag实例，直接返回
            return value
        if isinstance(value, str) and value in ["left", "right"]:
            value = value.lower()
            if value in cls._instances:  # 已存在则返回单例
                return cls._instances[value]
            instance = super().__new__(cls)  # 创建新实例
            cls._instances[value] = instance
            return instance
        raise ValueError(f"Invalid arm tag: {value}. Must be 'left' or 'right'.")  # 非法标签

    def __init__(self, value):
        """初始化ArmTag实例。

        Args:
            value (str): 手臂标签。
        """
        if isinstance(value, str):
            self.arm = value.lower()  # 统一转为小写

    @property
    def opposite(self):
        """获取相反的手臂标签。

        Returns:
            ArmTag: 相反的ArmTag实例。
        """
        return ArmTag("right") if self.arm == "left" else ArmTag("left")

    def __eq__(self, other):
        """判断两个ArmTag是否相等。

        Args:
            other (ArmTag|str): 另一个ArmTag或字符串。

        Returns:
            bool: 是否相等。
        """
        if isinstance(other, ArmTag):
            return self.arm == other.arm
        if isinstance(other, str):
            return self.arm == other.lower()
        return False

    def __hash__(self):
        """返回哈希值，使其可作为字典键。"""
        return hash(self.arm)

    def __repr__(self):
        """返回实例的字符串表示。"""
        return f"ArmTag('{self.arm}')"

    def __str__(self):
        """返回标签字符串。"""
        return self.arm

class Action:
    """动作类，描述机械臂的动作指令。

    Attributes:
        arm_tag (ArmTag): 动作对应的手臂标签。
        action (str): 动作类型，"move"或"gripper"。
        target_pose (list): 目标位姿，仅对"move"动作有效。
        target_gripper_pos (float): 夹爪目标位置，仅对"gripper"动作有效。
        args (dict): 其他额外参数。
    """

    arm_tag: ArmTag
    action: Literal["move", "gripper"]
    target_pose: list = None
    target_gripper_pos: float = None

    def __init__(
        self,
        arm_tag: ArmTag | Literal["left", "right"],
        action: Literal["move", "open", "close", "gripper"],
        target_pose: sapien.Pose | list | np.ndarray = None,
        target_gripper_pos: float = None,
        **args,
    ):
        """初始化Action实例。

        Args:
            arm_tag (ArmTag|str): 手臂标签。
            action (str): 动作类型，"move"、"open"、"close"、"gripper"。
            target_pose (Pose|list|ndarray, optional): 目标位姿，仅对"move"有效。
            target_gripper_pos (float, optional): 夹爪目标位置，仅对"gripper"有效。
            **args: 其他额外参数。

        Raises:
            ValueError: 动作类型非法。
            AssertionError: 必要参数缺失。
        """
        self.arm_tag = ArmTag(arm_tag)  # 转换为ArmTag实例
        if action != "move":
            # 非move动作，处理夹爪动作
            if action == "open":
                self.action = "gripper"
                # "open"默认夹爪位置为1.0
                self.target_gripper_pos = (target_gripper_pos if target_gripper_pos is not None else 1.0)
            elif action == "close":
                self.action = "gripper"
                # "close"默认夹爪位置为0.0
                self.target_gripper_pos = (target_gripper_pos if target_gripper_pos is not None else 0.0)
            elif action == "gripper":
                self.action = "gripper"
            else:
                raise ValueError(f"Invalid action: {action}. Must be 'open' or 'close'.")  # 非法动作
            assert (self.target_gripper_pos is not None), "target_gripper_pos cannot be None for gripper action."  # 夹爪动作必须有目标位置
        else:
            # move动作，必须有目标位姿
            self.action = "move"
            assert (target_pose is not None), "target_pose cannot be None for move action."
            self.target_pose = _tolist(target_pose)  # 转为list
        self.args = args  # 记录额外参数

    def __str__(self):
        """返回动作的字符串描述。

        Returns:
            str: 动作描述字符串。
        """
        result = f"{self.arm_tag}: {self.action}"
        if self.action == "move":
            result += f"({self.target_pose})"  # move动作显示目标位姿
        else:
            result += f"({self.target_gripper_pos})"  # gripper动作显示目标夹爪位置
        if self.args:
            result += f"    {self.args}"  # 显示额外参数
        return result
