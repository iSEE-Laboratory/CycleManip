import json
import os

def process(task_name, task_cfg):
    

  # 构建出路径, 从当前文件开始向上一级到data目录
  data_dir = os.path.join(os.path.dirname(__file__), "..", task_name, task_cfg)

  # 1. 检查数据文件夹下是否有instrctions文件夹, 如果有，重命名为instructions_backup
  if os.path.exists(os.path.join(data_dir, "instructions")):
      os.rename(
          os.path.join(data_dir, "instructions"),
          os.path.join(data_dir, "instructions_backup"),
      )

  # 2. 创建一个新的instructions文件夹
  os.makedirs(os.path.join(data_dir, "instructions"), exist_ok=True)


  # 读取loop_times.txt文件，获取循环次数(按空格分隔)
  with open(os.path.join(data_dir, "loop_times.txt"), "r") as f:
      loop_times = list(map(int, f.read().strip().split(" ")))

  # 构建instructions文件: json
  # 为每一个循环次数创建一个指令文件，包括十个seen和十个unseen，但都是一模一样的str(num)
  # 格式如下
  """
  {
    "seen": [
      "1",
      "2",
      "3",
      ...
    ],
    "unseen": [
      "1",
      "2",
      "3",
      ...
    ]
  }
  """

  for i, loop_time in enumerate(loop_times):
      instructions = {
          "seen": [str(loop_time) for _ in range(10)],
          "unseen": [str(loop_time) for _ in range(10)],
      }

      # 保存为json文件
      with open(os.path.join(data_dir, "instructions", f"episode{i}.json"), "w") as f:
          json.dump(instructions, f, indent=2)

  print("Instruction files generated successfully.")


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("task_name", help="Name of the task")
  parser.add_argument("task_cfg", help="Configuration of the task")
  args = parser.parse_args()
  process(args.task_name, args.task_cfg)
