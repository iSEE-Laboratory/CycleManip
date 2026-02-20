import numpy as np

# 从文本文件中读取数据
def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file.readlines()]
    return data

# 计算均值和标准差
def compute_statistics(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return mean, std_dev

# 主程序
file_path = '/home/liaohaoran/code/RoboTwin/policy/DP3/scripts/test.txt'  # 替换为你的txt文件路径
data = read_data_from_txt(file_path)
mean, std_dev = compute_statistics(data)

# 输出结果
print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")
