import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

class PeriodicOscillationAnalyzer:
    def __init__(self, max_harmonics=5):
        """
        初始化周期震荡分析器
        
        Parameters:
        max_harmonics: 傅里叶级数中使用的最大谐波数量
        """
        self.max_harmonics = max_harmonics
        
    def fourier_series(self, x, *coefficients):
        """
        傅里叶级数函数
        
        Parameters:
        x: 输入序列的索引
        coefficients: 傅里叶系数 [a0, a1, b1, a2, b2, ...]
        """
        result = coefficients[0]  # a0
        n_harmonics = (len(coefficients) - 1) // 2
        
        for n in range(1, n_harmonics + 1):
            a_n = coefficients[2*n - 1]
            b_n = coefficients[2*n]
            result += a_n * np.cos(2 * np.pi * n * x / len(x))
            result += b_n * np.sin(2 * np.pi * n * x / len(x))
            
        return result
    
    def fit_fourier_series(self, sequence):
        """
        使用傅里叶级数拟合序列
        
        Parameters:
        sequence: 输入序列
        
        Returns:
        fitted_curve: 拟合后的曲线
        coefficients: 傅里叶系数
        """
        x = np.arange(len(sequence))
        
        # 初始猜测：使用序列的均值作为a0，其他系数设为0
        initial_guess = [np.mean(sequence)]
        for i in range(self.max_harmonics):
            initial_guess.extend([0, 0])  # a_n, b_n
        
        try:
            # 使用曲线拟合找到最优傅里叶系数
            popt, _ = curve_fit(
                self.fourier_series, 
                x, 
                sequence, 
                p0=initial_guess,
                maxfev=10000
            )
            
            # 生成拟合曲线
            fitted_curve = self.fourier_series(x, *popt)
            
            return fitted_curve, popt
            
        except Exception as e:
            print(f"拟合失败: {e}")
            # 如果拟合失败，返回原始序列
            return sequence, initial_guess
    
    def detect_cycles(self, sequence, min_height_ratio=0.3, min_prominence_ratio=0.2):
        """
        检测完整周期数量
        
        Parameters:
        sequence: 输入序列
        min_height_ratio: 最小高度比例（相对于序列范围）
        min_prominence_ratio: 最小突出度比例
        
        Returns:
        cycle_count: 周期数量
        peaks: 峰值位置
        valleys: 谷值位置
        """
        # 1. 首先对序列进行平滑处理，减少噪声影响
        if len(sequence) > 11:
            smoothed_sequence = savgol_filter(sequence, 11, 3)
        else:
            smoothed_sequence = sequence
        
        # 2. 使用傅里叶级数拟合
        fitted_curve, _ = self.fit_fourier_series(smoothed_sequence)
        
        # 3. 在拟合曲线上检测峰值和谷值
        sequence_range = np.max(fitted_curve) - np.min(fitted_curve)
        min_height = np.min(fitted_curve) + min_height_ratio * sequence_range
        min_prominence = min_prominence_ratio * sequence_range
        
        # 找到峰值（局部最大值）
        peaks, peak_properties = find_peaks(
            fitted_curve, 
            height=min_height,
            prominence=min_prominence,
            distance=len(sequence)//20  # 最小峰值间距
        )
        
        # 找到谷值（局部最小值）- 在负序列上找峰值
        valleys, valley_properties = find_peaks(
            -fitted_curve, 
            height=-np.max(fitted_curve) + min_height_ratio * sequence_range,
            prominence=min_prominence,
            distance=len(sequence)//20
        )
        
        # 4. 统计完整周期
        cycle_count = 0
        valid_cycles = []
        
        # 确保有足够的极值点来形成周期
        if len(peaks) >= 2 and len(valleys) >= 1:
            # 找到第一个峰值或谷值作为起始点
            start_point = min(peaks[0], valleys[0]) if len(valleys) > 0 else peaks[0]
            
            # 交替检查峰值和谷值
            current_type = 'peak' if peaks[0] < valleys[0] else 'valley'
            cycle_start = start_point
            
            for i in range(1, max(len(peaks), len(valleys))):
                if current_type == 'peak' and i < len(valleys):
                    # 从峰值到谷值
                    current_type = 'valley'
                elif current_type == 'valley' and i < len(peaks):
                    # 从谷值到峰值 - 完成一个周期
                    current_type = 'peak'
                    cycle_count += 1
                    valid_cycles.append((cycle_start, peaks[i]))
                    cycle_start = peaks[i]
        
        return cycle_count, peaks, valleys, fitted_curve
    
    def analyze_sequence(self, sequence, plot_results=True):
        """
        完整分析序列
        
        Parameters:
        sequence: 输入序列
        plot_results: 是否绘制结果
        
        Returns:
        result: 分析结果字典
        """
        cycle_count, peaks, valleys, fitted_curve = self.detect_cycles(sequence)
        
        result = {
            'cycle_count': cycle_count,
            'peaks': peaks,
            'valleys': valleys,
            'fitted_curve': fitted_curve,
            'peak_count': len(peaks),
            'valley_count': len(valleys)
        }
        
        if plot_results:
            self.plot_analysis(sequence, result)
            
        return result
    
    def plot_analysis(self, sequence, result):
        """
        绘制分析结果
        """
        plt.figure(figsize=(12, 8))
        
        # 绘制原始序列
        plt.subplot(2, 1, 1)
        plt.plot(sequence, 'b-', alpha=0.7, label='原始序列')
        plt.plot(result['fitted_curve'], 'r-', linewidth=2, label='傅里叶拟合')
        plt.plot(result['peaks'], sequence[result['peaks']], 'ro', label='峰值')
        plt.plot(result['valleys'], sequence[result['valleys']], 'go', label='谷值')
        plt.title('序列分析和周期检测')
        plt.xlabel('索引')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制周期标记
        plt.subplot(2, 1, 2)
        plt.plot(sequence, 'b-', alpha=0.5, label='原始序列')
        plt.plot(result['fitted_curve'], 'r-', linewidth=2, label='傅里叶拟合')
        
        # 标记检测到的周期
        colors = plt.cm.tab10(np.linspace(0, 1, result['cycle_count']))
        for i in range(result['cycle_count'] - 1):
            start_idx = result['peaks'][i]
            end_idx = result['peaks'][i + 1]
            plt.axvspan(start_idx, end_idx, alpha=0.2, color=colors[i], 
                       label=f'周期 {i+1}' if i == 0 else "")
        
        plt.title(f'检测到的完整周期数: {result["cycle_count"]}')
        plt.xlabel('索引')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 使用示例
def generate_test_sequence(length=400, cycles=8, noise_level=0.2):
    """生成测试序列"""
    x = np.linspace(0, cycles * 2 * np.pi, length)
    # 基础正弦波
    signal = np.sin(x)
    # 添加二次谐波
    signal += 0.3 * np.sin(2 * x + 0.5)
    # 添加噪声
    noise = np.random.normal(0, noise_level, length)
    return signal + noise

if __name__ == "__main__":
    # 生成测试数据
    test_sequence = generate_test_sequence(400, cycles=8, noise_level=0.3)
    
    # 创建分析器
    analyzer = PeriodicOscillationAnalyzer(max_harmonics=5)
    
    # 分析序列
    result = analyzer.analyze_sequence(test_sequence, plot_results=True)
    
    print(f"检测到的完整周期数: {result['cycle_count']}")
    print(f"峰值数量: {result['peak_count']}")
    print(f"谷值数量: {result['valley_count']}")