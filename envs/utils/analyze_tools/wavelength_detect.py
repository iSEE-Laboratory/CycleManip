import numpy as np
from .peak_detect import peak_detect
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import matplotlib.patches as patches
from matplotlib.path import Path

def wavelength_detect(signal, plot_title="Wavelength Detection", 
                      median_rel_factor=4.0, slope_drop_factor=0.3, 
                      min_peak_distance=10, smoothing_window=18,
                      save_plot=True, save_path=None
                      ):
    """
    基于10月24日Copilot优化版本的高级波长检测函数
    
    核心特性：
    1. 使用guard机制保护峰值附近区域
    2. 以峰两侧最近波谷作为端点（若某侧无波谷则退化为基于guard的回退端点）
    3. 动态阈值与梯度可保留用于后续扩展或可视化
    4. 学术级别的可视化输出
    
    参数:
    - signal: 输入信号数组
    - plot_title: 图表标题
    - median_rel_factor: 中位数相关因子 (默认4.0)
    - slope_drop_factor: 斜率下降因子 (默认0.3)
    - min_peak_distance: 峰值最小间距
    - smoothing_window: 平滑窗口大小
    
    返回:
    - waves_info: 波长信息列表
    """
    
    # 信号预处理
    signal = np.array(signal)
    n = len(signal)
    
    # 应用Savitzky-Golay滤波进行平滑
    if len(signal) > smoothing_window:
        signal_smooth = savgol_filter(signal, smoothing_window, 3)
    else:
        signal_smooth = signal.copy()
    
    # 寻找峰值
    _, peaks = peak_detect(
                signal_smooth,
                smooth=False,
                save_plot=False,
            )
    peaks = np.array(peaks)
    
    if len(peaks) == 0:
        print("未检测到峰值")
        return []
    
    # 寻找波谷 (局部最小值)
    _, minima = peak_detect(
                -signal_smooth,
                smooth=False,
                save_plot=False,
            )
    minima = np.array(minima)
    
    waves_info = []
    
    for peak in peaks:
        p = peak
        
        guard = max(1, int(round(0.035 * n)))
        
        # === 左端点：取峰左侧最近波谷；若不存在则回退为 guard 边界 ===
        if p == 0:
            left_min = 0
        else:
            left_minima = minima[minima < p] if len(minima) > 0 else np.array([])
            if left_minima.size > 0:
                left_min = int(left_minima.max())
            else:
                # 无左波谷时回退：不越过信号边界，可选 guard 边界
                left_min = max(0, p - guard)
        
        # === 右端点：取峰右侧最近波谷；若不存在则回退为 guard 边界 ===
        if p >= n - 1:
            right_min = n - 1
        else:
            right_minima = minima[minima > p] if len(minima) > 0 else np.array([])
            if right_minima.size > 0:
                right_min = int(right_minima.min())
            else:
                # 无右波谷时回退：不越过信号边界，可选 guard 边界
                right_min = min(n - 1, p + guard)

        # 基本一致性保障：确保 left_min < p < right_min（若不满足，使用 guard 回退）
        if left_min >= p:
            left_min = max(0, p - guard)
        if right_min <= p:
            right_min = min(n - 1, p + guard)
        
        # 计算波长
        wavelength = right_min - left_min
        
        waves_info.append({
            'peak': p,
            'left_min': left_min,
            'right_min': right_min,
            'wavelength': wavelength
        })
    
    # === 学术级可视化 ===
    if save_plot:
        _save_wavelength_plot(signal, signal_smooth, peaks, minima, waves_info, plot_title, save_path)
    
    return waves_info

def _save_wavelength_plot(signal, signal_smooth, peaks, minima, waves_info, plot_title, save_path):
    """学术会议论文级别的波长检测可视化"""
    
    # 使用学术风格
    preferred_styles = ('seaborn-darkgrid', 'seaborn-v0_8-darkgrid', 'seaborn', 'ggplot', 'default')
    for s in preferred_styles:
        if s in plt.style.available:
            plt.style.use(s)
            break
    else:
        plt.style.use('default')
    
    # 创建上下布局的子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1]})
    
    # === 上图：信号与检测结果 ===
    # 绘制原始和平滑信号
    ax1.plot(signal, label='原始信号', color='C0', alpha=0.45, linewidth=1)
    ax1.plot(signal_smooth, label='平滑信号', color='C1', linewidth=2)
    
    # 标记峰值和波谷
    if len(peaks) > 0:
        ax1.plot(peaks, signal_smooth[peaks], 'o', color='C3', markersize=6, label=f'峰值 ({len(peaks)})')
    if len(minima) > 0:
        ax1.plot(minima, signal_smooth[minima], 'v', color='C2', markersize=5, label=f'波谷 ({len(minima)})')
    
    # 为每个波标注背景区域和连接线
    for w in waves_info:
        lm = int(w['left_min'])
        rm = int(w['right_min'])
        pk = int(w['peak'])
        wl = int(w['wavelength'])
        
        # 背景区域
        ax1.axvspan(lm, rm, color='C4', alpha=0.08)
        
        # 美化连线：先画阴影，再画贝塞尔曲线
        ax1.plot([lm, rm],
                 [signal_smooth[lm], signal_smooth[rm]],
                 color='black', linewidth=3.0, alpha=0.12,
                 solid_capstyle='round', zorder=2)
        
        # 二次贝塞尔曲线连接两个端点
        amp = float(np.max(signal_smooth) - np.min(signal_smooth)) if np.ptp(signal_smooth) != 0 else 1.0
        ctrl_y = float(min(signal_smooth[pk], signal_smooth[lm], signal_smooth[rm]) - 0.06 * amp)
        verts = [(lm, float(signal_smooth[lm])), (pk, ctrl_y), (rm, float(signal_smooth[rm]))]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)
        patch = patches.PathPatch(path,
                                edgecolor='C3', facecolor='none',
                                lw=1.6, linestyle=(0, (6, 3)),
                                alpha=0.9, zorder=4, capstyle='round')
        ax1.add_patch(patch)
        
        # 在端点处添加菱形标记
        ax1.scatter([lm, rm],
                   [signal_smooth[lm], signal_smooth[rm]],
                   marker='D', s=36, color='C2', edgecolor='white', linewidth=0.8, zorder=5)
        
        # 波长注释
        ax1.annotate(f'{wl}', xy=(pk, signal_smooth[pk]), xytext=(0, 14),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, color='black',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.6))
    
    ax1.set_title(f'{plot_title} — 信号分析')
    ax1.set_xlabel('索引')
    ax1.set_ylabel('幅度')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_xlim(0, max(0, len(signal) - 1))
    
    # === 下图：波长分布柱状图 ===
    if waves_info:
        wavelengths = [w['wavelength'] for w in waves_info]
        bars = ax2.bar(range(1, len(wavelengths) + 1), wavelengths,
                       color='C3', alpha=0.7, edgecolor='darkred', linewidth=1.2)

        # 计算纵坐标下截断值：最小波长的 4/5
        min_wl = min(wavelengths)
        y_lower = 0.8 * min_wl
        y_upper = max(wavelengths) * 1.05  # 适度上留白
        if y_lower >= y_upper:  # 退化情况（所有相等）
            y_lower = 0.95 * min_wl
            y_upper = 1.05 * min_wl
        ax2.set_ylim(y_lower, y_upper)

        # 在每个柱子上标注数值
        for i, (bar, wl) in enumerate(zip(bars, wavelengths)):
            height = bar.get_height()
            ax2.annotate(f'{wl}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

        ax2.set_title('波长分布统计')
        ax2.set_xlabel('波序号')
        ax2.set_ylabel('波长')
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.set_xticks(range(1, len(wavelengths) + 1))

    plt.tight_layout()
    plt.savefig(save_path if save_path else 'wavelength_detection.png',
                dpi=300, bbox_inches='tight')
    plt.show()

# 测试函数
if __name__ == "__main__":
    # 生成测试信号
    x = np.linspace(0, 4*np.pi, 200)
    signal_test = np.sin(x) + 0.5*np.sin(3*x) + 0.3*np.random.randn(200)
    
    # 运行检测
    results = wavelength_detect(signal_test)
    
    print("检测结果:")
    for i, wave in enumerate(results):
        print(f"波 {i+1}: 峰值={wave['peak']}, 左端={wave['left_min']}, 右端={wave['right_min']}, 波长={wave['wavelength']}")