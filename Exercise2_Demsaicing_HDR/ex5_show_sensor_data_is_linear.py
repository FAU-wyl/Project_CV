import os
import rawpy
import numpy as np
import matplotlib.pyplot as plt

"""
    任务要求:
    对每张图计算整张图所有像素的平均值（注意：不能按 R/G/B 通道分开算，要把所有 raw 数据一起平均）
    以曝光时间为 X 轴，平均像素值为 Y 轴作图
    验证得到的是一条直线
"""

def main():
    base = os.path.dirname(__file__)
    data_folder = os.path.join(base, 'exercise_2_data', '05')

    # Image files and corresponding exposure times (in seconds)
    images = [
        ('IMG_3044.CR3', 1/10),    # 1/10 seconds
        ('IMG_3045.CR3', 1/20),    # 1/20 seconds
        ('IMG_3046.CR3', 1/40),    # 1/40 seconds
        ('IMG_3047.CR3', 1/80),    # 1/80 seconds
        ('IMG_3048.CR3', 1/160),   # 1/160 seconds
        ('IMG_3049.CR3', 1/320),   # 1/320 seconds
    ]

    exposure_times = []
    average_values = []

    print("Processing images...")
    for filename, exp_time in images:
        filepath = os.path.join(data_folder, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            continue

        # Load raw data
        raw = rawpy.imread(filepath)
        raw_data = raw.raw_image_visible.astype(np.float32)

        # Compute mean of the entire raw data (not channel-wise)
        mean_val = raw_data.mean()

        exposure_times.append(exp_time)
        average_values.append(mean_val)

        print(f"  {filename}: exposure={exp_time:.6f}s, avg_raw={mean_val:.2f}")

    # Plot
    exposure_times = np.array(exposure_times)
    average_values = np.array(average_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(exposure_times, average_values, s=100, color='blue', label='Measured data')

    # Fit a line to verify linearity
    z = np.polyfit(exposure_times, average_values, 1)
    p = np.poly1d(z)
    fit_line = p(exposure_times)
    ax.plot(exposure_times, fit_line, 'r--', linewidth=2, label=f'Linear fit: y={z[0]:.1f}x+{z[1]:.1f}')

    # Calculate R^2 to show goodness of fit
    residuals = average_values - fit_line
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((average_values - average_values.mean())**2)
    r_squared = 1 - (ss_res / ss_tot)

    ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
    ax.set_ylabel('Average Raw Data Value', fontsize=12)
    ax.set_title('Sensor Linearity: Average Raw Value vs Exposure Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add R^2 to plot
    textstr = f'$R^2 = {r_squared:.4f}$'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    out_path = os.path.join(base, 'linearity_plot.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")

    # Display plot
    plt.show()

    print("\nLinearity Analysis:")
    print(f"  Slope: {z[0]:.2f} (raw value per second)")
    print(f"  Intercept: {z[1]:.2f}")
    print(f"  R² (goodness of fit): {r_squared:.4f}")
    print(f"  Linear relationship confirmed: {r_squared > 0.99}")

if __name__ == "__main__":
    main()

