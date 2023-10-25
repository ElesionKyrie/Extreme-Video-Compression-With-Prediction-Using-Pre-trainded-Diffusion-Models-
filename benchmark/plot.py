import re
import matplotlib.pyplot as plt
import numpy as np
#
#
# def read_data_from_file(file_path):
#     psnr_values = []
#     lpips_values = []
#     bpp_values = []
#
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             psnr_match = re.search(r'PSNR: ([\d.]+),', line)
#             lpips_match = re.search(r'LPIPS: ([\d.]+),', line)
#             bpp_match = re.search(r'BPP: ([\d.]+)', line)
#
#             if psnr_match and lpips_match and bpp_match:
#                 psnr_values.append(float(psnr_match.group(1)))
#                 lpips_values.append(float(lpips_match.group(1)))
#                 bpp_values.append(float(bpp_match.group(1)))
#
#     return psnr_values, lpips_values, bpp_values
#
#
# def plot_and_save(psnr_values, lpips_values, bpp_values):
#     # Plot PSNR vs BPP
#     plt.figure(figsize=(8, 6))
#     plt.plot(bpp_values, psnr_values, 'o-', label='PSNR')
#     plt.xlabel('BPP')
#     plt.ylabel('PSNR')
#     plt.title('PSNR vs BPP')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('psnr_plot.png')
#     plt.close()
#
#     # Plot LPIPS vs BPP
#     plt.figure(figsize=(8, 6))
#     plt.plot(bpp_values, lpips_values, 'o-', label='LPIPS')
#     plt.xlabel('BPP')
#     plt.ylabel('LPIPS')
#     plt.title('LPIPS vs BPP')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('lpips_plot.png')
#     plt.close()
#
#
# if __name__ == "__main__":
#     file_path = '/home/myworkstation/OUT/output_smm/out_frames_0_265/psnr_lpips_bpp.txt'
#     psnr_values, lpips_values, bpp_values = read_data_from_file(file_path)
#     plot_and_save(psnr_values, lpips_values, bpp_values)
import os

# 文件夹路径
folder_path = "/home/myworkstation/OUT/output_smm_823"

# 文件夹个数
num_folders = 3

# 指标个数
num_metrics = 4

# 每个txt文件的行数
num_lines = 52

# 创建一个空的三维数组
data_array = np.zeros((num_folders, num_metrics, num_lines))

# 遍历每个文件夹
for folder_idx in range(num_folders):
    txt_file_path = os.path.join(folder_path, f"out_frames_{folder_idx}_265", "psnr_lpips_fvd_bpp.txt")

    # 打开txt文件并逐行读取数据
    with open(txt_file_path, "r") as file:
        lines = file.readlines()
        for line_idx, line in enumerate(lines):
            parts = line.strip().split(", ")
            for metric_idx, part in enumerate(parts):
                metric_value = float(part.split(": ")[1])
                data_array[folder_idx, metric_idx, line_idx] = metric_value

# 打印数组的形状
# print(data_array.shape)
# print(data_array[0][2])
output_file_path = f"/home/myworkstation/OUT/output_smm_823/bench_265_{num_folders}.npy"
np.save(output_file_path, data_array)

