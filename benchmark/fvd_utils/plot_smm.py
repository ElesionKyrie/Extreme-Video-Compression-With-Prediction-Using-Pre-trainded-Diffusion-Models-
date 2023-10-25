import numpy as np
import matplotlib
# %matplotlib inline
from matplotlib import pyplot as plt
import scipy.spatial as spt
import pandas as pd
import seaborn as sns;sns.set()
from scipy.interpolate import interp1d
from itertools import chain

k = 3


def plot_metric_vs_bpp(metric_name, bpp_data_264, metric_data_264, bpp_data_265, metric_data_265,bpp_data_our, metric_data_our):
    # 初始化数据列表
    rewards_bpp_264 = []
    rewards_metric_264 = []

    # 根据数据范围和编解码器标签提取数据
    for x, y_values in zip(bpp_data_264[0:20], metric_data_264[0:20]):
        rewards_bpp_264.append([x] * len(y_values))
        rewards_metric_264.append(y_values)

    rewards_bpp_264 = list(chain.from_iterable(rewards_bpp_264))
    rewards_metric_264 = list(chain.from_iterable(rewards_metric_264))

    for i in range(len(rewards_metric_264)):
        rewards_metric_264[i] = rewards_metric_264[i].item()

    rewards_bpp_265 = []
    rewards_metric_265 = []

    # 根据数据范围和编解码器标签提取数据
    for x, y_values in zip(bpp_data_265[0:20], metric_data_265[0:20]):
        rewards_bpp_265.append([x] * len(y_values))
        rewards_metric_265.append(y_values)

    rewards_bpp_265 = list(chain.from_iterable(rewards_bpp_265))
    rewards_metric_265 = list(chain.from_iterable(rewards_metric_265))

    for i in range(len(rewards_metric_265)):
        rewards_metric_265[i] = rewards_metric_265[i].item()

    rewards_bpp_our = []
    rewards_metric_our = []

    # 根据数据范围和编解码器标签提取数据
    for x, y_values in zip(bpp_data_our[0:20], metric_data_our[0:20]):
        rewards_bpp_our.append([x] * len(y_values))
        rewards_metric_our.append(y_values)

    rewards_bpp_our = list(chain.from_iterable(rewards_bpp_our))
    rewards_metric_our = list(chain.from_iterable(rewards_metric_our))

    for i in range(len(rewards_metric_our)):
        rewards_metric_our[i] = rewards_metric_our[i].item()

    df_264 = pd.DataFrame(data={'BPP': rewards_bpp_264, f"{metric_name}": rewards_metric_264})
    df_265 = pd.DataFrame(data={'BPP': rewards_bpp_265, f"{metric_name}": rewards_metric_265})
    df_our = pd.DataFrame(data={'BPP': rewards_bpp_our, f"{metric_name}": rewards_metric_our})
    plt.figure()
    sns.lineplot(x="BPP", y=f"{metric_name}", data=df_264, label='H.264')
    sns.lineplot(x="BPP", y=f"{metric_name}", data=df_265, label='H.265')
    sns.lineplot(x="BPP", y=f"{metric_name}", data=df_our, label='Ours')
    plt.legend()
    plt.show()







def process_metrics_data(all_metrics, all_bpps):
    bppall_list_metrics = []
    metricsall_list = []
    step = 0.02
    for i, metrics_curve in enumerate(all_metrics):
        bpps_curve_metrics = all_bpps[i]

        if len(bpps_curve_metrics) == 0:
            print(f"Skipping curve {i} due to empty bpps_curve")
            continue

        interp_curve = interp1d(bpps_curve_metrics, metrics_curve, kind='linear')

        bpp_min_metrics = np.min(all_bpps[i])
        bpp_max_metrics = np.max(all_bpps[i])

        common_bpps_metrics = np.arange(np.ceil(bpp_min_metrics / step) * step, bpp_max_metrics, step)

        metrics_temp = interp_curve(common_bpps_metrics)
        bppall_list_metrics.append(common_bpps_metrics)
        metricsall_list.append(metrics_temp)

    for i in range(len(bppall_list_metrics)):
        for j in range(len(bppall_list_metrics[i])):
            bppall_list_metrics[i][j] = float("%.2f" % bppall_list_metrics[i][j])

    positions_dict_metrics = {}
    for i in range(1, int(1.0 / step)):
    # for i in range(1, 25):
        i_float = float("%.2f" % (i * step))
        i_positions = []
        for j in range(len(bppall_list_metrics)):
            if i_float in bppall_list_metrics[j]:
                i = find_positions(bppall_list_metrics[j], i_float)
                i_positions.append((j, i))
        if i_float in positions_dict_metrics:
            positions_dict_metrics[i_float].extend(i_positions)
        else:
            positions_dict_metrics[i_float] = i_positions

    #     print(positions_dict_metrics)

    avg_metrics_dict = {}

    bpp_image_metrics = []
    metrics_image = []
    for i_float, positions in positions_dict_metrics.items():
        metrics_values = []
        for position in positions:
            j, i = position
            metrics_values.append(metricsall_list[j][i])

        bpp_image_metrics.append(i_float)
        metrics_image.append(metrics_values)
        #         print(metrics_values)
        if metrics_values:
            avg_metrics = np.mean(metrics_values)
        else:
            avg_metrics = np.nan
        avg_metrics_dict[i_float] = avg_metrics

    i_float_values = list(avg_metrics_dict.keys())
    avg_metrics_values = list(avg_metrics_dict.values())

    return i_float_values, avg_metrics_values, bpp_image_metrics, metrics_image


def find_positions(lst, target_element):
    positions = []
    for i in range(len(lst)):
        if lst[i] == target_element:
            positions.append(i)
    return positions





# 原始数据
id_list = list(range(k))

all_h264_bpps_smm, all_h264_psnr_smm, all_h264_lpips_smm, all_h264_fvd_smm = [], [], [], []
all_h265_bpps_smm, all_h265_psnr_smm, all_h265_lpips_smm, all_h265_fvd_smm = [], [], [], []


for databatchidx in id_list:
    smm_bm_h264 = np.load(f"/home/myworkstation/OUT/output_smm_823/bench_264_{k}.npy")[databatchidx][:, 1:]
    # print(databatchidx)
    # print(smm_bm_h264.shape)
    smm_bm_h265 = np.load(f"/home/myworkstation/OUT/output_smm_823/bench_265_{k}.npy")[databatchidx]

    h264_bpp_smm = smm_bm_h264[3]
    h264_psnr_smm = smm_bm_h264[0]
    h264_lpips_smm = smm_bm_h264[1]
    h264_fvd_smm = smm_bm_h264[2]

    valid_indices_smm = np.where((h264_bpp_smm >= 0) & (h264_bpp_smm <= 1.2))[0]
    h264_bpp_valid_smm = h264_bpp_smm[valid_indices_smm]
    h264_psnr_valid_smm = h264_psnr_smm[valid_indices_smm]
    h264_lpips_valid_smm = h264_lpips_smm[valid_indices_smm]
    h264_fvd_valid_smm = h264_fvd_smm[valid_indices_smm]

    ###############################################################
    h265_bpp_smm = smm_bm_h265[3]
    h265_psnr_smm = smm_bm_h265[0]
    h265_lpips_smm = smm_bm_h265[1]
    h265_fvd_smm = smm_bm_h265[2]

    valid_indices_smm = np.where((h265_bpp_smm >= 0) & (h265_bpp_smm <= 1.2))[0]
    h265_bpp_valid_smm = h265_bpp_smm[valid_indices_smm]
    h265_psnr_valid_smm = h265_psnr_smm[valid_indices_smm]
    h265_lpips_valid_smm = h265_lpips_smm[valid_indices_smm]
    h265_fvd_valid_smm = h265_fvd_smm[valid_indices_smm]


    h264_bpps_smm = h264_bpp_valid_smm
    h264_psnr_smm = h264_psnr_valid_smm
    h264_lpips_smm = h264_lpips_valid_smm
    h264_fvd_smm = h264_fvd_valid_smm

    h265_bpps_smm = h265_bpp_valid_smm
    h265_psnr_smm = h265_psnr_valid_smm
    h265_lpips_smm = h265_lpips_valid_smm
    h265_fvd_smm = h265_fvd_valid_smm

    all_h264_bpps_smm.append(h264_bpps_smm)
    all_h264_psnr_smm.append(h264_psnr_smm)
    all_h264_lpips_smm.append(h264_lpips_smm)
    all_h264_fvd_smm.append(h264_fvd_smm)

    all_h265_bpps_smm.append(h265_bpps_smm)
    all_h265_psnr_smm.append(h265_psnr_smm)
    all_h265_lpips_smm.append(h265_lpips_smm)
    all_h265_fvd_smm.append(h265_fvd_smm)





# # 处理 h264 数据
i_float_values_h264_fvd_smm, avg_fvd_values_h264_smm, bpp_image_fvd_264_smm, fvd_image_264_smm  = process_metrics_data(all_h264_fvd_smm, all_h264_bpps_smm)
# print(all_h264_bpps)

# # # 处理 h265 数据
i_float_values_h265_fvd_smm, avg_fvd_values_h265_smm, bpp_image_fvd_265_smm, fvd_image_265_smm = process_metrics_data(all_h265_fvd_smm, all_h265_bpps_smm)


# print(avg_fvd_values_h265_smm)
# 绘制图像
plt.plot(i_float_values_h264_fvd_smm, avg_fvd_values_h264_smm, label='h264')
plt.plot(i_float_values_h265_fvd_smm, avg_fvd_values_h265_smm, label='h265')
plt.xlabel('BPPs')
plt.ylabel('Average FVD')
plt.title('Average FVD vs. BPPs')
plt.legend()
plt.show()



plot_metric_vs_bpp('FVD', bpp_image_fvd_264_smm, fvd_image_264_smm,bpp_image_fvd_265_smm,fvd_image_265_smm , )


# 处理 h264 数据
i_float_values_h264_psnr_smm, avg_psnr_values_h264_smm, bpp_image_psnr_264_smm, psnr_image_264_smm  = process_metrics_data(all_h264_psnr_smm, all_h264_bpps_smm)


# 处理 h265 数据
i_float_values_h265_psnr_smm, avg_psnr_values_h265_smm, bpp_image_psnr_265_smm, psnr_image_265_smm = process_metrics_data(all_h265_psnr_smm, all_h265_bpps_smm)


# 绘制图像
plt.plot(i_float_values_h264_psnr_smm, avg_psnr_values_h264_smm, label='h264')
plt.plot(i_float_values_h265_psnr_smm, avg_psnr_values_h265_smm, label='h265')
plt.xlabel('BPPs')
plt.ylabel('Average PSNR')
plt.title('Average PSNR vs. BPPs')
plt.legend()
plt.show()

plot_metric_vs_bpp('PSNR', bpp_image_psnr_264_smm, psnr_image_264_smm,bpp_image_psnr_265_smm,psnr_image_265_smm )


# 处理 h264 数据
i_float_values_h264_lpips_smm, avg_lpips_values_h264_smm, bpp_image_lpips_264_smm, lpips_image_264_smm   = process_metrics_data(all_h264_lpips_smm, all_h264_bpps_smm)

# 处理 h265 数据
i_float_values_h265_lpips_smm, avg_lpips_values_h265_smm, bpp_image_lpips_265_smm, lpips_image_265_smm   = process_metrics_data(all_h265_lpips_smm, all_h265_bpps_smm)
#

plt.plot(i_float_values_h264_lpips_smm, avg_lpips_values_h264_smm, label='h264')
plt.plot(i_float_values_h265_lpips_smm, avg_lpips_values_h265_smm, label='h265')
plt.xlabel('BPPs')
plt.ylabel('Average LPIPS')
plt.title('Average LPIPS vs. BPPs')
plt.legend()
plt.show()

plot_metric_vs_bpp('LPIPS', bpp_image_lpips_264_smm, lpips_image_264_smm,bpp_image_lpips_265_smm,lpips_image_265_smm )



# rewards_bpp_smm_lpips_our, rewards_metrics_smm_lpips_our = [], []
# rewards_bpp_smm_lpips_264, rewards_metrics_smm_lpips_264 = [], []
# rewards_bpp_smm_lpips_265, rewards_metrics_smm_lpips_265 = [], []
#
# # for x, y_values in zip(bpp_image_lpips_our_smm[0:10], fvd_image_264_smm[0:10]):
# #     rewards_bpp_smm_lpips_our.append([x] * len(y_values))
# #     rewards_metrics_smm_lpips_our.append(y_values)
#
# # 处理 rewards9_fvd_264 和 rewards10_fvd_264 数据
# for x, y_values in zip(bpp_image_lpips_264_smm[0:20], lpips_image_264_smm[0:20]):
#     rewards_bpp_smm_lpips_264.append([x] * len(y_values))
#     rewards_metrics_smm_lpips_264.append(y_values)
#
# # 处理 rewards9_fvd_265 和 rewards8_fvd_265 数据
# for x, y_values in zip(bpp_image_lpips_265_smm[0:20], lpips_image_265_smm[0:20]):
#     rewards_bpp_smm_lpips_265.append([x] * len(y_values))
#     rewards_metrics_smm_lpips_265.append(y_values)
#
# from itertools import chain
#
#
# # rewards_bpp_smm_lpips_our = list(chain.from_iterable(rewards_bpp_smm_lpips_our))
# # rewards_metrics_smm_lpips_our = list(chain.from_iterable(rewards_metrics_smm_lpips_our))
# #
# # for i in range(len(rewards_metrics_smm_lpips_our)):
# #     rewards_metrics_smm_lpips_our[i] = rewards_metrics_smm_lpips_our[i].item()
#
#
# rewards_bpp_smm_lpips_264 = list(chain.from_iterable(rewards_bpp_smm_lpips_264))
# rewards_metrics_smm_lpips_264 = list(chain.from_iterable(rewards_metrics_smm_lpips_264))
#
# for i in range(len(rewards_metrics_smm_lpips_264)):
#     rewards_metrics_smm_lpips_264[i] = rewards_metrics_smm_lpips_264[i].item()
#
#
# rewards_bpp_smm_lpips_265 = list(chain.from_iterable(rewards_bpp_smm_lpips_265))
# rewards_metrics_smm_lpips_265 = list(chain.from_iterable(rewards_metrics_smm_lpips_265))
#
# for i in range(len(rewards_metrics_smm_lpips_265)):
#     rewards_metrics_smm_lpips_265[i] = rewards_metrics_smm_lpips_265[i].item()
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 创建数据框
# # df1 = pd.DataFrame(data={'BPP': rewards_bpp_smm_lpips_our, "LPIPS": rewards_metrics_smm_lpips_our})
# df2 = pd.DataFrame(data={'BPP': rewards_bpp_smm_lpips_264, "LPIPS": rewards_metrics_smm_lpips_264})
# df3 = pd.DataFrame(data={'BPP': rewards_bpp_smm_lpips_265, "LPIPS": rewards_metrics_smm_lpips_265})
#
# # 绘制图形
# plt.figure()
# # sns.lineplot(x="BPP", y="LPIPS", data=df1, label='OURs')
# sns.lineplot(x="BPP", y="LPIPS", data=df2, label='H.264')
# sns.lineplot(x="BPP", y="LPIPS", data=df3, label='H.265')
# plt.legend()
# plt.show()
#







# k = 64时在打开
# array_265_smm = np.vstack((fvd_list_265, psnr_list_265, lpips_list_265,i_float_values_h265_lpips_smm))
#
# array_264_smm = np.vstack((avg_fvd_values_h264_smm, avg_psnr_values_h264_smm, avg_lpips_values_h264_smm,i_float_values_h264_lpips_smm))
# output_file_path_265 = f"/home/myworkstation/OUT/output_smm_final/array_265_smm.npy"
# np.save(output_file_path_265, array_265_smm)
# output_file_path_264 = f"/home/myworkstation/OUT/output_smm_final/array_264_smm.npy"
# np.save(output_file_path_264, array_264_smm)





