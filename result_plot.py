
import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns;
import os
sns.set()










def plot_line(x_values_new, y_values_new,x_values_264,y_values_264, x_values_265,y_values_265,x_label, y_label, title, save_path=None):

    plt.plot(x_values_new, y_values_new, label='Neural Network',color='red', marker='o', linestyle='-')
    plt.plot(x_values_264, y_values_264, label='H.264',color='blue', marker='o', linestyle='-')
    plt.plot(x_values_265, y_values_265, label='H.265',color='orange', marker='o', linestyle='-')


    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


    plt.legend()
    if save_path:
        plt.savefig(save_path)

    plt.close()



def plot(databatchidx, psnr_arr, lpips_arr, fvd_arr, output_path):


    all_h264_bpps, all_h264_psnr, all_h264_lpips, all_h264_fvd = [], [], [], []
    all_h265_bpps, all_h265_psnr, all_h265_lpips, all_h265_fvd = [], [], [], []


    city_bm_h264 = np.load("./bench_264_24.npy")[databatchidx]
    city_bm_h265 = np.load("./bench_265_24.npy")[databatchidx]


    h264_bpp = city_bm_h264[3]
    h264_psnr = city_bm_h264[0]
    h264_lpips = city_bm_h264[1]
    h264_fvd = city_bm_h264[2]
    valid_indices = np.where((h264_bpp >= 0) & (h264_bpp <= 1.2))[0]
    h264_bpp_valid = h264_bpp[valid_indices]
    h264_psnr_valid = h264_psnr[valid_indices]
    h264_lpips_valid = h264_lpips[valid_indices]
    h264_fvd_valid = h264_fvd[valid_indices]
    ###############################################################
    h265_bpp = city_bm_h265[3]
    h265_psnr = city_bm_h265[0]
    h265_lpips = city_bm_h265[1]
    h265_fvd = city_bm_h265[2]

    valid_indices = np.where((h265_bpp >= 0) & (h265_bpp <= 1.2))[0]
    h265_bpp_valid = h265_bpp[valid_indices]
    h265_psnr_valid = h265_psnr[valid_indices]
    h265_lpips_valid = h265_lpips[valid_indices]
    h265_fvd_valid = h265_fvd[valid_indices]

    ###############################################################

    h264_bpps = h264_bpp_valid
    h264_psnr = h264_psnr_valid
    h264_lpips = h264_lpips_valid
    h264_fvd = h264_fvd_valid

    h265_bpps = h265_bpp_valid
    h265_psnr = h265_psnr_valid
    h265_lpips = h265_lpips_valid
    h265_fvd = h265_fvd_valid
    ###############################################################
    all_h264_bpps.append(h264_bpps)
    all_h264_psnr.append(h264_psnr)
    all_h264_lpips.append(h264_lpips)
    all_h264_fvd.append(h264_fvd)

    all_h265_bpps.append(h265_bpps)
    all_h265_psnr.append(h265_psnr)
    all_h265_lpips.append(h265_lpips)
    all_h265_fvd.append(h265_fvd)
    ###############################################################





    plot_line(psnr_arr[0,:],psnr_arr[1,:],all_h264_bpps[0],all_h264_psnr[0],all_h265_bpps[0],all_h265_psnr[0], 'BPP', 'PSNR', f'BPP_PSNR_idx{databatchidx}',os.path.join(output_path, f"BPP_PSNR_idx{databatchidx}.png"))
    print("*"*70)
    print("绘制psnr")
    print(psnr_arr[0,:])
    print(psnr_arr[1,:])
    print(all_h264_bpps[0])
    print(all_h264_psnr[0])
    print("*"*70)

    plot_line( lpips_arr[0,:],lpips_arr[1,:],all_h264_bpps[0],all_h264_lpips[0],all_h265_bpps[0],all_h265_lpips[0], 'BPP', 'LPIPS', f'BPP_LPIPS_idx{databatchidx}',os.path.join(output_path, f"BPP_LPIPS_idx{databatchidx}.png"))
    print("*"*70)
    print("绘制lpips")
    print(lpips_arr[0,:])
    print(lpips_arr[1,:])
    print(all_h264_bpps[0])
    print(all_h264_lpips[0])
    print("*"*70)
    plot_line(fvd_arr[0,:],fvd_arr[1,:], all_h264_bpps[0], all_h264_fvd[0], all_h265_bpps[0],all_h265_fvd[0], 'BPP', 'FVD', f'BPP_FVD_idx{databatchidx}', os.path.join(output_path, f"BPP_FVD_idx{databatchidx}.png"))
    print("*" * 70)
    print("绘制fvd")
    print(fvd_arr[0, :])
    print(fvd_arr[1, :])
    print(all_h264_bpps[0])
    print(all_h264_fvd[0])
    print("*" * 70)





