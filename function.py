
from matplotlib import pyplot as plt
import scipy.spatial as spt
import seaborn as sns;
sns.set()
import cv2
import matplotlib
matplotlib.use('Agg')
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
import pandas as pd
import argparse
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from fvd_utils.calculate_fvd import *

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace








def save_output(gt, xge, q, thr, idx, output_dir):
    output = np.concatenate([gt, xge], axis=0)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, "city_output_npy_idx%d_q%d_thr%.2f.npy" % (idx, q, thr)), output)

    output = (output * 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_dir, "city_idx%d_q%d_thr%.2f.png" % (idx, q, thr)), output)



def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def plot_scatter(x_values, y_values, x_label, y_label, title):
    """
    绘制散点图

    Args:
    x_values (list): x 坐标值的列表
    y_values (list): y 坐标值的列表
    x_label (str): x 轴标签
    y_label (str): y 轴标签
    title (str): 图表标题

    Returns:
    None
    """
    # 创建散点图
    plt.scatter(x_values, y_values, label='Scatter Plot', color='blue', marker='o')

    # 添加标签和标题
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # 显示图例
    plt.legend()

    # 显示散点图
    plt.show()


def plot_line(x_values, y_values, x_label, y_label, title):
    """
    绘制线性图

    Args:
    x_values (list): x 坐标值的列表
    y_values (list): y 坐标值的列表
    x_label (str): x 轴标签
    y_label (str): y 轴标签
    title (str): 图表标题

    Returns:
    None
    """
    # 创建线性图
    plt.plot(x_values, y_values, label='Data Line', color='green', marker='o', linestyle='-')

    # 添加标签和标题
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # 显示图例
    plt.legend()


    plt.show()



def process_data_and_save(databatchidx,bpp_npy,psnr_npy,lpips_npy,fvd_npy,save_path):
    """
    新的部分
    """



    psnr_list = np.mean(psnr_npy, 1)#
    lpips_list = np.mean(lpips_npy, 1)


    all_bpps_psnr = []
    all_bpps_lpips = []
    all_psnr = []
    all_lpips = []


    points_psnr = np.stack([bpp_npy, psnr_list]).transpose(1, 0)
    hull = spt.ConvexHull(points=points_psnr)
    leftmost_point = np.argmin(points_psnr[hull.vertices, 0])
    highest_point = np.argmax(points_psnr[hull.vertices, 1])
    if highest_point > leftmost_point + 1:
        selected_indices = list(range(highest_point + 1, len(hull.vertices)))

    else:
        selected_indices = list(range(highest_point, leftmost_point + 1))
    selected_points = points_psnr[hull.vertices[selected_indices]]
    all_bpps_psnr.append(selected_points[:, 0])
    all_psnr.append(selected_points[:, 1])
    ###############################################################

    psnr_arr = np.vstack((all_bpps_psnr[0], all_psnr[0]))


    points_lpips = np.stack([bpp_npy, lpips_list]).transpose(1, 0)
    hull = spt.ConvexHull(points=points_lpips)
    lowest_point_lpips = np.argmin(points_lpips[hull.vertices, 1])

    leftest_point_lpips = np.argmin(points_lpips[hull.vertices, 0])

    if leftest_point_lpips >= lowest_point_lpips + 1:
        selected_indices_lpips = list(range(leftest_point_lpips + 1, len(hull.vertices)))
    else:
        selected_indices_lpips = list(range(leftest_point_lpips, lowest_point_lpips + 1))

    selected_points_lpips = points_lpips[hull.vertices[selected_indices_lpips]]
    all_bpps_lpips.append(selected_points_lpips[:, 0])
    all_lpips.append(selected_points_lpips[:, 1])

    lpips_arr = np.vstack((all_bpps_lpips[0], all_lpips[0]))

    all_bpps_fvd = []
    all_fvd = []
    points_fvd = np.stack([bpp_npy, fvd_npy]).transpose(1, 0)
    hull = spt.ConvexHull(points=points_fvd)
    lowest_point_fvd = np.argmin(points_fvd[hull.vertices, 1])
    leftest_point_fvd = np.argmin(points_fvd[hull.vertices, 0])

    if leftest_point_fvd > lowest_point_fvd + 1:
        selected_indices_fvd = list(range(leftest_point_fvd + 1, len(hull.vertices)))
        selected_indices_fvd.insert(0, leftest_point_fvd)
        selected_indices_fvd.append(lowest_point_fvd)

    else:

        selected_indices_fvd = list(range(leftest_point_fvd, lowest_point_fvd + 1))

    selected_points_fvd = points_fvd[hull.vertices[selected_indices_fvd]]

    all_bpps_fvd.append(selected_points_fvd[:, 0])
    all_fvd.append(selected_points_fvd[:, 1])

    fvd_arr = np.vstack((all_bpps_fvd[0], all_fvd[0]))



    np.save(os.path.join(save_path, f"psnr_{databatchidx}.npy"), psnr_arr)
    np.save(os.path.join(save_path, f"lpips_{databatchidx}.npy"), lpips_arr)
    np.save(os.path.join(save_path, f"fvd_{databatchidx}.npy"), fvd_arr)



    return psnr_arr, lpips_arr, fvd_arr