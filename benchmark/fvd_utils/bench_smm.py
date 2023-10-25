
from fvd_utils.my_utils import calculate_fvd
import cv2
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import os
import subprocess
import numpy as np
from PIL import Image
import torch
import os
import numpy as np
from eval_models import PerceptualLoss
import re
import matplotlib.pyplot as plt



def read_data_from_file(file_path):
    psnr_values = []
    lpips_values = []
    fvd_values = []
    bpp_values = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            psnr_match = re.search(r'PSNR: ([\d.]+),', line)
            lpips_match = re.search(r'LPIPS: ([\d.]+),', line)
            fvd_match = re.search(r'FVD: ([\d.]+),', line)
            bpp_match = re.search(r'BPP: ([\d.]+)', line)

            if psnr_match and lpips_match and fvd_match and bpp_match:
                psnr_values.append(float(psnr_match.group(1)))
                lpips_values.append(float(lpips_match.group(1)))
                fvd_values.append(float(fvd_match.group(1)))
                bpp_values.append(float(bpp_match.group(1)))

    return psnr_values, lpips_values, fvd_values, bpp_values




def plot_and_save(psnr_values, lpips_values, fvd_values,bpp_values , filepath):
    # Plot PSNR vs BPP
    plt.figure(figsize=(8, 6))
    plt.plot(bpp_values, psnr_values, 'o-', label='PSNR')
    plt.xlabel('BPP')
    plt.ylabel('PSNR')
    plt.title('PSNR vs BPP')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{filepath}/psnr_plot.png')
    plt.close()

    # Plot LPIPS vs BPP
    plt.figure(figsize=(8, 6))
    plt.plot(bpp_values, lpips_values, 'o-', label='LPIPS')
    plt.xlabel('BPP')
    plt.ylabel('LPIPS')
    plt.title('LPIPS vs BPP')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{filepath}/lpips_plot.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(bpp_values, fvd_values, 'o-', label='FVD')
    plt.xlabel('BPP')
    plt.ylabel('FVD')
    plt.title('FVD vs BPP')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{filepath}/fvd_plot.png')
    plt.close()


def check_normalize(array):
    # 获取数组的最小值和最大值
    min_val = np.min(array)
    max_val = np.max(array)

    # 检查最小值和最大值是否在0到255的范围内
    if min_val >= 0 and max_val <= 255:
        return "未归一化"
    elif min_val >= 0 and max_val <= 1:
        return "已归一化"
    else:
        return "未知范围"




def get_filesize(filepath: Union[Path, str]) -> int:
    return Path(filepath).stat().st_size


def calculate_psnr(original_images, compressed_images):
    orig_img = cv2.imread(original_images, cv2.IMREAD_GRAYSCALE)
    comp_img = cv2.imread(compressed_images, cv2.IMREAD_GRAYSCALE)
    mse = np.mean((orig_img - comp_img) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))

    return psnr

org_array  = np.load("/home/myworkstation/OUT/0x_gt.npy")


# print(org_array.shape)
# print(check_normalize(org_array))
for j in range(64):
    input_folder_video = f"/home/myworkstation/OUT/rawvideo_smm_gray/rawvideo_{j}"
    input_folder_image = f"/home/myworkstation/OUT/frame_smm/video_{j}"
    output_folder = f"/home/myworkstation/OUT/output_smm_823/out_frames_{j}_265"

    # org_fvd = (org_array[j, ...] / 255)
    # org_tensor = torch.tensor(org_fvd).unsqueeze(dim=0).repeat(2, 1, 1, 1, 1).to(torch.float)

    qp_range = range(52)
    psnrs = []
    lpips_mean = []
    bpp = []
    fvd = []
    for qp in qp_range:
        compressed_file = os.path.join(output_folder, f"compressed_qp{qp}.hevc")
        decompressed_folder = os.path.join(output_folder, f"decompressed_qp{qp}")
        os.makedirs(decompressed_folder, exist_ok=True)

        command = f"ffmpeg -f rawvideo -pix_fmt gray -s 64x64 -r 30 -i {input_folder_video}/video.yuv -c:v libx265 -preset veryfast -crf {qp} -tune zerolatency {compressed_file} -y"

        subprocess.run(command, shell=True)

        # decompression
        command = f"ffmpeg -i {compressed_file} -pix_fmt gray {decompressed_folder}/reconstructed%d.png"
        subprocess.run(command, shell=True)

        decompressed_frames = []
        org_frames = []
        for i in range(30):
            decompressed_frame_path = os.path.join(decompressed_folder, f"reconstructed{i + 1}.png")
            org_frame_path = os.path.join(input_folder_image, f"frame_{i:02d}.png")
            decompressed_frame = np.array(Image.open(decompressed_frame_path))
            org_frame = np.array(Image.open(org_frame_path))
            decompressed_frames.append(decompressed_frame)
            org_frames.append(org_frame)
        dec_array = np.stack(decompressed_frames)
        org_array = np.stack(org_frames)
        dec_fvd = (dec_array / 255)
        org_fvd = (org_array / 255)
        dec_tensor = torch.tensor(dec_fvd).unsqueeze(0).unsqueeze(2).repeat(2, 1, 1, 1, 1).to(torch.float)
        print(dec_tensor.shape)
        org_tensor = torch.tensor(org_fvd).unsqueeze(0).unsqueeze(2).repeat(2, 1, 1, 1, 1).to(torch.float)



        fvd.append(calculate_fvd(org_tensor, dec_tensor))

        psnr_qp = []
        lpips_qp = []
        # dec_array = np.empty((1, 30, 1, 64, 64))
        for i in range(30):
            original_file = os.path.join(input_folder_image, f"frame_{i:02d}.png")
            compressed_frame = os.path.join(decompressed_folder, f"reconstructed{i + 1}.png")
            # decoded_image = cv2.imread(compressed_frame, cv2.IMREAD_GRAYSCALE)

            # 将解码图像添加到数组中
            # dec_array[0, i, 0, :, :] = decoded_image

            psnr = calculate_psnr(original_file, compressed_frame)



            model_lpips = PerceptualLoss(model='net-lin', net='alex',
                                         device='cuda')
            # 计算两张灰度图像之间的 LPIPS
            tensor_org = (torch.Tensor(np.array(Image.open(original_file)))).to("cuda:0")

            tensor_dec = (torch.Tensor(np.array(Image.open(compressed_frame)))).to("cuda:0")


            lpips = model_lpips.forward(tensor_org, tensor_dec).cpu().detach().numpy()
            lpips_qp.append(lpips)

            psnr_qp.append(psnr)
        psnrs.append(np.mean(psnr_qp))
        lpips_mean.append(np.mean(lpips_qp))

        bpp.append(float((get_filesize(compressed_file) * 8 / (64 * 64 * 30))))
#purmute之后形状[1, 30, 1, 64, 64]



        with open(f"{output_folder}/psnr_lpips_fvd_bpp.txt", "w") as f:
            for psnr_val, lpips_val, fvd_val, bpp_val in zip(psnrs, lpips_mean, fvd, bpp):
                f.write(f"PSNR: {psnr_val}, LPIPS: {lpips_val}, FVD: {fvd_val}, BPP: {bpp_val}\n")

    file_path = f"{output_folder}/psnr_lpips_fvd_bpp.txt"
    psnr_values, lpips_values, fvd_values ,bpp_values = read_data_from_file(file_path)
    plot_and_save(psnr_values, lpips_values, fvd_values ,bpp_values,output_folder)







    #     print(f"Average PSNR for QP={qp}: {np.mean(psnr_qp):.2f}, bpp for QP={qp}: {bpp[-1]:.2f}")
    # print(bpp)
    # print(psnrs)
# # print(fvd)
# plt.plot(bpp, psnrs, marker='o')
# # plt.plot(bpp, psnr, marker='o')
# plt.title('BPP vs PSNR')
# plt.xlabel('BPP')
# plt.ylabel('PSNR')
# plt.show()
