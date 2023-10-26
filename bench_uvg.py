import cv2
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from lpips_utils.eval_models import PerceptualLoss
import re
import os
import subprocess
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, Union
from functools import partial
import torch
import torch.nn.functional as F
from fvd_utils.my_utils import calculate_fvd
from torch import Tensor
import enum
import argparse
import time
from fractions import Fraction
from typing import Any, Dict, Sequence, Union

from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import numpy as np


import torch
import torch.nn.functional as F

from torch import Tensor
import sys
import os
# module_dir = os.path.abspath('/home/myworkstation/PycharmProjects/benchmark/fvd_utils')
# sys.path.append(module_dir)


YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}





class VideoFormat(enum.Enum):
    YUV400 = "yuv400"  # planar 4:0:0 YUV
    YUV420 = "yuv420"  # planar 4:2:0 YUV
    YUV422 = "yuv422"  # planar 4:2:2 YUV
    YUV444 = "yuv444"  # planar 4:4:4 YUV
    RGB = "rgb"  # planar 4:4:4 RGB


def get_num_frms(file_size, width, height, video_format, dtype):
    w_sub, h_sub = subsampling[video_format]
    itemsize = np.array([0], dtype=dtype).itemsize

    frame_size = (width * height) + 2 * (
        round(width / w_sub) * round(height / h_sub)
    ) * itemsize

    total_num_frms = file_size // frame_size

    return total_num_frms

video_formats = {
    "yuv400": VideoFormat.YUV400,
    "yuv420": VideoFormat.YUV420,
    "420": VideoFormat.YUV420,
    "p420": VideoFormat.YUV420,
    "i420": VideoFormat.YUV420,
    "yuv422": VideoFormat.YUV422,
    "p422": VideoFormat.YUV422,
    "i422": VideoFormat.YUV422,
    "y42B": VideoFormat.YUV422,
    "yuv444": VideoFormat.YUV444,
    "p444": VideoFormat.YUV444,
    "y444": VideoFormat.YUV444,
}


framerate_to_fraction = {
    "23.98": Fraction(24000, 1001),
    "23.976": Fraction(24000, 1001),
    "29.97": Fraction(30000, 1001),
    "59.94": Fraction(60000, 1001),
}

file_extensions = {
    "yuv",
    "rgb",
    "raw",
}


subsampling = {
    VideoFormat.YUV400: (0, 0),
    VideoFormat.YUV420: (2, 2),
    VideoFormat.YUV422: (2, 1),
    VideoFormat.YUV444: (1, 1),
}


bitdepth_to_dtype = {
    8: np.uint8,
    10: np.uint16,
    12: np.uint16,
    14: np.uint16,
    16: np.uint16,
}


def make_dtype(format, value_type, width, height):
    # Use float division with rounding to account for oddly sized Y planes
    # and even sized U and V planes to match ffmpeg.

    w_sub, h_sub = subsampling[format]
    if h_sub > 1:
        sub_height = (height + 1) // h_sub
    elif h_sub:
        sub_height = round(height / h_sub)
    else:
        sub_height = 0

    if w_sub > 1:
        sub_width = (width + 1) // w_sub if w_sub else 0
    elif w_sub:
        sub_width = round(width / w_sub)
    else:
        sub_width = 0

    return np.dtype(
        [
            ("y", value_type, (height, width)),
            ("u", value_type, (sub_height, sub_width)),
            ("v", value_type, (sub_height, sub_width)),
        ]
    )




class RawVideoSequence(Sequence[np.ndarray]):
    """
    Generalized encapsulation of raw video buffer data that can hold RGB or
    YCbCr with sub-sampling.

    Args:
        data: Single dimension array of the raw video data.
        width: Video width, if not given it may be deduced from the filename.
        height: Video height, if not given it may be deduced from the filename.
        bitdepth: Video bitdepth, if not given it may be deduced from the filename.
        format: Video format, if not given it may be deduced from the filename.
        framerate: Video framerate, if not given it may be deduced from the filename.
    """

    def __init__(
        self,
        mmap: np.memmap,
        width: int,
        height: int,
        bitdepth: int,
        format: VideoFormat,
        framerate: int,
    ):
        self.width = width
        self.height = height
        self.bitdepth = bitdepth
        self.framerate = framerate

        if isinstance(format, str):
            self.format = video_formats[format.lower()]
        else:
            self.format = format

        value_type = bitdepth_to_dtype[bitdepth]
        self.dtype = make_dtype(
            self.format, value_type=value_type, width=width, height=height
        )
        self.data = mmap.view(self.dtype)

        self.total_frms = get_num_frms(mmap.size, width, height, format, value_type)

    @classmethod
    def new_like(
        cls, sequence: "RawVideoSequence", filename: str
    ) -> "RawVideoSequence":
        mmap = np.memmap(filename, dtype=bitdepth_to_dtype[sequence.bitdepth], mode="r")
        return cls(
            mmap,
            width=sequence.width,
            height=sequence.height,
            bitdepth=sequence.bitdepth,
            format=sequence.format,
            framerate=sequence.framerate,
        )

    @classmethod
    def from_file(
        cls,
        filename: str,
        width: int = None,
        height: int = None,
        bitdepth: int = None,
        format: VideoFormat = None,
        framerate: int = None,
    ) -> "RawVideoSequence":
        """
        Loads a raw video file from the given filename.

        Args:
            filename: Name of file to load.
            width: Video width, if not given it may be deduced from the filename.
            height: Video height, if not given it may be deduced from the filename.
            bitdepth: Video bitdepth, if not given it may be deduced from the filename.
            format: Video format, if not given it may be deduced from the filename.

        Returns (RawVideoSequence):
            A RawVideoSequence instance wrapping the file on disk with a
            np memmap.
        """
        # info = get_raw_video_file_info(filename)

        bitdepth = 8
        format = VideoFormat.YUV420
        height = 128
        width = 128
        framerate = 30

        if width is None or height is None or bitdepth is None or format is None:
            raise RuntimeError(f"Could not get sequence information {filename}")

        mmap = np.memmap(filename, dtype=bitdepth_to_dtype[bitdepth], mode="r")

        return cls(
            mmap,
            width=width,
            height=height,
            bitdepth=bitdepth,
            format=format,
            framerate=framerate,
        )

    def __getitem__(self, index: Union[int, slice]) -> Any:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def close(self):
        del self.data


Frame = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]

def to_tensors(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    max_value: int = 1,
    device: str = "cpu",
) -> Frame:
    return tuple(
        torch.from_numpy(np.true_divide(c, max_value, dtype=np.float32)).to(device)
        for c in frame
    )


def parse_metrics_file(file_path):
    # 打开文件
    with open(file_path, 'r') as file:
        content = file.read()

    # 初始化空列表用于存放各个指标
    psnr_list = []
    lpips_list = []
    fvd_list = []
    bpp_list = []

    # 分割内容为每一行
    lines = content.split('\n')

    # 遍历每一行并解析指标
    for line in lines:
        if line:
            parts = line.split(',')
            psnr = float(parts[0].split(':')[1].strip())
            # print(psnr)
            lpips = float(parts[1].split(':')[1].strip())
            # print(lpips)
            fvd = float(parts[2].split(':')[1].strip())
            # print(fvd)
            bpp = float(parts[3].split(':')[1].strip())
            # print(bpp)

            # 将解析的指标添加到相应的列表中
            psnr_list.append(psnr)
            lpips_list.append(lpips)
            fvd_list.append(fvd)
            bpp_list.append(bpp)

    # 返回四个列表
    return psnr_list, lpips_list, fvd_list, bpp_list



def get_filesize(filepath: Union[Path, str]) -> int:
    return Path(filepath).stat().st_size
#
def calculate_psnr_color(original_image, compressed_image):
    # Load original and compressed images
    orig_img = cv2.imread(original_image)
    comp_img = cv2.imread(compressed_image)
    # print(orig_img.shape)
    # print(comp_img.shape)
    # Check if the images have the same shape
    if orig_img.shape != comp_img.shape:
        raise ValueError("Original and compressed images must have the same dimensions.")

    # Calculate the mean squared error (MSE) for each channel
    mse_r = np.mean((orig_img[:, :, 0] - comp_img[:, :, 0]) ** 2)
    mse_g = np.mean((orig_img[:, :, 1] - comp_img[:, :, 1]) ** 2)
    mse_b = np.mean((orig_img[:, :, 2] - comp_img[:, :, 2]) ** 2)

    # Calculate the average MSE across all channels
    mse_avg = (mse_r + mse_g + mse_b) / 3

    # Calculate PSNR using the average MSE
    psnr = 20 * np.log10(255 / np.sqrt(mse_avg))

    return psnr



def _check_input_tensor(tensor: Tensor) -> None:
    if (
        not isinstance(tensor, Tensor)
        or not tensor.is_floating_point()
        or not len(tensor.size()) in (3, 4)
        or not tensor.size(-3) == 3
    ):
        raise ValueError(
            "Expected a 3D or 4D tensor with shape (Nx3xHxW) or (3xHxW) as input"
        )


def yuv_420_to_444(
    yuv: Tuple[Tensor, Tensor, Tensor],
    mode: str = "bilinear",
    return_tuple: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Convert a 420 input to a 444 representation.

    Args:
        yuv (torch.Tensor, torch.Tensor, torch.Tensor): 420 input frames in
            (Nx1xHxW) format
        mode (str): algorithm used for upsampling: ``'bilinear'`` |
            | ``'bilinear'`` | ``'nearest'`` Default ``'bilinear'``
        return_tuple (bool): return input as tuple of tensors instead of a
            concatenated tensor, 3 (Nx1xHxW) tensors instead of one (Nx3xHxW)
            tensor (default: False)

    Returns:
        (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): Converted
            444
    """
    if len(yuv) != 3 or any(not isinstance(c, torch.Tensor) for c in yuv):
        raise ValueError("Expected a tuple of 3 torch tensors")

    if mode not in ("bilinear", "bicubic", "nearest"):
        raise ValueError(f'Invalid upsampling mode "{mode}".')

    kwargs = {}
    if mode != "nearest":
        kwargs = {"align_corners": False}

    def _upsample(tensor):
        return F.interpolate(tensor, scale_factor=2, mode=mode, **kwargs)

    y, u, v = yuv
    u, v = _upsample(u), _upsample(v)
    if return_tuple:
        return y, u, v
    return torch.cat((y, u, v), dim=1)

def ycbcr2rgb(ycbcr: Tensor) -> Tensor:
    """YCbCr to RGB conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    Args:
        ycbcr (torch.Tensor): 3D or 4D floating point RGB tensor

    Returns:
        rgb (torch.Tensor): converted tensor
    """
    _check_input_tensor(ycbcr)

    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = torch.cat((r, g, b), dim=-3)
    return rgb














# width = args.width
# height = args.width
# bitdepth = 8
# framerate = args.framerate

# 直接提供视频格式
video_format = VideoFormat.YUV420



# 定义视频压缩和性能计算函数
def compress_and_evaluate(input_folder, output_folder, codec, qp_range, pix_fmt, width_in, height_in, framerate_in ,project_str ,bitdepth = 8):
    os.makedirs(output_folder, exist_ok=True)

    psnrs = []
    lpips_mean = []
    bpp = []
    fvd = []
    for qp in qp_range:
        compressed_file = os.path.join(output_folder, f"compressed_qp{qp}_{codec}.mp4")
        decompressed_folder = os.path.join(output_folder, f"decompressed_qp{qp}")
        os.makedirs(decompressed_folder, exist_ok=True)

        command = f"ffmpeg -framerate {framerate_in} -video_size {width_in}x{height_in} -i {input_folder} -c:v {codec} -preset veryfast -s 128x128 -crf {qp} -pix_fmt {pix_fmt} -tune zerolatency {compressed_file} -y"
        subprocess.run(command, shell=True)

        output_file = os.path.join(decompressed_folder, "reconstructed.yuv")
        command = f"ffmpeg -i {compressed_file} -pix_fmt yuv420p {output_file} -y"
        subprocess.run(command, shell=True)



        # command = f"ffmpeg -i {compressed_file} -pix_fmt yuv420p{decompressed_folder}/reconstructed.yuv"
        # subprocess.run(command, shell=True)



        # filename_org = f"{input_folder}/{project_str}_128x128_30Hz_8bit_yuv420p8.yuv"  # 替换为实际的文件路径

        mmap1 = np.memmap(input_folder,
                          dtype=np.uint8, mode="r")
        mmap2 = np.memmap(output_file,
                          dtype=np.uint8, mode="r")

        org_sequence = RawVideoSequence(mmap1, width_in, height_in, bitdepth, video_format, framerate_in)
        dec_sequence = RawVideoSequence(mmap2, width_in, height_in, bitdepth, video_format, framerate_in)

        org_subfolder = os.path.join(output_folder, f'org_images_{qp}')
        dec_subfolder = os.path.join(output_folder, f'dec_images_{qp}')


        psnr_qp = []
        lpips_qp = []
        org_frames = []
        dec_frames = []

        for i in range(30):
            max_val = 2 ** bitdepth - 1

            org_frame_new = org_sequence[i]
            org_frame = to_tensors(org_frame_new, device="cuda")
            org_frame = tuple(p.unsqueeze(0).unsqueeze(0) for p in org_frame)  # type: ignore
            org_rgb_01 = ycbcr2rgb(yuv_420_to_444(org_frame, mode="bicubic").true_divide(max_val))

            dec_frame_new = dec_sequence[i]
            dec_frame = to_tensors(dec_frame_new, device="cuda")
            dec_frame = tuple(p.unsqueeze(0).unsqueeze(0) for p in dec_frame)  # type: ignore
            dec_rgb_01 = ycbcr2rgb(yuv_420_to_444(dec_frame, mode="bicubic").true_divide(max_val))


            org_rgb = (org_rgb_01 * max_val).clamp(0, max_val).round()
            dec_rgb = (dec_rgb_01 * max_val).clamp(0, max_val).round()


            org_pil_image = Image.fromarray(org_rgb[0, :, :, :].cpu().numpy().astype('uint8').transpose(1, 2, 0))

            dec_pil_image = Image.fromarray(dec_rgb[0, :, :, :].cpu().numpy().astype('uint8').transpose(1, 2, 0))


            org_frames.append(org_rgb_01)
            dec_frames.append(dec_rgb_01)

            os.makedirs(org_subfolder, exist_ok=True)
            os.makedirs(dec_subfolder, exist_ok=True)


            # 保存图像到子文件夹
            org_pil_image.save(os.path.join(org_subfolder, f'org_image_{i}.png'))  # 保存原始图像
            dec_pil_image.save(os.path.join(dec_subfolder, f'dec_image_{i}.png'))  # 保存解码后的图像
            # print(org_rgb)

            mse_rgb = (org_rgb - dec_rgb).pow(2).mean()
            psnr = 10 * np.log10((255 ** 2) / mse_rgb.cpu().numpy())
            psnr_qp.append(psnr)
            # print(psnr)
            model_lpips = PerceptualLoss(model='net-lin', net='alex',
                                         device='cuda')
            lpips = model_lpips.forward(org_rgb, dec_rgb)

            lpips_qp.append(lpips.cpu().detach().numpy())

            psnr_qp.append(psnr)
        psnrs.append(np.mean(psnr_qp))
        lpips_mean.append(np.mean(lpips_qp))

        org_frames_tensor = torch.stack(org_frames, dim=0)
        org_frames_tensor = org_frames_tensor.permute(1, 0, 2, 3, 4).repeat(2, 1, 1, 1, 1)

        dec_frames_tensor = torch.stack(dec_frames, dim=0)
        dec_frames_tensor = dec_frames_tensor.permute(1, 0, 2, 3, 4).repeat(2, 1, 1, 1, 1)
        fvd.append(calculate_fvd(org_frames_tensor, dec_frames_tensor))


        # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(org_frames_tensor)


        bpp.append(float((get_filesize(compressed_file) * bitdepth / (width_in * height_in * framerate_in))))

        with open(f"{output_folder}/psnr_lpips_fvd_bpp.txt", "w") as f:
            for psnr_val, lpips_val, fvd_val, bpp_val in zip(psnrs, lpips_mean, fvd, bpp):
                f.write(f"PSNR: {psnr_val}, LPIPS: {lpips_val}, FVD: {fvd_val}, BPP: {bpp_val}\n")

    # file_path = f"{output_folder}/psnr_lpips_bpp.txt"



# width = args.width
# height = args.width
# bitdepth = 8
# framerate = args.framerate



import subprocess

def convert_images_to_yuv(input_path, output_path, project_str,num_images=30):
    # 创建一个空列表，用于存储图像文件的路径
    image_list = []
    # 生成图像文件路径并添加到列表中
    # for i in range(num_images):
    #     img_path = f"{input_path}/frame{i+1}.png"
    #     image_list.append(img_path)
    print("*"*70)
    print(f"{input_path}/frame%d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", "30",
        "-f", "image2",
        "-i", f"{input_path}/frame%d.png",
        "-c:v", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-an",
        f"{output_path}/{project_str}.yuv",
    ]
    subprocess.run(cmd)
    # for attempt in range(3):
    #     try:
    #         path = f"{input_path}/frame%d.png"
    #         if os.path.exists(path):
    #             print(f"{path} exists.")
    #         else:
    #             print(f"{path} does not exist.")        # 设置命令行参数
    #         cmd = [
    #             "ffmpeg",
    #             "-y",
    #             "-framerate", "30",
    #             "-f", "image2",
    #             "-i", f"{input_path}/frame%d.png",
    #             "-c:v", "rawvideo",
    #             "-pix_fmt", "yuv420p",
    #             "-an",
    #             f"{output_path}/{project_str}.yuv",
    #         ]
    #         subprocess.run(cmd)
    #     except FileNotFoundError:
    #         print(f"File not found: {input_path}")
    #         return None
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         if attempt < 3 - 1:
    #             print(f"Retrying in {1} seconds...")
    #             time.sleep(1)
    #     # 执行命令行


# 定义center_crop函数
def center_crop(image):
    h, w, c = image.shape
    new_h, new_w = h if h < w else w, w if w < h else h
    r_min, r_max = h // 2 - new_h // 2, h // 2 + new_h // 2
    c_min, c_max = w // 2 - new_w // 2, w // 2 + new_w // 2
    return image[r_min:r_max, c_min:c_max, :]

# 定义read_video函数
def read_video(video_files, image_size):
    frames = []
    for file in video_files:
        frame = cv2.imread(file)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_cc = center_crop(img)
        pil_im = Image.fromarray(img_cc)
        pil_im_rsz = pil_im.resize((image_size, image_size), Image.LANCZOS)
        frames.append(np.array(pil_im_rsz))

    return frames


def plot_and_save_graph(bpp_values_1, psnr_values_1, bpp_values_2, psnr_values_2, label_1, label_2, metric_name, output_folder):
    plt.figure(figsize=(10, 6))
    plt.plot(bpp_values_1, psnr_values_1, marker='o', label=label_1)
    plt.plot(bpp_values_2, psnr_values_2, marker='x', label=label_2)
    plt.xlabel('BPP')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs BPP')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, f"{metric_name.lower()}_vs_bpp_bench.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        add_help=False,
    )
    # Common options.
    parser.add_argument("--dataset", type=str, default="/media/myworkstation/文档/Dataset/UVG/UVG_6", help="dataset path")

    parser.add_argument(
        "--output_path",
        type=str,
        default="/media/myworkstation/文档/output_uvg_bench/UVG_6",
        help="result output path",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="img width",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="img height",
    )

    parser.add_argument(
        "--framerate",
        type=int,
        default=30,
        help="video framerate",
    )

    parser.add_argument(
        "--project_str",
        type=str,
        choices=["city", "smm", "uvg"],
        help="Select one of the provided strings (city, smm, uvg)",
    )

    args = parser.parse_args()

    # input_folder_image = f"/home/myworkstation/OUT/output_images/rawvideo/rawvideo_{j}"
    input_folder = args.dataset
    output_folder = args.output_path
    convert_images_to_yuv(input_folder, output_folder, args.project_str)
    # time.sleep(10)
    input_yuv = os.path.join(output_folder, f"{args.project_str}.yuv")
    # input_yuv = output_folder
    # input_folder = f"/home/myworkstation/OUT/output_images/video/video_{j}"

    output_folder_264 = os.path.join(output_folder, f"out_frames_264")
    output_folder_265 = os.path.join(output_folder, f"out_frames_265")

    qp_range = range(52)

    # Define the LPIPS model
    # lpips_model = PerceptualLoss(model='net-lin', net='alex', device='cuda')

    # Compress and evaluate using H.264
    compress_and_evaluate(input_yuv, output_folder_264, 'libx264', qp_range, "yuv420p",args.width, args.height, args.framerate, args.project_str)

    # Compress and evaluate using H.265 (commented command)
    compress_and_evaluate(input_yuv, output_folder_265, 'libx265', qp_range, "yuv420p",args.width, args.height, args.framerate,args.project_str)






    psnr_values_1 ,lpips_values_1, fvd_values_1,bpp_values_1  = parse_metrics_file(os.path.join(output_folder_264, f"psnr_lpips_fvd_bpp.txt"))

    psnr_values_2 ,lpips_values_2,fvd_values_2, bpp_values_2  = parse_metrics_file(os.path.join(output_folder_265, f"psnr_lpips_fvd_bpp.txt"))

    # metrics_array_264 = np.empty((4, 52))
    #
    # # 填充数组，按照顺序填充 psnr, lpips, fvd, bpp
    # metrics_array_264[0, :] = psnr_values_1
    # metrics_array_264[1, :] = lpips_values_1
    # metrics_array_264[2, :] = fvd_values_1
    # metrics_array_264[3, :] = bpp_values_1
    #
    # metrics_array_265 = np.empty((4, 52))
    #
    # # 填充数组，按照顺序填充 psnr, lpips, fvd, bpp
    # metrics_array_265[0, :] = psnr_values_2
    # metrics_array_265[1, :] = lpips_values_2
    # metrics_array_265[2, :] = fvd_values_2
    # metrics_array_265[3, :] = bpp_values_2
    #
    # np.save(os.path.join(output_folder,"bench_uvg_264.npy"),metrics_array_264)
    # np.save(os.path.join(output_folder,"bench_uvg_265.npy"), metrics_array_265)
    metrics_array_264 = np.array([psnr_values_1, lpips_values_1, fvd_values_1, bpp_values_1])
    metrics_array_265 = np.array([psnr_values_2, lpips_values_2, fvd_values_2, bpp_values_2])
    # print(metrics_array_264.shape)
    # 假设 output_folder 是保存文件的目录
    np.save(os.path.join(output_folder, "bench_uvg_264.npy"), metrics_array_264)
    np.save(os.path.join(output_folder, "bench_uvg_265.npy"), metrics_array_265)
    # 绘制PSNR关于BPP的曲线图
    plot_and_save_graph(bpp_values_1, psnr_values_1, bpp_values_2, psnr_values_2, '264', '265', 'PSNR',
                        output_folder_265)

    # 绘制LPIPS关于BPP的曲线图
    plot_and_save_graph(bpp_values_1, lpips_values_1, bpp_values_2, lpips_values_2, '264', '265', 'LPIPS',
                        output_folder_265)

    # 绘制FVD关于BPP的曲线图
    plot_and_save_graph(bpp_values_1, fvd_values_1, bpp_values_2, fvd_values_2, '264', '265', 'FVD', output_folder_265)


if __name__ == "__main__":
    main()