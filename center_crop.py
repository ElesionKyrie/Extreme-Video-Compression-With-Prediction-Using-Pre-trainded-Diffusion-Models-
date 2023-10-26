# import cv2
# import numpy as np
# from PIL import Image
#
# # 定义center_crop函数
# def center_crop(image):
#     h, w, c = image.shape
#     new_h, new_w = h if h < w else w, w if w < h else h
#     r_min, r_max = h // 2 - new_h // 2, h // 2 + new_h // 2
#     c_min, c_max = w // 2 - new_w // 2, w // 2 + new_w // 2
#     return image[r_min:r_max, c_min:c_max, :]
#
# # 定义read_video函数
# def read_video(video_files, image_size):
#     frames = []
#     for file in video_files:
#         frame = cv2.imread(file)
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img_cc = center_crop(img)
#         pil_im = Image.fromarray(img_cc)
#         pil_im_rsz = pil_im.resize((image_size, image_size), Image.LANCZOS)
#         frames.append(np.array(pil_im_rsz))
#
#     return frames
# all_arrays = []
#
# for i in range(7):
#     # 文件路径
#     image_dir = f'/home/myworkstation/下载/archive (1)/uvg_raw_{i}'
#     image_size = 128
#     num_images = 30
#
#     # 构建文件路径列表
#     image_files = [f"{image_dir}/frame{i+1}.png" for i in range(num_images)]
#
#     # 读取并处理图像
#     processed_frames = read_video(image_files, image_size)
#
#     # 将处理后的图像堆叠成数组
#     processed_frames = np.expand_dims(processed_frames, axis=0)
#     processed_frames = np.array(processed_frames)
#     frames_array = processed_frames.transpose(0, 1, 4, 2, 3)  # 调整维度顺序
#
#     # 添加到列表
#     all_arrays.append(frames_array)
#
# resulting_array = np.concatenate(all_arrays, axis=0)
#
# # 打印数组形状
# print(resulting_array.shape)
#
# # 保存拼接后的数组
# np.save('/media/myworkstation/文档/Dataset/UVG/UVG_all.npy', resulting_array)

import numpy as np
from PIL import Image
import os
import os
import subprocess
import numpy as np
from PIL import Image
# 加载数组
# array_path = '/home/myworkstation/项目win/citynpy/citynpy.npy'
# array_path = '/media/myworkstation/文档/Dataset/UVG/UVG.npy'

array_path = "/media/myworkstation/文档/Dataset/UVG/UVG_all.npy"

array = np.load(array_path)
for j in range(0, 7):
# 获取第一个视频的30帧
    video_frames = array[j, :30]

    # 逐帧保存为PNG图像
    output_folder = f'/media/myworkstation/文档/Dataset/UVG/UVG_{j}/'
    os.makedirs(output_folder, exist_ok=True)

    for i, frame in enumerate(video_frames):

        frame_path = os.path.join(output_folder, f'frame{i}.png')
        frame_rgb = frame.transpose(1, 2, 0)  # 调整通道顺序为 (128, 128, 3)

        pil_image = Image.fromarray(frame_rgb.astype(np.uint8))
        pil_image.save(frame_path)
