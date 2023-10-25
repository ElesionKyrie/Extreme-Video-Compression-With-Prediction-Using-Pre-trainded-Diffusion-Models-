import os
import re
import numpy as np

# 获取output文件夹下的所有文件夹名
output_folder = '/home/myworkstation/OUT/output_test4'
subfolders = [f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))]

# 初始化空列表用于存放数据
data_list = []

# 遍历每个文件夹
for subfolder in subfolders:
    subfolder_path = os.path.join(output_folder, subfolder)
    txt_file_path = os.path.join(subfolder_path, '265_fvd_bpp.txt')

    # 检查文件是否存在
    if os.path.exists(txt_file_path):
        fvd_values = []
        bpp_values = []

        # 从txt文件中读取数据
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                fvd_match = re.search(r'FVD: ([\d.]+), BPP: ([\d.]+)', line)
                if fvd_match:
                    fvd_values.append(float(fvd_match.group(1)))
                    bpp_values.append(float(fvd_match.group(2)))

        # 将FVD和BPP数据存入一个二维数组
        subfolder_data = np.array([fvd_values, bpp_values])

        # 添加二维数组到列表中
        data_list.append(subfolder_data)

# 将数据列表转换为三维数组
data_3d = np.array(data_list)
print(data_3d.shape)

# 保存三维数组为npy文件
output_file_path = '/home/myworkstation/OUT/output_test4/fvd_data_265.npy'
np.save(output_file_path, data_3d)

print("数据已保存到", output_file_path)
