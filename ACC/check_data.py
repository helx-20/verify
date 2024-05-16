import os
import numpy as np
path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data3_critical_json'
name_list = os.listdir(path)
exist_list = np.zeros((int(len(name_list)/3)))
for name in name_list:
    if name[:3] == 'Uin':
        idx = int(name[4:-5])
        if idx-1 < int(len(name_list)/3):
            exist_list[idx-1] = 1
            print(np.sum(exist_list)/int(len(name_list)/3))