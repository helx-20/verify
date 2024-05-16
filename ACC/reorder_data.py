import os
import numpy as np
path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data3_uncritical_json'
name_list = os.listdir(path)
exist_list = np.zeros((int(len(name_list)/3)))
for name in name_list:
    if name[:3] == 'Uin':
        idx = int(name[4:-5])
        if idx-1 < int(len(name_list)/3):
            exist_list[idx-1] = 1
            print(np.sum(exist_list)/int(len(name_list)/3))
reorder_list = []
for name in name_list:
    if name[:3] == 'Uin':
        idx = int(name[4:-5])
        if idx > int(len(name_list)/3):
            reorder_list.append(idx)
            print(len(reorder_list))
reorder_list.sort()
reorder_list = reorder_list[::-1]
for i in range(int(len(name_list)/3)):
    if exist_list[i] == 0:
        idx = reorder_list.pop()
        os.rename(os.path.join(path,'Rc_{}.json'.format(idx)), os.path.join(path,'Rc_{}.json'.format(i+1)))
        os.rename(os.path.join(path,'Rc_convex_{}.json'.format(idx)), os.path.join(path,'Rc_convex_{}.json'.format(i+1)))
        os.rename(os.path.join(path,'Uin_{}.json'.format(idx)), os.path.join(path,'Uin_{}.json'.format(i+1)))
        exist_list[i] = 1
        print(np.sum(exist_list)/int(len(name_list)/3))
