import json
import numpy as np
import math
max_len = 1

def load_data(idx):
    with open("/mnt/mnt1/linxuan/nnv/ACC/train_data_uncritical_json/{}.json".format(idx), 'r') as file:
        data = json.load(file)
    return data

def select_range(idx):
    data = []
    label = []
    max_range = math.floor(5*np.random.rand())
    
    data_tmp = load_data(idx)
    if isinstance(data_tmp, list):
        for i in range(5):
            data.append(np.array(data_tmp[0]['Uin'])[[i,i],[0,i]])
        for i in range(len(data_tmp)):
            label.append(np.array([data_tmp[i]['Rc'][0],np.sum(np.abs(np.array(data_tmp[i]['Rc'][1:])))]))
    else:
        for i in range(5):
            data.append(np.array(data_tmp['Uin'])[[i,i],[0,i]])
        label.append(np.array([data_tmp['Rc'][0],np.sum(np.abs(np.array(data_tmp['Rc'][1:])))]))

    label_new = []
    adjusted = np.zeros((len(label)))
    for i in range(len(label)):
        min_value = 1000
        min_idx = -1
        for j in range(len(label)):
            if adjusted[j] == 1:
                continue
            else:
                if label[j][0] < min_value:
                    min_idx = j
                    min_value = label[j][0]
        adjusted[min_idx] = 1
        label_new.append(label[min_idx])
    #print(label_new)
    return data, label_new

num = 1763451
data = np.zeros((num,5,2))
label = np.zeros((num,max_len,2))
max_num = 0
for idx in range(num):
    print(idx)
    data_tmp, label_tmp = select_range(idx+1)
    for i in range(len(data_tmp)):
        data[idx,i,:] = np.array(data_tmp[i])
    for i in range(len(label_tmp)):
        label[idx,i,:] = np.array(label_tmp[i])
    if len(label_tmp) > max_num:
        max_num = len(label_tmp)
        print(max_num)

np.save("/mnt/mnt1/linxuan/nnv/ACC/dataset/uncritical_data.npy", {'data':data, 'label':label})
