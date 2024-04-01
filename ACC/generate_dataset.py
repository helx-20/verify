import json
import numpy as np
import math
max_len = 192

def load_data(idx):
    with open("train_data_critical_json/{}.json".format(idx), 'r') as file:
        data = json.load(file)
    return data

def select_range(idx):
    data = []
    label = []
    max_range = math.floor(5*np.random.rand())
    for idx_left in range(max_range):
        if idx - max_range + idx_left < 1:
            continue
        else:
            data_tmp = load_data(idx-max_range+idx_left)
            if isinstance(data_tmp, list):
                data.append(np.array(data_tmp[0]['Uin'])[:,0])
                for i in range(len(data_tmp)):
                    label.append(np.array([data_tmp[i]['Rc'][0],np.sum(np.abs(np.array(data_tmp[i]['Rc'][1:])))]))
            else:
                data.append(np.array(data_tmp['Uin'])[:,0])
                label.append(np.array([data_tmp['Rc'][0],np.sum(np.abs(np.array(data_tmp['Rc'][1:])))]))

    data_tmp = load_data(idx)
    if isinstance(data_tmp, list):
        data.append(np.array(data_tmp[0]['Uin'])[:,0])
        for i in range(len(data_tmp)):
            label.append(np.array([data_tmp[i]['Rc'][0],np.sum(np.abs(np.array(data_tmp[i]['Rc'][1:])))]))
    else:
        data.append(np.array(data_tmp['Uin'])[:,0])
        label.append(np.array([data_tmp['Rc'][0],np.sum(np.abs(np.array(data_tmp['Rc'][1:])))]))

    for idx_right in range(max_range):
        if idx + idx_right + 1 > 507958:
            break
        else:
            data_tmp = load_data(idx+idx_right+1)
            if isinstance(data_tmp, list):
                data.append(np.array(data_tmp[0]['Uin'])[:,0])
                for i in range(len(data_tmp)):
                    label.append(np.array([data_tmp[i]['Rc'][0],np.sum(np.abs(np.array(data_tmp[i]['Rc'][1:])))]))
            else:
                data.append(np.array(data_tmp['Uin'])[:,0])
                label.append(np.array([data_tmp['Rc'][0],np.sum(np.abs(np.array(data_tmp['Rc'][1:])))]))
    
    return data, label

data = np.zeros((507958,max_len,5))
label = np.zeros((507958,max_len,2))
for idx in range(507958):
    print(idx)
    data_tmp, label_tmp = select_range(idx+1)
    for i in range(len(data_tmp)):
        data[idx,i,:] = np.array(data_tmp[i])
    for i in range(len(label_tmp)):
        label[idx,i,:] = np.array(label_tmp[i])

np.save("dataset/critical_data.npy", {'data':data, 'label':label})