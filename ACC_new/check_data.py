import os
name_list = os.listdir('/mnt/mnt1/linxuan/nnv/ACC/new_train_data2_critical_json')
for i in range(126036):
    if 'Rc_{}.json'.format(i+1) not in name_list:
        print('Rc_{}.json'.format(i+1))
    if 'Uin_{}.json'.format(i+1) not in name_list:
        print('Uin_{}.json'.format(i+1))