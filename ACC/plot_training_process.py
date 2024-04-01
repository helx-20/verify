import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os

with open("uncritical.log", 'r') as file:
    lines = file.readlines()

loss = []
epoch = []
epoch_idx = 0
for i in range(len(lines)):
    if "Test set" in lines[i]:
        epoch.append(epoch_idx)
        epoch_idx += 1
        data_tmp = lines[i].split(":")
        loss_tmp = float(data_tmp[-1])
        loss.append(loss_tmp)

rcParams["font.size"] = 12

plt.plot(epoch,loss)
plt.title("loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("uncritical_net_loss.png",dpi=300)
plt.close("all")