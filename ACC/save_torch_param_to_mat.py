import torch
from scipy.io import savemat
from model.critical_net_for_matlab import Critical_Net
state_dict = torch.load('saved_model/critical_net2_state_dict_5.5_428.pth')
params_dict = {key: param.cpu().numpy() for key, param in state_dict.items()}
savemat('critical_state_dict.mat',{'params':params_dict},long_field_names=True)
#model = Critical_Net(input_dim_V=5, input_dim_C=323, output_dim=2, m_tokens_in=164)
#model.load_state_dict(torch.load("saved_model/critical_net2_state_dict.pth"))
#data = torch.rand(1,164,323+5)
#torch.jit.save(torch.jit.trace(model.forward,data),'test.pt')