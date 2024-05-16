function use_torch_model()
    pyenv('Version','/home/linxuan/anaconda3/envs/NDE/bin/python');
    py.sys.setdlopenflags(int32(bitor(int64(py.os.RTLD_LAZY),int64(py.os.RTLD_DEEPBIND))));
    %gpuDevice(1)

    net = py.model.critical_net.Critical_Net(input_dim_V=py.int(5), input_dim_C=py.int(323), output_dim=py.int(2), m_tokens_in=py.int(164));
    %data = py.torch.tensor(py.numpy.array(rand(1,164,5+323)),dtype=py.torch.float32);
    
    %C = py.torch.tensor(py.numpy.array(rand(1,164,323)),dtype=py.torch.float32);
    %net = importNetworkFromPyTorch("test.pt")
    %py.torch.jit.save(py.torch.jit.trace(net,data),'test.pt')
    params = load('critical_state_dict.mat').params;
    keys = fieldnames(params);
    params_new = net.state_dict();
    for idx = 1:length(keys)
        %idx
        key = keys{idx};
        params_new{key} = py.torch.tensor(py.numpy.array(params.(key)));
    end
    net.load_state_dict(params_new);
    %py.torch.jit.save(py.torch.jit.trace(net,data),'test.pt')
    net.eval();
    net.to(py.torch.device('cuda:0'));
    data = py.torch.tensor(py.numpy.array(rand(1,164,5)),dtype=py.torch.float32);
    C = py.torch.tensor(py.numpy.array(rand(1,164,323)),dtype=py.torch.float32);
    data = data.to(py.torch.device('cuda:0'));
    C = C.to(py.torch.device('cuda:0'));
    for i = 1:10
        net(data,C);
    end
    %tic;
    for i = 1:200
        net(data,C);
        tic;
        net(data,C);
        toc
    end
    %toc/100
end