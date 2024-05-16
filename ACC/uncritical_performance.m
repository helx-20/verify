function uncritical_performance()

    save_path = '/mnt/mnt1/linxuan/nnv/ACC/uncritical_3steps_performance_json/';

    net = load_NN_from_mat('controller_main.mat');
    net.reachMethod = 'approx-star';
    net2 = load_NN_from_mat('controller_main.mat');
    net2.reachMethod = 'exact-star';

    % Load plant from neuralODE function @tanh_plant
    reachStep = 0.01;
    controlPeriod = 0.1;
    states = 8;
    C = eye(states); C(7,7) = 0; C(end) = 0;
    plant = NonLinearODE(8,1,@tanh_plant,reachStep,controlPeriod,C);
    plant.options.tensorOrder = 2;
    
    pyenv('Version','/home/linxuan/anaconda3/envs/NDE/bin/python');
    py.sys.path().append('/home/linxuan/nnv-master/code/nnv/examples/NNV2.0/Submission/CAV2023/NeuralODEs/ACC')
    py.sys.setdlopenflags(int32(bitor(int64(py.os.RTLD_LAZY),int64(py.os.RTLD_DEEPBIND))));
    %gpuDevice(1)

    NN = py.model.uncritical_net.Uncritical_Net(input_dim_V=py.int(5), input_dim_C=py.int(323), output_dim=py.int(2), m_tokens_in=py.int(164));
    params = load('uncritical_state_dict.mat').params;
    keys = fieldnames(params);
    params_new = NN.state_dict();
    for idx = 1:length(keys)
        key = keys{idx};
        params_new{key} = py.torch.tensor(py.numpy.array(params.(key)));
    end
    NN.load_state_dict(params_new);
    NN.eval();
    NN.to(py.torch.device('cuda:0'));

    %% Reachability analysis

    % Set reachability options
    v_ego_range = [20, 40];
    d_x_range = [5, 200];
    d_v_range = [-20, 20];
    % input set 
    U = Star(0,0);
    map_mat = [0 0 0 0 1 0 0 0;
                1 0 0 -1 0 0 0 0;
                0 1 0 0 -1 0 0 0];
    U_fix = Star([30;1],[30;1]); % vset and tgap
    % Perform reachability analysis
    N = 3;
    %parpool('Processes',16);
    for idx = 56:200
        idx
        try 
            usable = true; 
            i = v_ego_range(1) + (v_ego_range(2) - v_ego_range(1)) * rand();
            j = 10 + i + (d_x_range(2) - 10 - i) * rand();
            k = d_v_range(1) + (d_v_range(2) - d_v_range(1)) * rand();
            lb = [300-1*rand();i+k-1*rand();0;300-j;i-1*rand();0;0;-2]; %lower bound
            ub = [300+1*rand();i+k+1*rand();0;300-j;i+1*rand();0;0;-2]; % upper bound
            X0 = Star(lb,ub);
            for tmp_t = 1:2
                R1 = plant.stepReachStar(X0,U);
                X0 = R1(end);
                if length(X0) > 1
                    usable = false;
                    break
                end
                ppp = X0.affineMap(map_mat,[]);
                Uin = U_fix.concatenate(ppp);
                Rc = net.reach(Uin);
                [xmin, xmax] = Rc.getRange(1);
                Rc = Star(xmin, xmax);
                x08 = X0.affineMap([0 0 0 0 0 0 0 1],[]);
                X0 = X0.affineMap(C(1:6,:),[]); % Get set for variables 1 to 6
                X0 = X0.concatenate(Rc); % Add state/input 7 (a_ego)
                X0 = X0.concatenate(x08);
            end
            % NN method
            NN_tmp = NN;
            X0_NN = X0;
            for tmp_t = 1:N
                R1 = plant.stepReachStar(X0_NN,U);
                X0_NN = R1(end);
                if length(X0_NN) > 1
                    usable = false;
                    break
                end
                ppp = X0_NN.affineMap(map_mat,[]);
                Uin = U_fix.concatenate(ppp);
                data = py.torch.tensor(py.numpy.array(Uin.V),dtype=py.torch.float32).reshape(py.int(1),py.int(5),py.int(164)).transpose(py.int(1),py.int(2));
                data = data.to(py.torch.device('cuda:0'));
                C_d = py.torch.tensor(py.numpy.array([Uin.C,Uin.d]),dtype=py.torch.float32).reshape(py.int(1),py.int(323),py.int(164)).transpose(py.int(1),py.int(2));
                C_d = C_d.to(py.torch.device('cuda:0'));
                output = double(NN_tmp(data,C_d).detach().cpu().reshape(py.int(-1)).tolist());
                Rc_NN = Star(output(1)-output(2),output(1)+output(2));
                x08 = X0_NN.affineMap([0 0 0 0 0 0 0 1],[]);
                X0_NN = X0_NN.affineMap(C(1:6,:),[]); % Get set for variables 1 to 6
                X0_NN = X0_NN.concatenate(Rc_NN); % Add state/input 7 (a_ego)
                X0_NN = X0_NN.concatenate(x08);
            end
            if ~usable
                continue
            else
                new_data = jsonencode(output);
                fileID = fopen(strcat(save_path,'Rc_NN_',num2str(idx),'.json'), 'w');
                fwrite(fileID, new_data);
                fclose(fileID);
            end

            % approx method
            X0_approx = X0;
            for tmp_t = 1:N
                R1 = plant.stepReachStar(X0_approx,U);
                X0_approx = R1(end);
                ppp = X0_approx.affineMap(map_mat,[]);
                Uin = U_fix.concatenate(ppp);
                Rc_approx = net.reach(Uin);
                x08 = X0_approx.affineMap([0 0 0 0 0 0 0 1],[]);
                X0_approx = X0_approx.affineMap(C(1:6,:),[]); % Get set for variables 1 to 6
                X0_approx = X0_approx.concatenate(Rc_approx); % Add state/input 7 (a_ego)
                X0_approx = X0_approx.concatenate(x08);
            end
            full_Rc_approx = cell(length(Rc_approx),1);
            for tmp_i = 1:length(Rc_approx)
                set = Rc_approx(tmp_i);
                [xmin, xmax] = set.getRange(1);
                full_Rc_approx{tmp_i} = [(xmin+xmax)/2,(xmax-xmin)/2];
            end
            new_data = jsonencode(full_Rc_approx);
            fileID = fopen(strcat(save_path,'Rc_approx_',num2str(idx),'.json'), 'w');
            fwrite(fileID, new_data);
            fclose(fileID);

        catch exception
            disp("error")
        end
    end

end