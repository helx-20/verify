function verify()
    net = load_NN_from_mat('controller_main.mat');
    net.reachMethod = 'approx-star';

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

    critical_net = py.model.critical_net.Critical_Net(input_dim_V=py.int(5), input_dim_C=py.int(323), output_dim=py.int(2), m_tokens_in=py.int(164));
    params = load('uncritical_state_dict.mat').params;
    keys = fieldnames(params);
    params_new = critical_net.state_dict();
    for idx = 1:length(keys)
        key = keys{idx};
        params_new{key} = py.torch.tensor(py.numpy.array(params.(key)));
    end
    critical_net.load_state_dict(params_new);
    critical_net.eval();
    critical_net.to(py.torch.device('cuda:0'));

    uncritical_net = py.model.uncritical_net.Uncritical_Net(input_dim_V=py.int(5), input_dim_C=py.int(323), output_dim=py.int(2), m_tokens_in=py.int(164));
    params = load('uncritical_state_dict.mat').params;
    keys = fieldnames(params);
    params_new = uncritical_net.state_dict();
    for idx = 1:length(keys)
        key = keys{idx};
        params_new{key} = py.torch.tensor(py.numpy.array(params.(key)));
    end
    uncritical_net.load_state_dict(params_new);
    uncritical_net.eval();
    uncritical_net.to(py.torch.device('cuda:0'));

    %% Reachability analysis

    % Set reachability options
    % initial condition of x_lead
    xlead = [99 101];
    % initial condition of v_lead
    v_lead = [31 32];
    % initial condition of x_ego
    x_ego = [10 11]; 
    % initial condition of v_ego
    v_ego = [30 31];
    % input set 
    U = Star(0,0);
    % initial state set
    lb = [xlead(1);v_lead(1);0;x_ego(1);v_ego(1);0;0;-2]; %lower bound
    ub = [xlead(2);v_lead(2);0;x_ego(2);v_ego(2);0;0;-2]; % upper bound
    X0 = Star(lb,ub); %
    map_mat = [0 0 0 0 1 0 0 0;
                1 0 0 -1 0 0 0 0;
                0 1 0 0 -1 0 0 0];
    U_fix = Star([30;1],[30;1]); % vset and tgap
    % Perform reachability analysis
    trajR = X0;
    trajU = [];
    N = 30;
    % Start computation
    t = tic;
    error = 0;
    for k=1:N
        k
        tic;
        R1 = plant.stepReachStar(X0,U); % reachability of plant
        toc
        X0 = R1(end); % Get only last reach set (reach set at control period)
        trajR = [trajR X0]; % Keep track of trajectory of NNCS
        ppp = X0.affineMap(map_mat,[]);
        Uin = U_fix.concatenate(ppp);
        if 1
            data = py.torch.tensor(py.numpy.array(Uin.V),dtype=py.torch.float32).reshape(py.int(1),py.int(5),py.int(164)).transpose(py.int(1),py.int(2));
            data = data.to(py.torch.device('cuda:0'));
            C_d = py.torch.tensor(py.numpy.array([Uin.C,Uin.d]),dtype=py.torch.float32).reshape(py.int(1),py.int(323),py.int(164)).transpose(py.int(1),py.int(2));
            C_d = C_d.to(py.torch.device('cuda:0'));
            tic;
            output = double(uncritical_net(data,C_d).detach().cpu().reshape(py.int(-1)).tolist())
            toc
            Rc = Star(output(1)-output(2),output(1)+output(2));
            %tic;
            Rc2 = net.reach(Uin);
            %toc
            [xmin, xmax] = Rc2.getRange(1)
            error_tmp = max(abs(xmin-(output(1)-output(2))),abs(xmax-(output(1)+output(2))));
            if error_tmp > error
                error = error_tmp
            end
        else
            Rc = net.reach(Uin);
        end
        %net.reach(Uin);
        trajU = [trajU Rc];
        x08 = X0.affineMap([0 0 0 0 0 0 0 1],[]);
        X0 = X0.affineMap(C(1:6,:),[]); % Get set for variables 1 to 6
        X0 = X0.concatenate(Rc); % Add state/input 7 (a_ego)
        X0 = X0.concatenate(x08);
    end
    rT = toc(t);

    % Save results
    if 1
        %save('results_tanhplant.mat','rT',"trajR","trajU");
        
        %% Visualize results

        % Transform reach sets for visualization
        t_gap = 1;
        D_default = 10;
        alp = 1;
        outAll = [];
        safe_dis = [];
        for i=1:length(trajR)
            outAll = [outAll trajR(i).affineMap([1 0 0 -1 0 0 0 0], [])]; % distance between cars
            safe_dis = [safe_dis trajR(i).affineMap([0 0 0 0 alp*t_gap 0 0 0], alp*D_default)]; % safe distance
        end
        times = 0:0.1:0.1*N; % to plot in x-axis (time)

        % Create figure
        f = figure;
        hold on;
        pb = plot(0,85,'m');
        pr = plot(0,85,'k');
        Star.plotRanges_2D(outAll,1,times,'k');
        hold on;
        Star.plotRanges_2D(safe_dis,1,times,'m');

        % Enhance figure for paper
        ax = gca; % Get current axis
        ax.XAxis.FontSize = 15; % Set font size of axis
        ax.YAxis.FontSize = 15;
        xlabel('Time (s)');
        ylabel('Distance (m)')
        ylim([40 110])
        legend([pr,pb],{'rel dist','safe dist'},"Location","best",'FontSize',14);

        %saveas(f, "/home/linxuan/nnv-master/code/nnv/examples/NNV2.0/Submission/CAV2023/NeuralODEs/ACC/uncritical_verify_approx.png")
    end

end