function nn_reach_test()
    
    % Reachability analysis of Adaptive Cruise Control with a neuralODE (nonlinear) as a plant model

    % safety specification: relative distance > safe distance
    % dis = x_lead - x_ego  
    % safe distance between two cars, see here 
    % https://www.mathworks.com/help/mpc/examples/design-an-adaptive-cruise-control-system-using-model-predictive-control.html
    % dis_safe = D_default + t_gap * v_ego;


    %% Load objects
    
    % Load controller
    net = load_NN_from_mat('controller_main.mat');
    net.reachMethod = 'exact-star';

    % Load plant from neuralODE function @tanh_plant
    reachStep = 0.01;
    controlPeriod = 0.1;
    states = 8;
    C = eye(states); C(7,7) = 0; C(end) = 0;
    plant = NonLinearODE(8,1,@tanh_plant,reachStep,controlPeriod,C);
    plant.options.tensorOrder = 2;
    
    D_default = 10;
    t_gap = 1;
    map_mat = [0 0 0 0 1 0 0 0;
                1 0 0 -1 0 0 0 0;
                0 1 0 0 -1 0 0 0];
    U = Star(0,0);
    U_fix = Star([30;t_gap],[30;t_gap]); % vset and tgap

    %% Reachability analysis

    v_ego_range = [20, 40];
    d1 = 0.5;
    d_x_range = [5, 30];
    d2 = 0.5;
    d_v_range = [-20, 20];
    d3 = 0.5;
    idx = 0;
    time = 0;
    for i = v_ego_range(1):5:v_ego_range(2)
        for j = d_x_range(1):5:d_x_range(2)
            for k = d_v_range(1):5:d_v_range(2)
                lb = [300-0.8*rand();30+k/2-0.4*rand();0;300-i;30-k/2-0.8*rand();0;0;-2]; %lower bound
                ub = [300+0.8*rand();30+k/2+0.4*rand();0;300-i;30-k/2+0.8*rand();0;0;-2]; % upper bound
                X0 = Star(lb,ub);
                R1 = plant.stepReachStar(X0,U);
                X0 = R1(end);
                ppp = X0.affineMap(map_mat,[]);
                Uin = U_fix.concatenate(ppp);
                if idx == 0
                    Rc = net.reach(Uin);
                end
                tic;
                Rc = net.reach(Uin);
                time = time + toc;
                idx = idx + 1;
                time / idx
            end
        end
    end
    time = time / idx
    disp("end")
end