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

    % Load plant from neuralODE function @tanh_plant
    reachStep = 0.01;
    controlPeriod = 0.1;
    states = 8;
    C = eye(states); C(7,7) = 0; C(end) = 0;
    plant = NonLinearODE(8,1,@tanh_plant,reachStep,controlPeriod,C);
    plant.options.tensorOrder = 2;
    

    %% Reachability analysis

    v_ego_range = [20, 40];
    d1 = 0.5;
    d_x_range = [5, 30];
    d2 = 0.5;
    d_v_range = [-20, 20];
    d3 = 0.5;
    U_fix = Star([30;1],[30;1]);
    idx = 0;
    time = 0;
    for i = v_ego_range(1):5:v_ego_range(2)
        for j = d_x_range(1):5:d_x_range(2)
            for k = d_v_range(1):5:d_v_range(2)
                ppp = Star([i-d1/2-0.4*rand();j-d2/2-0.8*rand();k-d3/2-0.4*rand()],[i+d1/2+0.4*rand();j+d2/2+0.8*rand();k+d3/2+0.4*rand()]);
                Uin = U_fix.concatenate(ppp);
                tic;
                Rc = net.reach(Uin);
                time = time + toc
                idx = idx + 1;
            end
        end
    end
    time = time / idx
    disp("end")
end