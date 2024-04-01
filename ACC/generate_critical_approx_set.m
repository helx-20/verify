function generate_critical_approx_set()
    
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
    
    U_fix = Star([30;1],[30;1]);
    origin_file_path = '/mnt/mnt1/linxuan/nnv/ACC/train_data3_critical_json/';
    file_path = '/mnt/mnt1/linxuan/nnv/ACC/train_data3_critical_approx_json/';
    for idx = 1:219186
        all_data = jsondecode(fileread(strcat(origin_file_path,num2str(idx),'.json')));
        data = all_data(1).Uin;
        ppp = Star([data(3,1)-data(3,4);data(4,1)-data(4,5);data(5,1)-data(5,6)],[data(3,1)+data(3,4);data(4,1)+data(4,5);data(5,1)+data(5,6)]);
        Uin = U_fix.concatenate(ppp);
        Rc = net.reach(Uin);
        Rc_V = cell(length(Rc),1);
        for Rc_i = 1:length(Rc)
            Rc_V{Rc_i} = Rc(Rc_i).V;
        end
        new_data = jsonencode(struct('Uin',Uin.V,'Rc',Rc_V));
        fileID = fopen(strcat(file_path,num2str(idx),'.json'), 'w');
        fwrite(fileID, new_data);
        fclose(fileID);
        if mod(idx,1000) == 0
            idx
        end
    end
end