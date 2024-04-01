function generate_train_set()
    
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
    
    D_default = 5;
    t_gap = 1;
    v_ego_range = [20, 40];
    d1 = 0.5;
    d_x_range = [2, 300];
    d2 = 0.5;
    d_v_range = [-20, 20];
    d3 = 0.5;
    U_fix = Star([30;1],[30;1]);
    idx = 1;
    file_path = '/mnt/mnt1/linxuan/nnv/ACC/train_data_uncritical_json/';
    for i = v_ego_range(1):d1:v_ego_range(2)
        for j = d_x_range(1):d2:d_x_range(2)
            for k = d_v_range(1):d3:d_v_range(2)
                if j - (D_default + t_gap * i) >= 0
                    
                    ppp = Star([i-d1/2-0.4*rand();j-d2/2-0.8*rand();k-d3/2-0.4*rand()],[i+d1/2+0.4*rand();j+d2/2+0.8*rand();k+d3/2+0.4*rand()]);
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

                    idx = idx + 1;
                    if mod(idx,1000) == 0
                        idx,i,j,k
                    end
                    %save(strcat("/mnt/mnt1/nnv/ACC/train_data_critical/",num2str(idx),".mat"),"Uin","Rc")
                end
            end
        end
    end

end