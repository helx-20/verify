function generate_critical_not_identified_set()
    
    % Reachability analysis of Adaptive Cruise Control with a neuralODE (nonlinear) as a plant model

    % safety specification: relative distance > safe distance
    % dis = x_lead - x_ego  
    % safe distance between two cars, see here 
    % https://www.mathworks.com/help/mpc/examples/design-an-adaptive-cruise-control-system-using-model-predictive-control.html
    % dis_safe = D_default + t_gap * v_ego;


    %% Load objects
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
    
    % Load controller
    net = load_NN_from_mat('controller_main.mat');
    net.reachMethod = 'exact-star';

    origin_file_path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data2_critical_not-identified_json/';
    file_path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data2_critical_not-identified_json/';
    data_list1 = jsondecode(fileread(strcat(origin_file_path,'data_list.json')));
    data_list2 = jsondecode(fileread(strcat(origin_file_path,'data_list2.json')));
    data_list = jsondecode(fileread(strcat(origin_file_path,'data_list3.json')));
    %parpool('Processes',64);
    parfor tmp = 1:size(data_list,1)
        i = round(data_list(tmp,3)*2)/2;
        j = round(data_list(tmp,4)*2)/2;
        k = round(data_list(tmp,5)*2)/2;

        idx = 10 * size(data_list1,1) + 5 * size(data_list2,1) + (tmp - 1) * 20;
        for tmp2 = 1:20
            lb = [300-0.8*rand();i+k-0.4*rand();0;300-j;i-0.8*rand();0;0;-2]; %lower bound
            ub = [300+0.8*rand();i+k+0.4*rand();0;300-j;i+0.8*rand();0;0;-2]; % upper bound
            X0 = Star(lb,ub);
            R1 = plant.stepReachStar(X0,U);
            X0 = R1(end);
            ppp = X0.affineMap(map_mat,[]);
            Uin = U_fix.concatenate(ppp);
            Rc = net.reach(Uin);
            [row, col, value] = find(Uin.C);
            Uin_C = [row,col,value];
            Rc_V = cell(length(Rc),1);
            Rc_C = cell(length(Rc),1);
            Rc_d = cell(length(Rc),1);
            for Rc_i = 1:length(Rc)
                Rc_V{Rc_i} = Rc(Rc_i).V;
                [row, col, value] = find(Rc(Rc_i).C);
                Rc_C{Rc_i} = [row,col,value];
                Rc_d{Rc_i} = Rc(Rc_i).d;
            end
            new_data = jsonencode(struct('Uin_V',Uin.V,'Uin_C',Uin_C,'Uin_d',Uin.d));
            fileID = fopen(strcat(file_path,'Uin_',num2str(idx+tmp2),'.json'), 'w');
            fwrite(fileID, new_data);
            fclose(fileID);
            new_data = jsonencode(struct('Rc_V',Rc_V,'Rc_C',Rc_C,'Rc_d',Rc_d));
            fileID = fopen(strcat(file_path,'Rc_',num2str(idx+tmp2),'.json'), 'w');
            fwrite(fileID, new_data);
            fclose(fileID);
        end

        if mod(tmp,100) == 0
            tmp
        end
    end
end