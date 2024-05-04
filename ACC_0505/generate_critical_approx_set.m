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
    net.reachMethod = 'approx-star';

    origin_file_path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data2_critical_json/';
    file_path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data2_critical_approx_json/';
    %parpool('Processes',64);
    parfor idx = 1:255717
        all_data = jsondecode(fileread(strcat(origin_file_path,'Uin_',num2str(idx),'.json')));
        V = all_data.Uin_V;
        d = all_data.Uin_d;
        C_value = all_data.Uin_C;
        C = full(sparse(C_value(:,1),C_value(:,2),C_value(:,3)));
        Uin = Star(V,C,d);
        Uin.predicate_lb = -ones(163,1);
        Uin.predicate_ub = ones(163,1);
        Rc = net.reach(Uin);
        Rc_V = cell(length(Rc),1);
        Rc_C = cell(length(Rc),1);
        Rc_d = cell(length(Rc),1);
        for Rc_i = 1:length(Rc)
            Rc_V{Rc_i} = Rc(Rc_i).V;
            [row, col, value] = find(Rc(Rc_i).C);
            Rc_C{Rc_i} = [row,col,value];
            Rc_d{Rc_i} = Rc(Rc_i).d;
        end
        new_data = jsonencode(struct('Rc_V',Rc_V,'Rc_C',Rc_C,'Rc_d',Rc_d));
        fileID = fopen(strcat(file_path,'Rc_',num2str(idx),'.json'), 'w');
        fwrite(fileID, new_data);
        fclose(fileID);

        if mod(idx,1000) == 0
            idx
        end
    end
end