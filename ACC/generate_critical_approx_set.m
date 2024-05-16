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

    origin_file_path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data3_critical_json/';
    file_path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data3_critical_approx_json/';
    %parpool('Processes',64);
    parfor idx = 1:593528
        all_data = jsondecode(fileread(strcat(origin_file_path,'Uin_',num2str(idx),'.json')));
        V = all_data.Uin_V;
        d = all_data.Uin_d;
        C_value = all_data.Uin_C;
        C = full(sparse(C_value(:,1),C_value(:,2),C_value(:,3)));
        Uin = Star(V,C,d);
        Uin.predicate_lb = -ones(163,1);
        Uin.predicate_ub = ones(163,1);
        Rc = net.reach(Uin);

        x_lower_bound = 100;
        x_upper_bound = -100;
        full_Rc = cell(length(Rc),1);
        for tmp_i = 1:length(Rc)
            set = Rc(tmp_i);
            [xmin, xmax] = set.getRange(1);
            full_Rc{tmp_i} = [(xmin+xmax)/2,(xmax-xmin)/2];
            if xmin < x_lower_bound
                x_lower_bound = xmin;
            end
            if xmax > x_upper_bound
                x_upper_bound = xmax;
            end
        end

        new_data = jsonencode([(x_upper_bound+x_lower_bound)/2,(x_upper_bound-x_lower_bound)/2]);
        fileID = fopen(strcat(file_path,'Rc_convex_',num2str(idx),'.json'), 'w');
        fwrite(fileID, new_data);
        fclose(fileID);

        if mod(idx,1000) == 0
            idx
        end
    end
end