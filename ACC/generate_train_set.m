function generate_train_set()
    
    % Reachability analysis of Adaptive Cruise Control with a neuralODE (nonlinear) as a plant model

    % safety specification: relative distance > safe distance
    % dis = x_lead - x_ego  
    % safe distance between two cars, see here 
    % https://www.mathworks.com/help/mpc/examples/design-an-adaptive-cruise-control-system-using-model-predictive-control.html
    % dis_safe = D_default + t_gap * v_ego;

    parpool('Processes',64);
    % Load objects
    %pause(3600)
    rng(0); % random seed
    % Load controller
    net = load_NN_from_mat('controller_main.mat');
    net.reachMethod = 'approx-star';

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

    v_ego_range = [20, 40];
    d1 = 0.5;
    d_x_range = [5, 200];
    d2 = 1;
    d_v_range = [-20, 20];
    d3 = 1;
    file_path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data3_uncritical_json/';
    for i = v_ego_range(1):d1:v_ego_range(2)
        for j = d_x_range(1):d2:d_x_range(2)
            parfor tmp = 1:round((d_v_range(2)-d_v_range(1))/d3)+1
                k = d_v_range(1)+d3*(tmp-1);
                %idx = 5*(round((i-20)/d1*((26+6+i)/d2)/2)*(40/d3+1)+round((j-5)/d2)*(40/d3+1)+round((k+20)/d3))+1;
                %idx = 5*(round((i-20)/d1*(50/d2+1)*(40/d3+1))+round((j-5)/d2)*(40/d3+1)+round((k+20)/d3))+1;
                %idx = round((i-20)/d1*(298*2+2-36.5-i))*(40*2+1)+round((j-(D_default+t_gap*i))/d2)*(40*2+1)+round((k+20)/d3)+1;
                idx = 5*(round((i-20)/d1*(295/d2+1))*(40/d3+1)+round((j-5)/d2)*(40/d3+1)+round((k+20)/d3))+1;
                %if idx <= 875000
                %   continue
                %end
                if j - (D_default + t_gap * i) >= 0
                    lb = [300-1*rand();i+k-1*rand();0;300-j;i-1*rand();0;0;-2]; %lower bound
                    ub = [300+1*rand();i+k+1*rand();0;300-j;i+1*rand();0;0;-2]; % upper bound
                    X0 = Star(lb,ub);
                    for tmp_t = 1:5
                        R1 = plant.stepReachStar(X0,U);
                        X0 = R1(end);
                        ppp = X0.affineMap(map_mat,[]);
                        Uin = U_fix.concatenate(ppp);
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
                        Rc = Star(x_lower_bound,x_upper_bound);

                        new_data = jsonencode(full_Rc);
                        fileID = fopen(strcat(file_path,'Rc_',num2str(idx),'.json'), 'w');
                        fwrite(fileID, new_data);
                        fclose(fileID);

                        new_data = jsonencode([(x_upper_bound+x_lower_bound)/2,(x_upper_bound-x_lower_bound)/2]);
                        fileID = fopen(strcat(file_path,'Rc_convex_',num2str(idx),'.json'), 'w');
                        fwrite(fileID, new_data);
                        fclose(fileID);

                        [row, col, value] = find(Uin.C);
                        Uin_C = [row,col,value];
                        new_data = jsonencode(struct('Uin_V',Uin.V,'Uin_C',Uin_C,'Uin_d',Uin.d));
                        fileID = fopen(strcat(file_path,'Uin_',num2str(idx),'.json'), 'w');
                        fwrite(fileID, new_data);
                        fclose(fileID);

                        x08 = X0.affineMap([0 0 0 0 0 0 0 1],[]);
                        X0 = X0.affineMap(C(1:6,:),[]); % Get set for variables 1 to 6
                        X0 = X0.concatenate(Rc); % Add state/input 7 (a_ego)
                        X0 = X0.concatenate(x08);
                        idx = idx + 1;
                    
                        %idx = idx + 1;
                        if mod(idx,1000) == 0
                            idx,i,j,k
                        end
                    end
                end
            end
        end
    end
end