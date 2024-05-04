function generate_train_set()
    
    % Reachability analysis of Adaptive Cruise Control with a neuralODE (nonlinear) as a plant model

    % safety specification: relative distance > safe distance
    % dis = x_lead - x_ego  
    % safe distance between two cars, see here 
    % https://www.mathworks.com/help/mpc/examples/design-an-adaptive-cruise-control-system-using-model-predictive-control.html
    % dis_safe = D_default + t_gap * v_ego;

    %parpool('Processes',60);
    % Load objects
    %pause(3600)
    rng(1); % random seed
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

    v_ego_range = [20, 40];
    d1 = 0.5;
    d_x_range = [2, 300];
    d2 = 0.5;
    d_v_range = [-20, 20];
    d3 = 0.5;
    %idx = 1;
    file_path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data2_append_critical_json/';
    for i = v_ego_range(1):d1:v_ego_range(2)
        for j = d_x_range(1):d2:d_x_range(2)
            parfor tmp = 1:round((d_v_range(2)-d_v_range(1))/d3)+1
                k = d_v_range(1)+d3*(tmp-1);
                %k = d_v_range(1):d3:d_v_range(2)
                idx = 255717+round((i-20)/d1*(36.5+i))*(40*2+1)+round((j-2)/d2)*(40*2+1)+round((k+20)/d3)+1;
                %idx = round((i-20)/d1*(298*2+2-36.5-i))*(40*2+1)+round((j-(D_default+t_gap*i))/d2)*(40*2+1)+round((k+20)/d3)+1;
                %if idx <= 875000
                %   continue
                %end
                if j - (D_default + t_gap * i) <= 0
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
                    fileID = fopen(strcat(file_path,'Uin_',num2str(idx),'.json'), 'w');
                    fwrite(fileID, new_data);
                    fclose(fileID);
                    new_data = jsonencode(struct('Rc_V',Rc_V,'Rc_C',Rc_C,'Rc_d',Rc_d));
                    fileID = fopen(strcat(file_path,'Rc_',num2str(idx),'.json'), 'w');
                    fwrite(fileID, new_data);
                    fclose(fileID);

                    %idx = idx + 1;
                    if mod(idx,1000) == 0
                        idx,i,j,k
                    end
                end
            end
        end
    end

end