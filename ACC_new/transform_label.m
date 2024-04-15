function transform_label()

    origin_file_path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data_uncritical_json/';
    %file_path = '/mnt/mnt1/linxuan/nnv/ACC/new_train_data_critical_json/';
    parpool('Processes',64);
    parfor idx = 1:1730241
        idx
        all_data = jsondecode(fileread(strcat(origin_file_path,'Rc_',num2str(idx),'.json')));
        x_lower_bound = 100;
        x_upper_bound = -100;
        for i = 1:length(all_data)
            V = all_data(i).Rc_V';
            d = all_data(i).Rc_d;
            C_value = all_data(i).Rc_C;
            C = full(sparse(C_value(:,1),C_value(:,2),C_value(:,3)));
            set = Star(V,C,d);
            [xmin, xmax] = set.getRange(1);
            if xmin < x_lower_bound
                x_lower_bound = xmin;
            end
            if xmax > x_upper_bound
                x_upper_bound = xmax;
            end
            new_data = jsonencode([(x_upper_bound+x_lower_bound)/2,(x_upper_bound-x_lower_bound)/2]);
            fileID = fopen(strcat(origin_file_path,'Rc_convex_',num2str(idx),'.json'), 'w');
            fwrite(fileID, new_data);
            fclose(fileID);
        end
        %new_set = Star.get_convex_hull(new_set);
        %disp(new_set)
    end
end