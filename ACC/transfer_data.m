function transfer_data()
    idx = 1;
    while 1
        file_path = '/home/linxuan/nnv-master/code/nnv/examples/NNV2.0/Submission/CAV2023/NeuralODEs/ACC/train_data_critical/';
        file_path2 = '/home/linxuan/nnv-master/code/nnv/examples/NNV2.0/Submission/CAV2023/NeuralODEs/ACC/train_data_critical_json/';
        data = load(strcat(file_path,num2str(idx),'.mat'));
        Rc_V = cell(length(data.Rc),1);
        for i = 1:length(data.Rc)
            Rc_V{i} = data.Rc(i).V;
        end
        new_data = jsonencode(struct('Uin',data.Uin.V,'Rc',Rc_V));
        fileID = fopen(strcat(file_path2,num2str(idx),'.json'), 'w');
        fwrite(fileID, new_data);
        fclose(fileID);
        idx = idx + 1
    end
end