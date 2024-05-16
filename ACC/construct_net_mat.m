function construct_net_mat()
    structure = [5,20,20,20,5];
    W = {};
    b = {};
    act_fcns = '';
    for i = 1:length(structure)-1
        W{i} = rand(structure(i+1),structure(i))*2-1;
        b{i} = rand(structure(i+1),1)*2-1;
        if i == length(structure)-1
            act_fcns(i,:) = 'linear';
        else
            act_fcns(i,:) = 'relu  ';
        end
    end
    save("controller_main12.mat","act_fcns","b","W")
end