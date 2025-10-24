function [InData, Class, Target, train_ind, test_ind] =  ExtractInfo4Training(basepath, sbjs_folder, P)
    filedir = fullfile(basepath,sbjs_folder);
    load(filedir);
    InData = DataBalanced.InData;
    Class = DataBalanced.Class;
    Target = DataBalanced.Target;
    prova = DataBalanced.prova;      
    [m,n] = size(InData);
    idx = randperm(n);
    train_ind = idx(1:round(P*n));
    test_ind = idx(round(P*n)+1:end);
end