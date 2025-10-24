% Script to train and test the Sp_DAE network with 70-30 split mode

% Download the input data from the following link:
% https://alumnisssup-my.sharepoint.com/:f:/g/personal/alessandro_filippeschi_santannapisa_it/ElqXa34kySFLmYiYqYS_b34BpYkvwlWBlHBQVMNfmTQpHA?e=9iFXFc

% Set the right input folder (inputdir)
% Set the subject's numerosity (sbjs_folder)
% Set the output folder (outputdir)
% Set how many network to train and test (numnet)

inputdir = './Input';
outputdir = './Networks/SpDAE';

numHU = [100, 300, 500, 700, 900, 1100, 1300, 2000, 2500, 3000, 3500];
maxE = [100, 300, 500, 700, 1000, 1300, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500];

numnet = 1;

base = dir(inputdir);
for I1 = 1:length(base)
    sbjs_folder = base(I1).name;
    sbjs_folder_dir = [inputdir,filesep,sbjs_folder];
    if exist(sbjs_folder_dir,'dir')
        if sbjs_folder == "." || sbjs_folder == ".."
            continue
        elseif sbjs_folder == "2" || sbjs_folder == "2_3" || sbjs_folder == "2_3_4" || sbjs_folder == "2_3_4_10" || sbjs_folder == "2_3_4_10_9" || sbjs_folder == "2_3_4_10_9_6"  || ...
               sbjs_folder == "2_3_4_10_9_6_7" || sbjs_folder == "2_3_4_10_9_6_7_8" || sbjs_folder == "2_3_4_10_9_6_7_8_5"  || sbjs_folder == "2_3_4_10_9_6_7_8_5_1" || ...
               sbjs_folder == "2_3_4_10_9_6_7_8_5_1_13" || sbjs_folder == "2_3_4_10_9_6_7_8_5_1_13_12"  || sbjs_folder == "2_3_4_10_9_6_7_8_5_1_13_12_14" || sbjs_folder == "2_3_4_10_9_6_7_8_5_1_13_12_14_11"
            sbjs_folder
            und = 0;
            for ii = 1:size(sbjs_folder,2)
                if sbjs_folder(ii) == '_'
                    und = und + 1;
                end
            end
% Balancing the data
            P = 0.70;
            clear InData Class Target train_ind test_ind 
            [InData, Class, Target, train_ind, test_ind] = ExtractInfo4Training(inputdir, sbjs_folder, P);
            clear TrainData TrainClass TrainClass_cat TestData TestClass TestClass_cat  
            inputSize = numel(InData{1,1});
% Training data
            TrainData = InData(:,train_ind); 
            TrainT = Target(:,train_ind);
            % Turn the test data into vectors and put them in a matrix
            TrainDataM = zeros(inputSize,numel(TrainData));
            for i = 1:numel(TrainData)
                TrainDataM(:,i) = TrainData{i}(:);
            end
% Testing data
            TestData = InData(:,test_ind);
            TestT = Target(:,test_ind);
            % Turn the test data into vectors and put them in a matrix
            TestDataM = zeros(inputSize,numel(TestData));
            for i = 1:numel(TestData)
                TestDataM(:,i) = TestData{i}(:);
            end
% Set the parameters and train the networks
            for kkk = 1:numnet
                for i = 1:size(numHU,2)
                    for j = 1:size(maxE,2)
                        numHiddenUnits = numHU(1,i);
                        maxepoch = maxE(1,j);

                        subj_fld_name = ['SpDAE_' num2str(und) '_' num2str(numHiddenUnits) 'HU_' num2str(maxepoch) 'Ep_' num2str(kkk)]
                        subj_fld_name = fullfile(outputdir,subj_fld_name);

                        tic
                        autoenc1 = trainAutoencoder(TrainData, numHiddenUnits, 'MaxEpochs', maxepoch);
                        autoenc1time = toc/60

                        TrainDataR = predict(autoenc1,TrainData);
                        mseError = mse(cell2mat(TrainData) - cell2mat(TrainDataR))
                        TestingAccuracy = sqrt(mseError)                %   Calculate testing accuracy (RMSE)

                        tic
                        feat1 = encode(autoenc1,TrainData);
                        feat1time = toc/60;
                        tic
                        softnet = trainSoftmaxLayer(feat1,TrainT);
                        softnettime = toc/60;
                        stackednet = stack(autoenc1, softnet);
% Evaluate the performances 
                        TestOut = stackednet(TestDataM);
                        TrainOut = stackednet(TrainDataM);
                        % Plot confusion matrix
                        figure; plotconfusion(TestT,TestOut,'TestData',TrainT,TrainOut,'TrainData');
                        % Calculate confusion metrics 
                        [c,cm,ind,per] = confusion(TestT,TestOut);
                        Overall_accuracy = 100 - c*100 % True Positive Rate
                        for ii = 1:size(cm,2)
                            Precisions(ii,1) = cm(ii,ii) / sum(cm(:,ii));    % Precision
                            Recalls(ii,1) = cm(ii,ii) / sum(cm(ii,:));       % Recall
                            F1scores(ii,1) = 2*Precisions(ii,1) * Recalls(ii,1) / (Precisions(ii,1) + Recalls(ii,1)); 
                        end
                        Precision = mean(Precisions)*100;
                        Recall = mean(Recalls)*100;
                        F1score = mean(F1scores)*100
                        % Receiver Operating Characteristic
                        [tpr,fpr,thresholds] = roc(TestT,TestOut); %figure; plotroc(TestT,TestOut)
                        for iii = 1:size(fpr,2)
                            AUC(iii) = trapz(cell2mat(fpr(1,iii)),cell2mat(tpr(1,iii)));
                        end
                        AUCm = mean(AUC)*100;
% Save the network                                 
                        
                        save((subj_fld_name),'inputdir','sbjs_folder','InData','Class','Target','P','train_ind','test_ind', 'TestOut', ...
                            'numHiddenUnits','maxepoch','inputSize','autoenc1','autoenc1time','feat1','Overall_accuracy','F1score','Precision','Recall','AUCm','subj_fld_name','feat1time','softnet','softnettime')
                    end
                end
            end
        end
    end
end

