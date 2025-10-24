% Script to train and test the BiLSTM network with 70-30 split mode

% Download the input data from the following link:
% https://alumnisssup-my.sharepoint.com/:f:/g/personal/alessandro_filippeschi_santannapisa_it/ElqXa34kySFLmYiYqYS_b34BpYkvwlWBlHBQVMNfmTQpHA?e=9iFXFc

% Set the right input folder (inputdir)
% Set the subject's numerosity (sbjs_folder)
% Set the output folder (outputdir)
% Set how many network to train and test (numnet)


inputdir = './Input';
outputdir = './Networks/BiLSTM';
numnet = 1;

miniBatchSize = 128;
numHU = [100, 300, 500, 700, 900];
maxE = [100, 300, 500, 700, 1000, 1500, 2000, 3000, 3500];

base = dir(inputdir);
for I1 = 1:length(base)
    sbjs_folder = base(I1).name;
    sbjs_folder_dir = fullfile(inputdir,sbjs_folder);
    if exist(sbjs_folder_dir,'dir') || exist(sbjs_folder_dir,'file')
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
            P = 0.70;
            clear InData Class Target train_ind test_ind 
            [InData, Class, Target, train_ind, test_ind] = ExtractInfo4Training(inputdir, sbjs_folder, P);            
            clear TrainData TrainClass TrainClass_cat TestData TestClass TestClass_cat        
% Training data
            TrainData = InData(:,train_ind); 
            TrainData = TrainData';
            TrainClass = Class(:,train_ind);
            TrainClass = TrainClass';
            TrainClass_cat = categorical(TrainClass);
% Testing data
            TestData = InData(:,test_ind);
            TestData = TestData';
            TestClass = Class(:,test_ind);
            TestClass = TestClass';
            TestClass_cat = categorical(TestClass);

            for kkk = 1:numnet
                for i = 1:size(numHU,2)
                    numHiddenUnits = numHU(1,i);
                    for j = 1:size(maxE,2)                        
                        maxEpochs = maxE(1,j);
                        inputSize = size(InData{1,1},1);    % Number of inputs
                        numClasses = max(Class);            % Number of classes
                        
                        subj_fld_name = ['bilstm_' num2str(und) '_' num2str(numHiddenUnits) 'HU_' num2str(maxEpochs) 'Ep_' num2str(kkk)] 
                        subj_fld_name = fullfile(outputdir,subj_fld_name);

                        layers = [ ...
                            sequenceInputLayer(inputSize)
                            bilstmLayer(numHiddenUnits,'OutputMode','last')
                            fullyConnectedLayer(numClasses)
                            softmaxLayer
                            classificationLayer];
                        
                        options = trainingOptions('adam', ... 
                            'ExecutionEnvironment','gpu', ...
                            'GradientThreshold',1, ...    
                            'MiniBatchSize',miniBatchSize, ...    
                            'MaxEpochs',maxEpochs, ... 
                            'SequenceLength','longest', ...
                            'Verbose',1, ...
                            'Shuffle','never', ...
                            'Plots','training-progress');

                        tic
                        lstmnet = trainNetwork(TrainData, TrainClass_cat, layers, options);
                        lstmnettime = toc/60
    % Evaluate the performances                
                        TestT = Target(:,test_ind);
                        TestOut = predict(lstmnet,TestData);
                        TestOut = TestOut';
                        % Calculate the predicted classes
                        TestClass_cat_Pred = classify(lstmnet,TestData, 'MiniBatchSize',miniBatchSize);
                        TrainClass_cat_Pred = classify(lstmnet,TrainData, 'MiniBatchSize',miniBatchSize);
                        % Plot confusion matrix
                        figure; plotconfusion(TestClass_cat,TestClass_cat_Pred,'TestData',TrainClass_cat,TrainClass_cat_Pred,'TrainData')
                        [c,cm,ind,per] = confusion(TestT,TestOut);
                        Overall_accuracy = 100 - c*100 % True Positive Rate
                        for iii = 1:size(cm,2)
                            Precisions(iii,1) = cm(iii,iii) / sum(cm(:,iii));    % Precision
                            Recalls(iii,1) = cm(iii,iii) / sum(cm(iii,:));       % Recall
                            F1scores(iii,1) = 2*Precisions(iii,1) * Recalls(iii,1) / (Precisions(iii,1) + Recalls(iii,1)); 
                        end
                        Precision = mean(Precisions)*100
                        Recall = mean(Recalls)*100
                        F1score = mean(F1scores)*100
                        % Receiver Operating Characteristic
                        [tpr,fpr,thresholds] = roc(TestT,TestOut); %figure; plotroc(TestT,TestOut)
                        for iiii = 1:size(fpr,2)
                            AUC(iiii) = trapz(cell2mat(fpr(1,iiii)),cell2mat(tpr(1,iiii)));
                        end
                        AUCm = mean(AUC)*100
                        % Save the network                             
                        save((subj_fld_name),'inputdir','sbjs_folder','InData','Class','Target','P','train_ind','test_ind','TestOut','miniBatchSize', ...
                                                    'numHiddenUnits','maxEpochs','inputSize','numClasses','options','layers','lstmnet','lstmnettime','Overall_accuracy','F1score','Precision','Recall','AUCm','subj_fld_name')      
                    end
                end
            end
        end
    end
end