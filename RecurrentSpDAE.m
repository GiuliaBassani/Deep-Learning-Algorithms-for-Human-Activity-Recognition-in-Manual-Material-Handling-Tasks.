% Script to train and test the Sp_DAE network with 70-30 split mode
% This script can run only after having trained the autoencoder you want to use in the Recurrent Sp-DAE

% Set the output directory (outputdir)
% Set how many network to train and test (numnet)
% Set the autecoder directory (SpDAEdir)
% Set the autoencoder you want to load (SpDAE)
% Set the hidden units of the BiLSTM (numHU)
% Set the number of epochs of the BiLSTM (maxE)


outputdir = './Networks/RecurrentSpDAE';
SpDAEdir = './Networks/SpDAE';    % SpDAE directory
SpDAE = 'SpDAEName.mat';          % SpDAE to load

miniBatchSize = 128;

maxE = 300;                       % Number of epochs of the BiLSTM
numHU = 1000;                     % Hidden units of the BiLSTM

numnet = 1;                       % How many network you want to train
     
% Load the selected autoencoder
clear autoenc_vars
autoenc_vars = load(fullfile(SpDAEdir, SpDAE)); 
% Number of subjects
und = 0;
for ii = 1:size(autoenc_vars.sbjs_folder,2)
    if autoenc_vars.sbjs_folder(ii) == '_'
        und = und + 1;
    end
end

% Training data
TrainData = autoenc_vars.InData(:,autoenc_vars.train_ind)';
TrainClass = autoenc_vars.Class(:,autoenc_vars.train_ind);
TrainClass = TrainClass';
TrainClass_cat = categorical(TrainClass);
% Testing data
TestData = autoenc_vars.InData(:,autoenc_vars.test_ind)';
TestClass = autoenc_vars.Class(:,autoenc_vars.test_ind);
TestClass = TestClass';
TestClass_cat = categorical(TestClass);

for kkk = 1:numnet
    for i = 1:size(numHU,2)
        for j = 1:size(maxE,2)
            numHiddenUnits = numHU(1,i);        % Number of hidden units
            maxEpochs = maxE(1,j);
            numClasses = max(autoenc_vars.Class);            % Number of classes
            numfeat = size(autoenc_vars.feat1,1);            % Number of features extracted from the autoencoder    
            
            Bi = ['B_' num2str(numHiddenUnits) 'HU_' num2str(maxEpochs) 'Ep'];
            SpDAE_u = strfind(SpDAE,'_');          
            subj_fld_name = ['RecurrentSpDAE_' num2str(und) '_A_' SpDAE(SpDAE_u(2)+1:SpDAE_u(4)-1) '_' num2str(Bi) '_' num2str(kkk)]
            subj_fld_name = fullfile(outputdir,subj_fld_name);
    
            layers = [ ...
                featureInputLayer(numfeat,'Name','features')
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
    
            feat1 = autoenc_vars.feat1';
            tic
            lstmnet = trainNetwork(feat1, TrainClass_cat, layers, options);
            lstmnettime = toc/60
    % Evaluate the performances  
            feat_T = encode(autoenc_vars.autoenc1,TestData);
            feat_T = feat_T';
    
            TestT = autoenc_vars.Target(:,autoenc_vars.test_ind);
            TestOut = predict(lstmnet,feat_T);
            TestOut = TestOut';
            % Calculate the predicted classes
            TestClass_cat_Pred = classify(lstmnet,feat_T, 'MiniBatchSize',miniBatchSize);
            TrainClass_cat_Pred = classify(lstmnet,feat1, 'MiniBatchSize',miniBatchSize);
            % Plot confusion matrix
            figure; plotconfusion(TestClass_cat,TestClass_cat_Pred,'TestData',TrainClass_cat,TrainClass_cat_Pred,'TrainData')
            
            [c,cm,ind,per] = confusion(TestT,TestOut);
            Overall_accuracy = 100 - c*100 % True Positive Rate
            for i = 1:size(cm,2)
                Precisions(i,1) = cm(i,i) / sum(cm(:,i));    % Precision
                Recalls(i,1) = cm(i,i) / sum(cm(i,:));       % Recall
                F1scores(i,1) = 2*Precisions(i,1) * Recalls(i,1) / (Precisions(i,1) + Recalls(i,1)); % F1-score: The best value for f1 score is 1 and the worst is 0.
            end
            Precision = mean(Precisions)*100
            Recall = mean(Recalls)*100
            F1score = mean(F1scores)*100
            % Receiver Operating Characteristic
            [tpr,fpr,thresholds] = roc(TestT,TestOut); %figure; plotroc(TestT,TestOut)
            for i = 1:size(fpr,2)
                AUC(i) = trapz(cell2mat(fpr(1,i)),cell2mat(tpr(1,i)));
            end
            AUCm = mean(AUC)*100

            sbjs_folder = autoenc_vars.sbjs_folder; 
            InData = autoenc_vars.InData;
            Class = autoenc_vars.Class;
            Target = autoenc_vars.Target;
            train_ind = autoenc_vars.train_ind;
            test_ind = autoenc_vars.test_ind;
            autoenc1 = autoenc_vars.autoenc1;
                                    
            save((subj_fld_name),'SpDAE','miniBatchSize','sbjs_folder','InData','Class','Target','train_ind','test_ind','numHiddenUnits','maxEpochs', ...
                                'numClasses','numfeat','subj_fld_name','layers','options','feat1','lstmnet','lstmnettime','autoenc1', ...
                                'Overall_accuracy','F1score','Precision','Recall','AUCm')      
        end
    end
end
