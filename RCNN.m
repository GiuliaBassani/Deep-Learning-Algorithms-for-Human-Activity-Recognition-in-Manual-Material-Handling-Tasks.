% Script to train and test the RCNN network with 70-30 split mode

% Download the input data from the following link:
% https://alumnisssup-my.sharepoint.com/:f:/g/personal/alessandro_filippeschi_santannapisa_it/ElqXa34kySFLmYiYqYS_b34BpYkvwlWBlHBQVMNfmTQpHA?e=9iFXFc


% Set the right input folder (inputfolder)
% Set the subject's numerosity (sbjs_folder)
% Set the output folder (outputfolder)
% Set how many network to train and test (numnet)

inputfolder = './Input';
outputfolder = './Networks/RCNN';
numnet = 1;

miniBatchSize = 128;
numHU = [100, 300, 500, 700, 900];
maxE = [100, 300, 500, 700, 1000];
NumF = 32;
DilFact = 2;
Str = 4;
Pad = 0;

base = dir(inputfolder);
for I1 = 1:length(base)
    sbjs_folder = base(I1).name;
    sbjs_folder_dir = fullfile(inputfolder,sbjs_folder);
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
            P = 0.70;
            clear InData Class Target train_ind test_ind 
            [InData, Class, Target, train_ind, test_ind] = ExtractInfo4Training(inputfolder, sbjs_folder, P);   
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
                        for k = 1:size(NumF,2)
                            for ii = 1:size(DilFact,2)
                                for jj = 1:size(Str,2)
                                    for kk = 1:size(Pad,2)                                       
                                        NumF_ = NumF(1,k);
                                        DilFact_ = DilFact(1,ii);
                                        Str_ = Str(1,jj);
                                        if DilFact_ ~= 4
                                            Pad_ = Pad(1,kk);
                                            Pad_1 = Pad_;
                                        else
                                            Pad_ = Pad(1,kk);
                                            Pad_1 = 1;
                                        end
                                        
                                        subj_fld_name = ['RCNN_' num2str(und) '_' num2str(numHiddenUnits) 'HU_' num2str(maxEpochs) 'Ep_' ...
                                            num2str(NumF_) '_' num2str(DilFact_) '_' num2str(Str_) '_' num2str(Pad_1) '_' num2str(kkk)] 
                                        subj_fld_name = fullfile(outputfolder,subj_fld_name);
                                        
                                        inputSize = [size(InData{1,1},1) size(InData{1,1},2) 1]; % Number of inputs
                                        numClasses = max(Class);                                 % Number of classes                                    
    
                                        layers = [
                                            sequenceInputLayer(inputSize,'Name','input')
                                            sequenceFoldingLayer('Name','fold')
                                            convolution2dLayer(3,NumF_,'Padding',Pad_1,'Name','conv1')
                                            instanceNormalizationLayer('Name','norm1')
                                            reluLayer('Name','relu1')
                                            convolution2dLayer(3,NumF_,'Padding',Pad_,'DilationFactor',DilFact_,'Stride',[1 Str_],'Name','conv2')
                                            instanceNormalizationLayer('Name','norm2')
                                            reluLayer('Name','relu2')
                                            sequenceUnfoldingLayer('Name','unfold')
                                            flattenLayer('Name','flatten')      
                                            gruLayer(numHiddenUnits,'OutputMode','last','Name','last')
                                            fullyConnectedLayer(numClasses,'Name','fc')
                                            softmaxLayer('Name','soft')
                                            classificationLayer('Name','class')];
    
                                        options = trainingOptions('adam', ... 
                                            'ExecutionEnvironment','gpu', ...
                                            'GradientThreshold',1, ...    
                                            'MaxEpochs',maxEpochs, ... 
                                            'SequenceLength','longest', ...
                                            'Verbose',1, ...
                                            'Shuffle','never', ...
                                            'Plots','training-progress');
    
                                        lgraph = layerGraph(layers);
                                        lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');
                                        % figure
                                        % plot(lgraph)
    
                                        tic
                                        cnnnet = trainNetwork(TrainData,TrainClass_cat,lgraph,options);
                                        cnnnettime = toc/60
    
                                        TestT = Target(:,test_ind);
                                        TestOut = predict(cnnnet,TestData);
                                        TestOut = TestOut';
                                        % Calculate the predicted classes
                                        TestClass_cat_Pred = classify(cnnnet,TestData);
                                        TrainClass_cat_Pred = classify(cnnnet,TrainData);
                                        % Plot confusion matrix
                                        figure; plotconfusion(TestClass_cat,TestClass_cat_Pred,'TestData',TrainClass_cat,TrainClass_cat_Pred,'TrainData')
                                        [c,cm,ind,per] = confusion(TestT,TestOut);
                                        Overall_accuracy = 100 - c*100 % True Positive Rate
                                        for i_ = 1:size(cm,2)
                                            Precisions(i_,1) = cm(i_,i_) / sum(cm(:,i_));    % Precision
                                            Recalls(i_,1) = cm(i_,i_) / sum(cm(i_,:));       % Recall
                                            F1scores(i_,1) = 2*Precisions(i_,1) * Recalls(i_,1) / (Precisions(i_,1) + Recalls(i_,1)); 
                                        end
                                        Precision = mean(Precisions)*100
                                        Recall = mean(Recalls)*100
                                        F1score = mean(F1scores)*100
                                        % Receiver Operating Characteristic
                                        [tpr,fpr,thresholds] = roc(TestT,TestOut); %figure; plotroc(TestT,TestOut)
                                        for iii = 1:size(fpr,2)
                                            AUC(iii) = trapz(cell2mat(fpr(1,iii)),cell2mat(tpr(1,iii)));
                                        end
                                        AUCm = mean(AUC)*100
                                        % Save the network                                                                    
                                        save((subj_fld_name),'inputfolder','sbjs_folder','InData','Class','Target','P','train_ind','test_ind','TestOut','miniBatchSize', ...
                                                    'numHiddenUnits','maxEpochs','inputSize','numClasses','options','layers','cnnnettime','cnnnet','Overall_accuracy','F1score','Precision','Recall','AUCm','subj_fld_name', 'DilFact_', 'lgraph', 'NumF_', 'Pad_', 'Pad_1','Str_') 
                                    end
                                end
                            end
                        end                   
                    end
                end
            end
        end
    end
end     
