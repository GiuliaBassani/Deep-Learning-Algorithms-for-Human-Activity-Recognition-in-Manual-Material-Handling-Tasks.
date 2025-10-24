% Script to train and test the RCNN network with LOSO mode

% Set the right input folder (inputdir)
% Set the output folder (outputdir)
% Set the RCNN directory (blstmdir)
% Set the RCNN name you want to load (rcnnnet)
% Set the test subject file (sbj)

inputdir = './Input';
outputdir = './Networks/RCNN_LOSO';
blstmdir = './Networks/RCNN';
rcnnnet = "RCNNname.mat"
sbj = '11.mat';

% Load trained BiLSTM network
RCNN = load(fullfile(blstmdir, rcnnnet));
% Load test data
sbjfile = fullfile(inputdir,sbj);
[TestData, TestClass, TestClass_cat, TestT] = loadTestData(sbjfile);

RCNN_u = strfind(RCNN.subj_fld_name,'_');  
und = RCNN.subj_fld_name(RCNN_u(1)+1:RCNN_u(2)-1);

TestOut = predict(RCNN.cnnnet,TestData);
TestOut = TestOut';
N = size(TestOut,2);
% Calculate the predicted classes
TestClass_cat_Pred = classify(RCNN.cnnnet,TestData);

TrainData = RCNN.InData(:,RCNN.train_ind); 
TrainData = TrainData';
TrainClass = RCNN.Class(:,RCNN.train_ind);
TrainClass = TrainClass';
TrainClass_cat = categorical(TrainClass);

TrainClass_cat_Pred = classify(RCNN.cnnnet,TrainData);
% Plot confusion matrix
figure; plotconfusion(TestClass_cat,TestClass_cat_Pred,'TestData',TrainClass_cat,TrainClass_cat_Pred,'TrainData')
% Calculate confusion metrics 

[c,cm,ind,per] = confusion(TestT,TestOut);
Overall_accuracy = 100 - c*100; % True Positive Rate
for ii = 1:size(cm,2)
    Precisions(ii,1) = cm(ii,ii) / sum(cm(:,ii));    % Precision
    Recalls(ii,1) = cm(ii,ii) / sum(cm(ii,:));       % Recall
    F1scores(ii,1) = 2*Precisions(ii,1) * Recalls(ii,1) / (Precisions(ii,1) + Recalls(ii,1)); % F1-score: The best value for f1 score is 1 and the worst is 0.
    FN(ii,1) = cm(ii,ii) / Recalls(ii,1) - cm(ii,ii);
    TN(ii,1) = N - (sum(cm(:,ii)) + FN(ii,1));
    Accuracies(ii,1) = (cm(ii,ii) + TN(ii,1)) / N;    % Accuracies
end
Precision = mean(Precisions)*100
Recall = mean(Recalls)*100
F1score = mean(F1scores)*100

for ii = 1:size(F1scores,1)
    switch ii
        case 1
            F1score_field_name = 'F1score_N';
        case 2
            F1score_field_name = 'F1score_So';
        case 3
            F1score_field_name = 'F1score_Ao';
        case 4
            F1score_field_name = 'F1score_Sa';
        case 5
            F1score_field_name = 'F1score_Aa';
        case 6
            F1score_field_name = 'F1score_Mo';
        case 7
            F1score_field_name = 'F1score_Ca';
    end
    F1scores_classes.(F1score_field_name) = F1scores(ii)*100;
end
for ii = 1:size(Accuracies,1)
    switch ii
        case 1
            Accuracy_field_name = 'Accuracy_N';
        case 2
            Accuracy_field_name = 'Accuracy_So';
        case 3
            Accuracy_field_name = 'Accuracy_Ao';
        case 4
            Accuracy_field_name = 'Accuracy_Sa';
        case 5
            Accuracy_field_name = 'Accuracy_Aa';
        case 6
            Accuracy_field_name = 'Accuracy_Mo';
        case 7
            Accuracy_field_name = 'Accuracy_Ca';
    end
    Accuracies_classes.(Accuracy_field_name) = Accuracies(ii)*100;
end

for ii = 1:size(Precisions,1)
    switch ii
        case 1
            Precision_field_name = 'Precision_N';
        case 2
            Precision_field_name = 'Precision_So';
        case 3
            Precision_field_name = 'Precision_Ao';
        case 4
            Precision_field_name = 'Precision_Sa';
        case 5
            Precision_field_name = 'Precision_Aa';
        case 6
            Precision_field_name = 'Precision_Mo';
        case 7
            Precision_field_name = 'Precision_Ca';
    end
    Precisions_classes.(Precision_field_name) = Precisions(ii)*100;
end

[tpr,fpr,thresholds] = roc(TestT,TestOut); %figure; plotroc(TestT,TestOut)
for ii = 1:size(fpr,2)
    AUC(ii) = trapz(cell2mat(fpr(1,ii)),cell2mat(tpr(1,ii)));
end
AUCs_classes.AUCm_diffSubj = mean(AUC)*100;
% Save  
sbj_p = strfind(sbj,'.');
subj_fld_name = [RCNN.subj_fld_name, sbj(1:sbj_p(1)-1)]
save((subj_fld_name),'inputdir','F1scores_classes', ...
                       'Overall_accuracy','F1score','Precision','Recall','AUCm','F1scores_classes', 'Accuracies_classes', 'Precisions_classes', 'rcnnnet', 'subj_fld_name')      
