clear all
clc
close all
addpath('/media/forAll/Processed_Emotions/ForDataSetArticle/eeglab2019_2')

ChlInform = [{'1','R'};{'2','R'};{'3','R'};{'4','R'};{'5','R'};{'6','C'};{'7','L'};{'8','R'};{'9','R'};{'10','R'};{'11','C'};{'12','L'};{'13','L'};{...
'14','R'};{'15','C'};{'16','C'};{'17','C'};{'18','L'};{'19','L'};{'20','L'};{'21','L'};{'22','L'};{'23','L'};{'24','L'};{'25','L'};{'26','L'};{'27','L'};{...
'28','L'};{'29','L'};{'30','L'};{'31','L'};{'32','L'};{'33','L'};{'34','L'};{'35','L'};{'36','L'};{'37','L'};{'38','L'};{'39','L'};{'40','L'};{'41','L'};{...
'42','L'};{'43','L'};{'44','L'};{'45','L'};{'46','L'};{'47','L'};{'48','L'};{'49','L'};{'50','L'};{'51','L'};{'52','L'};{'53','L'};{'54','L'};{'55','C'};{...
'56','L'};{'57','L'};{'58','L'};{'59','L'};{'60','L'};{'61','L'};{'62','C'};{'63','L'};{'64','L'};{'65','L'};{'66','L'};{'67','L'};{'68','C'};{'69','L'};{...
'70','L'};{'71','L'};{'72','C'};{'73','L'};{'74','L'};{'75','C'};{'76','R'};{'77','R'};{'78','R'};{'79','R'};{'80','R'};{'81','R'};{'82','C'};{'83','R'};{...
'84','R'};{'85','R'};{'86','R'};{'87','R'};{'88','R'};{'89','R'};{'90','R'};{'91','R'};{'92','R'};{'93','R'};{'94','R'};{'95','R'};{'96','R'};{'97','R'};{...
'98','R'};{'99','R'};{'100','R'};{'101','R'};{'102','R'};{'103','R'};{'104','R'};{'105','R'};{'106','R'};{'107','R'};{'108','R'};{'109','R'};{'110','R'};{...
'111','R'};{'112','R'};{'113','R'};{'114','R'};{'115','R'};{'116','R'};{'117','R'};{'118','R'};{'119','R'};{'120','R'};{'121','R'};{'122','R'};{...
'123','R'};{'124','R'};{'125','R'};{'126','R'};{'127','L'};{'128','L'}];
%load('EEG_test_v2.mat'); %load the sample dataset which is is located in the parent folder
cd('/media/forAll/Processed_Emotions/ForDataSetArticle/')
electData = importdata('10-10ToHydroCel.csv', ',',1);
for i = 2 : length(electData.data)+1
    electData.textdata{i, 2} = electData.data(i-1);
end

old_ = 0;
arr_ = [];
for i = 2 : length(electData.data)+1    
    new_ = electData.textdata{i, 2};
    if new_ - old_ > 1
        arr_ = [arr_, (old_+1:new_-1)];
    end
    old_ = new_;
end

nameElct = length(electData.textdata);
for i = 1 : length(arr_)
    electData.textdata{nameElct+i, 1} = strcat('E',num2str(arr_(i)));
    electData.textdata{nameElct+i, 2} = arr_(i);
end
%py.importlib.import_module('matchingEmotion')
rootFolder = pwd;
sourceFolder = fullfile(rootFolder);
dataFolder = '/media/forAll/Processed_Emotions/ToTransfer_Zuddler/ForMSAAnalysis/AllBands/EEGConnAnalysisEmt_Base';
%sourceFolder = fullfile(rootFolder,'PreProcessed');
cd(sourceFolder)
eeglab

EEG = pop_mffimport(fullfile(sourceFolder, 'mit003_bio_20191201_051113.mff'),{'type'});
EEG = eeg_checkset( EEG );
EEG = pop_select( EEG, 'channel',{'E1' 'E2' 'E3' 'E4' 'E5' 'E6' 'E7' 'E8' 'E9' 'E10' 'E11' 'E12' 'E13' 'E14' 'E15' 'E16' 'E17' 'E18' 'E19' 'E20' 'E21' 'E22' 'E23' 'E24' 'E25' 'E26' 'E27' 'E28' 'E29' 'E30' 'E31' 'E32' 'E33' 'E34' 'E35' 'E36' 'E37' 'E38' 'E39' 'E40' 'E41' 'E42' 'E43' 'E44' 'E45' 'E46' 'E47' 'E48' 'E49' 'E50' 'E51' 'E52' 'E53' 'E54' 'E55' 'E56' 'E57' 'E58' 'E59' 'E60' 'E61' 'E62' 'E63' 'E64' 'E65' 'E66' 'E67' 'E68' 'E69' 'E70' 'E71' 'E72' 'E73' 'E74' 'E75' 'E76' 'E77' 'E78' 'E79' 'E80' 'E81' 'E82' 'E83' 'E84' 'E85' 'E86' 'E87' 'E88' 'E89' 'E90' 'E91' 'E92' 'E93' 'E94' 'E95' 'E96' 'E97' 'E98' 'E99' 'E100' 'E101' 'E102' 'E103' 'E104' 'E105' 'E106' 'E107' 'E108' 'E109' 'E110' 'E111' 'E112' 'E113' 'E114' 'E115' 'E116' 'E117' 'E118' 'E119' 'E120' 'E121' 'E122' 'E123' 'E124' 'E125' 'E126' 'E127' 'E128'});
EEG = eeg_checkset( EEG );

EEG.FC.parameters.metric = 'fcmetric_PLI';
EEG.FC.parameters.bands = [[{'[0.5 4]'}, {'Band'}]; [{'[4 8]'}, {'Band'}]; [{'[8 12]'}, {'Band'}]...
    ; [{'[13 20]'}, {'Band'}]; [{'[20 30]'}, {'Band'}]; [{'[30 40]'}, {'Band'}]];
Groups = dir(fullfile(dataFolder, 'Group*'));
frqInf = {'Delta', 'Theta', 'Alpha', 'LBeta', 'UBeta', 'Gamma'};

for grp_= 1 : length(Groups)
    leftArr = [];
    centerArr = [];
    rightArr = [];
    
    fullPath = fullfile(Groups(grp_).folder, Groups(grp_).name);
    for frq_ = 1 : 6
        for segId = 0 : 8        
            fileName = fullfile(fullPath, strcat('New_EEGConnectomePlots_phaseRandomConnAnalEEG_ConnectedElectrodes_',frqInf{frq_},'_','seg-',num2str(segId),'.PNG'));
            %if ~ isfile(fileName)
            if isfile(fileName)
                if isfile(fullfile(fullPath, strcat('phaseRandomConnAnalEEG_ConnectedElectrodes_',frqInf{frq_},'_','seg-',num2str(segId),'.mat')))
                    load(fullfile(fullPath, strcat('phaseRandomConnAnalEEG_ConnectedElectrodes_',frqInf{frq_},'_','seg-',num2str(segId),'.mat')))
                    load(fullfile(fullPath, strcat('phaseRandomConnAnalEEG_emotion_analysis_orig_tstatistics_',frqInf{frq_},'_','seg-',num2str(segId),'.mat')))

                    if ~ length(conn_Remains)
                        continue
                    end
                    %if size(conn_Remains, 2) == 1
                    %    conn_Remains = [conn_Remains(1,1), conn_Remains(2,1)];
                    %end
                    allNodes = [conn_Remains(1,:)+1, conn_Remains(2,:)+1];
                    allNodes = unique(allNodes);
                    laterality = [];
                    left = 0;
                    center = 0;
                    right  = 0;
                    for nd_ = 1 : length(allNodes)
                        if str2num(ChlInform{allNodes(nd_), 1}) ~= allNodes(nd_)
                            disp('stop here')
                        end
                        if ChlInform{allNodes(nd_), 2} == 'L'
                            left = left + 1;
                        elseif ChlInform{allNodes(nd_), 2} == 'C'
                            center = center + 1;
                        elseif ChlInform{allNodes(nd_), 2} == 'R'
                            right = right + 1;
                        end
                    end
                   
                    leftArr = [leftArr, left];
                    centerArr = [centerArr, center];
                    rightArr = [rightArr, right];
                    %continue
                    
                    if size(conn_Remains,2) ~= 1                
                        ConnAdjMat = zeros(128, 128);
                        for i = 1 : length(conn_Remains)
                            disp([conn_Remains(1,i)+1, conn_Remains(2,i)+1])
                            disp(origLabeling(conn_Remains(1,i)+1, conn_Remains(2,i)+1))
                            ConnAdjMat(conn_Remains(1, i)+1, conn_Remains(2, i)+1) = origLabeling(conn_Remains(1,i)+1, conn_Remains(2,i)+1);
                            %ConnAdjMat(conn_Remains(1, i)+1, conn_Remains(2, i)+1) = 1; % +1 because python indexing starts from 0                
                        end 
                        ConnAdjMat = ConnAdjMat + ConnAdjMat';

                        EEG.FC.PLI.Band.adj_matrix = ConnAdjMat;            
                        eeglab_path=which('eeglab');
                        eeglab_path=strrep(eeglab_path,'eeglab.m','');
                        fieldnames = fields(EEG.FC);
                        metrics_file = dir([eeglab_path 'plugins/FCLAB1.0.0/FC_metrics/fcmetric_*.m']);

                        for i = 1:length(metrics_file)
                            measure_full = metrics_file(i,:).name;
                            fcmetrics{i} = measure_full(10:end-2);
                        end

                        fieldnames = intersect(fields(EEG.FC), fcmetrics);            

                        fieldnames_freq=fields(EEG.FC.(fieldnames{1}));
                        if ~isempty(EEG.chanlocs)
                            ds.chanPairs=[];
                            ds.connectStrength=[];

                            for i=1:EEG.nbchan-1
                                for j=i+1:EEG.nbchan
                                    ds.chanPairs=[ds.chanPairs; i j];
                                    ds.connectStrength=[ds.connectStrength...
                                        EEG.FC.(fieldnames{1}).(fieldnames_freq{1}).adj_matrix(i,j)];
                                end
                            end

                            %OldRange = (max(ds.connectStrength) - min(ds.connectStrength))  ;
                            %NewRange = (1 - 0)  ;
                            %ds.connectStrength = (((ds.connectStrength - min(ds.connectStrength)) * NewRange) / OldRange) + 0;

                            ds.connPairs = conn_Remains;
                            ds.labels = electData.textdata;
                            figure();
                            title(frqInf{frq_}); 
                            sprintf("%s %s %d ", Groups(grp_).name, frqInf{frq_}, segId)
                            topoplot_connect(ds, EEG.chanlocs, 'hot');    
                            minStatsVal = round(min(ds.connectStrength(ds.connectStrength>0)), 2);
                            annotation('textbox', [0.47, 0.75, 0.1, 0.1], 'String', num2str(minStatsVal),'FitBoxToText','on', 'LineStyle', 'none', 'FontSize', 23)
                            %annotation('textbox', [0.47, 0.71, 0.1, 0.1], 'String', num2str(left),'FitBoxToText','on', 'LineStyle', 'none', 'FontSize', 17)
                            %annotation('textbox', [0.5, 0.71, 0.1, 0.1], 'String', num2str(center),'FitBoxToText','on', 'LineStyle', 'none', 'FontSize', 17)
                            %annotation('textbox', [0.53, 0.71, 0.1, 0.1], 'String', num2str(right),'FitBoxToText','on', 'LineStyle', 'none', 'FontSize', 17)
                            print(gcf, fileName, '-dpng','-r600');
                            close all
                        end
                     end
                end
            end
        end
    end
    %[Lp, Lh, Lstats] = signrank(leftArr, rightArr, 'tail', 'left');
    %[Rp, Rh, Rstats] = signrank(leftArr, rightArr, 'tail', 'right');
    %leftTail = [Lp, Lh, Lstats.zval]
    %rightTail = [Rp, Rh, Rstats.zval]
    %save(fullfile(dataFolder, strcat(Groups(grp_).name, '_LeftTail.mat')), 'leftTail')
    %save(fullfile(dataFolder, strcat(Groups(grp_).name, '_RightTail.mat')), 'rightTail')
end