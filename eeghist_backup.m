% EEGLAB history file generated on the 03-Aug-2020
% ------------------------------------------------
{
%pyenv('Version', 'C:\Users\Sudhakar\anaconda3\envs\matlabEnv\python.exe')
addpath('C:\Processed_Emotions\eeglab2019_2_this\')
}

clear all
close all

cd('C:\Processed_Emotions')
%py.importlib.import_module('matchingEmotion')

rootFolder = pwd;
sourceFolder = fullfile(rootFolder,'Done');
%sourceFolder = fullfile(rootFolder,'PreProcessed');
firstPartL = 1;
firstPartH = 600;
secondPartL = 601;
secondPartH = 1200;
eventCounter = 0;
ECGPrefix = strcat(num2str(firstPartL), '-',num2str(firstPartH), '-',num2str(secondPartL), '-',num2str(secondPartH), '_');
if isfile(strcat(ECGPrefix, 'baselineFirstHalfArrLowf_MedianlogPowerNoSquare_1-30-80.mat'))

    cd(sourceFolder)    
AllSubjects_ = dir('mit*');
save(fullfile(rootFolder, 'AllSubjects.mat'), 'AllSubjects_')
ECGFlag = 1;

baselineFirstHalfArrLowf = [];
baselineSecondHalfArrLowf = [];
stimulusFirstHalfArrLowf = [];
stimulusSecondHalfArrLowf = [];

baselineFirstHalfArrMidf = [];
baselineSecondHalfArrMidf = [];
stimulusFirstHalfArrMidf = [];
stimulusSecondHalfArrMidf = [];
AllValence = [];
AllArousal = [];


for SubDIdx = 1 : length(AllSubjects_)
    
    FivePointScale = {''};
    %need to check neutral timing at many points.
    
    eeglab
    global EEG;
    EEG.etc.eeglabvers = '2019.1'; % this tracks which version of EEGLAB is being used, you may ignore it
    EEG = eeg_checkset( EEG );    
    sourceFolder = fullfile(rootFolder,'Done');
    %sourceFolder = fullfile(rootFolder,'PreProcessed');
    subject = AllSubjects_(SubDIdx, 1).name   
    subjectCSVDataDir = fullfile(rootFolder,'csvFiles', subject);

    subjectFolder = fullfile(sourceFolder, subject);
    %targetFolder = fullfile(rootFolder, 'PreProcessedWithPrestimulus');
    targetFolder = fullfile(rootFolder, 'PreProcessedWholeICAThenCreatingEventFiles');
    subTarFolder = fullfile(targetFolder, subject);
    if ~ isdir(subTarFolder)
        mkdir(subTarFolder)
    end
  
    %if ~(isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_FinalProcessed.set'))))
    %if (isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_FinalProcessed.set'))))

        cd(subjectFolder)
        subFileStruct = dir(strcat(subject, '_test*'));
        initials = strsplit(subFileStruct(1,1).name, '.');
        cd(rootFolder)        
        dataFileName = strcat(initials{1,1}, '.mff');
        if ECGFlag == 0
            EEG = pop_mffimport({fullfile(subjectFolder, dataFileName)},{'type'});
            locationRef = EEG.chanlocs;
            display('Always check if locations in EEG.chanlocs and EEG.urchanlocs are same or not. Because there is a bug in eeg_checkchanlocs function. It by default take +X as the nose direction whereas in EGI it is +Y direction.')
            pause(2)
            EEG = eeg_checkset( EEG );
            EEG = pop_select( EEG, 'channel',{'E1' 'E2' 'E3' 'E4' 'E5' 'E6' 'E7' 'E8' 'E9' 'E10' 'E11' 'E12' 'E13' 'E14' 'E15' 'E16' 'E17' 'E18' 'E19' 'E20' 'E21' 'E22' 'E23' 'E24' 'E25' 'E26' 'E27' 'E28' 'E29' 'E30' 'E31' 'E32' 'E33' 'E34' 'E35' 'E36' 'E37' 'E38' 'E39' 'E40' 'E41' 'E42' 'E43' 'E44' 'E45' 'E46' 'E47' 'E48' 'E49' 'E50' 'E51' 'E52' 'E53' 'E54' 'E55' 'E56' 'E57' 'E58' 'E59' 'E60' 'E61' 'E62' 'E63' 'E64' 'E65' 'E66' 'E67' 'E68' 'E69' 'E70' 'E71' 'E72' 'E73' 'E74' 'E75' 'E76' 'E77' 'E78' 'E79' 'E80' 'E81' 'E82' 'E83' 'E84' 'E85' 'E86' 'E87' 'E88' 'E89' 'E90' 'E91' 'E92' 'E93' 'E94' 'E95' 'E96' 'E97' 'E98' 'E99' 'E100' 'E101' 'E102' 'E103' 'E104' 'E105' 'E106' 'E107' 'E108' 'E109' 'E110' 'E111' 'E112' 'E113' 'E114' 'E115' 'E116' 'E117' 'E118' 'E119' 'E120' 'E121' 'E122' 'E123' 'E124' 'E125' 'E126' 'E127' 'E128' 'E129'});
            EEG = eeg_checkset( EEG );

            for idx_ = 1 : length(EEG.urchanlocs)
                EEG.urchanlocs(1, idx_).index_ = idx_;
            end
            originalData = EEG.data;
            origChanLocs = EEG.chanlocs;
            origEvent = EEG.event;

            %%

                % Creating Eventset.
                stimulus = 'yes';
                if strmatch(stimulus,'no') % For Baseline Raw Data
                    event_info = {'base'};
                elseif strmatch(stimulus,'yes') % For trial data
                    fixl = strmatch('fixL', { EEG.event.type });        
                    for indFx = 1 : length(fixl)
                        if strncmp('stm', EEG.event(fixl(indFx)+1).type,3)
                            fixInd = fixl(indFx);
                            break;
                        end
                    end

                    fixInd = fixl(1);
                    while 1                        
                        %if strfind(EEG.event(fixInd).type, 'clic')
                        if strfind(EEG.event(fixInd).type, 'clic')
                            %event_info = {'base', EEG.event(fixInd).type, eventInfo{2}};                            
                            event_info = {'base', EEG.event(fixInd).type, EEG.event(fixl(indFx)+1).type};       
                            break
                        end
                        fixInd = fixInd + 1;
                    end                    
                end  

                stmIdx = strmatch(event_info{1,3},{ EEG.event.type }, 'exact');

                for idx_1 = 1 : length(stmIdx)
                    stmArray(idx_1) = {EEG.event(1, stmIdx(idx_1)).label};
                end
                save(fullfile(subTarFolder, 'stmArray.mat'), 'stmArray')
                py.importlib.import_module('readingEventEmotions')
                py.readingEventEmotions.readingEventEmotions(subjectCSVDataDir, subTarFolder, length(stmArray)-1, subject)
                AllEmotions = load(fullfile(subTarFolder, 'AllEmotions.mat'));        

                clcIdx = strmatch(event_info{1,2},{ EEG.event.type }, 'exact')    
                stmIdx = strmatch(event_info{1,3},{ EEG.event.type }, 'exact')

                for emtIdx_ = 1 : size(AllEmotions.AllEmotions,1)
                    emtDescr = AllEmotions.AllEmotions(emtIdx_, :)
                    temp1Arr = split(emtDescr, 'Trial-');
                    try
                        temp2Arr = split(temp1Arr{2,1}, 'Click-');
                    catch
                        continue;
                    end
                    trialNumber = str2num(temp2Arr{1,1})
                    temp3Arr = split(temp2Arr{2,1}, 'Five');
                    clickNumber = str2num(temp3Arr{1,1})
                    try
                        insrtIdx = stmIdx(trialNumber) + clickNumber + 1
                    catch
                        display('Stop');
                    end
                    EEG.event(1, insrtIdx).emtInfo = emtDescr;
                end

                %{
                newEmtList = {};
                emtDescIdx = 1;
                foundIdx = 0;
                emtCounter = 1;
                %while emtDescIdx <= size(AllEmotions.AllEmotions,1) 
                subStringArr = {};
                while emtDescIdx <= length(clcIdx) 

                    for sIdx = 1 : length(stmIdx)
                        if sIdx == length(stmIdx)
                            index = sIdx;        
                        elseif (clcIdx(emtDescIdx) > stmIdx(sIdx)) && (clcIdx(emtDescIdx) < stmIdx(sIdx+1))        
                            index = sIdx;
                            break;
                        end
                    end

                    subString = strcat('Trial-', num2str(index)); 

                    if ~ length(strmatch(subString, subStringArr))
                        start = emtDescIdx
                        subStringArr = [subStringArr, {subString}];
                    end

                    try
                        mainString = AllEmotions.AllEmotions(emtCounter, :)        
                        for strMatch = 1 : length(mainString)-length(subString)
                            if all(subString==mainString(strMatch:strMatch+length(subString)-1))
                                break;
                            end
                        end

                        if strMatch == length(mainString)-length(subString)
                            newEmtList(emtDescIdx) = {''}  
                            emtDescIdx = emtDescIdx + 1;
                        elseif strMatch < length(mainString)-length(subString)
                            arr_ = split(mainString, 'Click-');
                            arr_ = split(arr_{2,1}, 'Five');
                            temp_C = arr_(1);
                            clickNumber = str2num(temp_C{1,1});
                            %newEmtList(emtDescIdx) = {mainString};
                            newEmtList(start+clickNumber) = {mainString};
                            emtCounter = emtCounter + 1;
                            emtDescIdx = start+clickNumber+1;
                        end       
                    catch
                        if size(AllEmotions.AllEmotions,1) < emtCounter
                            newEmtList(emtDescIdx) = {''};
                            emtDescIdx = emtDescIdx + 1
                            emtCounter = emtCounter + 1
                        end
                    end
                end

                % Special Case to be considered here. If user has entered his own emotion
                % then it is better to remove that click.

                if length(clcIdx) ~= length(newEmtList)
                    error('Please Check As the Number of Clicks Are not Equal to Number of Emotions')
                end

                for idxxx = 1 : length(clcIdx)
                    EEG.event(1, clcIdx(idxxx)).emtInfo = newEmtList{idxxx};
                end    
                %}

            if strmatch('mit108', subject)
                EEG.event(end)=[]
            end
            stmArr = strmatch(event_info{1,3}, {EEG.event.type});
            clcArr = strmatch(event_info{1,2}, {EEG.event.type});

            Nindx_ = strmatch(event_info{3},{ EEG.event.type }, 'exact');
            evtCnt = 1;
            for Nind = 1 : length(Nindx_)
                arr__ = strsplit(EEG.event(Nindx_(Nind)).label, '_');
                %if str2num(arr__{1, end}) > 1
                arr__{1, end} = num2str(evtCnt);
                if length(arr__) > 2
                    EEG.event(Nindx_(Nind)).label = strcat(arr__{1, 1}, '_', arr__{1, 2}, '_', arr__{1, end});
                else
                    EEG.event(Nindx_(Nind)).label = strcat(arr__{1, 1}, '_', arr__{1, end});
                end

                evtCnt = evtCnt + 1;        
            end

            pwd_ = pwd;
            cd(rootFolder)    
            for evtIdx = 1 : length(EEG.event)
                if length(EEG.event(evtIdx).emtInfo)
                    temp1Arr = split(EEG.event(evtIdx).emtInfo, 'Trial-');
                    emotion_ = temp1Arr{1,1}
                    temp2 = split(temp1Arr(2,1), 'Click-');
                    temp3Arr = split(temp2(2,1), 'Five');
                    trialNumber = str2num(temp2{1,1})
                    clickNumber = str2num(temp3Arr{1,1})
                    temp3 = split(EEG.event(stmArr(trialNumber)).label, '_')
                    insrtIdx = stmArr(str2num(temp3{end, 1}))+clickNumber+1

                    if strmatch(EEG.event(evtIdx).emtInfo, EEG.event(insrtIdx).emtInfo)                
                        display(subject)
                        py.importlib.import_module('matchingEmotion')
                        py.matchingEmotion.matchingEmotion(trialNumber, clickNumber, emotion_, subjectCSVDataDir, subject, length(stmArray)-1, rootFolder)

                    else
                        display(subject)
                        error('Events are not matching')
                    end            
                end
            end            
                %%
            if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40.set')))
                EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40.set'),'filepath', subTarFolder);
            else                        
                %Band-pass filter the data with low cut-off frequency equal to 4.0 Hz or 1.0 Hz and high cut-off frequency equal to 40.0 Hz. It is to remove channel drift.        
                EEG = pop_eegfiltnew(EEG, 'locutoff',1,'hicutoff',40,'plotfreqz',1,'chantype',{'EEG'});
                pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40.set'),'filepath', subTarFolder);
            end

            %EEG = pop_loadset('filename','mit003_channelBand1_40.set','filepath','C:\Processed_Emotions\Done\mit003\NewResponses\');
            EEG = eeg_checkset( EEG );
            %pop_eegplotMod( EEG, 1, 1, 1, 'title_', 'After Applying Bandpass filter, Before Extracting Relevent Event');
            %figure; title('After Applying Bandpass filter, Before Extracting Relevent Event');pop_spectopo(EEG, 1, [0  704040], 'EEG' , 'percent', 15, 'freq', [5 10 15 20 30], 'freqrange',[0 40],'electrodes','on');

            %%
            %Extract 3 events: Pre-trial period, the whole stimulus(trial) period, and the subject response marker (mouse clicks) inside the stimulus (trial).

            close all;
            time_range = [-6.1 1.1];
            baselineTime = [10 70];
            fixLStmTime = [-3 0];
            totalTime = time_range(2)-time_range(1);    

            if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40EventExt.set')))
                EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40EventExt.set'),'filepath',subTarFolder);
            else
                cd(rootFolder)
                py.importlib.import_module('gettingTheNeutralTime')
                py.gettingTheNeutralTime.gettingTheNeutralTime(rootFolder, subjectCSVDataDir)
                neutral_ = load(fullfile(subjectCSVDataDir, 'Neutral_2.mat'));
                EEG = pop_rmdat_baseClick( EEG, event_info, time_range , 0, 0, baselineTime, neutral_, fixLStmTime, subject);
                pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40EventExt.set'),'filepath',subTarFolder);

            end
            EEG = eeg_checkset( EEG );

            for evtIdx = 1 : length(EEG.event)
                if length(EEG.event(evtIdx).emtInfo)
                    temp1Arr = split(EEG.event(evtIdx).emtInfo, 'Trial-');
                    emotion_ = temp1Arr{1,1}
                    temp2 = split(temp1Arr(2,1), 'Click-');
                    temp3Arr = split(temp2(2,1), 'Five');
                    trialNumber = str2num(temp2{1,1})
                    clickNumber = str2num(temp3Arr{1,1})

                    display(subject)
                    py.importlib.import_module('matchingEmotion')
                    py.matchingEmotion.matchingEmotion(trialNumber, clickNumber, emotion_, subjectCSVDataDir, subject, length(stmArray)-1, rootFolder)
                end
            end        
            %pop_eegplotMod( EEG, 1, 1, 1, 'title_', 'After Extracting Relevent Event and Before CleanLine');
            %figure; title('After Extracting Relevent Event and Before CleanLine');pop_spectopo(EEG, 1, [0  704040], 'EEG' , 'percent', 15, 'freq', [5 10 15 20 30], 'freqrange',[1 40],'electrodes','on');

            %%
            %Remove the line noise using the CleanLine function.
            close all;
            if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_cleanline.set')))
                EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40_cleanline.set'),'filepath', subTarFolder);
            else
                EEG = pop_cleanline(EEG, 'bandwidth',2,'chanlist',[1:129] ,'computepower',1,'legacy',0,'linefreqs',50,'normSpectrum',0,'p',0.01,'pad',2,'plotfigures',0,'scanforlines',1,'sigtype','Channels','taperbandwidth',2,'tau',100,'verb',1,'winsize',4,'winstep',1);
                p_ = gcp;
                delete(p_);
                EEG = eeg_checkset( EEG );
                pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_cleanline.set'),'filepath',subTarFolder);

            end

            for i = 1 : length(EEG.chanlocs)
                if (EEG.chanlocs(i).X == locationRef(i).X) & (EEG.chanlocs(i).Y == locationRef(i).Y) & (EEG.chanlocs(i).Z == locationRef(i).Z)
                    continue
                else
                    error('something is wrong with the channel locations')
                end
            end
            if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_ManualChanSampleRej.set')))
                EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40_ManualChanSampleRej.set'),'filepath',subTarFolder);
            else
                oldEvent = EEG.event;
                % Since the whole event is being deleted so don't select anysegment
                % from neutral otherwise the whole event will be deleted.
                pop_eegplotMod( EEG, 1, 1, 1, 'title_', 'After Applying CleanLine');
                disp(subject)
                [noTime, noChannel] = rejectData(subject);
                [EEG deletedChl1 dataToDelete1] = pop_select(EEG, 'notime', noTime, 'nochannel', noChannel);
                pop_eegplotMod( EEG, 1, 1, 1, 'title_', 'After Removing Some Data Visually Bad');
                fid=fopen(fullfile(subTarFolder, 'MyFile.txt'),'w') 
                fprintf(fid, 'Deleted Channels Are: \n')
                fprintf(fid, '%d \t', deletedChl1);
                fprintf(fid, '\n')
                fprintf(fid, 'Deleted Time Durations Are: \n')            
                fprintf(fid, '%f \t', [dataToDelete1]');
                fclose(fid);true
                %{
                time_range = [-6 1];
                baselineTime = [10 70];
                fixLStmTime = [-3 0];
                totalTime = time_range(2)-time_range(1);
                neutral_ = [];
                EEG = pop_rmdat_baseClick( EEG, event_info, time_range , 0, 0, baselineTime, neutral_, fixLStmTime);
                %}

                %{
                [EEG deletedChl2 dataToDelete2] = pop_select(EEG);  
                EEG_01 = pop_select(EEG_01, 'nochannel', deletedChl2, 'notime', dataToDelete2);
                EEG = eeg_checkset( EEG );
                EEG_01 = eeg_checkset( EEG_01 );
                totaldelchl = [deletedChl1 deletedChl2];
                totaldeldata = [dataToDelete1 dataToDelete2];
                %}
                EEG.DelChalMan = deletedChl1;
                EEG.DelDatMan = dataToDelete1;
                newEvent = EEG.event;
                EEG.summary.ReducedEvents.BeforeFirstICA = length(newEvent)-length(oldEvent);
                pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_ManualChanSampleRej.set'),'filepath',subTarFolder);

            end
            %pop_eegplotMod( EEG, 1, 1, 1, 'title_', 'After Removing Some Data Visually Bad');
            %pop_eegplotMod( EEG_01, 1, 1, 1, 'title_', 'After Removing Some Data Visually Bad 01');
            %figure; title('After Applying CleanLine and Removing Bad Channels and Data');pop_spectopo(EEG, 1, [0  704040], 'EEG' , 'percent', 15, 'freq', [5 10 15 20 30], 'freqrange',[0 40],'electrodes','on');
            %figure; title('After Applying CleanLine and Removing Bad Channels and Data 01');pop_spectopo(EEG_01, 1, [0  704040], 'EEG' , 'percent', 15, 'freq', [5 10 15 20 30], 'freqrange',[0 40],'electrodes','on');

            %%
            %Eye artifact removal by computing ICA. This is needed before the next step otherwise the Clean Raw Data function in EEGLab removes data points having eye artifacts.
            close all
            oldEvent = EEG.event;
            eventWise = 0;
            if eventWise == 1
                clcIdx2 = strmatch(event_info{1,2},{ EEG.event.type }, 'exact');
                eventInfo2 = {'base'};
                for clc_ = 1 : length(clcIdx2)
                    tmpEmt = split(EEG.event(clcIdx2(clc_)).emtInfo, 'Trial');
                    emtt_ = tmpEmt{1};
                    if length(emtt_)
                        tmpT = split(tmpEmt{2}, 'Click');
                        trial_ = strcat('trial', tmpT{1}, '_');
                    else
                        if strmatch('neutral', EEG.event(clcIdx2(clc_)).label)
                            trial_ = '';
                            emtt_ = 'neutral';
                        else
                            trial_ = '';
                        end
                    end
                    EEG.event(clcIdx2(clc_)).type = strcat(trial_,EEG.event(clcIdx2(clc_)).type, '_', num2str(clc_), '_', emtt_);
                    EEG.event(clcIdx2(clc_)).code = strcat(trial_,EEG.event(clcIdx2(clc_)).code, '_', num2str(clc_), '_', emtt_);
                    eventInfo2(clc_+1) = {EEG.event(clcIdx2(clc_)).type};
                end

                [ALLEEG EEG index] = eeg_store(ALLEEG, EEG);
                for evt_ = 1 : length(eventInfo2)
                    if evt_ == 1
                        EEG = pop_rmdatOrigBase( EEG, {'base'}, [0 60], 0);
                    else
                        EEG = pop_rmdatIndStim( EEG, eventInfo2(evt_), [-6 1], 0);
                    end
                    %[ALLEEG EEG index] = eeg_store(ALLEEG, EEG);

                    if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_EyeArtRemoved_',eventInfo2{evt_},'.set')))
                        EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40_EyeArtRemoved_',eventInfo2{evt_},'.set'),'filepath', subTarFolder);
                    else
                        EEG = eeg_checkset( EEG );
                        EEG = pop_runica(EEG, 'icatype', 'runica', 'chanind',(1:EEG.nbchan));
                        pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_ICA_',eventInfo2{evt_},'.set'),'filepath',subTarFolder);
                        EEG = eeg_checkset( EEG );

                        %pop_runica(finallyCorrected, 'icatype', 'runica', 'chanind',(1:finallyCorrected.nbchan));
                        EEG = pop_iclabel(EEG, 'default');
                        pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_icLabel_',eventInfo2{evt_},'.set'),'filepath',subTarFolder);

                        EEG = eeg_checkset( EEG );
                        max_ = max(EEG.etc.ic_classification.ICLabel.classifications, [], 2);
                        Eyeindx = [];

                        for maxInd = 1 : length(max_)
                            temp_ = find(EEG.etc.ic_classification.ICLabel.classifications(maxInd, :) == max_(maxInd));
                            if (temp_ == 3) && (EEG.etc.ic_classification.ICLabel.classifications(maxInd, 1) < 0.3)         
                                Eyeindx = [Eyeindx maxInd];
                            end
                        end

                        EEG.icaactOrig = EEG.icaact;
                        EEG.icawinvOrig = EEG.icawinv;
                        EEG.icasphereOrig = EEG.icasphere;
                        EEG.icaweightsOrig = EEG.icaweights;
                        EEG.icachansindOrig = EEG.icachansind;

                        Eyeindx
                        EEG = pop_subcomp( EEG, Eyeindx);    

                        EEG.FirstICA.Artifact.Eye = Eyeindx;
                        %pop_viewprops( EEG, 0, [1:size(EEG.icaact,1)], {'freqrange', [2 40]}, {}, 1, 'ICLabel' , 'C:\Processed_Emotions', 'first');
                        % Rechecking for Channel Rejection there.
                        pop_eegplotMod( EEG, 1, 1, 1, 'title_', 'After Removing Eye Components');        
                        %figure; title('After Removing Eye Components');pop_spectopo(EEG, 1, [0  704040], 'EEG' , 'percent', 15, 'freq', [5 10 15 20 30], 'freqrange',[1 40],'electrodes','on');    
                        newEvent = EEG.event;

                        %totaldelchl = [totaldelchl deletedCh3];        
                        %totaldeldata = [totaldeldata dataToDelete3];
                        %EEG.DelChalMan = [EEG.DelChalMan deletedCh3];
                        %EEG.DelDatMan = [EEG.DelDatMan dataToDelete3];

                        newEvent = EEG.event;
                        EEG.summary.ReducedEvents.AfterFirstICAandSubsequentRemovalofBadDataOnceMore = length(oldEvent)-length(newEvent);
                        pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_EyeArtRemoved_',eventInfo2{evt_},'.set'),'filepath', subTarFolder);
                    end

                    EEG.icaact = [];
                    EEG.icawinv = [];
                    EEG.icasphere = [];
                    EEG.icaweights = [];
                    EEG.icachansind = [];
                    EEG.etc.icaweights_beforerms = [];
                    EEG.etc.icasphere_beforerms = [];
                    EEG.etc.ic_classification = [];    
                    %pop_eegplotMod( EEG, 1, 1, 1, 'title_', 'After Removing Eye Components and Reassuring for Bad Channels');
                    %figure; title('After Removing Eye Components and Reassuring for Bad Channels');pop_spectopo(EEG, 1, [0  704040], 'EEG' , 'percent', 15, 'freq', [5 10 15 20 30], 'freqrange',[0 40],'electrodes','on');

                    %%
                    % This is so that we can have all the channel indices together after
                    % the automatic bad channel rejection. Because anyway whatever channels
                    % has been deleted previously are going to be assigned with vector 0
                    % which will be taken care by automatic artifact rejection.
                    if length(EEG.DelChalMan) ~= 0
                        ManDelChl = sort(EEG.DelChalMan);
                        dupData = ones(size(EEG.data, 1)+length(ManDelChl), size(EEG.data, 2));
                        dupData(1:ManDelChl(1)-1, :) = EEG.data(1:ManDelChl(1)-1, :);
                        for chIdx_ = 1 : length(ManDelChl)
                            dupData(ManDelChl(chIdx_), :) = zeros(1, EEG.pnts);
                            %EEG.chanlocs(1, ManDelChl(chIdx_)).labels
                        end

                        chIdx_ = ManDelChl(1)+1;
                        EEGdataCnt = ManDelChl(1);
                        while chIdx_ < size(EEG.data,1)+ length(ManDelChl)
                            if mean(dupData(chIdx_, :)) == 1
                                dupData(chIdx_, :) = EEG.data(EEGdataCnt, :);
                                EEGdataCnt = EEGdataCnt + 1;
                            else
                                display(dupData(chIdx_, :));
                            end
                            chIdx_ = chIdx_ + 1;
                        end
                        EEG.data = dupData;
                        toCheck = 0;
                        for chIdx_ = 1 : length(ManDelChl)
                            toCheck = toCheck + sum(EEG.data(ManDelChl(chIdx_),:));
                            if toCheck == 0
                                disp('New Data Is ok')
                            else
                                error(strcat('There is a problem with the new data. Please check the index-', num2str(ManDelChl(chIdx_))))
                            end                    
                        end

                        EEG.data(end, :) = zeros(1, EEG.pnts);
                        EEG.nbchan = size(EEG.data,1);
                        delete dupData
                        EEG.chanlocs = origChanLocs;
                    end

                    %Automatic channel rejection to reject bad channels. (Cross-verification visually)
                    if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_AfterRejectingBadChannels_',eventInfo2{evt_},'.set')))
                        EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40_AfterRejectingBadChannels_',eventInfo2{evt_},'.set'),'filepath', subTarFolder);
                    else
                        EEG = clean_artifacts(EEG, 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion','off','WindowCriterion','off','BurstRejection','on','Distance','Euclidian');

                        temp = [];
                        delChlCnt = 0;
                        for chIdx = 1 : length(EEG.etc.clean_channel_mask)
                            if EEG.etc.clean_channel_mask(chIdx) == 0
                                delChlCnt = delChlCnt + 1;
                                temp(delChlCnt) = chIdx;
                            end
                        end
                        EEG.deletedChl = temp;
                        temp = sort([temp EEG.DelChalMan]);
                        %EEG.etc.clean_channel_mask = ones(129);
                        temp = unique(temp);
                        for delIdx = 1 : length(temp)

                            %EEG.etc.clean_channel_mask(locationRef(1, temp(delIdx)).index_) = 0;
                            EEG.chaninfo.nodatchans(1, end+1).labels = getfield(locationRef(1, temp(delIdx)), 'labels');
                            EEG.chaninfo.nodatchans(1, end).description = getfield(locationRef(1, temp(delIdx)), 'description');
                            EEG.chaninfo.nodatchans(1, end).X = getfield(locationRef(1, temp(delIdx)), 'X');
                            EEG.chaninfo.nodatchans(1, end).Y = getfield(locationRef(1, temp(delIdx)), 'Y');
                            EEG.chaninfo.nodatchans(1, end).Z = getfield(locationRef(1, temp(delIdx)), 'Z');
                            EEG.chaninfo.nodatchans(1, end).identifier = getfield(locationRef(1, temp(delIdx)), 'identifier');
                            EEG.chaninfo.nodatchans(1, end).type = getfield(locationRef(1, temp(delIdx)), 'type');
                            EEG.chaninfo.nodatchans(1, end).ref = getfield(locationRef(1, temp(delIdx)), 'ref');
                            EEG.chaninfo.nodatchans(1, end).sph_theta = getfield(locationRef(1, temp(delIdx)), 'sph_theta');
                            EEG.chaninfo.nodatchans(1, end).sph_phi = getfield(locationRef(1, temp(delIdx)), 'sph_phi');
                            EEG.chaninfo.nodatchans(1, end).sph_radius = getfield(locationRef(1, temp(delIdx)), 'sph_radius');
                            EEG.chaninfo.nodatchans(1, end).theta = getfield(locationRef(1, temp(delIdx)), 'theta');
                            EEG.chaninfo.nodatchans(1, end).radius = getfield(locationRef(1, temp(delIdx)), 'radius');
                            EEG.chaninfo.nodatchans(1, end).urchan = [];
                            EEG.chaninfo.nodatchans(1, end).datachan = 1;
                        end

                        EEG = eeg_checkset( EEG );
                        pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_AfterRejectingBadChannels_',eventInfo2{evt_},'.set'),'filepath', subTarFolder);
                    end
                    EEG = eeg_checkset( EEG );
                    %pop_eegplotMod( EEG, 1, 1, 1, 'title_', 'After Rejecting Bad Channels');    
                    %figure; title('After Rejectng Bad Channels');pop_spectopo(EEG, 1, [0  704040], 'EEG' , 'percent', 15, 'freq', [5 10 15 20 30], 'freqrange',[0 40],'electrodes','on');

                    %%
                    %Channel Interpolation and Reference the data to average.
                    % To Check whether pop_reref is working properly execute the file
                    % JustForChecking.m

                    close all;
                    if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_AfterRereferencing_',eventInfo2{evt_},'.set')))
                        EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40_AfterRereferencing_',eventInfo2{evt_},'.set'),'filepath', subTarFolder);
                    else
        %% 
        %********Error Caution: Interpolation will not work properly if the location of channels are not corrent. Please check it  thoroughly
                        EEG = pop_interp(EEG, EEG.chaninfo.nodatchans(4:end-1));
                        %EEG = pop_interp(EEG); %Select Channel other than PNS and ref for interpolation here

                        if EEG.nbchan < 129
                            EEG.nbchan = EEG.nbchan+1;
                            EEG.data(end+1,:) = zeros(1, EEG.pnts);
                        end
                        EEG.chanlocs(1,EEG.nbchan).labels = 'initialReference';
                        EEG = pop_reref( EEG, []);
                        EEG = pop_select( EEG,'nochannel',{'initialReference'});

                        %{
                        for delIdx = 1 : length(EEG.deletedChl)-1

                            EEG.chaninfo.nodatchans(1, end+1).labels = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'labels');
                            EEG.chaninfo.nodatchans(1, end).description = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'description');
                            EEG.chaninfo.nodatchans(1, end).X = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'X');
                            EEG.chaninfo.nodatchans(1, end).Y = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'Y');
                            EEG.chaninfo.nodatchans(1, end).Z = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'Z');
                            EEG.chaninfo.nodatchans(1, end).identifier = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'identifier');
                            EEG.chaninfo.nodatchans(1, end).type = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'type');
                            EEG.chaninfo.nodatchans(1, end).ref = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'ref');
                            EEG.chaninfo.nodatchans(1, end).sph_theta = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'sph_theta');
                            EEG.chaninfo.nodatchans(1, end).sph_phi = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'sph_phi');
                            EEG.chaninfo.nodatchans(1, end).sph_radius = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'sph_radius');
                            EEG.chaninfo.nodatchans(1, end).theta = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'theta');
                            EEG.chaninfo.nodatchans(1, end).radius = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'radius');
                            EEG.chaninfo.nodatchans(1, end).urchan = [];
                            EEG.chaninfo.nodatchans(1, end).datachan = 1;    
                            EEG = pop_select( EEG,'nochannel',{locationRef(1, EEG.deletedChl(delIdx)).labels});

                        end
                        %}
                        EEG = eeg_checkset( EEG );
                        pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_AfterRereferencing_',eventInfo2{evt_},'.set'),'filepath', subTarFolder);
                    end
                    %%
                    %Reapplication of ICA on the clean data to find out brain components.
                    %Applying ICLabel on ICA components. Cross checking the components marked “other” and retaining them if the brain component is contributing for more than 30 percent and spectrum and topography looks suspicious of having brain activity.   
                    if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_FinalPCA_',eventInfo2{evt_},'.set')))
                        EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40_FinalPCA_',eventInfo2{evt_},'.set'),'filepath', subTarFolder);
                    else
                        EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',0,'interrupt','on');
                        EEG = eeg_checkset( EEG );
                        EEG = pop_iclabel(EEG, 'default');
                        pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_FinalPCA_',eventInfo2{evt_},'.set'),'filepath', subTarFolder);
                    end

                    if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_FinalProcessed_',eventInfo2{evt_},'.set')))
                        EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40_FinalProcessed_',eventInfo2{evt_},'.set'),'filepath', subTarFolder);
                    else
                        EEG.icaactOrig = EEG.icaact;
                        EEG.icawinvOrig = EEG.icawinv;
                        EEG.icasphereOrig = EEG.icasphere;
                        EEG.icaweightsOrig = EEG.icaweights;
                        EEG.icachansindOrig = EEG.icachansind;

                        max_ = max(EEG.etc.ic_classification.ICLabel.classifications, [], 2);
                        Artifactindx = [];
                        Signalindx = [];

                        for maxInd = 1 : length(max_)
                            temp_ = find(EEG.etc.ic_classification.ICLabel.classifications(maxInd, :) == max_(maxInd));
                            if (temp_ == 7) || (temp_ == 1)
                                if(EEG.etc.ic_classification.ICLabel.classifications(maxInd, 1) > 0.25)         
                                    Signalindx = [Signalindx maxInd];
                                else
                                    Artifactindx = [Artifactindx maxInd];
                                end
                            else
                                Artifactindx = [Artifactindx maxInd];
                            end
                        end

                        EEG.SecondICA.Artifact = Artifactindx;
                        EEG.SecondICA.Signal = Signalindx;

                        %pop_viewprops( EEG, 0, [1:size(EEG.icaact,1)], {'freqrange', [2 40]}, {}, 1, 'ICLabel' , rootFolder, 'first');
                        EEG = eeg_checkset( EEG );
                        % See Variance for all the components For other more than 25 percent
                        EEG = pop_subcomp( EEG, Artifactindx);
                        EEG = eeg_checkset( EEG );
                        %EEG = pop_interp(EEG, EEG.chaninfo.nodatchans(5:end));
                        EEG = eeg_checkset( EEG );
            %{
                        neutralIdx_ = find([EEG.event.urevent] == EEG.neutralUrevent);    
                        stimType = strmatch('STIM', {EEG.event.tracktype});
                        if length(stimType) == 0
                            stimType = strmatch('EVNT', {EEG.event.classid});
                        end
                        EEG.event(stimType(neutralIdx_)).type = 'Neutral';          
                        neutLat = (EEG.event(stimType(neutralIdx_+1)).latency-EEG.event(stimType(neutralIdx_-1)).latency);
                        if neutLat < (3500+(EEG.srate*totalTime)) && (~ all(EEG.event(stimType(neutralIdx_-1)).type == 'bend'))
                            display('neutral Event is not right')
                        end
            %}
                        EEG.summary.FirstICAEyeComp = EEG.FirstICA.Artifact.Eye;
                        EEG.summary.SecondICABrainComp = EEG.SecondICA.Signal;
                        EEG.summary.SecondICAArtComp = EEG.SecondICA.Artifact;
                        %EEG.summary.AutomaticDelSamp = EEG.deletedsample;
                        EEG.summary.AutomaticDelSamp = [];
                        %EEG.summary.AutomaticDelSamp = [];
                        EEG.summary.TotalDelChannel = EEG.deletedChl;
                        EEG.summary.ManDelChannel = EEG.DelChalMan;
                        EEG.summary.numberEvents = strmatch(event_info{1,2}, {EEG.event.type});
                        baseIndex = strmatch('base', {EEG.event.type});
                        bendIndex = strmatch('bend', {EEG.event.type});
                        %EEG.summary.baseDuration = (EEG.event(bendIndex).latency-EEG.event(baseIndex).latency)/EEG.srate;

                        pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_FinalProcessed_',eventInfo2{evt_},'.set'),'filepath', subTarFolder);
                    end
                    dupData = zeros(size(EEG.data,1), size(EEG.data,2));
                    dupLocs = EEG.chanlocs;
                    for chanIdx_ = 1 : length(EEG.chanlocs)   
                        rightChan = strcat('E', num2str(chanIdx_));
                        if strmatch(rightChan, EEG.chanlocs(chanIdx_).labels)
                            dupData(chanIdx_, :) = EEG.data(chanIdx_, :);
                        else
                            tmp_ = EEG.data(chanIdx_, :); % Current index which has to be shifted to the location given in the twente file. the variable dest will have that index.
                            for chanIdx_2 = 1 : length(EEG.chanlocs)
                                another_rightChan = strcat('E', num2str(chanIdx_2));
                                if length(strmatch(another_rightChan, EEG.chanlocs(chanIdx_).labels, 'exact')) % where is the current location in twente.                    
                                    dest1 = chanIdx_2; %location about current channel of data in twente. It will tell me where to swap the current location.
                                    disp(strcat(EEG.chanlocs(chanIdx_).labels, ' is in location-', num2str(chanIdx_), ' in the current EEG data ', ' which has to go-', num2str(dest1), ' in dupData according to the twente'))
                                    dupData(dest1, :) = tmp_;

                                    dupLocs(dest1).type = EEG.chanlocs(chanIdx_).type;
                                    dupLocs(dest1).labels = EEG.chanlocs(chanIdx_).labels;
                                    dupLocs(dest1).description = EEG.chanlocs(chanIdx_).description;
                                    dupLocs(dest1).sph_theta = EEG.chanlocs(chanIdx_).sph_theta;
                                    dupLocs(dest1).sph_phi = EEG.chanlocs(chanIdx_).sph_phi;
                                    dupLocs(dest1).theta = EEG.chanlocs(chanIdx_).theta;
                                    dupLocs(dest1).radius = EEG.chanlocs(chanIdx_).radius;
                                    dupLocs(dest1).sph_radius = EEG.chanlocs(chanIdx_).sph_radius;
                                    dupLocs(dest1).X = EEG.chanlocs(chanIdx_).X;
                                    dupLocs(dest1).Y = EEG.chanlocs(chanIdx_).Y;
                                    dupLocs(dest1).Z = EEG.chanlocs(chanIdx_).Z;
                                    dupLocs(dest1).ref = EEG.chanlocs(chanIdx_).ref;
                                    dupLocs(dest1).urchan = EEG.chanlocs(chanIdx_).urchan;
                                    break;
                                end
                            end
                            for chanIdx_2 = 1 : length(EEG.chanlocs)
                                if length(strmatch(rightChan, EEG.chanlocs(chanIdx_2).labels, 'exact')) % where is the twente location in the current data.
                                    source = chanIdx_2; %location about channel which has to be placed in the current place in the data. 
                                    disp(strcat(rightChan, ' is in location-', num2str(source), ' in the current EEG data ', ' which has to go-', num2str(chanIdx_), ' in dupData according to the twente'))
                                    dupData(chanIdx_, :) = EEG.data(source, :);

                                    dupLocs(chanIdx_).type = EEG.chanlocs(source).type;
                                    dupLocs(chanIdx_).labels = EEG.chanlocs(source).labels;
                                    dupLocs(dest1).description = EEG.chanlocs(chanIdx_).description;
                                    dupLocs(chanIdx_).sph_theta = EEG.chanlocs(source).sph_theta;
                                    dupLocs(chanIdx_).sph_phi = EEG.chanlocs(source).sph_phi;
                                    dupLocs(chanIdx_).theta = EEG.chanlocs(source).theta;
                                    dupLocs(chanIdx_).radius = EEG.chanlocs(source).radius;
                                    dupLocs(chanIdx_).sph_radius = EEG.chanlocs(source).sph_radius;
                                    dupLocs(chanIdx_).X = EEG.chanlocs(source).X;
                                    dupLocs(chanIdx_).Y = EEG.chanlocs(source).Y;
                                    dupLocs(chanIdx_).Z = EEG.chanlocs(source).Z;
                                    dupLocs(chanIdx_).ref = EEG.chanlocs(source).ref;
                                    dupLocs(chanIdx_).urchan = EEG.chanlocs(source).urchan;                    
                                    break;
                                end                
                            end 
                        end
                    end
                    EEG.data = dupData;
                    EEG.chanlocs = dupLocs;            
                    pop_saveset(EEG, 'filename',strcat('corrected_', subject, '_channelBand1_40_FinalProcessed_',eventInfo2{evt_},'.set'),'filepath', subTarFolder);            
                    eegData = EEG.data;
                    try
                        Temotion = split(eventInfo2{evt_}, '_');
                    catch
                        disp('stop')
                    end
                    emotion = Temotion{end};
                    if strmatch('base', emotion, 'exact')
                        save(fullfile(subTarFolder, strcat('Baseline', '_', subject)), 'eegData')
                    else
                        save(fullfile(subTarFolder, strcat(emotion, '_', subject, Temotion{1},'_Click-',Temotion{end-1})), 'eegData')            
                    end
                    [ALLEEG EEG index] = pop_newset(ALLEEG, EEG, length(ALLEEG),'retrieve', 1,'study',0);
                    % Plot energy topoplot and psd plot here then save them
                end
            else
                if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_EyeArtRemoved.set')))
                    EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40_EyeArtRemoved.set'),'filepath', subTarFolder);
                else
                    EEG = eeg_checkset( EEG );
                    EEG = pop_runica(EEG, 'icatype', 'runica', 'chanind',(1:EEG.nbchan));
                    pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_ICA.set'),'filepath',subTarFolder);
                    EEG = eeg_checkset( EEG );

                    %pop_runica(finallyCorrected, 'icatype', 'runica', 'chanind',(1:finallyCorrected.nbchan));
                    EEG = pop_iclabel(EEG, 'default');
                    pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_icLabel.set'),'filepath',subTarFolder);

                    EEG = eeg_checkset( EEG );
                    max_ = max(EEG.etc.ic_classification.ICLabel.classifications, [], 2);
                    Eyeindx = [];

                    for maxInd = 1 : length(max_)
                        temp_ = find(EEG.etc.ic_classification.ICLabel.classifications(maxInd, :) == max_(maxInd));
                        if (temp_ == 3) && (EEG.etc.ic_classification.ICLabel.classifications(maxInd, 1) < 0.3)         
                            Eyeindx = [Eyeindx maxInd];
                        end
                    end

                    EEG.icaactOrig = EEG.icaact;
                    EEG.icawinvOrig = EEG.icawinv;
                    EEG.icasphereOrig = EEG.icasphere;
                    EEG.icaweightsOrig = EEG.icaweights;
                    EEG.icachansindOrig = EEG.icachansind;

                    Eyeindx
                    EEG = pop_subcomp( EEG, Eyeindx);    

                    EEG.FirstICA.Artifact.Eye = Eyeindx;
                    %pop_viewprops( EEG, 0, [1:size(EEG.icaact,1)], {'freqrange', [2 40]}, {}, 1, 'ICLabel' , 'C:\Processed_Emotions', 'first');
                    % Rechecking for Channel Rejection there.
                    pop_eegplotMod( EEG, 1, 1, 1, 'title_', 'After Removing Eye Components');        
                    %figure; title('After Removing Eye Components');pop_spectopo(EEG, 1, [0  704040], 'EEG' , 'percent', 15, 'freq', [5 10 15 20 30], 'freqrange',[1 40],'electrodes','on');    
                    newEvent = EEG.event;

                    %totaldelchl = [totaldelchl deletedCh3];        
                    %totaldeldata = [totaldeldata dataToDelete3];
                    %EEG.DelChalMan = [EEG.DelChalMan deletedCh3];
                    %EEG.DelDatMan = [EEG.DelDatMan dataToDelete3];

                    newEvent = EEG.event;
                    EEG.summary.ReducedEvents.AfterFirstICAandSubsequentRemovalofBadDataOnceMore = length(oldEvent)-length(newEvent);
                    pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_EyeArtRemoved.set'),'filepath', subTarFolder);
                end

                EEG.icaact = [];
                EEG.icawinv = [];
                EEG.icasphere = [];
                EEG.icaweights = [];
                EEG.icachansind = [];
                EEG.etc.icaweights_beforerms = [];
                EEG.etc.icasphere_beforerms = [];
                EEG.etc.ic_classification = [];    
                %pop_eegplotMod( EEG, 1, 1, 1, 'title_', 'After Removing Eye Components and Reassuring for Bad Channels');
                %figure; title('After Removing Eye Components and Reassuring for Bad Channels');pop_spectopo(EEG, 1, [0  704040], 'EEG' , 'percent', 15, 'freq', [5 10 15 20 30], 'freqrange',[0 40],'electrodes','on');

                %%
                % This is so that we can have all the channel indices together after
                % the automatic bad channel rejection. Because anyway whatever channels
                % has been deleted previously are going to be assigned with vector 0
                % which will be taken care by automatic artifact rejection.
                if length(EEG.DelChalMan) ~= 0
                    ManDelChl = sort(EEG.DelChalMan);
                    dupData = ones(size(EEG.data, 1)+length(ManDelChl), size(EEG.data, 2));
                    dupData(1:ManDelChl(1)-1, :) = EEG.data(1:ManDelChl(1)-1, :);
                    for chIdx_ = 1 : length(ManDelChl)
                        dupData(ManDelChl(chIdx_), :) = zeros(1, EEG.pnts);
                        %EEG.chanlocs(1, ManDelChl(chIdx_)).labels
                    end

                    chIdx_ = ManDelChl(1)+1;
                    EEGdataCnt = ManDelChl(1);
                    while chIdx_ < size(EEG.data,1)+ length(ManDelChl)
                        if mean(dupData(chIdx_, :)) == 1
                            dupData(chIdx_, :) = EEG.data(EEGdataCnt, :);
                            EEGdataCnt = EEGdataCnt + 1;
                        else
                            display(dupData(chIdx_, :));
                        end
                        chIdx_ = chIdx_ + 1;
                    end
                    EEG.data = dupData;
                    toCheck = 0;
                    for chIdx_ = 1 : length(ManDelChl)
                        toCheck = toCheck + sum(EEG.data(ManDelChl(chIdx_),:));
                        if toCheck == 0
                            disp('New Data Is ok')
                        else
                            error(strcat('There is a problem with the new data. Please check the index-', num2str(ManDelChl(chIdx_))))
                        end                    
                    end

                    EEG.data(end, :) = zeros(1, EEG.pnts);
                    EEG.nbchan = size(EEG.data,1);
                    delete dupData
                    EEG.chanlocs = origChanLocs;
                end

                for i = 1 : length(EEG.chanlocs)
                    if (EEG.chanlocs(i).X == locationRef(i).X) & (EEG.chanlocs(i).Y == locationRef(i).Y) & (EEG.chanlocs(i).Z == locationRef(i).Z)
                        continue
                    else
                        error('something is wrong with the channel locations')
                    end
                end

                %Automatic channel rejection to reject bad channels. (Cross-verification visually)
                if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_AfterRejectingBadChannels.set')))
                    EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40_AfterRejectingBadChannels.set'),'filepath', subTarFolder);
                else
                    EEG = clean_artifacts(EEG, 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion','off','WindowCriterion','off','BurstRejection','on','Distance','Euclidian');

                    temp = [];
                    delChlCnt = 0;
                    for chIdx = 1 : length(EEG.etc.clean_channel_mask)
                        if EEG.etc.clean_channel_mask(chIdx) == 0
                            delChlCnt = delChlCnt + 1;
                            temp(delChlCnt) = chIdx;
                        end
                    end
                    EEG.deletedChl = temp;
                    temp = sort([temp EEG.DelChalMan]);
                    %EEG.etc.clean_channel_mask = ones(129);
                    temp = unique(temp);
                    for delIdx = 1 : length(temp)

                        %EEG.etc.clean_channel_mask(locationRef(1, temp(delIdx)).index_) = 0;
                        EEG.chaninfo.nodatchans(1, end+1).labels = getfield(locationRef(1, temp(delIdx)), 'labels');
                        EEG.chaninfo.nodatchans(1, end).description = getfield(locationRef(1, temp(delIdx)), 'description');
                        EEG.chaninfo.nodatchans(1, end).X = getfield(locationRef(1, temp(delIdx)), 'X');
                        EEG.chaninfo.nodatchans(1, end).Y = getfield(locationRef(1, temp(delIdx)), 'Y');
                        EEG.chaninfo.nodatchans(1, end).Z = getfield(locationRef(1, temp(delIdx)), 'Z');
                        EEG.chaninfo.nodatchans(1, end).identifier = getfield(locationRef(1, temp(delIdx)), 'identifier');
                        EEG.chaninfo.nodatchans(1, end).type = getfield(locationRef(1, temp(delIdx)), 'type');
                        EEG.chaninfo.nodatchans(1, end).ref = getfield(locationRef(1, temp(delIdx)), 'ref');
                        EEG.chaninfo.nodatchans(1, end).sph_theta = getfield(locationRef(1, temp(delIdx)), 'sph_theta');
                        EEG.chaninfo.nodatchans(1, end).sph_phi = getfield(locationRef(1, temp(delIdx)), 'sph_phi');
                        EEG.chaninfo.nodatchans(1, end).sph_radius = getfield(locationRef(1, temp(delIdx)), 'sph_radius');
                        EEG.chaninfo.nodatchans(1, end).theta = getfield(locationRef(1, temp(delIdx)), 'theta');
                        EEG.chaninfo.nodatchans(1, end).radius = getfield(locationRef(1, temp(delIdx)), 'radius');
                        EEG.chaninfo.nodatchans(1, end).urchan = [];
                        EEG.chaninfo.nodatchans(1, end).datachan = 1;
                    end

                    EEG = eeg_checkset( EEG );
                    pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_AfterRejectingBadChannels.set'),'filepath', subTarFolder);
                end
                EEG = eeg_checkset( EEG );

                %pop_eegplotMod( EEG, 1, 1, 1, 'title_', 'After Rejecting Bad Channels');    
                %figure; title('After Rejectng Bad Channels');pop_spectopo(EEG, 1, [0  704040], 'EEG' , 'percent', 15, 'freq', [5 10 15 20 30], 'freqrange',[0 40],'electrodes','on');

                %%
                %Channel Interpolation and Reference the data to average.
                % To Check whether pop_reref is working properly execute the file
                % JustForChecking.m

                for ch__ = 4:length(EEG.chaninfo.nodatchans)-1
                    idxx = strmatch(EEG.chaninfo.nodatchans(ch__).labels, {locationRef.labels}, 'exact');
                    if (EEG.chaninfo.nodatchans(ch__).X == locationRef(idxx).X) & (EEG.chaninfo.nodatchans(ch__).Y == locationRef(idxx).Y) & (EEG.chaninfo.nodatchans(ch__).Z == locationRef(idxx).Z)
                        continue
                    else
                        error('something is wrong with the channel locations')
                    end
                end

                close all;
                if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_AfterRereferencing.set')))
                    EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40_AfterRereferencing.set'),'filepath', subTarFolder);
                else
    %% 
    %********Error Caution: Interpolation will not work properly if the location of channels are not corrent. Please check it  thoroughly

                    EEG = pop_interp(EEG, EEG.chaninfo.nodatchans(4:end-1));
                    %EEG = pop_interp(EEG); %Select Channel other than PNS and ref for interpolation here

                    for ch__ = 4:length(EEG.chaninfo.nodatchans)-1
                        idxx = strmatch(EEG.chaninfo.nodatchans(ch__).labels, {locationRef.labels}, 'exact');
                        if (EEG.chaninfo.nodatchans(ch__).X == locationRef(idxx).X) & (EEG.chaninfo.nodatchans(ch__).Y == locationRef(idxx).Y) & (EEG.chaninfo.nodatchans(ch__).Z == locationRef(idxx).Z)
                            continue
                        else
                            error('something is wrong with the channel locations')
                        end
                    end

                    if EEG.nbchan < 129
                        EEG.nbchan = EEG.nbchan+1;
                        EEG.data(end+1,:) = zeros(1, EEG.pnts);
                    end
                    EEG.chanlocs(1,EEG.nbchan).labels = 'initialReference';
                    EEG = pop_reref( EEG, []);
                    EEG = pop_select( EEG,'nochannel',{'initialReference'});

                    %{
                    for delIdx = 1 : length(EEG.deletedChl)-1

                        EEG.chaninfo.nodatchans(1, end+1).labels = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'labels');
                        EEG.chaninfo.nodatchans(1, end).description = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'description');
                        EEG.chaninfo.nodatchans(1, end).X = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'X');
                        EEG.chaninfo.nodatchans(1, end).Y = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'Y');
                        EEG.chaninfo.nodatchans(1, end).Z = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'Z');
                        EEG.chaninfo.nodatchans(1, end).identifier = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'identifier');
                        EEG.chaninfo.nodatchans(1, end).type = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'type');
                        EEG.chaninfo.nodatchans(1, end).ref = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'ref');
                        EEG.chaninfo.nodatchans(1, end).sph_theta = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'sph_theta');
                        EEG.chaninfo.nodatchans(1, end).sph_phi = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'sph_phi');
                        EEG.chaninfo.nodatchans(1, end).sph_radius = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'sph_radius');
                        EEG.chaninfo.nodatchans(1, end).theta = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'theta');
                        EEG.chaninfo.nodatchans(1, end).radius = getfield(locationRef(1, EEG.deletedChl(delIdx)), 'radius');
                        EEG.chaninfo.nodatchans(1, end).urchan = [];
                        EEG.chaninfo.nodatchans(1, end).datachan = 1;    
                        EEG = pop_select( EEG,'nochannel',{locationRef(1, EEG.deletedChl(delIdx)).labels});

                    end
                    %}
                    EEG = eeg_checkset( EEG );
                    pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_AfterRereferencing.set'),'filepath', subTarFolder);
                end
                %%
                %Reapplication of ICA on the clean data to find out brain components.
                %Applying ICLabel on ICA components. Cross checking the components marked “other” and retaining them if the brain component is contributing for more than 30 percent and spectrum and topography looks suspicious of having brain activity.   
                if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_FinalPCA.set')))
                    EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40_FinalPCA.set'),'filepath', subTarFolder);
                else
                    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',0,'interrupt','on');
                    EEG = eeg_checkset( EEG );
                    EEG = pop_iclabel(EEG, 'default');
                    pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_FinalPCA.set'),'filepath', subTarFolder);
                end

                if isfile(fullfile(subTarFolder, strcat(subject, '_channelBand1_40_FinalProcessed.set')))
                    EEG = pop_loadset('filename',strcat(subject, '_channelBand1_40_FinalProcessed.set'),'filepath', subTarFolder);
                else
                    EEG.icaactOrig = EEG.icaact;
                    EEG.icawinvOrig = EEG.icawinv;
                    EEG.icasphereOrig = EEG.icasphere;
                    EEG.icaweightsOrig = EEG.icaweights;
                    EEG.icachansindOrig = EEG.icachansind;

                    max_ = max(EEG.etc.ic_classification.ICLabel.classifications, [], 2);
                    Artifactindx = [];
                    Signalindx = [];

                    for maxInd = 1 : length(max_)
                        temp_ = find(EEG.etc.ic_classification.ICLabel.classifications(maxInd, :) == max_(maxInd));
                        if (temp_ == 7) || (temp_ == 1)
                            if(EEG.etc.ic_classification.ICLabel.classifications(maxInd, 1) > 0.25)         
                                Signalindx = [Signalindx maxInd];
                            else
                                Artifactindx = [Artifactindx maxInd];
                            end
                        else
                            Artifactindx = [Artifactindx maxInd];
                        end
                    end

                    EEG.SecondICA.Artifact = Artifactindx;
                    EEG.SecondICA.Signal = Signalindx;

                    %pop_viewprops( EEG, 0, [1:size(EEG.icaact,1)], {'freqrange', [2 40]}, {}, 1, 'ICLabel' , rootFolder, 'first');
                    EEG = eeg_checkset( EEG );
                    % See Variance for all the components For other more than 25 percent
                    EEG = pop_subcomp( EEG, Artifactindx);
                    EEG = eeg_checkset( EEG );
                    %EEG = pop_interp(EEG, EEG.chaninfo.nodatchans(5:end));
                    EEG = eeg_checkset( EEG );

                    neutralIdx_ = find([EEG.event.urevent] == EEG.neutralUrevent);    
                    stimType = strmatch('STIM', {EEG.event.tracktype});
                    if length(stimType) == 0
                        stimType = strmatch('EVNT', {EEG.event.classid});
                    end
                    EEG.event(stimType(neutralIdx_)).type = 'Neutral';          
                    neutLat = (EEG.event(stimType(neutralIdx_+1)).latency-EEG.event(stimType(neutralIdx_-1)).latency);
                    if neutLat < (3500+(EEG.srate*totalTime)) && (~ all(EEG.event(stimType(neutralIdx_-1)).type == 'bend'))
                        display('neutral Event is not right')
                    end

                    EEG.summary.FirstICAEyeComp = EEG.FirstICA.Artifact.Eye;
                    EEG.summary.SecondICABrainComp = EEG.SecondICA.Signal;
                    EEG.summary.SecondICAArtComp = EEG.SecondICA.Artifact;
                    %EEG.summary.AutomaticDelSamp = EEG.deletedsample;
                    EEG.summary.AutomaticDelSamp = [];
                    EEG.summary.AutomaticDelSamp = [];
                    EEG.summary.TotalDelChannel = EEG.deletedChl;
                    EEG.summary.ManDelChannel = EEG.DelChalMan;
                    EEG.summary.numberEvents = strmatch(event_info{1,2}, {EEG.event.type});
                    baseIndex = strmatch('base', {EEG.event.type});
                    bendIndex = strmatch('bend', {EEG.event.type});
                    EEG.summary.baseDuration = (EEG.event(bendIndex).latency-EEG.event(baseIndex).latency)/EEG.srate;

                    pop_saveset(EEG, 'filename',strcat(subject, '_channelBand1_40_FinalProcessed.set'),'filepath', subTarFolder);
                end
                dupData = zeros(size(EEG.data,1), size(EEG.data,2));
                dupLocs = EEG.chanlocs;
                for chanIdx_ = 1 : length(EEG.chanlocs)   
                    rightChan = strcat('E', num2str(chanIdx_));
                    if strmatch(rightChan, EEG.chanlocs(chanIdx_).labels)
                        dupData(chanIdx_, :) = EEG.data(chanIdx_, :);
                    else
                        tmp_ = EEG.data(chanIdx_, :); % Current index which has to be shifted to the location given in the twente file. the variable dest will have that index.
                        for chanIdx_2 = 1 : length(EEG.chanlocs)
                            another_rightChan = strcat('E', num2str(chanIdx_2));
                            if length(strmatch(another_rightChan, EEG.chanlocs(chanIdx_).labels, 'exact')) % where is the current location in twente.                    
                                dest1 = chanIdx_2; %location about current channel of data in twente. It will tell me where to swap the current location.
                                disp(strcat(EEG.chanlocs(chanIdx_).labels, ' is in location-', num2str(chanIdx_), ' in the current EEG data ', ' which has to go-', num2str(dest1), ' in dupData according to the twente'))
                                dupData(dest1, :) = tmp_;

                                dupLocs(dest1).type = EEG.chanlocs(chanIdx_).type;
                                dupLocs(dest1).labels = EEG.chanlocs(chanIdx_).labels;
                                dupLocs(dest1).description = EEG.chanlocs(chanIdx_).description;
                                dupLocs(dest1).sph_theta = EEG.chanlocs(chanIdx_).sph_theta;
                                dupLocs(dest1).sph_phi = EEG.chanlocs(chanIdx_).sph_phi;
                                dupLocs(dest1).theta = EEG.chanlocs(chanIdx_).theta;
                                dupLocs(dest1).radius = EEG.chanlocs(chanIdx_).radius;
                                dupLocs(dest1).sph_radius = EEG.chanlocs(chanIdx_).sph_radius;
                                dupLocs(dest1).X = EEG.chanlocs(chanIdx_).X;
                                dupLocs(dest1).Y = EEG.chanlocs(chanIdx_).Y;
                                dupLocs(dest1).Z = EEG.chanlocs(chanIdx_).Z;
                                dupLocs(dest1).ref = EEG.chanlocs(chanIdx_).ref;
                                dupLocs(dest1).urchan = EEG.chanlocs(chanIdx_).urchan;
                                break;
                            end
                        end
                        for chanIdx_2 = 1 : length(EEG.chanlocs)
                            if length(strmatch(rightChan, EEG.chanlocs(chanIdx_2).labels, 'exact')) % where is the twente location in the current data.
                                source = chanIdx_2; %location about channel which has to be placed in the current place in the data. 
                                disp(strcat(rightChan, ' is in location-', num2str(source), ' in the current EEG data ', ' which has to go-', num2str(chanIdx_), ' in dupData according to the twente'))
                                dupData(chanIdx_, :) = EEG.data(source, :);

                                dupLocs(chanIdx_).type = EEG.chanlocs(source).type;
                                dupLocs(chanIdx_).labels = EEG.chanlocs(source).labels;
                                dupLocs(dest1).description = EEG.chanlocs(chanIdx_).description;
                                dupLocs(chanIdx_).sph_theta = EEG.chanlocs(source).sph_theta;
                                dupLocs(chanIdx_).sph_phi = EEG.chanlocs(source).sph_phi;
                                dupLocs(chanIdx_).theta = EEG.chanlocs(source).theta;
                                dupLocs(chanIdx_).radius = EEG.chanlocs(source).radius;
                                dupLocs(chanIdx_).sph_radius = EEG.chanlocs(source).sph_radius;
                                dupLocs(chanIdx_).X = EEG.chanlocs(source).X;
                                dupLocs(chanIdx_).Y = EEG.chanlocs(source).Y;
                                dupLocs(chanIdx_).Z = EEG.chanlocs(source).Z;
                                dupLocs(chanIdx_).ref = EEG.chanlocs(source).ref;
                                dupLocs(chanIdx_).urchan = EEG.chanlocs(source).urchan;                    
                                break;
                            end                
                        end 
                    end
                end
                EEG.data = dupData;
                EEG.chanlocs = dupLocs;            

                for i = 1 : length(EEG.chanlocs)
                    if (EEG.chanlocs(i).X == locationRef(i).X) & (EEG.chanlocs(i).Y == locationRef(i).Y) & (EEG.chanlocs(i).Z == locationRef(i).Z)
                        continue
                    else
                        error('something is wrong with the channel locations')
                    end
                end

                if strmatch('E129', EEG.chaninfo.nodatchans(end).labels)
                    EEG.chaninfo.nodatchans(4)= EEG.chaninfo.nodatchans(end);
                end

                deleInfo = 0;
                for ch_s = length(EEG.chaninfo.nodatchans):-1:5
                    EEG.chaninfo.nodatchans(ch_s) = [];
                    deleInfo = deleInfo + 1;
                end
                pop_saveset(EEG, 'filename',strcat('corrected_', subject, '_channelBand1_40_FinalProcessed.set'),'filepath', subTarFolder);            
                %[ALLEEG EEG index] = pop_newset(ALLEEG, EEG, length(ALLEEG),'retrieve', 1,'study',0);
                % Plot energy topoplot and psd plot here then save them            
            end
            %pop_eegplotMod( EEG, 1, 1, 1, 'title_', 'After Removing Bad Components');
            %figure; title('After Removing Bad Components');pop_spectopo(EEG, 1, [0  704040], 'EEG' , 'percent', 15, 'freq', [5 10 15 20 30], 'freqrange',[1 40],'electrodes','on');
            %savefig('Final_Data.fig');
            clcIdx = strmatch('clic', {EEG.event.type});
            pwd_ = pwd;
            cd(rootFolder)    
            for evtIdx = 1 : length(EEG.event)
                if length(EEG.event(evtIdx).emtInfo)
                    temp1Arr = split(EEG.event(evtIdx).emtInfo, 'Trial-');
                    emotion_ = temp1Arr{1,1}
                    temp2 = split(temp1Arr(2,1), 'Click-');
                    temp3Arr = split(temp2(2,1), 'Five');
                    trialNumber = str2num(temp2{1,1})
                    clickNumber = str2num(temp3Arr{1,1})

                    display(subject)
                    py.importlib.import_module('matchingEmotion')
                    py.matchingEmotion.matchingEmotion(trialNumber, clickNumber, emotion_, subjectCSVDataDir, subject, length(stmArray)-1, rootFolder)
                end
            end
            %else 
            if eventWise == 0
                EEG = pop_loadset('filename',strcat('corrected_', subject, '_channelBand1_40_FinalProcessed.set'),'filepath', subTarFolder);
                [ALLEEG EEG index] = eeg_store(ALLEEG, EEG);
                EEG.filename = 'AllBands';
                ALLEEG.filename = 'AllBands';
                %{
                Y = fft(ALLEEG(1, 6).data(1,:));
                L = length(ALLEEG(1, 6).data(1,:));
                P2 = abs(Y/L);
                P1 = P2(1:L/2+1);
                P1(2:end-1) = 2*P1(2:end-1);
                f = ALLEEG(1, 6).srate*(0:(L/2))/L;
                figure;
                plot(f,P1) 
                title('Upper Beta')
                xlabel('f (Hz)')
                ylabel('|P1(f)|')                
                figure;
                %}
                clcIdx2 = [strmatch({'clic'}, {EEG.event.type}); strmatch({'Neutral'}, {EEG.event.type})];
                eventInfo2 = {'base'};
                for clc_ = 1 : length(clcIdx2)
                    tmpEmt = split(EEG.event(clcIdx2(clc_)).emtInfo, 'Trial');
                    emtt_ = tmpEmt{1};
                    if length(emtt_)
                        tmpT = split(tmpEmt{2}, 'Click');
                        trial_ = strcat('trial', tmpT{1}, '_');
                    else
                        if strmatch('neutral', EEG.event(clcIdx2(clc_)).label)
                            trial_ = '';
                            emtt_ = 'neutral';
                        else
                            trial_ = '';
                        end
                    end
                    EEG.event(clcIdx2(clc_)).type = strcat(trial_,EEG.event(clcIdx2(clc_)).type, '_', num2str(clc_), '_', emtt_);
                    EEG.event(clcIdx2(clc_)).code = strcat(trial_,EEG.event(clcIdx2(clc_)).code, '_', num2str(clc_), '_', emtt_);
                    eventInfo2(clc_+1) = {EEG.event(clcIdx2(clc_)).type};
                end  

                [ALLEEG EEG index] = eeg_store(ALLEEG, EEG, 1);

                EEG_Delta = pop_eegfiltnew(EEG, 'locutoff',1,'hicutoff',4,'plotfreqz',1,'chantype',{'EEG'});
                EEG_Delta.filename = 'Delta';
                [ALLEEG EEG index] = eeg_store(ALLEEG, EEG_Delta, 2);

                [ALLEEG EEG index] = pop_newset(ALLEEG, EEG, length(ALLEEG),'retrieve', 1,'study',0);
                EEG_Theta = pop_eegfiltnew(EEG, 'locutoff',4,'hicutoff',8,'plotfreqz',1,'chantype',{'EEG'});
                EEG_Theta.filename = 'Theta';
                [ALLEEG EEG index] = eeg_store(ALLEEG, EEG_Theta, 3);

                [ALLEEG EEG index] = pop_newset(ALLEEG, EEG, length(ALLEEG),'retrieve', 1,'study',0);
                EEG_Alpha = pop_eegfiltnew(EEG, 'locutoff',8,'hicutoff',13,'plotfreqz',1,'chantype',{'EEG'});
                EEG_Alpha.filename = 'Alpha';
                [ALLEEG EEG index] = eeg_store(ALLEEG, EEG_Alpha, 4);

                [ALLEEG EEG index] = pop_newset(ALLEEG, EEG, length(ALLEEG),'retrieve', 1,'study',0);
                EEG_LBeta = pop_eegfiltnew(EEG, 'locutoff',14,'hicutoff',20,'plotfreqz',1,'chantype',{'EEG'});
                EEG_LBeta.filename = 'Lower Beta';
                [ALLEEG EEG index] = eeg_store(ALLEEG, EEG_LBeta, 5);

                [ALLEEG EEG index] = pop_newset(ALLEEG, EEG, length(ALLEEG),'retrieve', 1,'study',0);
                EEG_UBeta = pop_eegfiltnew(EEG, 'locutoff',20,'hicutoff',30,'plotfreqz',1,'chantype',{'EEG'});
                EEG_UBeta.filename = 'Upper Beta';
                [ALLEEG EEG index] = eeg_store(ALLEEG, EEG_UBeta, 6);

                [ALLEEG EEG index] = pop_newset(ALLEEG, EEG, length(ALLEEG),'retrieve', 1,'study',0);
                EEG_Gamma = pop_eegfiltnew(EEG, 'locutoff',30,'hicutoff',40,'plotfreqz',1,'chantype',{'EEG'});
                EEG_Gamma.filename = 'Gamma';
                [ALLEEG EEG index] = eeg_store(ALLEEG, EEG_Gamma, 7);

                [ALLEEG EEG index] = pop_newset(ALLEEG, EEG, length(ALLEEG),'retrieve', 1,'study',0);
                EEG_ThetaAlphaLowerBeta = pop_eegfiltnew(EEG, 'locutoff',4,'hicutoff',20,'plotfreqz',1,'chantype',{'EEG'});
                EEG_ThetaAlphaLowerBeta.filename = 'ThetaAlphaLowerBeta';
                [ALLEEG EEG index] = eeg_store(ALLEEG, EEG_ThetaAlphaLowerBeta, 8);

                [ALLEEG EEG index] = pop_newset(ALLEEG, EEG, length(ALLEEG),'retrieve', 1,'study',0);
                EEG_AlphaLowerBeta = pop_eegfiltnew(EEG, 'locutoff',8,'hicutoff',20,'plotfreqz',1,'chantype',{'EEG'});
                EEG_AlphaLowerBeta.filename = 'AlphaLowerBeta';
                [ALLEEG EEG index] = eeg_store(ALLEEG, EEG_AlphaLowerBeta, 9);

                [ALLEEG EEG index] = pop_newset(ALLEEG, EEG, length(ALLEEG),'retrieve', 1,'study',0);
                EEG_DeltaThetaAlpha = pop_eegfiltnew(EEG, 'locutoff',1,'hicutoff',13,'plotfreqz',1,'chantype',{'EEG'});
                EEG_DeltaThetaAlpha.filename = 'DeltaThetaAlpha';
                [ALLEEG EEG index] = eeg_store(ALLEEG, EEG_DeltaThetaAlpha, 10);

                for EEGidx = 1 : length(ALLEEG)
                    [ALLEEG EEG index] = pop_newset(ALLEEG, EEG, length(ALLEEG),'retrieve', EEGidx,'study',0);

                    for idx = 1 : length(eventInfo2)
                        if strmatch('base', eventInfo2(idx), 'exact')
                            EEG = pop_rmdatOrigBase( EEG, eventInfo2(idx), [0 60], 0);
                        else
                            EEG = pop_rmdatIndStim( EEG, eventInfo2(idx), [-6 1], 0);
                        end   

                    %clcIdx_ = [strmatch({'clic'}, {EEG.event.type}); strmatch({'Neutral'}, {EEG.event.type})];
                    %timeRange = [6 1];
                    %for idx = 1 : length(clcIdx_)
                        if strmatch('trial-', eventInfo2(idx))
                            emtTemp = split(eventInfo2(idx), '_');
                            emotion = emtTemp{end};
                            emtFold = 'WithoutIndividualICA';

                            try
                                trialTemp = split(emtTemp{1}, 'trial-');
                            catch
                                display('error')
                            end
                            trial = trialTemp{2};
                            %clickTemp = split(trialTemp{2}, 'Five');
                            click = emtTemp{3};
                            eegData = EEG.data;
                            dest_ = fullfile(targetFolder, emtFold, EEG.filename);
                        else
                            if strmatch({'Neutral'}, eventInfo2(idx))
                                emotion = 'Neutral';
                                trial = '';
                                click = '';
                                eegData = EEG.data;
                                dest_ = fullfile(targetFolder, emotion, EEG.filename);
                            elseif strmatch({'base'}, eventInfo2(idx))
                                baselineTime = 7;
                                emotion = 'Baseline';
                                trial = '';
                                click = '';
                                baseIdx = strmatch({'base'}, {EEG.event.type});
                                bendIdx = strmatch({'bend'}, {EEG.event.type});
                                baseS = fix(EEG.event(baseIdx).latency);
                                baseE = fix(EEG.event(bendIdx).latency);
                                a = baseS+EEG.srate;
                                b = ((baseE/EEG.srate)-baselineTime)*EEG.srate;
                                baseS = (b-a).*rand(1,1) + a;
                                baseE = baseS+(baselineTime*EEG.srate);
                                eegData = EEG.data(:, baseS:baseE);
                                dest_ = fullfile(targetFolder, emotion, EEG.filename);
                            end
                            emtFold = emotion;                
                        end
                        if ~isdir(dest_)
                            mkdir(dest_);
                        end    

                        for i = 1 : length(EEG.chanlocs)
                            if (EEG.chanlocs(i).X == locationRef(i).X) & (EEG.chanlocs(i).Y == locationRef(i).Y) & (EEG.chanlocs(i).Z == locationRef(i).Z)
                                continue
                            else
                                error('something is wrong with the channel locations')
                            end
                        end
                        if strmatch('Baseline', emotion, 'exact')
                            pop_saveset(EEG, 'filename',strcat('corrected_', subject, '_channelBand1_40_FinalProcessed_',eventInfo2{idx},'.set'),'filepath', dest_);            
                            %EEG = pop_select(EEG, 'time', [9 16]);
                            EEG = pop_select(EEG, 'time', [baseS/EEG.srate baseE/EEG.srate]);
                            EEG.FinalBaselineTime = [baseS/EEG.srate baseE/EEG.srate];
                            pop_saveset(EEG, 'filename',strcat('corrected_',num2str(baselineTime),'Sec_', subject, '_channelBand1_40_FinalProcessed_',eventInfo2{idx},'.set'),'filepath', dest_);            
                            break
                        elseif strmatch('Neutral', emotion, 'exact')
                            pop_saveset(EEG, 'filename',strcat('corrected_', subject, '_channelBand1_40_FinalProcessed_',eventInfo2{idx},'.set'),'filepath', dest_);                                
                        else
                            try
                                pop_saveset(EEG, 'filename',strcat('corrected_', subject, '_channelBand1_40_FinalProcessed_',eventInfo2{idx},'.set'),'filepath', dest_);            
                            catch
                                disp('stop here')
                            end
                        end
                        if strmatch('Baseline', emotion, 'exact')
                            save(fullfile(dest_, strcat('Baseline', '_', subject)), 'eegData')
                        else
                            save(fullfile(dest_, strcat(emotion, '_', subject,'Trial-',trial,'Click-',num2str(str2num(click)))), 'eegData')
                        end
                        [ALLEEG EEG index] = pop_newset(ALLEEG, EEG, length(ALLEEG),'retrieve', EEGidx,'study',0);
                    end        
                end
            end

            fileId = fopen(fullfile(subTarFolder,'summary.txt'), 'w');
            %fprintf(fileId, 'FirstICAEyeComp-%d \n SecondICABrainComp-%d \n SecondICAArtComp-%d \n AutomaticDelSamp-%d \n TotalDelChannel-%d \n ManDelChannel-%d \n Number of Events-%d \n Base Duration-%f', ...
            %    [length(EEG.summary.FirstICAEyeComp), length(EEG.summary.SecondICABrainComp), length(EEG.summary.SecondICAArtComp), length(EEG.summary.AutomaticDelSamp), length(EEG.summary.TotalDelChannel), length(EEG.summary.ManDelChannel), length(EEG.summary.numberEvents), EEG.summary.baseDuration]);
            fclose(fileId);
            fileId = fopen('currentSubject.txt', 'w');
            fprintf(fileId, '%s', subject);
            fclose(fileId);
        else
            %% 

            G1 = {'Happy','Amused','Joyous','Delighted'};
            G2 = {'Adventorous','Aroused','Excited','Passionate','Lust'};
            G3 = {'Tense','Alarmed','Startled'};
            G4 = {'Hate','Disgust'};
            G5 = {'Afraid'};
            G6 = {'Angry','Distress'};
            G7 = {'Miserable','Melancholic','Depressed','Despondent','Dissatisfied','Taken Aback'};
            G8 = {'Sad'};
            network = {'coreAffCorResult','coreAssCorResult','LangResult','extroResult','exeCResult'};
            % Doing ECG Related Calculation
            frqs_ = {'Theta','Alpha','LowerBeta','UpperBeta','Gamma'};
            sourceData = 'C:\Processed_Emotions\GrangerCausality';
            baselineTime = 7;
            dest_ = fullfile(targetFolder, 'Baseline', 'AllBands');
            eventDuration = [-6 1];
            if ~ isfile(fullfile(subTarFolder, strcat('ECG_', subject, '_channelBand1_40_FinalProcessed.set')))
                
                EEG = eeg_checkset( EEG );
                EEG = pop_loadset('filename',strcat('corrected_', subject, '_channelBand1_40_FinalProcessed.set'),'filepath', subTarFolder);
                DeletedData = [];
                if length(EEG.DelDatMan)
                    disp(subject);
                    DeletedData = EEG.DelDatMan;
                end
                channel_1ToCompare = EEG.data(1,:);
                %% Extracting Event Related Information and Baseline Time
                % Duration from PreProcessed EEG File.
                FilteredEvents = EEG.event;
                dest_ = fullfile(targetFolder, 'Baseline', 'AllBands');
                EEG = pop_loadset('filename',strcat('corrected_',num2str(baselineTime),'Sec_', subject, '_channelBand1_40_FinalProcessed_','base.set'),'filepath', dest_);            
                BaseDuration = EEG.FinalBaselineTime;
                clearvars -except AllValence AllArousal channel_1ToCompare BaseDuration DeletedData FilteredEvents subjectFolder subject subjectCSVDataDir subTarFolder targetFolder rootFolder sourceFolder AllSubjects_ dataFileName baselineFirstHalfArrLowf baselineSecondHalfArrLowf stimulusFirstHalfArrLowf stimulusSecondHalfArrLowf baselineFirstHalfArrMidf baselineSecondHalfArrMidf stimulusFirstHalfArrMidf stimulusSecondHalfArrMidf
                %% Loading ECG Data                
                warning('The event time must be matched with the timing with EEG processing.')
                EEG = pop_mffimport({fullfile(subjectFolder, dataFileName)},{'type'});
                locationRef = EEG.chanlocs;            
                EEG = pop_select( EEG, 'channel',{'ECG'});
                EEG = pop_eegfiltnew(EEG, 'locutoff', 0.6, 'hicutoff',40,'plotfreqz',1,'chantype',{'PNS'});
                baselineStart = strmatch('base', {EEG.event.type}, 'exact');
                baselineEnd = strmatch('bend', {EEG.event.type}, 'exact');
                %% Keeping only ECG data for baseline time and Event related
                % time available in PreProcessed EEG File.
                EEG.event(baselineStart).latency = EEG.event(baselineStart).latency + (10*EEG.srate);
                TimeToKeep = [EEG.event(baselineStart).latency EEG.event(baselineEnd).latency];
                for Idxs_ = 1 : length(FilteredEvents)
                    if strmatch('clic', FilteredEvents(Idxs_).type, 'exact') %length(FilteredEvents(Idxs_).emtInfo)
                        evtIdx_ = strmatch(FilteredEvents(Idxs_).begintime, {EEG.event.begintime});
                        TimeToKeep = [TimeToKeep; [EEG.event(evtIdx_).latency+(eventDuration(1)*EEG.srate) EEG.event(evtIdx_).latency+(eventDuration(2)*EEG.srate)]]
                    end   
                end
                EEG = pop_select(EEG, 'time', (TimeToKeep-1)/EEG.srate);
                %% Removing the baseline data which is also removed in the PreProcessed EEG data.
                baselineStart = strmatch('base', {EEG.event.type}, 'exact');
                baselineEnd = strmatch('bend', {EEG.event.type}, 'exact');            
                TimeNotToKeep = [];
                for delIdx = 1 : size(DeletedData,1)
                    try
                        if DeletedData(delIdx,2) < EEG.event(baselineEnd).latency/EEG.srate
                            TimeNotToKeep = [TimeNotToKeep; DeletedData(delIdx, :)]
                        end
                    catch
                        disp('Stop here')
                    end
                end       
                if length(TimeNotToKeep)
                    EEG = pop_select(EEG, 'notime', (TimeNotToKeep));
                end

                % After deleting the data check that the length of the baseline
                % in EEG and ECG data must be same then only proceed.
                EEGBaseStart = strmatch('base', {FilteredEvents.type}); % Baseline Start of EEG
                EEGBaseEnd = strmatch('bend', {FilteredEvents.type}); % Baseline End of EEG
                baselineStart = strmatch('base', {EEG.event.type}, 'exact'); % Baseline start of ECG
                baselineEnd = strmatch('bend', {EEG.event.type}, 'exact');  % Baseline start of ECG         

                if ~(abs((EEG.event(baselineEnd).latency-EEG.event(baselineStart).latency)-(FilteredEvents(EEGBaseEnd).latency-FilteredEvents(EEGBaseStart).latency)) < 10)
                    disp('Stop Here')
                end
                clickIdxECGFile = strmatch('clic', {EEG.event.type});
                clickIdxEEGFile = strmatch('clic', {FilteredEvents.type});
                %if length(clickIdxECGFile) == length(clickIdxEEGFile)
                    for Idxs_ = 1 : length(clickIdxEEGFile)
                        evtIdx_ = strmatch(FilteredEvents(clickIdxEEGFile(Idxs_)).begintime, {EEG.event.begintime});
                        if evtIdx_ ~= clickIdxECGFile(Idxs_)
                            disp('I would check if assignment is right. That means FilteredEvents(clickIdxEEGFile(Idxs_)).begintime should be matched with EEG.event(evtIdx_).begintime. It may be error in assignment or may be not')
                        end
                        EEG.event(evtIdx_).emtInfo = FilteredEvents(clickIdxEEGFile(Idxs_)).emtInfo;
                        %EEG.event(clickIdxECGFile(Idxs_)).emtInfo = FilteredEvents(clickIdxEEGFile(Idxs_)).emtInfo;
                    end
                %else
                %    disp('Number of Clicks are not equal')
                %end

                close all
                EEG = pop_cleanline(EEG, 'bandwidth',2,'chanlist',[1] ,'computepower',1,'legacy',0,'linefreqs',[50 100 150],'normSpectrum',0,'p',0.01,'pad',2,'plotfigures',0,'scanforlines',1,'sigtype','Channels','taperbandwidth',2,'tau',100,'verb',1,'winsize',4,'winstep',1);
                pop_eegplotMod( EEG, 1, 1, 1, 'title_', 'ECG Data');
                pop_saveset(EEG, 'filename',strcat('ECG_', subject, '_channelBand1_40_FinalProcessed.set'),'filepath', subTarFolder);
            else                
                EEG = pop_loadset('filename',strcat('ECG_', subject, '_channelBand1_40_FinalProcessed.set'),'filepath', subTarFolder);
                BaseDuration = EEG.FinalBaselineTime;
            end            
            %% Saving Data in mat format
            
            eegData = EEG.data(BaseDuration(1)*EEG.srate:BaseDuration(2)*EEG.srate);
            if SubDIdx == 1
                ECGBaseline = eegData;
            else
                ECGBaseline = ECGBaseline+eegData;
            end
            stft(eegData,EEG.srate,'Window',kaiser(250,5),'OverlapLength',75,'FFTLength',250);                
            set(gcf,'Units','inches');
            screenposition = [0 0 30 15];
            set(gcf,...
                'PaperPosition',[0 0 screenposition(3:4)],...
                'PaperSize',[screenposition(3:4)]);    
            %orient(fig,'landscape')                
            print(gcf,fullfile(subTarFolder, strcat(ECGPrefix, 'ECGBaseline_', subject)),'-dpdf')  
                
            low_=1;
            high_=80;
            fftLength = 500;
            N = fftLength;
            partition = 30;
            lastF = 80;
            
            Y = fft(eegData(firstPartL:firstPartH), fftLength);
            P2 = abs(Y/fftLength);
            %P2 = (1/(EEG.srate*N)) * abs(Y).^2;
            P1 = P2(1:N/2+1);
            P1(2:end-1) = 2*P1(2:end-1);
            baselineFirstHalf = P1;
            %baselineFirstHalfArrLowf = [baselineFirstHalfArrLowf; median(log(baselineFirstHalf(1:partition)).^2)];
            %baselineFirstHalfArrMidf = [baselineFirstHalfArrMidf; median(log(baselineFirstHalf(partition:lastF)).^2)];
            baselineFirstHalfArrLowf = [baselineFirstHalfArrLowf; median(log(baselineFirstHalf(1:partition)))];
            baselineFirstHalfArrMidf = [baselineFirstHalfArrMidf; median(log(baselineFirstHalf(partition:lastF)))];            
            clear Y P2 P1 
            Y = fft(eegData(secondPartL:secondPartH), fftLength);
            P2 = abs(Y/fftLength);
            %P2 = (1/(EEG.srate*N)) * abs(Y).^2;
            P1 = P2(1:N/2+1);                
            %P1 = P2(1:fftLength/2+1);
            P1(2:end-1) = 2*P1(2:end-1);
            baselineSecondHalf = P1;
            %baselineSecondHalfArrLowf = [baselineSecondHalfArrLowf; median(log(baselineSecondHalf(1:partition)).^2)];
            %baselineSecondHalfArrMidf = [baselineSecondHalfArrMidf; median(log(baselineSecondHalf(partition:lastF)).^2)];
            baselineSecondHalfArrLowf = [baselineSecondHalfArrLowf; median(log(baselineSecondHalf(1:partition)))];
            baselineSecondHalfArrMidf = [baselineSecondHalfArrMidf; median(log(baselineSecondHalf(partition:lastF)))];            
            f = 0:EEG.srate/N:EEG.srate/2;
            %f = EEG.srate*(0:fftLength/2)/fftLength;
            clear Y P2 P1
            save(fullfile(subTarFolder, strcat('ECG_Baseline', '_', subject)), 'eegData')
            clear eegData
            
            clickIdxECGFile = strmatch('clic', {EEG.event.type});
            for Idxs_ = 1 : length(clickIdxECGFile)                
                eventLatency = EEG.event(clickIdxECGFile(Idxs_)).latency;
                tagInfo = EEG.event(clickIdxECGFile(Idxs_)).emtInfo;            
                if length(tagInfo)
                    tmp_1 = split(tagInfo, 'Trial-');
                    emotion = tmp_1{1};
                    tmp_2 = split(tmp_1{2}, 'Click-');
                    trial = tmp_2{1};
                    tmp_3 = split(tmp_2{2}, 'Five');
                    click = tmp_3{1};
                    tmp_4 = split(tmp_3{2}, 'Val-');
                    tmp_5 = split(tmp_4{2}, 'Arsl-');
                    valence = str2num(tmp_5{1});
                    tmp_6 = split(tmp_5{2}, 'M(');
                    arousal = str2num(tmp_6{1});
                    tmp_7 = split(tmp_6{2}, 'Familiarity-');
                    tmp_8 = split(tmp_7{2}, 'Relevance-');
                    Famil = str2num(tmp_8{1});
                    tmp_9 = split(tmp_8{2}, 'interrupted-');
                    Relev = str2num(tmp_9{1});
                else
                    continue
                end
                
                eegData = EEG.data(eventLatency+(eventDuration(1)*EEG.srate):(eventLatency+EEG.srate));
                disp(emotion)
                    
                if length(strmatch(emotion, G1, 'exact')) | length(strmatch(emotion, G2, 'exact')) | length(strmatch(emotion, G3, 'exact')) | length(strmatch(emotion, G6, 'exact'))
                    if ~exist('ECGStimHF', 'var')
                        ECGStimHF = eegData;
                        eventCounterHF = 1;
                    else
                        ECGStimHF = ECGStimHF+eegData;
                        eventCounterHF = eventCounterHF + 1;
                    end                
                elseif length(strmatch(emotion, G4, 'exact')) | length(strmatch(emotion, G5, 'exact')) | length(strmatch(emotion, G7, 'exact')) | length(strmatch(emotion, G8, 'exact'))
                    disp(emotion)
                    if ~exist('ECGStimLF', 'var')
                        ECGStimLF = eegData;
                        eventCounterLF = 1;
                    else
                        ECGStimLF = ECGStimLF+eegData;
                        eventCounterLF = eventCounterLF + 1;
                    end
                else
                    continue
                end 

                if ~exist('ECGStim', 'var')
                    ECGStim = eegData;
                    eventCounter = 1;
                else
                    ECGStim = ECGStim+eegData;
                    eventCounter = eventCounter + 1;
                end 
                
                stft(eegData,EEG.srate,'Window',kaiser(250,5),'OverlapLength',75,'FFTLength',250);                
                set(gcf,'Units','inches');
                screenposition = [0 0 30 15];
                set(gcf,...
                    'PaperPosition',[0 0 screenposition(3:4)],...
                    'PaperSize',[screenposition(3:4)]);    
                %orient(fig,'landscape')                  
                print(gcf,fullfile(subTarFolder, strcat(ECGPrefix,'STFTECGStimuli_', emotion, '_', subject,'Trial-',trial,'Click-',num2str(str2num(click)),'_MedianlogPowerNoSquare_','1-30-80')),'-dpdf')  
                %segments = [1 250; 175 325; 250 500; 425 675; 600 850; 775 1025; 950 1200]
                Y = fft(eegData(firstPartL:firstPartH), fftLength);
                P2 = abs(Y/fftLength);
                %P2 = (1/(EEG.srate*N)) * abs(Y).^2;
                P1 = P2(1:N/2+1);                
                %P1 = P2(1:fftLength/2+1);
                P1(2:end-1) = 2*P1(2:end-1);
                stimulusFirstPart = P1;            
                %stimulusFirstHalfArrLowf = [stimulusFirstHalfArrLowf; median(log(stimulusFirstPart(1:partition)).^2)];
                %stimulusFirstHalfArrMidf = [stimulusFirstHalfArrMidf; median(log(stimulusFirstPart(partition:lastF)).^2)];
                stimulusFirstHalfArrLowf = [stimulusFirstHalfArrLowf; median(log(stimulusFirstPart(1:partition)))];
                stimulusFirstHalfArrMidf = [stimulusFirstHalfArrMidf; median(log(stimulusFirstPart(partition:lastF)))];                
                clear Y P2 P1
                
                Y = fft(eegData(secondPartL:secondPartH), fftLength);
                P2 = abs(Y/fftLength);
                %P2 = (1/(EEG.srate*N)) * abs(Y).^2;
                P1 = P2(1:N/2+1);                
                %P1 = P2(1:fftLength/2+1);
                P1(2:end-1) = 2*P1(2:end-1);
                stimulusSecondPart = P1; 
                %stimulusSecondHalfArrLowf = [stimulusSecondHalfArrLowf; median(log(stimulusSecondPart(1:partition)).^2)];
                %stimulusSecondHalfArrMidf = [stimulusSecondHalfArrMidf; median(log(stimulusSecondPart(partition:lastF)).^2)];
                stimulusSecondHalfArrLowf = [stimulusSecondHalfArrLowf; median(log(stimulusSecondPart(1:partition)))];
                stimulusSecondHalfArrMidf = [stimulusSecondHalfArrMidf; median(log(stimulusSecondPart(partition:lastF)))];                
                clear Y P2 P1
                %{
                figure('units','normalized','outerposition',[0 0 1 1]);
                plot(f(low_:high_), log(baselineFirstHalf(low_:high_)), 'r-', 'DisplayName', strcat('baselineFirstPart: ', num2str(baselineFirstHalfArrMidf(end))));
                hold on
                plot(f(low_:high_), log(baselineSecondHalf(low_:high_)), 'r--', 'DisplayName', strcat('baselineSecondPart: ', num2str(baselineSecondHalfArrMidf(end))));
                hold on
                plot(f(low_:high_), log(stimulusFirstPart(low_:high_)), 'b-', 'DisplayName', strcat('stimulusFirstPart: ', num2str(stimulusFirstHalfArrMidf(end))));
                hold on
                plot(f(low_:high_), log(stimulusSecondPart(low_:high_)), 'b--', 'DisplayName', strcat('stimulusSecondPart: ', num2str(stimulusSecondHalfArrMidf(end))));
                hold on
                legend;
                title(strcat('ECG ', emotion, ' ', subject,'Trial-',trial,'Click-',num2str(str2num(click))))
                saveas(gca, fullfile(subTarFolder, strcat('ECG_', emotion, '_', subject,'Trial-',trial,'Click-',num2str(str2num(click)),'_MedianlogPowerNoSquare_','1-30-80')), 'jpeg')
                close all
                %}                                
                save(fullfile(subTarFolder, strcat('ECG_', emotion, '_', subject,'Trial-',trial,'Click-',num2str(str2num(click)),'_MedianlogPowerNoSquare_','1-30-80')), 'eegData')
                %save(fullfile(subTarFolder,strcat('ECG_', subject, '_channelBand1_40_FinalProcessed_base')), 'baselineData');
            end            
            EEG = eeg_checkset( EEG );
        end
    display(subject)
    clearvars -except AllValence AllArousal baselineFirstHalfArrLowf baselineSecondHalfArrLowf stimulusFirstHalfArrLowf ...
        stimulusSecondHalfArrLowf baselineFirstHalfArrMidf baselineSecondHalfArrMidf stimulusFirstHalfArrMidf ...
        stimulusSecondHalfArrMidf firstPartL firstPartH secondPartL secondPartH ECGPrefix ECGBaseline ECGStim eventCounter ...
        eventCounterHF eventCounterLF ECGStimHF ECGStimLF

    close all    
    cd('C:\Processed_Emotions')
    rootFolder = pwd;    
    AllSubjects_ = load(fullfile(rootFolder, 'AllSubjects.mat'));
    AllSubjects_ = AllSubjects_.AllSubjects_; 
    ECGFlag = 1;
end
    familiarityTar = 'C:\Processed_Emotions\ECG_Files\CorrelationWithECG'; % Don't confuse with the variable name. It is not for familiarity wise correlation.
    save(strcat(ECGPrefix, 'baselineFirstHalfArrLowf_MedianlogPowerNoSquare_1-30-80.mat'), 'baselineFirstHalfArrLowf');
    save(strcat(ECGPrefix, 'baselineFirstHalfArrMidf_MedianlogPowerNoSquare_1-30-80.mat'), 'baselineFirstHalfArrMidf');
    save(strcat(ECGPrefix, 'baselineSecondHalfArrLowf_MedianlogPowerNoSquare_1-30-80.mat'), 'baselineSecondHalfArrLowf');
    save(strcat(ECGPrefix, 'baselineSecondHalfArrMidf_MedianlogPowerNoSquare_1-30-80.mat'), 'baselineSecondHalfArrMidf');

    save(strcat(ECGPrefix, 'stimulusFirstHalfArrLowf_MedianlogPowerNoSquare_1-30-80.mat'), 'stimulusFirstHalfArrLowf');
    save(strcat(ECGPrefix, 'stimulusFirstHalfArrMidf_MedianlogPowerNoSquare_1-30-80.mat'), 'stimulusFirstHalfArrMidf');
    save(strcat(ECGPrefix, 'stimulusSecondHalfArrLowf_MedianlogPowerNoSquare_1-30-80.mat'), 'stimulusSecondHalfArrLowf');
    save(strcat(ECGPrefix, 'stimulusSecondHalfArrMidf_MedianlogPowerNoSquare_1-30-80.mat'), 'stimulusSecondHalfArrMidf');
else
    familiarityTar = 'C:\Processed_Emotions\ECG_Files\CorrelationWithECG'; % Don't confuse with the variable name. It is not for familiarity wise correlation.
    baselineFirstHalfArrLowf = load(strcat(ECGPrefix, 'baselineFirstHalfArrLowf_MedianlogPowerNoSquare_1-30-80.mat'));
    baselineFirstHalfArrLowf = baselineFirstHalfArrLowf.baselineFirstHalfArrLowf;
    baselineFirstHalfArrMidf = load(strcat(ECGPrefix, 'baselineFirstHalfArrMidf_MedianlogPowerNoSquare_1-30-80.mat'));
    baselineFirstHalfArrMidf = baselineFirstHalfArrMidf.baselineFirstHalfArrMidf;
    baselineSecondHalfArrLowf = load(strcat(ECGPrefix, 'baselineSecondHalfArrLowf_MedianlogPowerNoSquare_1-30-80.mat'));
    baselineSecondHalfArrLowf = baselineSecondHalfArrLowf.baselineSecondHalfArrLowf;
    baselineSecondHalfArrMidf = load(strcat(ECGPrefix, 'baselineSecondHalfArrMidf_MedianlogPowerNoSquare_1-30-80.mat'));
    baselineSecondHalfArrMidf = baselineSecondHalfArrMidf.baselineSecondHalfArrMidf;

    stimulusFirstHalfArrLowf = load(strcat(ECGPrefix, 'stimulusFirstHalfArrLowf_MedianlogPowerNoSquare_1-30-80.mat'));
    stimulusFirstHalfArrLowf = stimulusFirstHalfArrLowf.stimulusFirstHalfArrLowf;
    stimulusFirstHalfArrMidf = load(strcat(ECGPrefix, 'stimulusFirstHalfArrMidf_MedianlogPowerNoSquare_1-30-80.mat'));
    stimulusFirstHalfArrMidf = stimulusFirstHalfArrMidf.stimulusFirstHalfArrMidf;
    stimulusSecondHalfArrLowf = load(strcat(ECGPrefix, 'stimulusSecondHalfArrLowf_MedianlogPowerNoSquare_1-30-80.mat'));
    stimulusSecondHalfArrLowf = stimulusSecondHalfArrLowf.stimulusSecondHalfArrLowf;
    stimulusSecondHalfArrMidf = load(strcat(ECGPrefix, 'stimulusSecondHalfArrMidf_MedianlogPowerNoSquare_1-30-80.mat'));
    stimulusSecondHalfArrMidf = stimulusSecondHalfArrMidf.stimulusSecondHalfArrMidf;
end
if ~isfile(fullfile(familiarityTar, strcat(ECGPrefix, 'ECGBaselineOverall', '.mat')))
    ECGBaseline = ECGBaseline / 40;
    save(fullfile(familiarityTar, strcat(ECGPrefix, 'ECGBaselineOverall', '.mat')), 'ECGBaseline');
else
    ECGBaseline = load(fullfile(familiarityTar, strcat(ECGPrefix, 'ECGBaselineOverall', '.mat')));   
    ECGBaseline = ECGBaseline.ECGBaseline;
end
stft(ECGBaseline,250,'Window',kaiser(250,5),'OverlapLength',75,'FFTLength',250);                
set(gcf,'Units','inches');
screenposition = [0 0 30 15];
set(gcf,...
    'PaperPosition',[0 0 screenposition(3:4)],...
    'PaperSize',[screenposition(3:4)]);    
%orient(fig,'landscape')                
print(gcf,fullfile(familiarityTar, strcat(ECGPrefix, 'ECGBaselineOverall')),'-dpdf')

if ~isfile(fullfile(familiarityTar, strcat(ECGPrefix, 'ECGStimuliOverall', '.mat')))
    ECGStim = ECGStim/eventCounter;
    save(fullfile(familiarityTar, strcat(ECGPrefix, 'ECGStimuliOverall', '.mat')), 'ECGStim')
else
    ECGStim = load(fullfile(familiarityTar, strcat(ECGPrefix, 'ECGStimuliOverall', '.mat')));
    ECGStim=ECGStim.ECGStim;
end

if ~isfile(fullfile(familiarityTar, strcat(ECGPrefix, 'ECGStimuliOverallHF', '.mat')))
    ECGStimHF = ECGStimHF/eventCounterHF;
    save(fullfile(familiarityTar, strcat(ECGPrefix, 'ECGStimuliOverallHF', '.mat')), 'ECGStimHF')
else
    ECGStimHF = load(fullfile(familiarityTar, strcat(ECGPrefix, 'ECGStimuliOverallHF', '.mat')));
    ECGStimHF=ECGStimHF.ECGStimHF;
end
if ~isfile(fullfile(familiarityTar, strcat(ECGPrefix, 'ECGStimuliOverallLF', '.mat')))
    ECGStimLF = ECGStimLF/eventCounterLF;
    save(fullfile(familiarityTar, strcat(ECGPrefix, 'ECGStimuliOverallLF', '.mat')), 'ECGStimLF')
else
    ECGStimLF = load(fullfile(familiarityTar, strcat(ECGPrefix, 'ECGStimuliOverallLF', '.mat')));
    ECGStimLF=ECGStimLF.ECGStimLF;
end

stft(ECGStim,250,'Window',kaiser(250,5),'OverlapLength',75,'FFTLength',250);                
set(gcf,'Units','inches');
screenposition = [0 0 30 15];
set(gcf,...
    'PaperPosition',[0 0 screenposition(3:4)],...
    'PaperSize',[screenposition(3:4)]);    
%orient(fig,'landscape')                
print(gcf,fullfile(familiarityTar, strcat(ECGPrefix, 'ECGStimuliOverall')),'-dpdf') 

stft(ECGStimHF,250,'Window',kaiser(250,5),'OverlapLength',75,'FFTLength',250);                
set(gcf,'Units','inches');
screenposition = [0 0 30 15];
set(gcf,...
    'PaperPosition',[0 0 screenposition(3:4)],...
    'PaperSize',[screenposition(3:4)]);    
%orient(fig,'landscape')                
print(gcf,fullfile(familiarityTar, strcat(ECGPrefix, 'ECGStimuliOverallHF')),'-dpdf') 
                
stft(ECGStimLF,250,'Window',kaiser(250,5),'OverlapLength',75,'FFTLength',250);                
set(gcf,'Units','inches');
screenposition = [0 0 30 15];
set(gcf,...
    'PaperPosition',[0 0 screenposition(3:4)],...
    'PaperSize',[screenposition(3:4)]);    
%orient(fig,'landscape')                
print(gcf,fullfile(familiarityTar, strcat(ECGPrefix, 'ECGStimuliOverallLF')),'-dpdf') 

diffSecond_FirstMidfStim = stimulusSecondHalfArrMidf-stimulusFirstHalfArrMidf;
[B, I] = sort(stimulusSecondHalfArrMidf-stimulusFirstHalfArrMidf);
lower_ = quantile(diffSecond_FirstMidfStim, 0.05);
upper_ = quantile(diffSecond_FirstMidfStim, 0.95);
ToRemoveIdx_ = [I(B<lower_); I(B>upper_)];
%ToRemoveIdx_=I(B<lower_);
stimulusSecondHalfArrMidfCorr = [];
stimulusFirstHalfArrMidfCorr = [];
for i = 1 : length(stimulusSecondHalfArrMidf)
    if any(i==ToRemoveIdx_)
        continue
    else
        stimulusSecondHalfArrMidfCorr = [stimulusSecondHalfArrMidfCorr, stimulusSecondHalfArrMidf(i)];
        stimulusFirstHalfArrMidfCorr = [stimulusFirstHalfArrMidfCorr, stimulusFirstHalfArrMidf(i)];
    end
end
%[h,p,ci,stats] = ttest(stimulusSecondHalfArrMidfCorr, stimulusFirstHalfArrMidfCorr,'tail','right');
%disp([h,p,ci])
%disp(stats)
fig=figure('units','normalized','outerposition',[0 0 1 1]);
stimulusSecondHalfArrMidfCorrSubMean = [];
stimulusFirstHalfArrMidfCorrSubMean = [];
for i = 1 : 2000
    randIdcs = randperm(length(stimulusSecondHalfArrMidfCorr),30);
    stimulusSecondHalfArrMidfCorrSubMean = [stimulusSecondHalfArrMidfCorrSubMean, mean(stimulusSecondHalfArrMidfCorr(randIdcs))];
    stimulusFirstHalfArrMidfCorrSubMean = [stimulusFirstHalfArrMidfCorrSubMean, mean(stimulusFirstHalfArrMidfCorr(randIdcs))];
end    
disp('Right: stimulusSecondHalfArrMidfCorrSubMean')
[h,p,ci,stats]=ttest(stimulusSecondHalfArrMidfCorrSubMean, stimulusFirstHalfArrMidfCorrSubMean, 'tail', 'right');
disp([h,p]); disp(ci);
disp(stats); 
fid = fopen(fullfile(familiarityTar, strcat(ECGPrefix, 'EEGSeg_ECGPartWiseSignTest.txt')),'a+');
fprintf(fid,'Right: stimulusSecondHalfArrMidfCorrSubMean - H:%f P:%f CI:[%f,%f] tstat:%f df:%f sd:%f\n',[h,p,ci,stats.tstat,stats.df,stats.sd]);
fclose(fid);        
disp('Left: stimulusSecondHalfArrMidfCorrSubMean')
[h,p,ci,stats]=ttest(stimulusSecondHalfArrMidfCorrSubMean, stimulusFirstHalfArrMidfCorrSubMean, 'tail', 'left');
disp([h,p]); disp(ci);
disp(stats);
fid = fopen(fullfile(familiarityTar, strcat(ECGPrefix, 'EEGSeg_ECGPartWiseSignTest.txt')),'a+');
fprintf(fid,'Left: stimulusSecondHalfArrMidfCorrSubMean - H:%f P:%f CI:[%f,%f] tstat:%f df:%f sd:%f\n',[h,p,ci,stats.tstat,stats.df,stats.sd]);
fclose(fid);
subplot(3,2,1);hist(stimulusSecondHalfArrMidfCorrSubMean);xlabel('Mean Distribution');ylabel('frequency');title('SecondHalfMid')
subplot(3,2,2);hist(stimulusFirstHalfArrMidfCorrSubMean);xlabel('Mean Distribution');ylabel('frequency');title('FirstHalfMid')    

diffSecond_FirstLowfStim = stimulusSecondHalfArrLowf-stimulusFirstHalfArrLowf;
[B, I] = sort(stimulusSecondHalfArrLowf-stimulusFirstHalfArrLowf);
lower_ = quantile(diffSecond_FirstLowfStim, 0.05);
upper_ = quantile(diffSecond_FirstLowfStim, 0.95);
ToRemoveIdx_ = [I(B<lower_); I(B>upper_)];
%ToRemoveIdx_=I(B<lower_); 
stimulusSecondHalfArrLowfCorr = [];
stimulusFirstHalfArrLowfCorr = [];
for i = 1 : length(stimulusSecondHalfArrMidf)
    if any(i==ToRemoveIdx_)
        continue
    else
        stimulusSecondHalfArrLowfCorr = [stimulusSecondHalfArrLowfCorr, stimulusSecondHalfArrLowf(i)];
        stimulusFirstHalfArrLowfCorr = [stimulusFirstHalfArrLowfCorr, stimulusFirstHalfArrLowf(i)];
    end
end    
stimulusSecondHalfArrLowfSubMean = [];
stimulusFirstHalfArrLowfSubMean = [];
for i = 1 : 2000
    randIdcs = randperm(length(stimulusSecondHalfArrLowfCorr),30);
    stimulusSecondHalfArrLowfSubMean = [stimulusSecondHalfArrLowfSubMean, mean(stimulusSecondHalfArrLowfCorr(randIdcs))];
    stimulusFirstHalfArrLowfSubMean = [stimulusFirstHalfArrLowfSubMean, mean(stimulusFirstHalfArrLowfCorr(randIdcs))];
end    
disp('Right: stimulusSecondHalfArrLowfSubMean')
[h,p,ci,stats]=ttest(stimulusSecondHalfArrLowfSubMean, stimulusFirstHalfArrLowfSubMean, 'tail', 'right');
disp([h,p]); disp(ci);
disp(stats);
fid = fopen(fullfile(familiarityTar, strcat(ECGPrefix, 'EEGSeg_ECGPartWiseSignTest.txt')),'a+');
fprintf(fid,'Right: stimulusSecondHalfArrLowfSubMean - H:%f P:%f CI:[%f,%f] tstat:%f df:%f sd:%f\n',[h,p,ci,stats.tstat,stats.df,stats.sd]);
fclose(fid);
disp('Left: stimulusSecondHalfArrLowfSubMean')
[h,p,ci,stats]=ttest(stimulusSecondHalfArrLowfSubMean, stimulusFirstHalfArrLowfSubMean, 'tail', 'left');
disp([h,p]); disp(ci);
disp(stats);
fid = fopen(fullfile(familiarityTar, strcat(ECGPrefix, 'EEGSeg_ECGPartWiseSignTest.txt')),'a+');
fprintf(fid,'Left: stimulusSecondHalfArrLowfSubMean - H:%f P:%f CI:[%f,%f] tstat:%f df:%f sd:%f\n',[h,p,ci,stats.tstat,stats.df,stats.sd]);
fclose(fid);
subplot(3,2,3);hist(stimulusSecondHalfArrLowfSubMean);xlabel('Mean Distribution');ylabel('frequency');title('SecondHalfLow')
subplot(3,2,4);hist(stimulusFirstHalfArrLowfSubMean);xlabel('Mean Distribution');ylabel('frequency');title('FirstHalfLow')        
disp('right: stimulusSecondHalfArrLowfSubMean./stimulusSecondHalfArrMidfCorrSubMean')
[h,p,ci,stats]=ttest(stimulusSecondHalfArrLowfSubMean./stimulusSecondHalfArrMidfCorrSubMean, ...
    stimulusFirstHalfArrLowfSubMean./stimulusFirstHalfArrMidfCorrSubMean, 'tail', 'right');
disp([h,p]); disp(ci);
disp(stats); 
fid = fopen(fullfile(familiarityTar, strcat(ECGPrefix, 'EEGSeg_ECGPartWiseSignTest.txt')),'a+');
fprintf(fid,'Right: stimulusSecondHalfArrLowfSubMean./stimulusSecondHalfArrMidfCorrSubMean - H:%f P:%f CI:[%f,%f] tstat:%f df:%f sd:%f\n',[h,p,ci,stats.tstat,stats.df,stats.sd]);
fclose(fid);
disp('left:stimulusSecondHalfArrLowfSubMean./stimulusSecondHalfArrMidfCorrSubMean')
[h,p,ci,stats]=ttest(stimulusSecondHalfArrLowfSubMean./stimulusSecondHalfArrMidfCorrSubMean, ...
    stimulusFirstHalfArrLowfSubMean./stimulusFirstHalfArrMidfCorrSubMean, 'tail', 'left')  ; 
subplot(3,2,5);hist(stimulusSecondHalfArrLowfSubMean./stimulusSecondHalfArrMidfCorrSubMean);xlabel('Mean Distribution');ylabel('frequency');title('SecondHalfLow/SecondHalfMid')
subplot(3,2,6);hist(stimulusFirstHalfArrLowfSubMean./stimulusFirstHalfArrMidfCorrSubMean);xlabel('Mean Distribution');ylabel('frequency');title('FirstHalfLow/FirstHalfMid')
disp([h,p]); disp(ci);
disp(stats); 
fid = fopen(fullfile(familiarityTar, strcat(ECGPrefix, 'EEGSeg_ECGPartWiseSignTest.txt')),'a+');
fprintf(fid,'Left: stimulusSecondHalfArrLowfSubMean./stimulusSecondHalfArrMidfCorrSubMean - H:%f P:%f CI:[%f,%f] tstat:%f df:%f sd:%f\n',[h,p,ci,stats.tstat,stats.df,stats.sd]);
fprintf(fid, '=================%s===============\n' , 'End Here')
fclose(fid);
set(fig,'Units','inches');
screenposition = [0 0 30 15];
set(fig,...
    'PaperPosition',[0 0 screenposition(3:4)],...
    'PaperSize',[screenposition(3:4)]);    
%orient(fig,'landscape')
print(fig,fullfile(familiarityTar, strcat(ECGPrefix, 'AllStimuli.pdf')),'-dpdf')  
disp('------------')
    
 %%% Baseline   
diffSecond_FirstMidfBase = baselineSecondHalfArrMidf-baselineFirstHalfArrMidf;
[B, I] = sort(diffSecond_FirstMidfBase);
lower_ = quantile(diffSecond_FirstMidfBase, 0.05);
upper_ = quantile(diffSecond_FirstMidfBase, 0.95);
ToRemoveIdx_B = [I(B<lower_); I(B>upper_)];
%ToRemoveIdx_=I(B<lower_);
baselineSecondHalfArrMidfCorr = [];
baselineFirstHalfArrMidfCorr = [];
for i = 1 : length(baselineSecondHalfArrMidf)
    if any(i==ToRemoveIdx_B)
        continue
    else
        baselineSecondHalfArrMidfCorr = [baselineSecondHalfArrMidfCorr, baselineSecondHalfArrMidf(i)];
        baselineFirstHalfArrMidfCorr = [baselineFirstHalfArrMidfCorr, baselineFirstHalfArrMidf(i)];
    end
end
%[h,p,ci,stats] = ttest(baselineSecondHalfArrMidfCorr, baselineFirstHalfArrMidfCorr, 'tail', 'right');
disp('Right:baselineSecondHalfArrMidfCorr')
[p,h,stats] = signrank(baselineSecondHalfArrMidfCorr, baselineFirstHalfArrMidfCorr, 'tail', 'right');
disp([h,p])
disp(stats)
fid = fopen(fullfile(familiarityTar, strcat(ECGPrefix, 'EEGSeg_ECGPartWiseSignTest.txt')),'a+');
fprintf(fid,'Right: baselineSecondHalfArrMidfCorr - H:%f P:%f zval:%f signedrank:%f\n',[h,p,stats.zval,stats.signedrank]);
fclose(fid);
disp('Left:baselineSecondHalfArrMidfCorr')
[p,h,stats] = signrank(baselineSecondHalfArrMidfCorr, baselineFirstHalfArrMidfCorr, 'tail', 'left');
disp([h,p])
disp(stats)
fid = fopen(fullfile(familiarityTar, strcat(ECGPrefix, 'EEGSeg_ECGPartWiseSignTest.txt')),'a+');
fprintf(fid,'Left: baselineSecondHalfArrMidfCorr - H:%f P:%f zval:%f signedrank:%f\n',[h,p,stats.zval,stats.signedrank]);
fclose(fid);
clear ToRemoveIdx_B B I 

diffSecond_FirstLowfBase = baselineSecondHalfArrLowf-baselineFirstHalfArrLowf;
[B, I] = sort(diffSecond_FirstLowfBase);
lower_ = quantile(diffSecond_FirstLowfBase, 0.05);
upper_ = quantile(diffSecond_FirstLowfBase, 0.95);
ToRemoveIdx_B = [I(B<lower_); I(B>upper_)];
%ToRemoveIdx_=I(B<lower_);
baselineSecondHalfArrLowfCorr = [];
baselineFirstHalfArrLowfCorr = [];
for i = 1 : length(baselineSecondHalfArrLowf)
    if any(i==ToRemoveIdx_B)
        continue
    else
        baselineSecondHalfArrLowfCorr = [baselineSecondHalfArrLowfCorr, baselineSecondHalfArrLowf(i)];
        baselineFirstHalfArrLowfCorr = [baselineFirstHalfArrLowfCorr, baselineFirstHalfArrLowf(i)];
    end
end
%[h,p,ci,stats] = ttest(baselineSecondHalfArrLowfCorr, baselineFirstHalfArrLowfCorr, 'tail', 'right');
disp('Right:baselineSecondHalfArrLowfCorr')
[p,h,stats] = signrank(baselineSecondHalfArrLowfCorr, baselineFirstHalfArrLowfCorr, 'tail', 'right');
disp([h,p])
disp(stats)
fid = fopen(fullfile(familiarityTar, strcat(ECGPrefix, 'EEGSeg_ECGPartWiseSignTest.txt')),'a+');
fprintf(fid,'Right: baselineSecondHalfArrLowfCorr - H:%f P:%f zval:%f signedrank:%f\n',[h,p,stats.zval,stats.signedrank]);
fclose(fid);
disp('Left:baselineSecondHalfArrLowfCorr')
[p,h,stats] = signrank(baselineSecondHalfArrLowfCorr, baselineFirstHalfArrLowfCorr, 'tail', 'left');
disp([h,p])
disp(stats)
fid = fopen(fullfile(familiarityTar, strcat(ECGPrefix, 'EEGSeg_ECGPartWiseSignTest.txt')),'a+');
fprintf(fid,'Left: baselineSecondHalfArrLowfCorr - H:%f P:%f zval:%f signedrank:%f\n',[h,p,stats.zval,stats.signedrank]);
fprintf(fid, '=================%s===============\n' , 'End Here')
fclose(fid);
disp('After this run temporalHistogramECG.py python script')
% temporalHistogramECG.py: Run it to create bar graph plotting p-value for
% different stistical conditioins across different combinations of first
% part and second part


%{
[h,p,ci,stats] = ttest(stimulusSecondHalfArrMidf, stimulusFirstHalfArrMidf,'tail','right');
[h,p,ci,stats] = ttest(stimulusSecondHalfArrLowf, stimulusFirstHalfArrLowf,'tail','right');
disp([h,p])
disp(ci)
disp(stats)
[h,p,ci,stats] = ttest(baselineSecondHalfArrLowf, baselineFirstHalfArrLowf, 'tail', 'right');
disp([h,p])
disp(ci)
disp(stats)
[h,p,ci,stats] = ttest(baselineSecondHalfArrMidf, baselineFirstHalfArrMidf, 'tail', 'right');
%}
