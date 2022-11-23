import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as  np
import glob
import pdb
import os
import pandas as pd
import scipy.io as io
from PIL import Image
import kendall_w as kw

sourceDir = '/media/forAll/Processed_Emotions/ToTransfer_Zuddler/ForMSAAnalysis/AllBands/EEGConnAnalysisEmt_Base'
EmotionFilesDir = '/home/zuddler/Processed_Emotions/EEGConnAnalysis'

RatingFile = pd.read_csv('/media/forAll/Processed_Emotions/ToTransfer_Zuddler/ForMSAAnalysis/ECGPowerByStimulus_StimWiseRatings.csv', index_col=0) # File is creating in program-/mnt/7CBFA0EC210FC340/Processed_Emotions/ECGEEGPowerCorrSubjectWise.py
count = 0
for sub_, trl_ in zip(RatingFile['subject'], RatingFile['trial']):
    RatingFile.loc[count, 'subTrials'] = sub_+'Trial-'+str(int(trl_))
    count = count + 1

## RatingFile = pd.read_csv('/media/forAll/Processed_Emotions/ToTransfer_Zuddler/ForMSAAnalysis/ECG_EEGChannel_Power_WithStimulusWiseRatings_Stimulus.csv', index_col=0) # File is creating in program-/mnt/7CBFA0EC210FC340/Processed_Emotions/ECGEEGPowerCorrSubjectWise.py

freqArr = ['UBeta'] #['Theta', 'Alpha', 'LBeta', 'UBeta', 'Gamma']
segIdxs = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
valenceRatingsGroupWise = [8.1, 7.6, 3.1, 1.8, 3.1, 1.9, 2.2, 2.2]
arousalRatingsGroupWise = [6.9, 7.8, 6.7, 6.6, 6.7, 6.9, 4.1, 4.3]
dominanRatingsGroupWise = [5.9, 6.7, 4.9, 4.7, 5.1, 5.2, 3.8, 4.3]
likingRatingsGroupWise = [4.6, 4.5, 2.9, 1.9, 2.7, 2.2, 2, 2.1]
familiaRatingsGroupWise = [3.9, 3.7, 2.9, 2.3, 2.7, 3.1, 2.2, 2.9]
relevanRatingsGroupWise = [3.5, 2.8, 2.1, 2, 2, 2, 1.9, 2.1]

def temporalDifferece():
    conds = ['Group-01','Group-02','Group-03','Group-04','Group-05','Group-06','Group-07','Group-08']
    for frq_ in freqArr:
        timeEvolve = {}
        linkDiffer = {}
        commLinks = {}
        RatingDir = {}

        for condIdx, cond_ in enumerate(conds):
            fullPath = os.path.join(sourceDir, cond_)
            EmtFilePath = os.path.join(EmotionFilesDir, cond_)
            print(cond_)

            for segId_ in np.arange(0,9):
                try:
                    connctOnly = io.loadmat(os.path.join(fullPath, 'phaseRandomConnAnalEEG_ConnectedElectrodes_'+frq_+'_'+'seg-'+str(segId_)+'.mat'))['conn_Remains']
                    connctOnly = np.transpose(connctOnly)
                    strLinks = [str(i[0])+'-'+str(i[1]) for i in connctOnly]

                    if segId_ == 0:
                        allLinks = strLinks
                    else:
                        allLinks.extend(strLinks)
                except:
                    continue

            newFiles = []
            for emtFile in glob.glob(os.path.join(EmtFilePath, '*seg-%s.npy' %str(0))):
                filePrefix = emtFile.split('seg-')[0]
                inRatingFile = filePrefix.split('/')[-1].split('-slor')[0].split('UBeta_')[1].split('_')[1].split('Click')[0]

                forIndexing = filePrefix.split('/')[-1] + '-' + cond_
                newFiles.extend([forIndexing])
                RatingDir[forIndexing] = []
                idxToSearch = np.where(RatingFile['subTrials'] == inRatingFile)[0]
                RatingDir[forIndexing].extend(np.unique(RatingFile.loc[idxToSearch, 'Valence']).tolist())
                RatingDir[forIndexing].extend(np.unique(RatingFile.loc[idxToSearch, 'Arousal']).tolist())
                RatingDir[forIndexing].extend(np.unique(RatingFile.loc[idxToSearch, 'Dominance']).tolist())
                RatingDir[forIndexing].extend(np.unique(RatingFile.loc[idxToSearch, 'Liking']).tolist())
                RatingDir[forIndexing].extend(np.unique(RatingFile.loc[idxToSearch, 'Familiarity']).tolist())
                RatingDir[forIndexing].extend(np.unique(RatingFile.loc[idxToSearch, 'Relevance']).tolist())

                timeEvolve[forIndexing] = []
                linkDiffer[forIndexing] = []
                commLinks[forIndexing] = []

                allLinks = np.unique(allLinks)
                allWeights = pd.DataFrame([], index=allLinks)

                for segId_ in np.arange(0,9):
                    connectionmatrix = np.load(filePrefix+'seg-'+str(segId_)+'.npy')
                    try:
                        connctOnly = io.loadmat(os.path.join(fullPath, 'phaseRandomConnAnalEEG_ConnectedElectrodes_'+frq_+'_'+'seg-'+str(segId_)+'.mat'))['conn_Remains']
                        connctOnly = np.transpose(connctOnly)
                        strLinks = [str(i[0])+'-'+str(i[1]) for i in connctOnly]

                        for lnk_ in strLinks:
                            allWeights.loc[lnk_, 'seg-%s' %str(segId_)] = connectionmatrix[int(lnk_.split('-')[0]), int(lnk_.split('-')[1])]
                    except:
                        continue

                allWeights.fillna(0, inplace=True)

                ################### Temporal Variability ###################
                for timeCol in allWeights.columns.values[:-1]:
                    print(timeCol)
                    nextTimeCol = 'seg-'+str(int(timeCol.split('-')[1])+1)
                    try:
                        timeDist = np.sqrt(np.sum(np.power(allWeights.loc[:, nextTimeCol]-allWeights.loc[:, timeCol], 2)))
                        timeEvolve[forIndexing].extend([np.round(timeDist, 2)])

                        linkDifference = sum(abs(allWeights.loc[:, nextTimeCol]-allWeights.loc[:, timeCol]))
                        linkDiffer[forIndexing].extend([linkDifference])

                        commonLink = sum((allWeights.loc[:, nextTimeCol]+allWeights.loc[:, timeCol]) > 1)
                        commLinks[forIndexing].extend([commonLink])

                    except:
                        timeDist = np.sqrt(np.sum(np.power(allWeights.loc[:, timeCol], 2)))
                        ## Why two time because if after seg-1, seg-3 is coming then
                        timeEvolve[forIndexing].extend([np.round(timeDist, 2)]) ## For seg-2 - seg-1
                        timeEvolve[forIndexing].extend([np.round(timeDist, 2)]) ## Since seg-2 is not present, the next iteration will go to seg-3 directly, hence this second one is for seg-2

                        linkDifference = sum(abs(allWeights.loc[:, timeCol]))
                        linkDiffer[forIndexing].extend([linkDifference])
                        linkDiffer[forIndexing].extend([linkDifference])

                        commonLink = sum((allWeights.loc[:, timeCol]) > 1)
                        commLinks[forIndexing].extend([commonLink])
                        commLinks[forIndexing].extend([commonLink])

                if timeCol == 'seg-6':
                    timeDist = np.sqrt(np.sum(np.power(allWeights.loc[:, 'seg-7'], 2)))
                    timeEvolve[forIndexing].extend([np.round(timeDist, 2)])

                    linkDifference = sum(abs(allWeights.loc[:, 'seg-7']))
                    linkDiffer[forIndexing].extend([linkDifference])

                    commonLink = sum((allWeights.loc[:, 'seg-7']) > 1)
                    commLinks[forIndexing].extend([commonLink])

            allRatingsVal = []
            allRatingsArl = []
            allRatingsDom = []
            allRatingsLik = []
            allRatingsFam = []
            allRatingsRel = []

            for key_ in newFiles:
                if len(RatingDir[key_]):
                    allRatingsVal.extend([RatingDir[key_][0]])
                    allRatingsArl.extend([RatingDir[key_][1]])
                    allRatingsDom.extend([RatingDir[key_][2]])
                    allRatingsLik.extend([RatingDir[key_][3]])
                    allRatingsFam.extend([RatingDir[key_][4]])
                    allRatingsRel.extend([RatingDir[key_][5]])

            avgVal = np.round(np.mean(allRatingsVal), 2)
            avgArl = np.round(np.mean(allRatingsArl), 2)
            avgDom = np.round(np.mean(allRatingsDom), 2)
            avgLik = np.round(np.mean(allRatingsLik), 2)
            avgFam = np.round(np.mean(allRatingsFam), 2)
            avgRel = np.round(np.mean(allRatingsRel), 2)

            for key_ in newFiles:
                if not len(RatingDir[key_]):
                    RatingDir[key_] = [avgVal, avgArl, avgDom, avgLik, avgFam, avgRel]

        timeEvolveFrame = pd.DataFrame.from_dict(timeEvolve)
        RatingDirFrame = pd.DataFrame.from_dict(RatingDir).transpose()
        # pdb.set_trace()
        # timeEvolveFrame = timeEvolveFrame.divide(timeEvolveFrame.max(axis=0).values)
        timeEvolveFrame = timeEvolveFrame.mean()

        timeEvolveFrame = pd.DataFrame(timeEvolveFrame, columns=['TV'])
        RatingDirFrame.rename({0:'Valence', 1:'Arousal', 2:'Dominance', 3:'Liking', 4:'Familiarity', 5:'Relevance'}, axis=1, inplace=True)
        timeEvolveFrame = pd.concat((timeEvolveFrame, RatingDirFrame), axis=1)

        chanIdx = {}
        for indx_ in timeEvolveFrame.index.values:
            timeEvolveFrame.loc[indx_, 'Group'] = indx_.split('_-')[1]
            timeEvolveFrame.loc[indx_, 'subject'] = indx_.split('Trial')[0].split('_')[2]
            timeEvolveFrame.loc[indx_, 'stimulus'] = indx_.split('_')[2].split('Click')[0]
            chanIdx[indx_] = indx_.split('-slor')[0]

        timeEvolveFrame.rename(chanIdx, axis=0, inplace=True)
        timeEvolveFrame.to_csv('TemporalVariability_ConnectomePlotsArrangement_EmotionFile.csv')
        tvGBY = timeEvolveFrame.groupby(by=['stimulus', 'subject']).mean()
        tvGBY.to_csv('TemporalVariability_ConnectomePlotsArrangement_StimulusWise.csv')
        pdb.set_trace()

        linkDiffStdFrame = pd.DataFrame()
        linkDiffStdFrame['std'] = np.argsort(linkDifferFrame.std())+1
        linkDiffStdFrame['GroupOrder'] = np.array(conds)[np.argsort(linkDifferFrame.std()).values]
        linkDiffStdFrame.set_index('GroupOrder', drop=True, inplace=True)

        timeVariableFrame = pd.DataFrame([])
        timeVariableFrame['tv'] = linkDifferFrame.std().round(2)
        timeVariableFrame['valence'] = valenceRatingsGroupWise
        timeVariableFrame['arousal'] = arousalRatingsGroupWise
        timeVariableFrame['dominance'] = dominanRatingsGroupWise
        timeVariableFrame['liking'] = likingRatingsGroupWise
        timeVariableFrame['familiarity'] = familiaRatingsGroupWise
        timeVariableFrame['relevance'] = relevanRatingsGroupWise
        timeVariableFrame.to_csv('TemporalVariability_ConnectomePlotsArrangement.csv')
        pdb.set_trace()

        ########################## Valence ######################
        ax = plt.subplot(gs[0, 0])
        sortedOrder = np.argsort(valenceRatingsGroupWise)
        for grp_ in linkDiffStdFrame.index.values:
            linkDiffStdFrame.loc[grp_, 'Valence'] = np.where(sortedOrder == int(grp_.split('Group-0')[1])-1)[0][0]+1

        pdb.set_trace()
        linkDiffStdFrame.plot.scatter(x='std', y='Valence', ax=ax)
        print('Valence')
        print(linkDiffStdFrame)
        d = np.polyfit(linkDiffStdFrame['std'], linkDiffStdFrame['Valence'], 1)
        f = np.poly1d(d)
        linkDiffStdFrame['predict'] = f(linkDiffStdFrame['std'])
        mse = np.round(np.sqrt(np.median(np.power(f(linkDiffStdFrame['std'])-linkDiffStdFrame['Valence'], 2))), 2)
        linkDiffStdFrame.plot.line('std', 'predict', ax=ax, color='Red', label = str(mse))
        linkDiffStdFrame.drop('Valence', axis=1, inplace=True)
        #ax.scatter(x=linkDiffStd, y=valenceRatingsGroupWise)
        ax.set_title('Valence', fontsize=25)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(np.arange(1, 9))
        ax.set_yticks(np.arange(1, 9))

        ########################## Arousal ######################
        ax = plt.subplot(gs[0, 1])
        sortedOrder = np.argsort(arousalRatingsGroupWise)
        for grp_ in linkDiffStdFrame.index.values:
            linkDiffStdFrame.loc[grp_, 'Arousal'] = np.where(sortedOrder == int(grp_.split('Group-0')[1])-1)[0][0]+1

        linkDiffStdFrame.plot.scatter('std', 'Arousal', ax=ax)
        print('Arousal')
        print(linkDiffStdFrame)
        d = np.polyfit(linkDiffStdFrame['std'], linkDiffStdFrame['Arousal'], 1)
        f = np.poly1d(d)
        linkDiffStdFrame['predict'] = f(linkDiffStdFrame['std'])
        mse = np.round(np.sqrt(np.median(np.power(f(linkDiffStdFrame['std'])-linkDiffStdFrame['Arousal'], 2))), 2)
        linkDiffStdFrame.plot.line('std', 'predict', ax=ax, color='Red', label = str(mse))

        linkDiffStdFrame.drop('Arousal', axis=1, inplace=True)
        #ax.scatter(x=linkDiffStd, y=valenceRatingsGroupWise)
        ax.set_title('Arousal', fontsize=25)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(np.arange(1, 9))
        ax.set_yticks(np.arange(1, 9))

        ########################## Dominance ######################
        ax = plt.subplot(gs[0, 2])
        sortedOrder = np.argsort(dominanRatingsGroupWise)
        for grp_ in linkDiffStdFrame.index.values:
            linkDiffStdFrame.loc[grp_, 'Dominance'] = np.where(sortedOrder == int(grp_.split('Group-0')[1])-1)[0][0]+1

        linkDiffStdFrame.plot.scatter('std', 'Dominance', ax=ax)
        print('Dominance')
        print(linkDiffStdFrame)
        d = np.polyfit(linkDiffStdFrame['std'], linkDiffStdFrame['Dominance'], 1)
        f = np.poly1d(d)
        linkDiffStdFrame['predict'] = f(linkDiffStdFrame['std'])
        mse = np.round(np.sqrt(np.median(np.power(f(linkDiffStdFrame['std'])-linkDiffStdFrame['Dominance'], 2))), 2)
        linkDiffStdFrame.plot.line('std', 'predict', ax=ax, color='Red', label = str(mse))

        linkDiffStdFrame.drop('Dominance', axis=1, inplace=True)
        #ax.scatter(x=linkDiffStd, y=valenceRatingsGroupWise)
        ax.set_title('Dominance', fontsize=25)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(np.arange(1, 9))
        ax.set_yticks(np.arange(1, 9))

        ########################## Liking ######################
        ax = plt.subplot(gs[1, 0])
        sortedOrder = np.argsort(likingRatingsGroupWise)
        for grp_ in linkDiffStdFrame.index.values:
            linkDiffStdFrame.loc[grp_, 'Liking'] = np.where(sortedOrder == int(grp_.split('Group-0')[1])-1)[0][0]+1

        linkDiffStdFrame.plot.scatter('std', 'Liking', ax=ax)
        print('Liking')
        print(linkDiffStdFrame)
        d = np.polyfit(linkDiffStdFrame['std'], linkDiffStdFrame['Liking'], 1)
        f = np.poly1d(d)
        linkDiffStdFrame['predict'] = f(linkDiffStdFrame['std'])
        mse = np.round(np.sqrt(np.median(np.power(f(linkDiffStdFrame['std'])-linkDiffStdFrame['Liking'], 2))), 2)
        linkDiffStdFrame.plot.line('std', 'predict', ax=ax, color='Red', label = str(mse))

        linkDiffStdFrame.drop('Liking', axis=1, inplace=True)
        #ax.scatter(x=linkDiffStd, y=valenceRatingsGroupWise)
        ax.set_title('Liking', fontsize=25)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(np.arange(1, 9))
        ax.set_yticks(np.arange(1, 9))

        ########################## Familiarity ######################
        ax = plt.subplot(gs[1, 1])
        sortedOrder = np.argsort(familiaRatingsGroupWise)
        for grp_ in linkDiffStdFrame.index.values:
            linkDiffStdFrame.loc[grp_, 'Familiarity'] = np.where(sortedOrder == int(grp_.split('Group-0')[1])-1)[0][0]+1

        linkDiffStdFrame.plot.scatter('std', 'Familiarity', ax=ax)
        print('Familiarity')
        print(linkDiffStdFrame)
        d = np.polyfit(linkDiffStdFrame['std'], linkDiffStdFrame['Familiarity'], 1)
        f = np.poly1d(d)
        linkDiffStdFrame['predict'] = f(linkDiffStdFrame['std'])
        mse = np.round(np.sqrt(np.median(np.power(f(linkDiffStdFrame['std'])-linkDiffStdFrame['Familiarity'], 2))), 2)
        linkDiffStdFrame.plot.line('std', 'predict', ax=ax, color='Red', label = str(mse))

        linkDiffStdFrame.drop('Familiarity', axis=1, inplace=True)
        #ax.scatter(x=linkDiffStd, y=valenceRatingsGroupWise)
        ax.set_title('Familiarity', fontsize=25)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(np.arange(1, 9))
        ax.set_yticks(np.arange(1, 9))

        ax = plt.subplot(gs[1, 2])
        sortedOrder = np.argsort(relevanRatingsGroupWise)
        for grp_ in linkDiffStdFrame.index.values:
            linkDiffStdFrame.loc[grp_, 'Relevance'] = np.where(sortedOrder == int(grp_.split('Group-0')[1])-1)[0][0]+1

        ########################## Relevance ######################
        linkDiffStdFrame.plot.scatter('std', 'Relevance', ax=ax)
        print('Relevance')
        print(linkDiffStdFrame)
        d = np.polyfit(linkDiffStdFrame['std'], linkDiffStdFrame['Relevance'], 1)
        f = np.poly1d(d)
        linkDiffStdFrame['predict'] = f(linkDiffStdFrame['std'])
        mse = np.round(np.sqrt(np.median(np.power(f(linkDiffStdFrame['std'])-linkDiffStdFrame['Relevance'], 2))), 2)
        linkDiffStdFrame.plot.line('std', 'predict', ax=ax, color='Red', label = str(mse))

        linkDiffStdFrame.drop('Relevance', axis=1, inplace=True)
        #ax.scatter(x=linkDiffStd, y=valenceRatingsGroupWise)
        ax.set_title('Relevance', fontsize=25)
        ax.set_ylabel('')
        ax.set_xlabel('Index of Sorted Deviation', fontsize=25)
        ax.set_xticks(np.arange(1, 9))
        ax.set_yticks(np.arange(1, 9))

        pdb.set_trace()
        print(linkDiffStdFrame)
        plt.savefig(os.path.join(sourceDir, 'RelationDynamicFCWithBehaviourScales.png'), bbox_inches='tight')
        plt.savefig(os.path.join(sourceDir, 'RelationDynamicFCWithBehaviourScales.pdf'), bbox_inches='tight')
        plt.close()
        plt.clf()

def hubsDifferece():
    conds = ['Group-01','Group-02','Group-03','Group-04','Group-05','Group-06','Group-07','Group-08']
    hubs = ['P9', 'T9', 'Fz', 'FC1', 'TP9', 'F10', 'O2', 'P1', 'Fpz', 'F2', 'Poz', 'P2', 'CP1', 'TP8', 'TP10'] # These hubs are assessed using program  172.17.25.241/media/forAll/Processed_Emotions/ToTransfer_Zuddler/ConnectionSummary.py and module allGroupsEEGConnMatrix()

    electrodeData = pd.read_csv('/media/forAll/Processed_Emotions/github/ForDataSetArticle/10-10ToHydroCel.csv', index_col=0)

    hubNodes = []
    for hub in hubs:
        if hub in electrodeData.index.values:
            hubNodes.extend(electrodeData.values[np.where(hub==electrodeData.index.values)[0][0]])
            print(hub)
        else:
            pdb.set_trace()

    hubNodes = np.sort(hubNodes)

    plt.figure(figsize=(20, 8.5))
    plt.rcParams['font.size']=20
    nRow = len(conds)
    nCol = 9
    gs = gridspec.GridSpec(nRow, nCol, height_ratios=np.ones(nRow), width_ratios=np.ones(nCol), left=0.02, right=0.99, top=0.995, bottom=0.006, hspace=0.99, wspace=0.2)
    row_ = 0
    col_ = 0

    for frq_ in freqArr:
        timeEvolve = {}
        linkDiffer = {}
        commLinks = {}
        for condIdx, cond_ in enumerate(conds):
            timeEvolve[cond_] = []
            linkDiffer[cond_] = []
            commLinks[cond_] = []

            fullPath = os.path.join(sourceDir, cond_)
            print(cond_)

            for segId_ in np.arange(0,9):
                try:
                    connctOnly = io.loadmat(os.path.join(fullPath, 'phaseRandomConnAnalEEG_ConnectedElectrodes_'+frq_+'_'+'seg-'+str(segId_)+'.mat'))['conn_Remains']
                    connctStat = io.loadmat(os.path.join(fullPath, 'phaseRandomConnAnalEEG_emotion_analysis_orig_tstatistics_'+frq_+'_'+'seg-'+str(segId_)+'.mat'))['origLabeling']
                    connctOnly = connctOnly.reshape(-1)

                    foundHubs = []
                    for nd_ in connctOnly:
                        nd_ = nd_+1 ## nd_+1 because python indexing starts from 0
                        if nd_ in hubNodes:
                            foundHubs.extend([nd_])

                    foundHubNodes = []
                    for nd_ in foundHubs:
                        foundHubNodes.extend(electrodeData.index.values[np.where(electrodeData.values==nd_)[0]])
                except:
                    continue

                dataFrame = pd.DataFrame(0, index=np.unique(foundHubNodes), columns=['count'])
                for nd_ in foundHubNodes:
                    dataFrame.loc[nd_, 'count'] = dataFrame.loc[nd_, 'count'] + 1

                ax = plt.subplot(gs[row_, col_])

                print(segId_)
                try:
                    dataFrame.plot.bar(ax = ax, fontsize=18)
                except:
                    if row_ == 0:
                        ax.set_title('seg-%s' %str(segId_), fontsize=18)
                    ax.set_xticklabels('')
                    ax.set_yticklabels('')
                    ax.legend([])
                    col_ = col_ + 1
                    continue

                ax.legend('')
                ax.tick_params(axis='x', labelrotation=50, pad=0, length = 0)
                col_ = col_ + 1

                if row_ == 0:
                    ax.set_title('seg-%s' %str(segId_), fontsize=18)

            row_ = row_ + 1
            col_ = 0

        print(sourceDir)
        plt.savefig(os.path.join(sourceDir, 'EvolutionHubs.png'), bbox_inches='tight')
        plt.savefig(os.path.join(sourceDir, 'EvolutionHubs.pdf'), bbox_inches='tight')
        plt.clf()
        plt.close()

def connectomePlotsArrang():
    conds = [['Group-01','Group-02','Group-03','Group-04'],['Group-05','Group-06','Group-07','Group-08']]
    for frq_ in freqArr:
        for condIdx, cond in enumerate(conds):

            plt.figure(figsize=(20, 8.5))
            plt.rcParams['font.size']=20
            nRow = len(cond)
            nCol = 9
            gs = gridspec.GridSpec(nRow, nCol, height_ratios=np.ones(nRow), width_ratios=np.ones(nCol), left=0.01, right=0.99, top=0.96, bottom=0.01, hspace=0.001, wspace=0.001)
            _row_ = 0
            _col_ = 0
            print(cond)

            for cond_ in cond:
                for segId_ in np.arange(0,9):
                    try:
                        segFile_ = os.path.join(sourceDir, cond_, 'New_EEGConnectomePlots_phaseRandomConnAnalEEG_ConnectedElectrodes_%s_seg-%s.PNG' %(frq_, segId_))
                        img_ = Image.open(segFile_)
                    except:
                        ax1 = plt.subplot(gs[_row_, _col_])
                        ax1.axis('off')
                        if _row_ == 0 and condIdx == 0:
                            ax1.set_title(str(segId_))

                        _col_ = _col_ + 1
                        continue

                    img_ = Image.open(segFile_)
                    width, height = img_.size
                    [left, top, right, bottom] = [1000, height/6, width/1.28, 7 * height / 9]
                    im1 = img_.crop((left, top, right, bottom))
                    #pdb.set_trace()
                    ax1 = plt.subplot(gs[_row_, _col_])
                    ax1.axis('off')
                    #ax1.set_yaxis('off')
                    ax1.imshow(im1)
                    if _col_ == 0:
                        ax1.set_ylabel(cond_)
                    if _row_ == 0 and condIdx == 0:
                        ax1.set_title(str(segId_))
                    _col_ = _col_ + 1

                _row_ = _row_ + 1
                _col_ = 0

            #plt.suptitle(frq_)
            plt.savefig(os.path.join(sourceDir, 'ConnectomePlotForFrequency-'+frq_+'-%s.png' %str(condIdx)), bbox_inches='tight')
            plt.savefig(os.path.join(sourceDir, 'ConnectomePlotForFrequency-'+frq_+'-%s.pdf' %str(condIdx)), bbox_inches='tight')
            plt.close()
            plt.clf()
