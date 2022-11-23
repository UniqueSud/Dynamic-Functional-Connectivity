import numpy as np
import os
import glob
import scipy.io as io
import scipy.signal as signal
from scipy.signal import butter, filtfilt, hilbert, welch
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from pyunicorn.timeseries import Surrogates
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from sklearn import manifold, datasets
import pdb

#time.sleep(1800000)
def PhaseCoherence(freq, timeSeries, FS):

    # Get parameters of input data
    nMeasures    = np.shape(timeSeries)[0]
    nSamples    = np.shape(timeSeries)[1]
    nSecs = nSamples / FS
    print('Number of measurements =', nMeasures)
    print('Number of time samples =', nSamples, '=', nSecs, 'seconds')

    # Calculate FFT for each measurement (spect is freq x measurements)
    spect = np.fft.fft(timeSeries, axis=1)

    # Normalise by amplitude
    spect = spect / abs(spect)

    # Find spectrum values for frequency bin of interest
    freqRes = 1 / nSecs;
    foibin = (np.round(freq / freqRes + 1) - 1).astype(np.int)
    spectFoi = spect[:,foibin]

    # Find individual phase angles per measurement at frequency of interest
    anglesFoi = np.arctan2(spectFoi.imag, spectFoi.real)

    # PC is root mean square of the sums of the cosines and sines of the angles
    PC = np.sqrt((np.sum(np.cos(anglesFoi)))**2 + (np.sum(np.sin(anglesFoi)))**2) / np.shape(anglesFoi)[0]

    # Print the value
    print('----------------------------------');
    print('Phase coherence value = ' + str("{0:.3f}".format(PC)));

    return PC

def angleCalc(t):
    dt_ = t[0]
    tmp_H = hilbert(dt_)  # dataPartFilt should either be in (1,nbSamples) or (nbSamples,)
    tmp_Ang = np.angle(tmp_H, deg=True)
    tmp_Amp = np.abs(tmp_H)

    return list([tmp_Ang])

sourceDir = '/data_hpc/home/ust/sudhakar/Processed_Emotions/ForMSAAnalysis/AllBands/CorrectedFiles/OnlyMatFiles'
target = os.path.join(sourceDir, 'EEGConnAnalysis')
os.chdir(sourceDir)

allConds = glob.glob('Baseline')
allConds.extend(glob.glob("Group-??"))

'''allConds = ['Baseline', 'Group-01', 'Group-02']
allConds = ['Group-03', 'Group-04']
allConds = ['Group-05', 'Group-06']
allConds = ['Group-07', 'Group-08']'''

segInf = [[0, 250], [175, 425], [350, 600], [525, 775], [700, 950], [875, 1125], [1050, 1300], [1225, 1475], [1400, 1650]]
bandInf = [[1, 4], [4, 8], [8, 13], [13, 20], [20, 30], [30, 40]]
bandName = ['Delta', 'Theta', 'Alpha', 'LBeta', 'UBeta', 'Gamma']
sampling_rate = 250

def PLVCalcEEG(allConds):
    ## Here I am Calculating Angles
    for cond_ in allConds:
        print(cond_)
        for file_ in glob.glob(os.path.join(cond_, "*.mat")):
            data = io.loadmat(file_)['data']
            targetDir = os.path.join(sourceDir, 'EEGConnAnalysis', file_.split('/')[0])
            if not os.path.isdir(targetDir):
                os.makedirs(targetDir)

            for segId in np.arange(len(segInf)):
                segData = data[:, segInf[segId][0]:segInf[segId][1]]
                for bandId in np.arange(len(bandInf)):

                    if not os.path.isfile('FilterCoeff_%s.npy' %bandName[bandId]):
                        filterCoef = signal.butter(5, np.array([bandInf[bandId][0], bandInf[bandId][1]]), 'bandpass', analog=False, fs=sampling_rate) # Creating filter with specific frequency band.
                        b = filterCoef[0]
                        a = filterCoef[1]
                        np.save('FilterCoeff_%s.npy' %bandName[bandId], filterCoef)
                        w, h = signal.freqz(b, a, fs=sampling_rate)  #, worN=np.arange(20), whole=True)

                        #w = w*(sampling_rate/(2*3.14))
                        #plt.plot(w[0:80], 20 * np.log10(abs(h[0:80])), 'b')
                        plt.plot(w, 20 * np.log10(abs(h)), 'b')
                        plt.title('Butterworth filter frequency response', fontsize=14)
                        plt.xlabel('Frequency [Hz]', fontsize=14)
                        plt.ylabel('Amplitude [dB]', color='b', fontsize=14)
                        plt.grid(which='both', axis='both')
                        plt.savefig(os.path.join('filterResponse-%s.png' %bandName[bandId]))
                        plt.clf()

                    else:
                        filterCoef = np.load('FilterCoeff_%s.npy' %bandName[bandId])

                    b = filterCoef[0]
                    a = filterCoef[1]
                    nCores = 35
                    dataPartFilt = signal.filtfilt(b, a, segData)

                    if not os.path.isfile(os.path.join(targetDir, bandName[bandId]+'_'+file_.split('/')[-1].split('.mat')[0]+'_angleData_seg-%s' %str(segId)+'.npy')):


                        if not os.path.isfile(bandName[bandId]+'_checkingFiltResp.png'):
                            from scipy.fft import fft
                            N = dataPartFilt.shape[1]
                            yf = fft(dataPartFilt[0, :])
                            yfOrig = fft(data[0, :])

                            T = 1.0/sampling_rate
                            xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
                            lastResort = len(xf[xf<40]) + 50
                            plt.plot(xf[0:lastResort], 2.0/N * np.abs(yf[0:lastResort]), 'b-', label=bandName[bandId])
                            plt.plot(xf[0:lastResort], 2.0/N * np.abs(yfOrig[0:lastResort]), 'c-', label='OriginalData')
                            plt.legend(loc='best')
                            #plt.ylim(-10, 50)
                            plt.grid()
                            #plt.show()
                            plt.savefig(bandName[bandId]+'_checkingFiltResp.png')
                            plt.close()


                        t0 = time.time()
                        print('Parallel Computation of Angle....')
                        tasks = []
                        for dt_ in dataPartFilt:
                            tasks.append([dt_])
                        results = Parallel(n_jobs=nCores)(delayed(angleCalc)(t) for t in tasks)
                        print('Done Parallel Calculation')
                        t1 = time.time()

                        print(t1-t0)
                        dtCount = 0
                        for dt_ in results:
                            tmp_ = np.reshape(dt_[0], (1, len(dt_[0])))

                            if dtCount == 0:
                                angle_data = tmp_
                            else:
                                angle_data = np.concatenate((angle_data, tmp_), axis=0)
                            dtCount = dtCount+1

                        np.save(os.path.join(targetDir, bandName[bandId]+'_'+file_.split('/')[-1].split('.mat')[0]+'_angleData_seg-%s' %str(segId)+'.npy'), angle_data)

                    else:
                        angle_data = np.load(os.path.join(targetDir, bandName[bandId]+'_'+file_.split('/')[-1].split('.mat')[0]+'_angleData_seg-%s' %str(segId)+'.npy'))

    ########### Calculating Phase Locking Value here among EEG Electrodes
                    if not os.path.isfile(os.path.join(targetDir, bandName[bandId]+'_'+file_.split('/')[-1].split('.mat')[0]+'_PLV_seg-%s' %str(segId)+'.npy')):
                        # Do the PLV formulation using python
                        # Calculate the angle and using these angles do the pLv calculation.
                        phase = np.zeros((angle_data.shape[0], angle_data.shape[0]))

                        for idx1, chan1 in enumerate(angle_data):
                            for idx2, chan2 in enumerate(angle_data):
                                try:
                                    angleDiff = (chan1-chan2)
                                    pdt = np.sum(np.exp(-1j*angleDiff))/len(chan2)
                                except:
                                    pdb.set_trace()

                                '''
                                The above one according to the article: "Phase locking value revisited: teaching new tricks to an old dog" should be done.
                                '''
                                phase[idx1, idx2] = np.angle(pdt)

                        fig, ax = plt.subplots(figsize=(1.5, 1.5))

                        ax.axis('off')
                        ax.imshow(phase)
                        numpyFileName = os.path.join(targetDir, bandName[bandId]+'_'+file_.split('/')[-1].split('.mat')[0]+'_PLV_seg-%s' %str(segId)+'.npy')
                        pngFileName = os.path.join(targetDir, bandName[bandId]+'_'+file_.split('/')[-1].split('.mat')[0]+'_PLV_seg-%s' %str(segId)+'.png')

                        np.save(numpyFileName, phase)
                        plt.savefig(pngFileName)
                        del phase

    #######################################
    ############### Calculation for Surrogate Data Generated after Phase Shuffling

                    obj = Surrogates(dataPartFilt)
                    surrogateData = obj.refined_AAFT_surrogates(original_data=dataPartFilt, n_iterations=200)

                    #if not os.path.isfile(os.path.join(targetDir, bandName[bandId]+'_'+file_.split('/')[-1].split('.mat')[0]+'_Surrogate_angleData_seg-%s' %str(segId)+'.npy')):
                    if not os.path.isfile(os.path.join(targetDir, bandName[bandId]+'_'+file_.split('/')[-1].split('.mat')[0]+'_Surrogate_angleData_seg-%s' %str(segId)+'.npy')):
                        if not os.path.isfile('FilterCoeff_%s.npy' %bandName[bandId]):
                            filterCoef = signal.butter(5, np.array([bandInf[bandId][0], bandInf[bandId][1]]), 'bandpass', analog=False, fs=sampling_rate) # Creating filter with specific frequency band.
                            b = filterCoef[0]
                            a = filterCoef[1]
                            np.save('FilterCoeff_%s.npy' %bandName[bandId], filterCoef)
                        else:
                            filterCoef = np.load('FilterCoeff_%s.npy' %bandName[bandId])

                        t0 = time.time()
                        print('Parallel Computation of Angle For Surrogate Data....')
                        tasks = []
                        for dt_ in surrogateData:
                            tasks.append([dt_])
                        results = Parallel(n_jobs=nCores)(delayed(angleCalc)(t) for t in tasks)
                        print('Done Parallel Calculation')
                        t1 = time.time()

                        print(t1-t0)
                        dtCount = 0
                        for dt_ in results:
                            tmp_ = np.reshape(dt_[0], (1, len(dt_[0])))

                            if dtCount == 0:
                                Sangle_data = tmp_
                            else:
                                Sangle_data = np.concatenate((Sangle_data, tmp_), axis=0)
                            dtCount = dtCount+1

                        np.save(os.path.join(targetDir, bandName[bandId]+'_'+file_.split('/')[-1].split('.mat')[0]+'_Surrogate__angleData_seg-%s' %str(segId)+'.npy'), Sangle_data)
                    else:
                        Sangle_data = np.load(os.path.join(targetDir, bandName[bandId]+'_'+file_.split('/')[-1].split('.mat')[0]+'_Surrogate__angleData_seg-%s' %str(segId)+'.npy'))

    ########### Calculating Phase Locking Value here among EEG Electrodes

                    if os.path.isfile(os.path.join(targetDir, bandName[bandId]+'_'+file_.split('/')[-1].split('.mat')[0]+'_Surrogate__PLV_seg-%s' %str(segId)+'.npy')):
                        # Do the PLV formulation using python
                        # Calculate the angle and using these angles do the pLv calculation.
                        Sphase = np.zeros((Sangle_data.shape[0], Sangle_data.shape[0]))

                        for idx1, chan1 in enumerate(Sangle_data):
                            for idx2, chan2 in enumerate(Sangle_data):
                                try:
                                    angleDiff = (chan1-chan2)
                                    pdt = np.sum(np.exp(-1j*angleDiff))/len(chan2)
                                except:
                                    pdb.set_trace()

                                '''
                                The above one according to the article: "Phase locking value revisited: teaching new tricks to an old dog" should be done.
                                '''
                                Sphase[idx1, idx2] = np.angle(pdt)

                        fig, ax = plt.subplots(figsize=(1.5, 1.5))

                        ax.axis('off')
                        ax.imshow(Sphase)
                        numpyFileName = os.path.join(targetDir, bandName[bandId]+'_'+file_.split('/')[-1].split('.mat')[0]+'_Surrogate__PLV_seg-%s' %str(segId)+'.npy')
                        pngFileName = os.path.join(targetDir, bandName[bandId]+'_'+file_.split('/')[-1].split('.mat')[0]+'_Surrogate__PLV_seg-%s' %str(segId)+'.png')

                        np.save(numpyFileName, Sphase)
                        plt.savefig(pngFileName)
                        del Sphase

def ConnectCalculatEEG(t):
    targetDir = t[0]
    bandName = t[1]
    bandId = t[2]
    segId = t[3]
    emt_file = t[4]

    noPermut = 5000
    alphaVal = 0.11
    #print(str(int(np.ceil(alphaVal*noPermut)))+' Largest Number')

    if not os.path.isfile(os.path.join(target,emt_file, 'ActivityBaselineDiff_%s_seg-%s.npy' %(bandName[bandId], str(segId)))):
        for idx, file_ in enumerate(glob.glob(os.path.join(targetDir, '%s*slor_PLV*seg-%s.npy' %(bandName[bandId], str(segId))))): # Here Normal Files:

            arr_ = file_.split('slor_PLV')
            #UBeta_Baseline_mit022-slor_Surrogate__angleData_seg-4
            print(file_.split('/')[-1])
            surrfile = arr_[0]+'slor_Surrogate__PLV'+arr_[1]
            print(surrfile.split('/')[-1])
            TorigData = np.load(file_)
            TsurrData = np.load(surrfile)

            TDiffData = TorigData-TsurrData
            TDiffData = np.reshape(TDiffData, (1, TDiffData.shape[0], TDiffData.shape[1]))
            #TorigData = np.reshape(TorigData, (1, TorigData.shape[0], TorigData.shape[1]))
            #TsurrData = np.reshape(TsurrData, (1, TsurrData.shape[0], TsurrData.shape[1]))
            if idx == 0:
                sampleSub = TDiffData
                #origData = TorigData
                #surrData = TsurrData
            else:
                sampleSub = np.concatenate((sampleSub, TDiffData), axis=0)
                #origData = np.concatenate((origData, TorigData), axis=0)
                #surrData = np.concatenate((surrData, TsurrData), axis=0)
        np.save(os.path.join(target,emt_file, 'ActivityBaselineDiff_%s_seg-%s.npy' %(bandName[bandId], str(segId))), sampleSub)
    else:
        sampleSub = np.load(os.path.join(target,emt_file, 'ActivityBaselineDiff_%s_seg-%s.npy' %(bandName[bandId], str(segId))))

    size_dt = sampleSub.shape
    #if not os.path.isfile(os.            pdb.set_trace()path.join(target,emt_file, 'CriticalValues&Threshold.npy')):
    if os.path.isfile(os.path.join(target,emt_file, 'CriticalValues&Threshold_%s_seg-%s.npy' %(bandName[bandId], str(segId)))):
        if not os.path.isfile(os.path.join(target,emt_file, 'emotion_analysis_orig_tstatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId)))):

            ## Find out mean here.
            MeanCalc = np.mean(sampleSub, axis=0)
            stdError = np.std(sampleSub, axis=0)/np.sqrt(size_dt[0])
            tstats = MeanCalc/stdError
            tstats[np.arange(0,128), np.arange(0,128)]=0
            print('Saving File')
            np.save(os.path.join(target,emt_file, 'emotion_analysis_orig_tstatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId))), tstats)
        else:
            tstats = np.load(os.path.join(target,emt_file, 'emotion_analysis_orig_tstatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId))))

### Read About Multistep Again.

        #flatArray = MeanCalc.flatten()
        #fifthMax = np.argsort(flatArray)[::-1][4]
        actMaxStat = np.max(tstats)

        if not os.path.isfile(os.path.join(target,emt_file, 'MaxStatsPermutationDistributionHistogram_%s_seg-%s.npy' %(bandName[bandId], str(segId)))):
            perMaxStat = []
            for shuffleCount in np.arange(noPermut):

                #MeanCalcPerm = np.zeros((size_dt[1],size_dt[2]), np.float32)
                noSampToFlip = np.random.randint(int(size_dt[0]/4),int(size_dt[0]/2))
                print(shuffleCount, noSampToFlip)
                samplToFlip = np.random.randint(0,size_dt[0],noSampToFlip)
                forProcessing = sampleSub.copy()

                for i in np.arange(size_dt[0]):
                    if i in samplToFlip:
                        forProcessing[i, :, :] = -forProcessing[i, :, :]

                #pdb.set_trace()
                MeanCalcPerm = np.mean(forProcessing, axis=0)
                stdErrorPerm = np.std(forProcessing, axis=0)/np.sqrt(size_dt[0])
                tstatsPerm = MeanCalcPerm/stdErrorPerm
                tstatsPerm[np.arange(0,128), np.arange(0,128)]=0
                perMaxStat.append(np.max(tstatsPerm))
                #perMaxStat.append(np.max(MeanCalcPerm))

            arr_ = np.array([i for i in perMaxStat]).squeeze()
            histVal = np.histogram(arr_, bins=100)
            np.save(os.path.join(target,emt_file, 'MaxStatsPermutationDistributionHistogram_%s_seg-%s.npy' %(bandName[bandId], str(segId))), histVal)
            np.save(os.path.join(target,emt_file, 'MaxStatsPermutationAArray_%s_seg-%s.npy' %(bandName[bandId], str(segId))), arr_)
        else:
            histVal = np.load(os.path.join(target,emt_file, 'MaxStatsPermutationDistributionHistogram_%s_seg-%s.npy' %(bandName[bandId], str(segId))), allow_pickle=True)
            arr_ = np.load(os.path.join(target,emt_file, 'MaxStatsPermutationAArray_%s_seg-%s.npy' %(bandName[bandId], str(segId))))

        idxs = np.where(histVal[1]>actMaxStat)
        np.save(os.path.join(target,emt_file, 'ActualMaxStatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId))), actMaxStat)
        p_val = sum(histVal[0][idxs[0][0]:])/noPermut
        arr_.sort()

        #### Why minus is here?
        critThrs = arr_[-int(np.ceil(alphaVal*noPermut)):][0]
        #print([p_val, alphaVal, critThrs, actMaxStat])
        np.save(os.path.join(target,emt_file, 'CriticalValues&Threshold_%s_seg-%s.npy' %(bandName[bandId], str(segId))), [p_val, alphaVal, critThrs, actMaxStat])
        if p_val < alphaVal:
            print('Reject The Null hypothesis')
            print(bandName[bandId], str(segId))
            print([p_val, alphaVal, critThrs, actMaxStat])
            print(np.where(tstats>=critThrs))

    else:
        [p_val, alphaVal, critThrs, actMaxStat] = np.load(os.path.join(target,emt_file, 'CriticalValues&Threshold_%s_seg-%s.npy' %(bandName[bandId], str(segId))))
        MaxStatsPermutationDistributionHistogram = np.load(os.path.join(target,emt_file, 'MaxStatsPermutationDistributionHistogram_%s_seg-%s.npy' %(bandName[bandId], str(segId))), allow_pickle=True)
        ActualMaxStatistics = np.load(os.path.join(target,emt_file, 'ActualMaxStatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId))))
        MeanCalc = np.load(os.path.join(target,emt_file, 'emotion_analysis_orig_tstatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId))))

        if p_val < alphaVal:
            pdb.set_trace()
            print('Reject The Null hypothesis')
            print(bandName[bandId], str(segId))
            print([p_val, alphaVal, critThrs, actMaxStat])
            print(np.where(MeanCalc>=critThrs))
        #else:
        #    print('Accept The Null hypothesis')

sourceDir = '/data_hpc/home/ust/sudhakar/Processed_Emotions/ForMSAAnalysis/AllBands/CorrectedFiles/OnlyMatFiles'
target = os.path.join(sourceDir, 'EEGConnAnalysis')
os.chdir(sourceDir)

allConds = glob.glob('Baseline')
allConds.extend(glob.glob("Group-??"))

allConds = ['Baseline', 'Group-01', 'Group-02']
allConds = ['Group-03', 'Group-04']
allConds = ['Group-05', 'Group-06']
allConds = ['Group-07', 'Group-08']

segInf = [[0, 250], [175, 425], [350, 600], [525, 775], [700, 950], [875, 1125], [1050, 1300], [1225, 1475], [1400, 1650]]
bandInf = [[1, 4], [4, 8], [8, 13], [13, 20], [20, 30], [30, 40]]
bandName = ['Delta', 'Theta', 'Alpha', 'LBeta', 'UBeta', 'Gamma']
sampling_rate = 250


def ConnectCalculatEEGEmt_Base(t):
    targetDirEmt = t[0]
    targetDirBase = t[1]
    bandName = t[2]
    bandId = t[3]
    segId = t[4]
    emt_file = t[5]

    noPermut = 5000
    alphaVal = 0.05
    #print(str(int(np.ceil(alphaVal*noPermut)))+' Largest Number')

    if not os.path.isfile(os.path.join(targetDirEmt, emt_file[0], 'phaseRandomConnAnalEEG_ActivityBaselineDiff_%s_seg-%s.npy' %(bandName[bandId], str(segId)))):
        for idx, file_ in enumerate(glob.glob(os.path.join(targetDirEmt, '%s*slor_PLV*seg-%s.npy' %(bandName[bandId], str(segId))))): # Here Normal Files:

            arr_ = file_.split('/')[-1].split('slor_PLV')
            idx_ = arr_[0].find('mit')
            subjName = arr_[0][idx_:idx_+6]
            frqBand = arr_[0].split('_')[0]
            seg_ = arr_[1].split('_')[1].split('.npy')[0]
            #UBeta_Baseline_mit022-slor_Surrogate__angleData_seg-4
            #print(file_.split('/')[-1])
            baseFile = os.path.join(targetDirBase, frqBand+'_Baseline_'+subjName+'-slor_PLV_'+seg_+'.npy')
            #print(baseFile.split('/')[-1])
            TorigData = np.load(file_)
            TsurrData = np.load(baseFile)

            TDiffData = TorigData-TsurrData
            TDiffData = np.reshape(TDiffData, (1, TDiffData.shape[0], TDiffData.shape[1]))
            #TorigData = np.reshape(TorigData, (1, TorigData.shape[0], TorigData.shape[1]))
            #TsurrData = np.reshape(TsurrData, (1, TsurrData.shape[0], TsurrData.shape[1]))
            if idx == 0:
                sampleSub = TDiffData
                #origData = TorigData
                #surrData = TsurrData
            else:
                sampleSub = np.concatenate((sampleSub, TDiffData), axis=0)
                #origData = np.concatenate((origData, TorigData), axis=0)
                #surrData = np.concatenate((surrData, TsurrData), axis=0)
        np.save(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_ActivityBaselineDiff_%s_seg-%s.npy' %(bandName[bandId], str(segId))), sampleSub)
    else:
        sampleSub = np.load(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_ActivityBaselineDiff_%s_seg-%s.npy' %(bandName[bandId], str(segId))))
        #np.save(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_ActivityBaselineDiff_%s_seg-%s.npy' %(bandName[bandId], str(segId))), sampleSub)

    size_dt = sampleSub.shape
    #if not os.path.isfile(os.            pdb.set_trace()path.join(target,emt_file, 'CriticalValues&Threshold.npy')):

    if os.path.isfile(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_CriticalValues&Threshold_%s_seg-%s.npy' %(bandName[bandId], str(segId)))):
        if not os.path.isfile(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_emotion_analysis_orig_tstatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId)))):

            ## Find out mean here.
            MeanCalc = np.mean(sampleSub, axis=0)
            stdError = np.std(sampleSub, axis=0)/np.sqrt(size_dt[0])
            tstats = MeanCalc/stdError
            tstats[np.arange(0,128), np.arange(0,128)]=0
            print('Saving File')
            np.save(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_emotion_analysis_orig_tstatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId))), tstats)
        else:
            tstats = np.load(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_emotion_analysis_orig_tstatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId))))
            io.savemat(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_emotion_analysis_orig_tstatistics_%s_seg-%s.mat' %(bandName[bandId], str(segId))), {'origLabeling':tstats})
            #np.save(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_emotion_analysis_orig_tstatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId))), tstats)

### Read About Multistep Again.
        actMaxStat = np.max(tstats)

        if not os.path.isfile(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_MaxStatsPermutationDistributionHistogram_%s_seg-%s.npy' %(bandName[bandId], str(segId)))):
            perMaxStat = []
            for shuffleCount in np.arange(noPermut):

                #MeanCalcPerm = np.zeros((size_dt[1],size_dt[2]), np.float32)
                noSampToFlip = np.random.randint(int(size_dt[0]/4),int(size_dt[0]/2))
                print(shuffleCount, noSampToFlip)
                samplToFlip = np.random.randint(0,size_dt[0],noSampToFlip)
                forProcessing = sampleSub.copy()

                for i in np.arange(size_dt[0]):
                    if i in samplToFlip:
                        forProcessing[i, :, :] = -forProcessing[i, :, :]

                #pdb.set_trace()
                MeanCalcPerm = np.mean(forProcessing, axis=0)
                stdErrorPerm = np.std(forProcessing, axis=0)/np.sqrt(size_dt[0])
                tstatsPerm = MeanCalcPerm/stdErrorPerm
                tstatsPerm[np.arange(0,128), np.arange(0,128)]=0
                perMaxStat.append(np.max(tstatsPerm))
                #perMaxStat.append(np.max(MeanCalcPerm))

            arr_ = np.array([i for i in perMaxStat]).squeeze()
            histVal = np.histogram(arr_, bins=100)
            np.save(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_MaxStatsPermutationDistributionHistogram_%s_seg-%s.npy' %(bandName[bandId], str(segId))), histVal)
            np.save(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_MaxStatsPermutationAArray_%s_seg-%s.npy' %(bandName[bandId], str(segId))), arr_)
        else:
            histVal = np.load(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_MaxStatsPermutationDistributionHistogram_%s_seg-%s.npy' %(bandName[bandId], str(segId))), allow_pickle=True)
            arr_ = np.load(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_MaxStatsPermutationAArray_%s_seg-%s.npy' %(bandName[bandId], str(segId))), allow_pickle=True)

        idxs = np.where(histVal[1]>actMaxStat)
        np.save(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_ActualMaxStatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId))), actMaxStat)

        if (len(idxs[0])):
            p_val = sum(histVal[0][idxs[0][0]:])/noPermut
        else:
            p_val = 1/noPermut

        arr_.sort()
        #### Why minus is here?
        critThrs = arr_[-int(np.ceil(alphaVal*noPermut)):][0]
        #print([p_val, alphaVal, critThrs, actMaxStat])
        np.save(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_CriticalValues&Threshold_%s_seg-%s.npy' %(bandName[bandId], str(segId))), [p_val, alphaVal, critThrs, actMaxStat])
        if p_val < alphaVal:
            print('Reject The Null hypothesis')
            print(bandName[bandId], str(segId))
            print([p_val, alphaVal, critThrs, actMaxStat])
            print(np.where(tstats>=critThrs))
            conn_Remains = np.where(tstats>=critThrs)
            np.save(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_ConnectedElectrodes_%s_seg-%s.npy' %(bandName[bandId], str(segId))), conn_Remains)
            io.savemat(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_ConnectedElectrodes_%s_seg-%s.mat' %(bandName[bandId], str(segId))), {'conn_Remains':conn_Remains})

    else:
        if (segId == 0) and (bandName[bandId]=='Gamma'):
            pdb.set_trace()

        [p_val, alphaVal, critThrs, actMaxStat] = np.load(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_CriticalValues&Threshold_%s_seg-%s.npy' %(bandName[bandId], str(segId))))
        MaxStatsPermutationDistributionHistogram = np.load(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_MaxStatsPermutationDistributionHistogram_%s_seg-%s.npy' %(bandName[bandId], str(segId))), allow_pickle=True)
        ActualMaxStatistics = np.load(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_ActualMaxStatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId))))
        MeanCalc = np.load(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_emotion_analysis_orig_tstatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId))))

        if p_val < alphaVal:
            print('Reject The Null hypothesis')
            print(bandName[bandId], str(segId))
            print([p_val, alphaVal, critThrs, actMaxStat])
            print(np.where(MeanCalc>=critThrs))
            conn_Remains = np.where(MeanCalc>=critThrs)
            np.save(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_ConnectedElectrodes_%s_seg-%s.npy' %(bandName[bandId], str(segId))), conn_Remains)
            io.savemat(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_ConnectedElectrodes_%s_seg-%s.mat' %(bandName[bandId], str(segId))), {'conn_Remains':conn_Remains})

        ## Doing research is for my natural survival. Niyat karma is the source of truth, knowledge and bliss. Lakhs of failure are more joyful than no action or quit.
        ## I found myself never ever happy in criticising anything or anybody. That means I can never be a critic but I am always happy about helping people. Politics is never ever my field.
        ## Lakhs of failure are acceptable then not moving. Manjoor hai lakhon asaphaltayen nahi hai manjoor na chalna, Isliye niyat karma karte chalo.
        ## Karma is for eternal happiness. That's why krishna said "Karma For The Sak eof Karma". Niyat Karma is blissful.
        #else:
        #    print('Accept The Null hypothesis')

sourceDir = '/data_hpc/home/ust/sudhakar/Processed_Emotions/ForMSAAnalysis/AllBands/CorrectedFiles/OnlyMatFiles'
target = os.path.join(sourceDir, 'EEGConnAnalysis')
os.chdir(sourceDir)

allConds = glob.glob('Baseline')
allConds.extend(glob.glob("Group-??"))

'''allConds = ['Baseline', 'Group-01', 'Group-02']
allConds = ['Group-03', 'Group-04']
allConds = ['Group-05', 'Group-06']
allConds = ['Group-07', 'Group-08']'''

segInf = [[0, 250], [175, 425], [350, 600], [525, 775], [700, 950], [875, 1125], [1050, 1300], [1225, 1475], [1400, 1650]]
bandInf = [[1, 4], [4, 8], [8, 13], [13, 20], [20, 30], [30, 40]]
bandName = ['Delta', 'Theta', 'Alpha', 'LBeta', 'UBeta', 'Gamma']
sampling_rate = 250

def permutationStats(emt_file):
    from joblib import Parallel, delayed

    tasks = []
    nCores = 1

    if isinstance(emt_file, str):
        targetDir = os.path.join(target, emt_file)
        for segId in np.arange(len(segInf)):
            for bandId in np.arange(len(bandInf)):
                tasks.append([targetDir, bandName, bandId, segId, emt_file])

        Parallel(n_jobs=nCores)(delayed(ConnectCalculatEEG)(t) for t in tasks)

    elif len(emt_file) > 1:
        target = os.path.join(sourceDir, 'EEGConnAnalysisEmt_Base')
        targetDirEmt = os.path.join(target, emt_file[0])
        targetDirBase = os.path.join(target, emt_file[1])
        for segId in np.arange(len(segInf)):
            for bandId in np.arange(len(bandInf)):
                tasks.append([targetDirEmt, targetDirBase, bandName, bandId, segId, emt_file])

        Parallel(n_jobs=nCores)(delayed(ConnectCalculatEEGEmt_Base)(t) for t in tasks)

########################### Reading Channel Locations ############################

def readChanLocs():
    import xml.etree.ElementTree as ET
    tree = ET.parse('/data_hpc/home/ust/sudhakar/Processed_Emotions/ForMSAAnalysis/mit003_bio_20191201_051113.mff/coordinates.xml')
    root = tree.getroot()

    chanLocs = pd.DataFrame([], index=np.arange(1, 133), columns=['chanName', 'x', 'y', 'z'])

    for child in root:
        print(child.tag, child.attrib)
        #pdb.set_trace()
        if child.tag.split('}')[1] == 'sensorLayout':
            for child1 in child:
                print('----------child1----------')
                print(child1.tag, child1.attrib)

                if child1.tag.split('}')[1] == 'sensors':
                    chanIdx = 1
                    for child2 in child1:
                        print('--------child2------------')
                        print(child2.tag, child2.attrib)

                        if child2.tag.split('}')[1] == 'sensor':
                            for child3 in child2:
                                print(child3.tag, child3.attrib)
                                if child3.tag.split('}')[1] == 'number':
                                    chanLocs.loc[chanIdx, 'chanName'] = 'E'+child3.text
                                if child3.tag.split('}')[1] == 'x':
                                    chanLocs.loc[chanIdx, 'x'] = float(child3.text)
                                if child3.tag.split('}')[1] == 'y':
                                    chanLocs.loc[chanIdx, 'y'] = float(child3.text)
                                if child3.tag.split('}')[1] == 'z':
                                    chanLocs.loc[chanIdx, 'z'] = float(child3.text)

                        chanIdx += 1

    ########### Now I have to perform 3d to 2d projections #################

    chanLocs.set_index('chanName', inplace=True)
    minnArray = []
    for chanIdx1 in chanLocs.index.values:
        minn = 1000
        for chanIdx2 in chanLocs.index.values:
            if chanIdx1 != chanIdx2:
                coordDist = np.sqrt((chanLocs.loc[chanIdx1, 'x']-chanLocs.loc[chanIdx2, 'x'])**2 + (chanLocs.loc[chanIdx1, 'y']-chanLocs.loc[chanIdx2, 'y'])**2
                + (chanLocs.loc[chanIdx1, 'z']-chanLocs.loc[chanIdx2, 'z'])**2)

                if minn > coordDist:
                    minn = np.round(coordDist, 2)

        minnArray.extend([minn])

    thrs = np.mean(minnArray) + (3*np.std(minnArray)) ### Threshold for long and short range connections

    longRange = {}
    shortRange = {}
    for chanIdx1 in chanLocs.index.values:
        longRange[chanIdx1] = []
        shortRange[chanIdx1] = []
        minn = 1000

        for chanIdx2 in chanLocs.index.values:
            if chanIdx1 != chanIdx2:
                coordDist = np.sqrt((chanLocs.loc[chanIdx1, 'x']-chanLocs.loc[chanIdx2, 'x'])**2 + (chanLocs.loc[chanIdx1, 'y']-chanLocs.loc[chanIdx2, 'y'])**2
                + (chanLocs.loc[chanIdx1, 'z']-chanLocs.loc[chanIdx2, 'z'])**2)

                if coordDist > thrs:
                    longRange[chanIdx1].extend([chanIdx2])
                else:
                    shortRange[chanIdx1].extend([chanIdx2])

    chanLocs2d = chanLocs.copy()
    chanLocs2d['x'] = chanLocs['x']/chanLocs['z']
    chanLocs2d['y'] = chanLocs['y']/chanLocs['z']
    chanLocs2d['z'] = chanLocs['z']/chanLocs['z']

    return list([longRange, shortRange, chanLocs])

#######################3 Here Connectivity Based Dynamic Correlation among Groups is being Calculated.
def connecBasedDynamicCorr():

    #from adjustText import adjust_text
    scale_ = ['Group-01', 'Group-02', 'Group-03', 'Group-04', 'Group-05', 'Group-06', 'Group-07', 'Group-08']
    target = os.path.join(sourceDir, 'EEGConnAnalysisEmt_Base')
    max_ = 0
    commNodes = {}
    commScore = {}
    LinkNodes = {}
    LinkScore = {}
    meanDist = {}
    medianDist = {}
    connDict = {}

    #fig = plt.figure(figsize=(20,7))
    fig = plt.figure(figsize=(20,10))
    plt.rcParams['font.size']=33
    nRows = 3
    nCols = 9
    gs = gridspec.GridSpec(nRows, nCols, height_ratios=np.ones(nRows), width_ratios=np.ones(nCols), hspace=0.15, wspace=0.45, left=0.015, right=0.975, bottom=0.04, top=0.99)
    row_ = 0
    col_ = 0
    renameDict = {'Group-01':'G1','Group-02':'G2','Group-03':'G3','Group-04':'G4','Group-05':'G5','Group-06':'G6','Group-07':'G7','Group-08':'G8'}

    #### Getting values from Channels' coordinates
    longRange, shortRange, chanLocs = readChanLocs()
    longShortEmtFFrqWiseDict = {}

    for bandId in np.arange(len(bandName)):
        for segId in np.arange(len(segInf)):
            ## First of all collect nodes forming edges for all the groups

            linkDataF = pd.DataFrame(0, index=['Group-01', 'Group-02', 'Group-03', 'Group-04', 'Group-05', 'Group-06', 'Group-07', 'Group-08'], columns=[''])
            longShortEmtF = pd.DataFrame(0, index=['Group-01', 'Group-02', 'Group-03', 'Group-04', 'Group-05', 'Group-06', 'Group-07', 'Group-08'], columns=['L', 'S'])

            if bandName[bandId] in ['Theta', 'LBeta', 'UBeta']:
                ax = plt.subplot(gs[row_, col_])

            for emtIdx, emt_file in enumerate(allConds[1:]):
                if 'Baseline' not in emt_file:
                    print(emt_file)
                    targetDirEmt = os.path.join(target, emt_file)
                    #origData = np.load(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_emotion_analysis_orig_tstatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId))))
                    try:
                        ## These are the significant connections
                        conn_Remains = np.load(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_ConnectedElectrodes_%s_seg-%s.npy' %(bandName[bandId], str(segId))))
                        io.savemat(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_ConnectedElectrodes_%s_seg-%s.mat' %(bandName[bandId], str(segId))), {'conn_Remains':conn_Remains})

                    except:
                        continue

                    for lnk_ in np.transpose(conn_Remains):
                        linkDataF.loc[emt_file, str(lnk_[0])+'-'+str(lnk_[1])] = 1

                        lnk_ = lnk_ + 1
                        if ('E'+str(lnk_[0]) in shortRange['E'+str(lnk_[1])]) or ('E'+str(lnk_[1]) in shortRange['E'+str(lnk_[0])]):
                            longShortEmtF.loc[emt_file, 'S'] = longShortEmtF.loc[emt_file, 'S'] + 1
                        elif ('E'+str(lnk_[0]) in longRange['E'+str(lnk_[1])]) or ('E'+str(lnk_[1]) in longRange['E'+str(lnk_[0])]):
                            longShortEmtF.loc[emt_file, 'L'] = longShortEmtF.loc[emt_file, 'L'] + 1
                        else:
                            pdb.set_trace()

                    if emtIdx == 0:
                        emtRunSeq = emt_file
                        connArr = conn_Remains
                    else:
                        emtRunSeq = emtRunSeq+','+emt_file
                        connArr = np.concatenate((connArr, conn_Remains), axis=1)

                    connDict[emt_file] = conn_Remains

            ########## This is to create figure based on longrange and short range connections.
            if bandName[bandId] in ['Theta', 'LBeta', 'UBeta']:

                if col_ == 0:
                    longShortEmtFFrqWiseDict[bandName[bandId]] = longShortEmtF.values
                else:
                    longShortEmtFFrqWiseDict[bandName[bandId]] = np.concatenate((longShortEmtFFrqWiseDict[bandName[bandId]], longShortEmtF.values), axis=0)

                longShortEmtF.plot.bar(ax=ax, stacked=True)

                '''if row_ == (nRows-1) and col_ == (nCols-1):
                    longShortEmtF.plot.bar(ax=ax, stacked=True)
                else:
                    longShortEmtF.plot.bar(ax=ax, stacked=True, legend=None)'''

                '''longShortEmtF.plot.scatter(x='L', y='S', ax=ax)
                #ax.set_ylim([-0.2, 3.2])

                for k, v in longShortEmtF.iterrows():
                    #texts.extend([plt.text(v[0], v[1], renameDict[k])])
                    ax.annotate(renameDict[k], v, xytext=(10,-5), textcoords='offset points', family='sans-serif', fontsize=15)

                #ax.tick_params( axis='both', which='both', left=False, right=False, bottom=False, top=False, labelbottom=False, labelleft=False)
                ax.set_xlabel('')
                if col_ == 0:
                    ax.set_ylabel(bandName[bandId], rotation=90)
                else:
                    ax.set_ylabel('')
                    #ax.set_yticklabels('')'''

                ax.set_xticklabels('')
                if row_ == (nRows-1) and col_ == (nCols-1):
                    print('')
                    ax.legend(fontsize=25, labelcolor='linecolor', columnspacing=0.5, handletextpad=0.1, framealpha=0.3, handlelength=0.6)
                else:
                    ax.legend().set_visible(False)
                col_ = col_ + 1

            ############################################

            linkDataF.fillna(0, inplace=True)
            #### Finding out Common Nodes
            NodesDict = {}
            for node_ in connArr.reshape(-1):
                if node_ not in NodesDict.keys():
                    NodesDict[node_] = 1
                else:
                    NodesDict[node_] = NodesDict[node_] + 1

            dfDegree = pd.DataFrame.from_dict(NodesDict, orient='index')
            commNodes['seg-%s_band-%s' %(str(segId), bandName[bandId])] = dfDegree.iloc[np.where(dfDegree>=4)[0]].index.values
            commNodes['seg-%s_band-%s' %(str(segId), bandName[bandId])] = np.concatenate((commNodes['seg-%s_band-%s' %(str(segId), bandName[bandId])], [len(dfDegree)]))

            commScore['seg-%s_band-%s' %(str(segId), bandName[bandId])] = dfDegree.iloc[np.where(dfDegree>=4)[0]].values.reshape(-1)
            commScore['seg-%s_band-%s' %(str(segId), bandName[bandId])] = np.concatenate((commScore['seg-%s_band-%s' %(str(segId), bandName[bandId])], [len(dfDegree)]))
            del NodesDict
            del dfDegree

            #### Finding out Common Links
            LinkDict = {}
            for link_ in np.transpose(connArr):
                linkStr = str(link_[0])+'-'+str(link_[1])
                #print(linkStr)
                if linkStr not in LinkDict.keys():
                    LinkDict[linkStr] = 1
                else:
                    LinkDict[linkStr] = LinkDict[linkStr] + 1

            dfDegree = pd.DataFrame.from_dict(LinkDict, orient='index')
            LinkNodes['seg-%s_band-%s' %(str(segId), bandName[bandId])] = dfDegree.iloc[np.where(dfDegree>=2)[0]].index.values
            LinkNodes['seg-%s_band-%s' %(str(segId), bandName[bandId])] = np.concatenate((LinkNodes['seg-%s_band-%s' %(str(segId), bandName[bandId])], [len(dfDegree)]))

            LinkScore['seg-%s_band-%s' %(str(segId), bandName[bandId])] = dfDegree.iloc[np.where(dfDegree>=2)[0]].values.reshape(-1)
            LinkScore['seg-%s_band-%s' %(str(segId), bandName[bandId])] = np.concatenate((LinkScore['seg-%s_band-%s' %(str(segId), bandName[bandId])], [len(dfDegree)]))
            del LinkDict
            del dfDegree

            ## Now collect the connections values to compare across the groups.
            DataF = pd.DataFrame(0, index = np.arange(connArr.shape[1]), columns = allConds[1:])
            for emtIdx, emt_file in enumerate(allConds[1:]):
                if 'Baseline' not in emt_file:
                    targetDirEmt = os.path.join(target, emt_file)
                    try:
                        origData = np.load(os.path.join(targetDirEmt, 'phaseRandomConnAnalEEG_emotion_analysis_orig_tstatistics_%s_seg-%s.npy' %(bandName[bandId], str(segId))))
                    except:
                        pdb.set_trace()

                    for idx_, connNodes in enumerate(np.transpose(connArr)):
                        linkDataF.loc[emt_file, str(connNodes[0])+'-'+str(connNodes[1])] = np.round(origData[connNodes[0], connNodes[1]], 3)
                        DataF.loc[idx_, emt_file] = np.round(origData[connNodes[0], connNodes[1]], 3)

            print(os.path.join(target, 'LinkWeightFrame_%s_seg-%s.npy' %(bandName[bandId], str(segId))))
            linkDataF.to_csv(os.path.join(target, 'LinkWeightFrame_%s_seg-%s.csv' %(bandName[bandId], str(segId))))

            distance = pd.DataFrame(0, index = np.sort(allConds[1:]), columns = np.sort(allConds[1:]))

            for emt_file1 in allConds[1:]:
                for emt_file2 in allConds[1:]:
                    diff_ = (DataF.loc[:, emt_file1] - DataF.loc[:, emt_file2])
                    distance.loc[emt_file1, emt_file2] = np.sqrt(sum(np.power(diff_,2)))


            distance.to_csv(os.path.join(target, 'freq_%s_seg-%s.csv' %(bandName[bandId], str(segId))))
            if max_ < np.max(distance.to_numpy()):
                max_ = np.max(distance.to_numpy())

            total_dist = 0
            totalDArr = []
            for scale_i in np.arange(len(scale_)):
                for scale_j in np.arange(scale_i+1, len(scale_)):
                    total_dist = total_dist + distance.loc[scale_[scale_i], scale_[scale_j]]
                    totalDArr.extend([distance.loc[scale_[scale_i], scale_[scale_j]]])

            total_dist = total_dist/len(totalDArr)
            meanDist['seg-%s_band-%s' %(str(segId), bandName[bandId])] = np.round(total_dist, 3)
            medianDist['seg-%s_band-%s' %(str(segId), bandName[bandId])] = np.round(np.median(totalDArr), 3)

        if bandName[bandId] in ['Theta', 'LBeta', 'UBeta']:
            row_ = row_ + 1
            col_ = 0

    plt.suptitle('Bars in each subgraph from left to right are representing emotion groups in order 1 to 8', y = -0.08, verticalalignment='bottom', fontsize=30)
    plt.savefig('LongShortFrame_New.png', bbox_inches='tight')
    plt.savefig('LongShortFrame_New.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    '''deltaFrame = pd.DataFrame(longShortEmtFFrqWiseDict['Delta'], columns=['D_L', 'D_S'])
    LongShortFrame = deltaFrame
    LongShortFrame = pd.concat((LongShortFrame, pd.DataFrame(longShortEmtFFrqWiseDict['Theta'], columns=['T_L', 'T_S'])), axis=1)
    LongShortFrame = pd.concat((LongShortFrame, pd.DataFrame(longShortEmtFFrqWiseDict['Alpha'], columns=['A_L', 'A_S'])), axis=1)
    LongShortFrame = pd.concat((LongShortFrame, pd.DataFrame(longShortEmtFFrqWiseDict['LBeta'], columns=['LB_L', 'LB_S'])), axis=1)
    LongShortFrame = pd.concat((LongShortFrame, pd.DataFrame(longShortEmtFFrqWiseDict['UBeta'], columns=['UB_L', 'UB_S'])), axis=1)
    LongShortFrame = pd.concat((LongShortFrame, pd.DataFrame(longShortEmtFFrqWiseDict['Gamma'], columns=['G_L', 'G_S'])), axis=1)
    LongShortFrame.to_csv('LongShortFrame.csv')'''

    print('I need to understand how the stats is created now')
    medianDistFrame = pd.DataFrame(0, index=['seg-0','seg-1','seg-2','seg-3','seg-4','seg-5','seg-6','seg-7','seg-8'], columns=bandName)
    meanDistFrame = pd.DataFrame(0, index=['seg-0','seg-1','seg-2','seg-3','seg-4','seg-5','seg-6','seg-7','seg-8'], columns=bandName)
    for key_ in medianDist.keys():
        idnx = key_.split('_')[0]
        clns = key_.split('_')[1].split('-')[1]
        medianDistFrame.loc[idnx, clns] = medianDist[key_]
        meanDistFrame.loc[idnx, clns] = meanDist[key_]

    plt.figure(figsize=(20,10))
    plt.rcParams['font.size']=14
    gs = gridspec.GridSpec(6, 10, height_ratios=np.ones(6), width_ratios=np.ones(10), hspace=0.04, wspace=0.04, left=0.015, right=0.975, bottom=0.04, top=0.99)
    row_ = 0
    col_ = 0
    renameDict = {'Group-01':'G1','Group-02':'G2','Group-03':'G3','Group-04':'G4','Group-05':'G5','Group-06':'G6','Group-07':'G7','Group-08':'G8'}

    allDistances = []
    deviationFrame = pd.DataFrame(0, index=['deviation'], columns=bandName)

    for bandId in np.arange(len(bandInf)):
        bandWiseArr = []
        for segId in np.arange(len(segInf)):
            distance = pd.read_csv(os.path.join(target, 'freq_%s_seg-%s.csv' %(bandName[bandId], str(segId))), index_col=0)
            ##pdb.set_trace()
            distance = distance.rename(columns=renameDict, index=renameDict)
            ax = plt.subplot(gs[row_,col_])
            if (row_==0) and (col_==0):
                sns.heatmap(distance, vmin=0, vmax=max_, xticklabels='', ax=ax, cbar=False)
            elif (row_ == 5) and (col_==8):
                sns.heatmap(distance, vmin=0, vmax=max_, yticklabels='', ax=ax, cbar=False)
            else:
                sns.heatmap(distance, vmin=0, vmax=max_, xticklabels='',yticklabels='', ax=ax, cbar=False)

            distance = distance.to_numpy().round(0).squeeze().flatten()
            bandWiseArr.extend(distance)
            print(f'Median for seg-%s and frequency-%s {np.median(distance)}' %(str(segId), bandName[bandId]))
            #print(np.median(bandWiseArr))
            col_ = col_ + 1

        ## Instead of heatmap plot histogram in the last map.

        ax = plt.subplot(gs[row_,9])
        bandWiseArr = np.array(bandWiseArr)
        bandWiseArr = bandWiseArr[np.where(bandWiseArr>0)[0]]
        deviation = np.std(bandWiseArr)
        bandWiseArr = bandWiseArr/deviation
        deviationFrame.loc['deviation', bandName[bandId]] = deviation
        allDistances.extend(bandWiseArr)

        medianDistFrame.loc[:, bandName[bandId]] = medianDistFrame.loc[:, bandName[bandId]]/deviation
        meanDistFrame.loc[:, bandName[bandId]] = meanDistFrame.loc[:, bandName[bandId]]/deviation

        ax.hist(bandWiseArr)
        ## Draw Median Here.
        medVal = np.round(np.mean(bandWiseArr),2)
        ax.axvline(x=medVal,label=str(medVal),color='r')
        plt.legend(loc='center left')
        #ax.tick_params(axis="y",direction="in", pad=-35)
        ax.tick_params(axis="y",left=False,right=True,labelright=True,labelleft=False)#, pad=-35)
        ax.tick_params(axis="x",bottom=False, top=True, direction="in", pad=-17, labeltop=True,labelbottom=False)
        # Which one has the max median

        '''if (row_ == 5):
            sns.heatmap(meanDist, vmin=0, yticklabels='', vmax=max_, ax=ax)
        else:
            sns.heatmap(meanDist, vmin=0, vmax=max_, xticklabels='',yticklabels='', ax=ax, cbar='False')'''

        row_ = row_ + 1
        col_ = 0

    plt.savefig(os.path.join(target, 'phaseRandomConnAnalEEG_distancePlot_SDNorm_New.pdf'))
    plt.savefig(os.path.join(target, 'phaseRandomConnAnalEEG_distancePlot_SDNorm_New.png'))
    plt.close()
    plt.clf()

    deviationFrame.to_csv(os.path.join(target, 'connectivityDeviation_New.csv'))

    medianDistFrame.plot.line(figsize=(20,10), fontsize=30, linewidth=5)#style={'linewidth':14})
    plt.legend(fontsize=25, labelcolor='linecolor', columnspacing=0.5, handletextpad=0.2)
    plt.savefig(os.path.join(target, 'medianDistFrameLinePlot_Program_phaseRandomConnAnalEEG_New.png'), bbox_inches='tight')
    plt.savefig(os.path.join(target, 'medianDistFrameLinePlot_Program_phaseRandomConnAnalEEG_New.pdf'), bbox_inches='tight')
    plt.close()
    plt.clf()

    meanDistFrame.plot.line(figsize=(20,10), fontsize=30, linewidth=5)#style={'linewidth':14})
    plt.legend(fontsize=25, labelcolor='linecolor', columnspacing=0, handletextpad=0.2)
    plt.savefig(os.path.join(target, 'meanDistFrameLinePlot_Program_phaseRandomConnAnalEEG_New.png'), bbox_inches='tight')
    plt.savefig(os.path.join(target, 'meanDistFrameLinePlot_Program_phaseRandomConnAnalEEG_New.pdf'), bbox_inches='tight')
    plt.close()
    plt.clf()

################## Histogram of common nodes across groups #################
    plt.figure(figsize=(20,10))
    plt.rcParams['font.size']=14
    gs = gridspec.GridSpec(6, 9, height_ratios=np.ones(6), width_ratios=np.ones(9), hspace=0.75, wspace=0.25, left=0.03, right=0.999, bottom=0.06, top=0.97)
    row_ = 0
    col_ = 0
    renameDict = {'Group-01':'G1','Group-02':'G2','Group-03':'G3','Group-04':'G4','Group-05':'G5','Group-06':'G6','Group-07':'G7','Group-08':'G8'}

    for bandId in np.arange(len(bandInf)):
        for segId in np.arange(len(segInf)):
            ax = plt.subplot(gs[row_,col_])

            ax.bar(np.arange(len(commScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][:-1])), commScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][:-1])
            ax.set_xticks(np.arange(len(commScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][:-1])))
            ax.set_xticklabels([str(i) for i in commNodes['seg-%s_band-%s' %(str(segId), bandName[bandId])][:-1]], rotation=90, fontsize=16)
            ax.set_title(str(commScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][-1]), loc='left', color='red')
            if len(commScore['seg-%s_band-%s' %(str(segId), bandName[bandId])]) == 1:
                ax.set_yticklabels([])
            '''try:
                ax.annotate(str(commScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][-1]), (0,max(commScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][:-1])+0.5), fontsize=20, color='red')
            except:
                ax.annotate(str(commScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][-1]), (0,0+0.5), fontsize=20, color='red')
                ax.set_yticklabels([])'''

            col_ = col_ + 1

        row_ = row_ + 1
        col_ = 0

    plt.savefig(os.path.join(target, 'phaseRandomConnAnalEEG_CommNodesPlot_New.pdf'))
    plt.savefig(os.path.join(target, 'phaseRandomConnAnalEEG_CommNodesPlot_New.png'))
    plt.close()
    plt.clf()

    LinkNodesFrame = pd.DataFrame(0, index=['seg-0','seg-1','seg-2','seg-3','seg-4','seg-5','seg-6','seg-7','seg-8'], columns=bandName)
    LinkScoreFrame = pd.DataFrame(0, index=['seg-0','seg-1','seg-2','seg-3','seg-4','seg-5','seg-6','seg-7','seg-8'], columns=bandName)
    commNodesFrame = pd.DataFrame(0, index=['seg-0','seg-1','seg-2','seg-3','seg-4','seg-5','seg-6','seg-7','seg-8'], columns=bandName)
    commScoreFrame = pd.DataFrame(0, index=['seg-0','seg-1','seg-2','seg-3','seg-4','seg-5','seg-6','seg-7','seg-8'], columns=bandName)
    totalNodesFrame = pd.DataFrame(0, index=['seg-0','seg-1','seg-2','seg-3','seg-4','seg-5','seg-6','seg-7','seg-8'], columns=bandName)
    totalLinksFrame = pd.DataFrame(0, index=['seg-0','seg-1','seg-2','seg-3','seg-4','seg-5','seg-6','seg-7','seg-8'], columns=bandName)

    for key_ in LinkNodes.keys():
        idnx = key_.split('_')[0]
        clns = key_.split('_')[1].split('-')[1]
        LinkNodesFrame.loc[idnx, clns] = len(LinkNodes[key_])-1
        LinkScoreFrame.loc[idnx, clns] = len(LinkScore[key_])-1
        totalLinksFrame.loc[idnx, clns] = '%s(%s)' %(str(LinkNodes[key_][-1]), str(len(LinkNodes[key_])-1))

        commNodesFrame.loc[idnx, clns] = len(commNodes[key_])-1
        commScoreFrame.loc[idnx, clns] = len(commScore[key_])-1
        totalNodesFrame.loc[idnx, clns] = '%s(%s)' %(str(commNodes[key_][-1]), str(len(commNodes[key_])-1))

    LinkScoreFrame.plot.line(figsize=(20,10), fontsize=30, linewidth=5)#style={'linewidth':14})
    plt.legend(fontsize=25, labelcolor='linecolor', columnspacing=0, handletextpad=0.2)
    plt.savefig(os.path.join(target, 'LinkScoreFrameLinePlot_Program_phaseRandomConnAnalEEG_New.png'), bbox_inches='tight')
    plt.savefig(os.path.join(target, 'LinkScoreFrameLinePlot_Program_phaseRandomConnAnalEEG_New.pdf'), bbox_inches='tight')
    plt.close()
    plt.clf()

    commScoreFrame.plot.line(figsize=(20,10), fontsize=30, linewidth=5)#style={'linewidth':14})
    plt.legend(fontsize=25, labelcolor='linecolor', columnspacing=0, handletextpad=0.2)
    plt.savefig(os.path.join(target, 'commScoreFrameLinePlot_Program_phaseRandomConnAnalEEG_New.png'), bbox_inches='tight')
    plt.savefig(os.path.join(target, 'commScoreFrameLinePlot_Program_phaseRandomConnAnalEEG_New.pdf'), bbox_inches='tight')
    plt.close()
    plt.clf()

    totalNodesFrame.to_csv(os.path.join(target, 'commNodesFrame_Program_phaseRandomConnAnalEEG_New.csv'))
    totalLinksFrame.to_csv(os.path.join(target, 'commLinksFrame_Program_phaseRandomConnAnalEEG_New.csv'))

################## Histogram of common links across groups #################
    plt.figure(figsize=(20,10))
    plt.rcParams['font.size']=16
    plt.rcParams['axes.titlepad'] = -1
    gs = gridspec.GridSpec(6, 9, height_ratios=np.ones(6), width_ratios=np.ones(9), hspace=1.9, wspace=0.25, left=0.02, right=0.999, bottom=0.09, top=0.97)
    row_ = 0
    col_ = 0
    renameDict = {'Group-01':'G1','Group-02':'G2','Group-03':'G3','Group-04':'G4','Group-05':'G5','Group-06':'G6','Group-07':'G7','Group-08':'G8'}

    for bandId in np.arange(len(bandInf)):
        for segId in np.arange(len(segInf)):
            ax = plt.subplot(gs[row_,col_])

            ax.bar(np.arange(len(LinkScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][:-1])), LinkScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][:-1])
            ax.tick_params(axis='both', which='both', pad=0)
            ax.set_xticks(np.arange(len(LinkScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][:-1])))
            ax.set_xticklabels([str(i) for i in LinkNodes['seg-%s_band-%s' %(str(segId), bandName[bandId])][:-1]], rotation=90, fontsize=15)
            print(bandId, segId)
            ax.set_title(str(LinkScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][-1]), loc='left', color='red')
            if len(LinkScore['seg-%s_band-%s' %(str(segId), bandName[bandId])]) == 1:
                ax.set_yticklabels([])
            '''try:
                ax.annotate(str(LinkScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][-1]), (-0.1, max(LinkScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][:-1])+0.2), fontsize=20, color='red')
            except:
                ax.annotate(str(LinkScore['seg-%s_band-%s' %(str(segId), bandName[bandId])][-1]), (-0.1,0.1), fontsize=20, color='red')
                ax.set_yticklabels([])'''

            col_ = col_ + 1

        row_ = row_ + 1
        col_ = 0

    plt.savefig(os.path.join(target, 'phaseRandomConnAnalEEG_CommLinksPlot_New.pdf'))
    plt.savefig(os.path.join(target, 'phaseRandomConnAnalEEG_CommLinksPlot_New.png'))
    plt.close()
    plt.clf()

###### Only Histogram without Distance matrices
    plt.figure(figsize=(20,3.5))
    plt.rcParams['font.size']=20
    gs = gridspec.GridSpec(1, 6, width_ratios=np.ones(6), hspace=0.04, wspace=0.3, left=0.04, right=0.99, bottom=0.1, top=0.88)
    row_ = 0
    col_ = 0
    renameDict = {'Group-01':'G1','Group-02':'G2','Group-03':'G3','Group-04':'G4','Group-05':'G5','Group-06':'G6','Group-07':'G7','Group-08':'G8'}

    allDistances = []
    allDistancesAllSegsComb = []
    for bandId in np.arange(len(bandInf)):
        meanDist = pd.DataFrame(0, index = np.sort(allConds[1:]), columns = np.sort(allConds[1:]))
        bandWiseArr = []
        for segId in np.arange(len(segInf)):
            distance = pd.read_csv(os.path.join(target, 'freq_%s_seg-%s.csv' %(bandName[bandId], str(segId))), index_col=0)
            ##pdb.set_trace()
            distance = distance.rename(columns=renameDict, index=renameDict)
            #ax = plt.subplot(gs[row_,col_])
            #print(row_, col_)
            #if (row_==0) and (col_==0):
            #    sns.heatmap(distance, vmin=0, vmax=max_, xticklabels='', ax=ax, cbar=False)
            #elif (row_ == 5) and (col_==8):
            #    sns.heatmap(distance, vmin=0, vmax=max_, yticklabels='', ax=ax, cbar=False)
            #else:
            #    sns.heatmap(distance, vmin=0, vmax=max_, xticklabels='',yticklabels='', ax=ax, cbar=False)

            distance = distance.to_numpy().round(0)
            upperTrangle = []
            for _row in np.arange(8):
                for _col in np.arange(_row+1, 8):
                    upperTrangle.extend([distance[_row, _col]])

            upperTrangle = np.array(upperTrangle)
            bandWiseArr.extend(upperTrangle.squeeze().flatten())
            col_ = col_ + 1
            _temp = upperTrangle.squeeze().flatten()
            _temp = np.reshape(_temp, (1, len(_temp)))
            if segId == 0:
                allSegsComb = _temp
            else:
                allSegsComb = np.concatenate((allSegsComb, _temp), axis=0)

        ## Instead of heatmap plot histogram in the last map.
        ax = plt.subplot(gs[row_])
        bandWiseArr = np.array(bandWiseArr)
        bandWiseArr = bandWiseArr[np.where(bandWiseArr>0)[0]]
        bandWiseArr = bandWiseArr/np.std(bandWiseArr)
        allSegsComb = np.divide(np.mean(allSegsComb, axis=0), np.std(allSegsComb, axis=0))
        allDistances.extend(bandWiseArr)
        allDistancesAllSegsComb.extend(allSegsComb)
        
        np.save(os.path.join(target, 'DifferenceBetweenEmotionNetworks_%s.npy' %bandName[bandId]), bandWiseArr)
        pd.DataFrame(bandWiseArr).to_csv(os.path.join(target, 'DifferenceBetweenEmotionNetworks_%s.csv' %bandName[bandId]))
        pd.DataFrame(allSegsComb).to_csv(os.path.join(target, 'DifferenceBetweenEmotionNetworks_allSegsComb_%s.csv' %bandName[bandId]))
        ax.hist(bandWiseArr)
        ## Draw Median Here.
        medVal = np.round(np.mean(bandWiseArr),2)
        print(bandName[bandId], medVal)
        ax.axvline(x=medVal,label=str(medVal),color='r')
        ax.set_title(bandName[bandId])
        plt.legend(loc='center left')
#        if bandId > 0:
#            ax.set_yticklabels([])
        #ax.tick_params(axis="y",direction="in", pad=-35)
        #ax.tick_params(axis="y",left=False,right=True,labelright=True,labelleft=False)#, pad=-35)
        #ax.tick_params(axis="x",bottom=False, top=True, direction="in", pad=-17, labeltop=True,labelbottom=False)
        # Which one has the max median

        '''if (row_ == 5):
            sns.heatmap(meanDist, vmin=0, yticklabels='', vmax=max_, ax=ax)
        else:
            sns.heatmap(meanDist, vmin=0, vmax=max_, xticklabels='',yticklabels='', ax=ax, cbar='False')'''

        row_ = row_ + 1
        col_ = 0

    pdb.set_trace()
    print(target)
    plt.savefig(os.path.join(target, 'phaseRandomConnAnalEEG_distanceHist_ccf_New.pdf'))
    plt.savefig(os.path.join(target, 'phaseRandomConnAnalEEG_distanceHist_ccf_New.png'))
    plt.close()
    plt.clf()

def SequentialboneferroniCorr(sortedAllPvals, sortedAllCompN):

    for rank, val_ in enumerate(sortedAllPvals):
        correP = 0.05/(len(sortedAllPvals)-(rank+1)+1)
        if val_ < correP:
            continue
        else:
            break

    return rank

def ConnDistStatsPlots():
    ## The inputs are created in R program in the acer laptop: /mnt/7CBFA0EC210FC340/ForMSAAnalysis/ConnDistSignificance.R

    nRows = 2
    nCols = 10
    if nRows == 2:
        plt.figure(figsize=(20,6))
    else:
        plt.figure(figsize=(20,10))

    plt.rcParams['font.size']=14
    gs = gridspec.GridSpec(nRows, nCols, height_ratios=np.ones(nRows), width_ratios=np.ones(nCols), hspace=0.75, wspace=0.1, left=0.015, right=0.975, bottom=0.04, top=0.99)
    row_ = 0
    col_ = 0

    target = os.path.join(sourceDir, 'EEGConnAnalysisEmt_Base')
    AllSegsComb = pd.read_csv(os.path.join(target, 'pValueConnectivityDistanceConsideringAllSegs.csv'), index_col=0)

    #################### These are the pre-requisites for boneferroni correction ###############
    allPVals = []
    allCompName = []
    for frq_ in bandName:
        data = pd.read_csv(os.path.join(target, 'pValueConnectivityDistancefor-%s.csv' %frq_), index_col=0)
        print(data.shape)
        for idx_ in data.index.values:
            for cols in data.columns.values:
                allPVals.extend([data.loc[idx_, cols]])
                allCompName.extend([frq_+'-'+idx_+'-'+str(cols)])

    #pdb.set_trace()
    for idx_ in AllSegsComb.index.values:
        for cols in AllSegsComb.columns.values:
            allPVals.extend([AllSegsComb.loc[idx_, cols]])
            allCompName.extend([idx_+'-'+str(cols)])

    #pdb.set_trace()
    sortedIdx = np.argsort(allPVals)
    sortedAllPvals = np.array(allPVals)[sortedIdx]
    sortedAllCompN = np.array(allCompName)[sortedIdx]
    ####################################
    #source:  Holm, S. 1979. A simple sequential rejective multiple test procedure. Scandinavian Journal of Statistics 6:65-70
    rank = SequentialboneferroniCorr(sortedAllPvals, sortedAllCompN)
    onlyBoneFCorrected = sortedAllCompN[0:rank]

    ####### Creating significance marked file for boneferroni correction when all segs are combined ###########
    AllSegCombsigMarkDf = pd.DataFrame('', index=bandName, columns=bandName)
    for bCorr in onlyBoneFCorrected:
        bCorrArr = bCorr.split('-')

        if len(bCorrArr) == 2:
            AllSegCombsigMarkDf.loc[bCorrArr[0], bCorrArr[1]] = '*'
    #################################################

    plotCount = 1
    #for frq_ in bandName:
    for frq_ in ['LBeta', 'UBeta']:
        data = pd.read_csv(os.path.join(target, 'pValueConnectivityDistancefor-%s.csv' %frq_), index_col=0)
        ####### Creating significant marker data frame after boneferroni correction ########
        frqIndex = np.where(frq_ == np.array(bandName))[0][0]
        if frqIndex > 0:
            indexDf = bandName[0:frqIndex]
            indexDf.extend(bandName[frqIndex+1:])

        else:
             indexDf = bandName[1:]

        sigMarkDf = pd.DataFrame('', index=indexDf, columns=[str(i) for i in np.arange(9)])

        for bCorr in onlyBoneFCorrected:
            bCorrArr = bCorr.split('-')

            if len(bCorrArr) > 2:
                if bCorrArr[0] == frq_:
                    sigMarkDf.loc[bCorrArr[1], bCorrArr[2]] = '*'

        #################################################

        for segId in np.arange(9):
            ax = plt.subplot(gs[row_, col_])
            data[str(segId)].plot.bar(ax=ax, fontsize=25)
            ax.set_ylim([0, 0.1])
            if col_ > 0:
                ax.set_yticklabels('')
            else:
                ax.set_ylabel(frq_, fontsize=25)

            if row_==0 and plotCount < 2:
                ax.set_title('seg-%s' %str(segId), fontsize=25)

            '''if row_ < nRows-1:
                ax.set_xticklabels('')'''
            col_ = col_+1

            ############ Putting astriks mark on top of each bar #############
            for p, sig in zip(ax.patches,sigMarkDf.loc[:, str(segId)]):
                #print(sig)
                #pdb.set_trace()
                if sig == '*':
                    ax.annotate('*', (p.get_x() + p.get_width(), p.get_height()), weight="bold",
                                ha='center', va='center', fontsize=35, color='gray', rotation=90, xytext=(0,10),
                                textcoords='offset points')
            #pdb.set_trace()

        ax1 = plt.subplot(gs[row_, col_])
        print(AllSegCombsigMarkDf.loc[frq_, :])
        AllSegsComb.loc[frq_, :].iloc[np.where(~np.isnan(AllSegsComb.loc[frq_, :]))[0]].plot.bar(ax=ax1, fontsize=25)
        if col_ > 0:
            ax1.set_yticklabels('')
        else:
            ax1.set_ylabel(frq_, fontsize=25)

        if row_==0 and plotCount < 2:
            ax1.set_title('AllSegComb', fontsize=25)

        ############ Putting astriks mark on top of each bar #############

        for p, sig in zip(ax1.patches,AllSegCombsigMarkDf.loc[frq_, indexDf]):
            #print(sig, p.get_x(), p.get_width(), p.get_height())
            if sig == '*':
                #print(p.get_x(), p.get_width(), p.get_height())
                ax1.annotate('*', (p.get_x() + p.get_width(), p.get_height()), weight="bold",
                            ha='center', va='center', fontsize=35, color='gray', rotation=90, xytext=(0,10),
                            textcoords='offset points')

        row_ = row_+1
        col_ = 0

        if not (row_ % 3):

            plt.savefig(os.path.join(target, 'pValueConnectivityDistanceBarPlotMarked-%s.png' %str(plotCount)), bbox_inches='tight')
            plt.savefig(os.path.join(target, 'pValueConnectivityDistanceBarPlotMarked-%s.pdf' %str(plotCount)), bbox_inches='tight')
            plotCount = plotCount + 1

            if plotCount < 3:
                plt.figure(figsize=(20,10))
                plt.rcParams['font.size']=14
                nRows = 3
                nCols = 10
                gs = gridspec.GridSpec(nRows, nCols, height_ratios=np.ones(nRows), width_ratios=np.ones(nCols), hspace=0.75, wspace=0.1, left=0.015, right=0.975, bottom=0.04, top=0.99)
                row_ = 0
                col_ = 0

    if nRows == 2:
        print(target)
        plt.savefig(os.path.join(target, 'pValueConnectivityDistanceBarPlotMarked.png' ), bbox_inches='tight')
        plt.savefig(os.path.join(target, 'pValueConnectivityDistanceBarPlotMarked.pdf' ), bbox_inches='tight')
