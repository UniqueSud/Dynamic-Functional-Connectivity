sourceDir = '/mnt/7CBFA0EC210FC340/ForMSAAnalysis/AllBands/CorrectedFiles/EEGConnAnalysisEmt_Base'
frqBands = c('Delta', 'Theta', 'Alpha', 'LBeta', 'UBeta', 'Gamma')
colToConsider = c(0:8)

mu = 5.44

####  All segments together ######
## These input files are create from /data_hpc/home/ust/sudhakar/Processed_Emotions/ForMSAAnalysis/phaseRandomConnAnalEEG.py and module is connecBasedDynamicCorr()
# delta = read.csv(file.path(sourceDir, 'DifferenceBetweenEmotionNetworks_Delta.csv'))
# theta = read.csv(file.path(sourceDir,'DifferenceBetweenEmotionNetworks_Theta.csv'))
# alpha = read.csv(file.path(sourceDir,'DifferenceBetweenEmotionNetworks_Alpha.csv'))
# LBeta = read.csv(file.path(sourceDir,'DifferenceBetweenEmotionNetworks_LBeta.csv'))
# UBeta = read.csv(file.path(sourceDir,'DifferenceBetweenEmotionNetworks_UBeta.csv'))
# Gamma = read.csv(file.path(sourceDir,'DifferenceBetweenEmotionNetworks_Gamma.csv'))

t.test(delta$X0, mu = 5.44, alternative = 'greater')
t.test(theta$X0, mu = 5.44, alternative = 'greater')
t.test(alpha$X0, mu = 5.44, alternative = 'greater')
t.test(LBeta$X0, mu = 5.44, alternative = 'greater')
t.test(UBeta$X0, mu = 5.44, alternative = 'greater')
t.test(Gamma$X0, mu = 5.44, alternative = 'greater')

t.test(theta$X0, UBeta$X0, paired = TRUE, alternative = 'less')
t.test(theta$X0, LBeta$X0, paired = TRUE, alternative = 'less')
t.test(LBeta$X0, UBeta$X0, paired = TRUE, alternative = 'less')


delta = read.csv(file.path(sourceDir, 'DifferenceBetweenEmotionNetworks_allSegsComb_Delta.csv'))
theta = read.csv(file.path(sourceDir,'DifferenceBetweenEmotionNetworks_allSegsComb_Theta.csv'))
alpha = read.csv(file.path(sourceDir,'DifferenceBetweenEmotionNetworks_allSegsComb_Alpha.csv'))
LBeta = read.csv(file.path(sourceDir,'DifferenceBetweenEmotionNetworks_allSegsComb_LBeta.csv'))
UBeta = read.csv(file.path(sourceDir,'DifferenceBetweenEmotionNetworks_allSegsComb_UBeta.csv'))
Gamma = read.csv(file.path(sourceDir,'DifferenceBetweenEmotionNetworks_allSegsComb_Gamma.csv'))

library(effsize)
t.test(delta$X0, mu = 8.24, alternative = 'greater')
cohen.d(delta$X0~., mu = 8.24)
t.test(theta$X0, mu = 8.24, alternative = 'greater')
cohen.d(theta$X0~., mu = 8.24)
t.test(alpha$X0, mu = 8.24, alternative = 'greater')
cohen.d(alpha$X0~., mu = 8.24)
t.test(LBeta$X0, mu = 8.24, alternative = 'greater')
cohen.d(LBeta$X0~., mu = 8.24)
t.test(UBeta$X0, mu = 8.24, alternative = 'greater')
cohen.d(UBeta$X0~., mu = 8.24)
t.test(Gamma$X0, mu = 8.24, alternative = 'greater')
cohen.d(Gamma$X0~., mu = 8.24)

t.test(theta$X0, UBeta$X0, paired = TRUE)#, alternative = 'less')
cohen.d(theta$X0, UBeta$X0, paired=TRUE)
t.test(theta$X0, LBeta$X0, paired = TRUE, alternative = 'less')
t.test(LBeta$X0, UBeta$X0, paired = TRUE, alternative = 'less')
cohen.d(LBeta$X0, UBeta$X0, paired=TRUE)

deviationData = read.csv(file.path(sourceDir, 'connectivityDeviation.csv'))
################ Segment wise statistics calculation  ################
mat = matrix(nrow=length(frqBands), ncol=length(frqBands))
staticFrame = data.frame(mat)
rownames(staticFrame) = frqBands
colnames(staticFrame) = frqBands

pValFrame = data.frame(mat)
rownames(pValFrame) = frqBands
colnames(pValFrame) = frqBands

for (frq_ in frqBands){
  firstArr = c()
  for (segId in colToConsider){
    distance = read.csv(file.path(sourceDir, paste('freq', frq_, paste('seg-',as.character(segId), '.csv', sep=''), sep='_')))
    
    for (i in c(1:8)){
      for (j in c(i+2:9)){
        firstArr = append(firstArr, distance[i, j])
      }
    }
  }
  firstArr = firstArr/deviationData[1, frq_]
  
  for (frq_2 in frqBands){
    if (frq_ != frq_2){
      secondArr = c()
      for (segId in colToConsider){
        distance2 = read.csv(file.path(sourceDir, paste('freq', frq_2, paste('seg-',as.character(segId), '.csv', sep=''), sep='_')))

        for (i in c(1:8)){
          for (j in c(i+2:9)){
            secondArr = append(secondArr, distance2[i, j])
          }
        }
      }
      secondArr = secondArr/deviationData[1, frq_2]
      
      res= wilcox.test(firstArr, secondArr, paired = TRUE, alternative = "greater")
      staticFrame[frq_, frq_2] = round(res$statistic, 4)
      pValFrame[frq_, frq_2] = round(res$p.value, 4)
      }
    }
}
print(staticFrame)
print(pValFrame)  
write.csv(pValFrame, file.path(sourceDir, paste('pValueConnectivityDistanceConsideringAllSegs', '.csv', sep='')))

################ Now Segment Wise
count = 1
for (frq_ in frqBands){
  newBand = frqBands[-count]
  count = count + 1
  mat = matrix(nrow=length(newBand), ncol=length(colToConsider))
  staticFrame = data.frame(mat)
  rownames(staticFrame) = newBand
  colnames(staticFrame) = colToConsider
  
  pValFrame = data.frame(mat)
  rownames(pValFrame) = newBand
  colnames(pValFrame) = colToConsider
  
  for (frq_2 in newBand){
    if (frq_ != frq_2){
      for (segId in colToConsider){
        distance = read.csv(file.path(sourceDir, paste('freq', frq_, paste('seg-',as.character(segId), '.csv', sep=''), sep='_')))
        distance2 = read.csv(file.path(sourceDir, paste('freq', frq_2, paste('seg-',as.character(segId), '.csv', sep=''), sep='_')))
        
        firstArr = c()
        secondArr = c()
        
        for (i in c(1:8)){
          for (j in c(i+2:9)){
            firstArr = append(firstArr, distance[i, j])
            secondArr = append(secondArr, distance2[i, j])
          }
        }
        firstArr = firstArr/deviationData[1, frq_]
        secondArr = secondArr/deviationData[1, frq_2]
        
        res= wilcox.test(firstArr, secondArr, paired = TRUE, alternative = "greater")
        staticFrame[frq_2, as.character(segId)] = round(res$statistic, 4)
        pValFrame[frq_2, as.character(segId)] = round(res$p.value, 4)
      }
    }
  }
  print(frq_)
  print(pValFrame)
  write.csv(pValFrame, file.path(sourceDir, paste('pValueConnectivityDistancefor-', frq_, '.csv', sep='')))
}

# Statistical significance analysis for dFC.    

