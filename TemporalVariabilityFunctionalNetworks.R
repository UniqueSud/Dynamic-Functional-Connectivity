library(lme4)
library(afex)
library(ggforce)
library(sjPlot)

sourceDir = '/mnt/7CBFA0EC210FC340/ForMSAAnalysis/'
setwd(sourceDir)

## This file is create in program: 
# For Groupwise calculation program: 172.17.25.241/media/forAll/Processed_Emotions/ToTransfer_Zuddler/ForMSAAnalysis/ConnectomePlotsArrangem.py and temporalDifferece()
# For Stimulus calculation program: 172.17.25.241/media/forAll/Processed_Emotions/ToTransfer_Zuddler/ForMSAAnalysis/ConnectomePlotsArrangem_StimulusWise.py and temporalDifferece()
# For Emotion Event wise calculation program: 172.17.25.241/media/forAll/Processed_Emotions/ToTransfer_Zuddler/ForMSAAnalysis/ConnectomePlotsArrangem_EmotionFile.py and temporalDifferece()

#tv = read.csv('TemporalVariability_ConnectomePlotsArrangement_EmotionFile.csv')
tv = read.csv('TemporalVariability_ConnectomePlotsArrangement_StimulusWise.csv')

# tv$TV = scale(tv$TV)
# tv$Valence = scale(tv$Valence)
# tv$Arousal = scale(tv$Arousal)
# tv$Dominance = scale(tv$Dominance)
# tv$Liking = scale(tv$Liking)
# tv$Familiarity = scale(tv$Familiarity)
# tv$Relevance = scale(tv$Relevance)

model_ = lmer(Valence ~ TV + (1|subject) + (1|stimulus), data=tv, control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))
df = as.data.frame(summary(model_)$coefficients)
confVals = confint(model_, method = "Wald")
confValIdx = which(row.names(confVals)=="(Intercept)")
df['2.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 1])
df['97.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 2])
df['oddRatio'] = exp(df['Estimate'])
df[2:nrow(df), 'odds'] = exp(df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate'])
df[2:nrow(df), 'log(odds)'] = df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate']
df[2:nrow(df), 'PHighArousal'] = df[2:nrow(df), 'odds']/(1+df[2:nrow(df), 'odds'])
df['oddRatio_2.5%'] = exp(df['2.5%'])
df['oddRatio_97.5%'] = exp(df['97.5%'])
df = round(df, 3)
write.csv(df, 'PredictionValenceFromTV.csv')

model_ = lmer(Arousal ~ TV + (1|subject) + (1|stimulus), data=tv, control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))
df = as.data.frame(summary(model_)$coefficients)
confVals = confint(model_, method = "Wald")
confValIdx = which(row.names(confVals)=="(Intercept)")
df['2.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 1])
df['97.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 2])
df['oddRatio'] = exp(df['Estimate'])
df[2:nrow(df), 'odds'] = exp(df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate'])
df[2:nrow(df), 'log(odds)'] = df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate']
df[2:nrow(df), 'PHighArousal'] = df[2:nrow(df), 'odds']/(1+df[2:nrow(df), 'odds'])
df['oddRatio_2.5%'] = exp(df['2.5%'])
df['oddRatio_97.5%'] = exp(df['97.5%'])
df = round(df, 3)
write.csv(df, 'PredictionArousalFromTV.csv')

model_ = lmer(Dominance ~ TV + (1|subject), data=tv)
df = as.data.frame(summary(model_)$coefficients)
confVals = confint(model_, method = "Wald")
confValIdx = which(row.names(confVals)=="(Intercept)")
df['2.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 1])
df['97.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 2])
df['oddRatio'] = exp(df['Estimate'])
df[2:nrow(df), 'odds'] = exp(df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate'])
df[2:nrow(df), 'log(odds)'] = df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate']
df[2:nrow(df), 'PHighArousal'] = df[2:nrow(df), 'odds']/(1+df[2:nrow(df), 'odds'])
df['oddRatio_2.5%'] = exp(df['2.5%'])
df['oddRatio_97.5%'] = exp(df['97.5%'])
df = round(df, 3)
write.csv(df, 'PredictionDominanceFromTV.csv')

model_ = lmer(Liking ~ TV + (1|subject), data=tv)
df = as.data.frame(summary(model_)$coefficients)
confVals = confint(model_, method = "Wald")
confValIdx = which(row.names(confVals)=="(Intercept)")
df['2.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 1])
df['97.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 2])
df['oddRatio'] = exp(df['Estimate'])
df[2:nrow(df), 'odds'] = exp(df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate'])
df[2:nrow(df), 'log(odds)'] = df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate']
df[2:nrow(df), 'PHighArousal'] = df[2:nrow(df), 'odds']/(1+df[2:nrow(df), 'odds'])
df['oddRatio_2.5%'] = exp(df['2.5%'])
df['oddRatio_97.5%'] = exp(df['97.5%'])
df = round(df, 3)
write.csv(df, 'PredictionLikingFromTV.csv')

model_ = lmer(Familiarity ~ TV + (1|subject), data=tv)
df = as.data.frame(summary(model_)$coefficients)
confVals = confint(model_, method = "Wald")
confValIdx = which(row.names(confVals)=="(Intercept)")
df['2.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 1])
df['97.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 2])
df['oddRatio'] = exp(df['Estimate'])
df[2:nrow(df), 'odds'] = exp(df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate'])
df[2:nrow(df), 'log(odds)'] = df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate']
df[2:nrow(df), 'PHighArousal'] = df[2:nrow(df), 'odds']/(1+df[2:nrow(df), 'odds'])
df['oddRatio_2.5%'] = exp(df['2.5%'])
df['oddRatio_97.5%'] = exp(df['97.5%'])
df = round(df, 3)
write.csv(df, 'PredictionFamiliarityFromTV.csv')

model_ = lmer(Relevance ~ TV + (1|subject), data=tv)
df = as.data.frame(summary(model_)$coefficients)
confVals = confint(model_, method = "Wald")
confValIdx = which(row.names(confVals)=="(Intercept)")
df['2.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 1])
df['97.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 2])
df['oddRatio'] = exp(df['Estimate'])
df[2:nrow(df), 'odds'] = exp(df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate'])
df[2:nrow(df), 'log(odds)'] = df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate']
df[2:nrow(df), 'PHighArousal'] = df[2:nrow(df), 'odds']/(1+df[2:nrow(df), 'odds'])
df['oddRatio_2.5%'] = exp(df['2.5%'])
df['oddRatio_97.5%'] = exp(df['97.5%'])
df = round(df, 3)
write.csv(df, 'PredictionRelevanceFromTV.csv')

########## Checking the effect of interaction between arousal and dominance on predicting high TV ######
tv[tv$TV<5.94 , 'TVGroup'] = 0
tv[tv$TV>=5.94 , 'TVGroup'] = 1
tv$TVGroup = as.factor(tv$TVGroup)

logModel = glmer(TVGroup ~ Arousal*Dominance + (1|subject), data=tv, family=binomial(link="logit"),
                 control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))

df = as.data.frame(summary(logModel)$coefficients)
confVals = confint(logModel, method = "Wald")
confValIdx = which(row.names(confVals)=="(Intercept)")
df['2.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 1])
df['97.5%'] = as.numeric(confVals[confValIdx:nrow(confVals), 2])
df['oddRatio'] = exp(df['Estimate'])
df[2:nrow(df), 'odds'] = exp(df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate'])
df[2:nrow(df), 'log(odds)'] = df["(Intercept)", 'Estimate']+df[2:nrow(df), 'Estimate']
df[2:nrow(df), 'PHighArousal'] = df[2:nrow(df), 'odds']/(1+df[2:nrow(df), 'odds'])
df['oddRatio_2.5%'] = exp(df['2.5%'])
df['oddRatio_97.5%'] = exp(df['97.5%'])
df = round(df, 3)
write.csv(df, 'PredictingTempVarFromArl*Dom.csv')

#p=plot_model(logModel, title='PredictingTempVar', type = "pred", terms = c("Arousal [1,2,3,4,5,6,7,8,9]", "Dominance [1,2,3,4,5,6,7,8,9]"))
p=plot_model(logModel, title='', type = "pred", terms = c("Arousal [1,2,3,4,5,6,7,8,9]", "Dominance [1,3,5,7,9]"))
p+theme_sjplot(base_size = 18, base_family = "") + 
  theme(axis.text=element_text(size=18, colour="black"), axis.title = element_text(size=18, colour = 'black')) + ylab("p(High TV)")+
  theme(legend.text=element_text(size=18), legend.direction="horizontal", legend.spacing.x = unit(0.2, 'cm'), legend.position=c(0.65, 1.03)) +
  scale_y_continuous(limits=c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1), n.breaks=5)
save_plot('PredictingTempVarFromArl*Dom.png', fig = last_plot(), width=20, height=10, label.color = "black", dpi=300)


############# Scales as the fixed effects ###########
OldRange = (5 - 1)  
NewRange = (9 - 1)  
NewValue = (((tv$Familiarity - 1) * NewRange) / OldRange) + 1

scaleArr = c('Valence', 'Arousal', 'Dominance', 'Familiarity', 'Relevance')
mat = data.frame(matrix(nrow=length(scaleArr), ncol=length(scaleArr)))
rownames(mat) = scaleArr
colnames(mat) = scaleArr
pVal = data.frame(matrix(nrow=length(scaleArr), ncol=length(scaleArr)))
rownames(pVal) = scaleArr
colnames(pVal) = scaleArr

for (scale_1 in scaleArr){
  for (scale_2 in scaleArr){
    first = as.numeric(unlist(tv[scale_1]))
    second = as.numeric(unlist(tv[scale_2]))
    print(paste(scale_1, scale_2))
    res = cor.test(first, second)
    mat[scale_1, scale_2] = res$estimate
    pVal[scale_1, scale_2] = res$p.value
  }
}

pVal = round(pVal, 3)
mat = round(mat, 3)

model_V = data.frame(summary(lmer(Valence ~  Arousal + Dominance + Liking + Familiarity + Relevance + (1|subject), data=tv))$coefficients)
print(model_V[model_V$Pr...t.. < 0.01, ])
model_A = data.frame(summary(lmer(Arousal ~  Valence + Dominance + Liking+ Familiarity + Relevance + (1|subject), data=tv))$coefficients)
print(model_A[model_A$Pr...t.. < 0.01, ])
model_D = data.frame(summary(lmer(Dominance ~  Valence + Arousal + Liking + Familiarity + Relevance + (1|subject), data=tv))$coefficients)
print(model_D[model_D$Pr...t.. < 0.01, ])
model_L = data.frame(summary(lmer(Liking ~ Valence + Arousal + Dominance + Familiarity + Relevance + (1|subject), data=tv))$coefficients)
print(model_L[model_L$Pr...t.. < 0.01, ])
model_F = data.frame(summary(lmer(Familiarity ~ Valence + Arousal + Dominance + Liking + Relevance + (1|subject), data=tv))$coefficients)
print(model_F[model_F$Pr...t.. < 0.01, ])
model_R = data.frame(summary(lmer(Relevance ~ Valence + Arousal + Dominance + Liking+ Familiarity + (1|subject), data=tv))$coefficients)
print(model_R[model_R$Pr...t.. < 0.01, ])

model_ = lm(cbind(Valence, Arousal, Dominance, Familiarity, Relevance) ~ TV, data=tv)
model_ = lmer(cbind(Valence, Arousal, Dominance, Familiarity, Relevance) ~ TV + (1|subject), data=tv)

model_ = lmer(TV ~ Valence + Arousal + Dominance + Liking + Familiarity + Relevance + (1|subject), data=tv)
model_ = lmer(TV ~ Valence + Arousal + Dominance + Familiarity + Relevance + (1|subject), data=tv)
model_ = lmer(TV ~ Valence + Arousal + Familiarity +  (1|subject), data=tv)
print(summary(model_))
stats_Pos = mixed(model_, REML=TRUE, data=tv, method = "KR")
print(stats_Pos)
confint(model_)

corrMatrix = as.matrix(round(cov2cor(vcov(model_)), 3))
write.csv(round(summary(model_)$coefficients, 3), 'temporal_variability_RatingScales.csv')
write.csv(round(as.matrix(stats_Pos$anova_table), 3), 'temporal_variability_RatingScales_Fstats.csv')
write.csv(corrMatrix, 'temporal_variability_RatingScales_CorrFixef.csv')


summary(lm(tv$Valence~tv$TV)) 
print('Arousal')

summary(lm(tv$Arousal~tv$TV)) 
cor.test(tv$Arousal, tv$TV)

print('Dominance')
tv$dominance = scale(tv$dominance)
summary(lm(tv$dominance~tv$tv)) 

print('Liking')
tv$liking = scale(tv$liking)
summary(lm(tv$liking~tv$tv)) 

print('Familiarity')
tv$familiarity = scale(tv$familiarity)
summary(lm(tv$familiarity~tv$tv)) 

print('Relevance')
tv$relevance = scale(tv$relevance)
summary(lm(tv$relevance~tv$tv)) 

summary(lm(tv$tv~tv$valence+tv$arousal+tv$dominance+tv$liking+tv$familiarity+tv$relevance))
