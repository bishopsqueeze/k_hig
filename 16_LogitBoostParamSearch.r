##------------------------------------------------------------------
## From the kaggle website:
##------------------------------------------------------------------

##------------------------------------------------------------------
## Load libraries
##------------------------------------------------------------------
library(data.table)
library(caret)
library(foreach)
library(doMC)
library(pROC)

##------------------------------------------------------------------
## register cores
##------------------------------------------------------------------
registerDoMC(4)

##------------------------------------------------------------------
## Clear the workspace
##------------------------------------------------------------------
rm(list=ls())

##------------------------------------------------------------------
## Set the working directory
##------------------------------------------------------------------
setwd("/Users/alexstephens/Development/kaggle/higgs/data/proc")

##------------------------------------------------------------------
## Source the utilities file
##------------------------------------------------------------------
source("/Users/alexstephens/Development/kaggle/higgs/k_hig/00_Utilities.r")

##------------------------------------------------------------------
## Load the training data
##------------------------------------------------------------------
loadfile <- "04_HiggsTrainExRbcLcNumNa.Rdata"
load(loadfile)

if (loadfile == c("04_HiggsTrainExRbcLc.Rdata")) {
    trainDescr  <- train.ex.rbc.lc
    trainClass  <- train.eval
} else if (loadfile == c("04_HiggsTrainExRbcNas.Rdata")) {
    trainDescr  <- train.ex.rbc.nas
    trainClass  <- train.eval
} else if (loadfile == c("04_HiggsTrainExRbcLcNumNa.Rdata")) {
    trainDescr  <- train.ex.rbc.lc.numna
    trainClass  <- train.eval
}


##******************************************************************
## Main
##******************************************************************

##------------------------------------------------------------------
## Transform data to data.frame for the fitting procedure
##------------------------------------------------------------------
trainClass.df   <- as.data.frame(trainClass)       ## test on a small sample
trainDescr.df   <- as.data.frame(trainDescr)       ## test on a small sample



##------------------------------------------------------------------
## Use a subset of the available data for the parameter search phase
##------------------------------------------------------------------

## define the fraction to use as a hold-out sample
p_ho    <- 0.90     ## use large fraction for sweeps

## define a partition index
set.seed(88888888)
samp.idx    <- createDataPartition(
                            y=trainClass.df[,c("label")],
                            times=1,
                            p = (1-p_ho),
                            list = TRUE)

## split data into training & hold-out
holdDescr    <- trainDescr.df[ -samp.idx$Resample1, ]
holdClass    <- trainClass.df[ -samp.idx$Resample1, ]
sampDescr    <- trainDescr.df[  samp.idx$Resample1, ]
sampClass    <- trainClass.df[  samp.idx$Resample1, ]



##------------------------------------------------------------------
## set-up the fit parameters using the pre-selected (stratified) samples
##------------------------------------------------------------------
num.cv      <- 10
num.repeat  <- 1
num.total   <- num.cv * num.repeat

## define the fit parameters
fitControl <- trainControl(
                    method="cv",
                    number=num.cv,
#verboseIter=TRUE,
classProbs=TRUE,
summaryFunction = twoClassSummary,
#summaryFunction = amsSummary,
                    savePredictions=TRUE)

##------------------------------------------------------------------
## for sweeps create a parameter grid and then step through each one,
## saving interim results so it doesn;t crap out on you and you can
## monitor progress
##------------------------------------------------------------------
## [1] Search a small space around the maximum
## rfGrid  <- expand.grid(.mtry=c(10,15,20))
## [2] expand the search
## rfGrid  <- expand.grid(.mtry=seq(5,15,2))
## nGrid   <- dim(rfGrid)[1]
## [3] Try with the two class summary function
#rfGrid  <- expand.grid(.mtry=seq(5,15,2))
#nGrid   <- dim(rfGrid)[1]
## [4] Try a tune length of 25
nGrid <- 1

##------------------------------------------------------------------
## perform the fit
##------------------------------------------------------------------
for (i in 1:nGrid) {
    
    ## define a filename
    tmp.filename <- paste("LogitBoost_sweep_ROC_mtry_tuneLength10_.Rdata",sep="")
    
    ## perform the fit
    tmp.fit      <- try(train(   x=sampDescr[,-1],
                                 y=sampClass[,c("label")],
                                 #weights = sampClass$weight,
                                 method="LogitBoost",
                                 trControl=fitControl,
                                 #verbose=TRUE,
                                 #metric = "AMS",
                                 metric = "ROC",
                                 tuneLength=20
                                 #tuneGrid=data.frame(.mtry=rfGrid[i,])
                                 ))
   
   ## score the hold-out sample & compute the AMS curve
   tmp.score    <- predict(tmp.fit, newdata=holdDescr[,-1], type="prob")[,c("s")]
   tmp.pred     <- predict(tmp.fit, newdata=holdDescr[,-1])
   tmp.breaks   <- quantile(tmp.score, probs=seq(0,1,0.01))
   tmp.ams      <- sapply(tmp.breaks, calcAmsCutoff, tmp.score, holdClass$label, holdClass$weight)
   
   ## save the results
   save(tmp.fit, samp.idx, tmp.score, tmp.ams, file=tmp.filename)
}




