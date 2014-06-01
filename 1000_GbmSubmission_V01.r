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
library(gbm)
library(plyr)

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
trainFile   <- "04_HiggsTrainExRbcLcNumNa.Rdata"
testFile    <- "04_HiggsTestExRbcLcNumNa.Rdata"

load(trainFile)
if (trainFile == c("04_HiggsTrainExRbcLcNumNa.Rdata")) {
    trainDescr  <- as.data.frame(train.ex.rbc.lc.numna)
    trainClass  <- as.data.frame(train.eval)
}

load(testFile)
if (testFile == c("04_HiggsTestExRbcLcNumNa.Rdata")) {
    testDescr  <- as.data.frame(test.ex.rbc.lc.numna)
}



##******************************************************************
## Main
##******************************************************************

## Version 1
v01.dir <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/gbm_full_depth9_trees450_shrink0.05_V01"

## Version 3
v03.dir <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/gbm_full_depth9_trees450_shrink0.05_V03"

## Version 4
v04.dir <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/gbm_full_depth9_trees450_shrink0.05_V04"


##------------------------------------------------------------------
## Review Version_01 fits
##------------------------------------------------------------------
## Note that CV was performed on varying subsets, so I can't
## compare results across models
##------------------------------------------------------------------
v01.files   <- dir(v01.dir)[grep(".Rdata$", dir(v01.dir))]
v01.num     <- length(v01.files)
v01.seq     <- seq(0,1,0.01)
v01.ams     <- matrix(0,nrow=length(v01.seq),ncol=v01.num)

for (i in 1:v01.num) {
    
    tmp.file    <- paste0(v01.dir,"/",v01.files[i])
    load(tmp.file)
    
    ## in this case, the sample index represents the training data
    holdDescr   <- trainDescr[-samp.idx$Resample1,]
    holdClass   <- trainClass[-samp.idx$Resample1,]
    
    ## score the hold-out sample & compute the AMS curve
    tmp.score    <- predict(tmp.fit, newdata=holdDescr[,-1], type="prob")[,c("s")]
    tmp.pred     <- predict(tmp.fit, newdata=holdDescr[,-1])
    tmp.breaks   <- quantile(tmp.score, probs=v01.seq)
    tmp.ams      <- sapply(tmp.breaks, calcAmsCutoff, tmp.score, holdClass$label, holdClass$weight)
    v01.ams[,i]  <- tmp.ams
    
    if (i == 1) {
        plot(v01.seq, v01.ams[,i], type="b", col=i)
    } else {
        points(v01.seq, v01.ams[,i], type="b", col=i)
    }
}





