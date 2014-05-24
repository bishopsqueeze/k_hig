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

##------------------------------------------------------------------
## register cores
##------------------------------------------------------------------
registerDoMC(2)

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
loadfile <- "04_HiggsTrainExRbcLc.Rdata"
load(loadfile)

if (loadfile == c("04_HiggsTrainExRbcLc.Rdata")) {
    trainDescr  <- train.ex.rbc.lc
    trainClass  <- train.eval
}

##******************************************************************
## Main
##******************************************************************

##------------------------------------------------------------------
##
##------------------------------------------------------------------

## check order of data
#if (sum(trainDescr[,eventid] - trainClass[,eventid]) == 0) {
#    trainDescr[,eventid:=NULL]
#    trainClass[,eventid:=NULL]
#}


## temporarily drop cols with NA
#na.cols     <- unlist(apply(train.sc, 2, function(x){sum(is.na(x))}))
#na.cols     <- names(na.cols[na.cols > 0])
#train.sc    <- train.sc[,-which(colnames(train.sc) %in% na.cols), with=FALSE]


##------------------------------------------------------------------
##
##------------------------------------------------------------------
nr  <- dim(trainDescr)[1]
ns  <- 10000

set.seed(123456789)
smp.idx    <- createDataPartition(
                    y=trainClass[,label],
                    times = 1,
                    p = 25000/nr,
                    list = TRUE,
                    groups = min(5, length(trainClass)))

##------------------------------------------------------------------
## set-up the tuning parameters
##------------------------------------------------------------------
gbmGrid    <- expand.grid(
                    .interaction.depth = c(2,3),
                    .n.trees = c(5, 10, 25, 50),
                    .shrinkage = c(0.01))



##------------------------------------------------------------------
## set-up the fit parameters using the pre-selected (stratified) samples
##------------------------------------------------------------------
num.cv      <- 3
num.repeat  <- 1
num.total   <- num.cv * num.repeat

## define the seeds to be used in the fits
#set.seed(123456789)
#seeds                               <- vector(mode = "list", length = (num.total + 1))
#for(k in 1:num.total) seeds[[k]]    <- sample.int(1000, nrow(gbmGrid))
#seeds[[num.total+1]]                <- sample.int(1000, 1)

## define the fit parameters
fitControl <- trainControl(
                    method="repeatedcv",
                    number=num.cv,
                    repeats=num.repeat,
                    classProbs=TRUE)

##------------------------------------------------------------------
## prep the data for fit
##------------------------------------------------------------------
trainClass.df   <- as.data.frame(trainClass)
trainDescr.df   <- as.data.frame(trainDescr)

##------------------------------------------------------------------
## perform the cross-validation fit
##------------------------------------------------------------------
tmp.fit <- try(train(   x=trainDescr.df[smp.idx[[1]],-grep("eventid",colnames(trainDescr))],
                        y=trainClass.df[smp.idx[[1]],c("label")],
                        method="gbm",
                        trControl=fitControl,
                        verbose=TRUE,
                        tuneLength=5))

tmp.pred <- predict(tmp.fit)
tmp.true <- unlist(trainClass[smp.idx[[1]],label])

##------------------------------------------------------------------
## perform the cross-validation fit
##------------------------------------------------------------------
tmp.fit <- try(train(   x=trainDescr.df[smp.idx[[1]],-grep("eventid",colnames(trainDescr))],
                        y=trainClass.df[smp.idx[[1]],c("label")],
                        method="ada",
                        trControl=fitControl,
                        verbose=TRUE,
                        tuneLength=5))







