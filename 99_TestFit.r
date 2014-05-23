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
load("04_HiggsCaretProcTrain.Rdata")
load("04_HiggsCaretProcTest.Rdata")


##******************************************************************
## Main
##******************************************************************

##------------------------------------------------------------------
##
##------------------------------------------------------------------

if (sum(train.sc[,eventid] - train.eval[,eventid]) == 0) {
    train.sc[,eventid:=NULL]
    train.eval[,eventid:=NULL]
}


## temporarily drop cols with NA
na.cols     <- unlist(apply(train.sc, 2, function(x){sum(is.na(x))}))
na.cols     <- names(na.cols[na.cols > 0])
train.sc    <- train.sc[,-which(colnames(train.sc) %in% na.cols), with=FALSE]


##------------------------------------------------------------------
##
##------------------------------------------------------------------
set.seed(123456789)
reg.idx    <- createDataPartition(train.eval[,label], p=0.50, list=TRUE)


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
num.cv      <- 10
num.repeat  <- 3
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
## perform the cross-validation fit
##------------------------------------------------------------------
tmp.fit <- try(train(   x=as.data.frame(train.sc[reg.idx[[1]],]),
                        y=as.factor(train.eval[reg.idx[[1]],label]),
                        method="gbm",
                        trControl=fitControl,
                        verbose=TRUE,
                        tuneLength=10))




