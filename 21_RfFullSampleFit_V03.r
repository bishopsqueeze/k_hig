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
## In this iteration, we will do the following:
##  1. First, we create a hold-out sample to be used across
##     all folds
##  1. Use 10-fold cross validation on the remaining dataset
##  2. Score each hold-out fold
##
## Since we are manually doing the folds, fit all datapoints
##------------------------------------------------------------------

##------------------------------------------------------------------
## Transform data to data.frame for the fitting procedure
##------------------------------------------------------------------
trainClass   <- as.data.frame(trainClass)
trainDescr   <- as.data.frame(trainDescr)

##------------------------------------------------------------------
## set-up the fit parameters using the pre-selected (stratified) samples
##------------------------------------------------------------------
num.cv      <- 5
num.repeat  <- 1
num.total   <- num.cv * num.repeat

##------------------------------------------------------------------
## define the fit control parameters
##------------------------------------------------------------------
fitControl <- trainControl(
                    method="cv",
                    number=num.cv,
                    savePredictions=FALSE)

##------------------------------------------------------------------
## for full fits use the best model from the sweeps
##------------------------------------------------------------------
rfGrid  <- expand.grid(.mtry=c(7))
nGrid   <- dim(rfGrid)[1]

##------------------------------------------------------------------
## Create the hold out data
##------------------------------------------------------------------

## hold-out percentage
ho.pct      <- 0.05

## define a partition index using the full dataset
set.seed(888888)
samp.idx    <- createDataPartition(
                    y=trainClass[,c("label")],
                    times=1,
                    p = (1-ho.pct),
                    list = TRUE)

## split the full dataset into HOLD-OUT (ho) and TRAIN/TEST (tt)
hoDescr     <- trainDescr[ -samp.idx$Resample1, ]
hoClass     <- trainClass[ -samp.idx$Resample1, ]
ttDescr     <- trainDescr[  samp.idx$Resample1, ]
ttClass     <- trainClass[  samp.idx$Resample1, ]

##------------------------------------------------------------------
## define the number of "iterations"
##------------------------------------------------------------------
num.iter   <- 10

##------------------------------------------------------------------
## create a set of folds using the remaining TRAIN/TEST data
##------------------------------------------------------------------
##fold.idx    <- createFolds(ttClass$label, k=num.folds, list=TRUE, returnTrain=FALSE)
##do.call("rbind",lapply(fold.idx, function(x){prop.table(table(trainClass[x,"label"]))}))

##------------------------------------------------------------------
## loop over each iteration and do a fit
##------------------------------------------------------------------
for (i in 1:num.iter) {
   
    ## define a filename
    tmp.filename <- paste("rf_full_mtry",rfGrid[1],"_fold",i,"_V03.Rdata",sep="")
    cat("Working on ... ", tmp.filename, "\n")
    
    ## perform the fit using the training data
    tmp.fit      <- try(train( x=ttDescr[,-1],
                               y=ttClass[,c("label")],
                               method="rf",
                               trControl=fitControl,
                               tuneGrid=data.frame(.mtry=rfGrid[1])
                               ))
   
   hold.score      <- predict(tmp.fit, newdata=hoDescr[,-1], type="prob")[,c("s")]
   hold.pred       <- predict(tmp.fit, newdata=hoDescr[,-1])
   hold.breaks     <- quantile(hold.score, probs=seq(0,1,0.01))
   hold.ams        <- sapply(hold.breaks, calcAmsCutoff, hold.score, hoClass[,"label"], hoClass[,"weight"])
   
   ## save the results
   save(tmp.fit, samp.idx, hold.score, hold.ams, file=tmp.filename)

}




