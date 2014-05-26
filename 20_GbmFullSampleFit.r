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
## Transform data to data.frame for the fitting procedure
##------------------------------------------------------------------
trainClass.df   <- as.data.frame(trainClass)
trainDescr.df   <- as.data.frame(trainDescr)

##------------------------------------------------------------------
## set-up the fit parameters using the pre-selected (stratified) samples
##------------------------------------------------------------------
num.cv      <- 10
num.repeat  <- 1
num.total   <- num.cv * num.repeat

##------------------------------------------------------------------
## define the fit control parameters
##------------------------------------------------------------------
fitControl <- trainControl(
                    method="cv",
                    number=num.cv,
                    verboseIter=TRUE,
                    savePredictions=FALSE)

##------------------------------------------------------------------
## for full fits use the best model from the sweeps
##------------------------------------------------------------------
gbmGrid <- expand.grid(.interaction.depth=c(9), .n.trees=c(450), .shrinkage=c(0.05))
nGrid   <- dim(gbmGrid)[1]

##------------------------------------------------------------------
## perform the fit; to avoid biases fit the model k times and then
## at some point combine the k versions to create the predicted
## outcome (because the true peak may be noisy)
##------------------------------------------------------------------

## set a seed
set.seed(4321)

## number of iterations to make
numFolds <- 10

## loop over the iterations and do a full fit
for (i in 2:numFolds) {
   
    ## define a filename
    tmp.filename <- paste("gbm_full_depth",gbmGrid[1,1],"_trees",gbmGrid[1,2],"_shrink",gbmGrid[1,3],"_fold",i,".Rdata",sep="")
   
    ## define a partition index using the full dataset
    samp.idx    <- createDataPartition(
                                        y=trainClass.df[,c("label")],
                                        times=1,
                                        p = (1-(1/numFolds)),
                                        list = TRUE)
    
    ## split the full dataset into training & hold-out
    holdDescr    <- trainDescr.df[ -samp.idx$Resample1, ]
    holdClass    <- trainClass.df[ -samp.idx$Resample1, ]
    sampDescr    <- trainDescr.df[  samp.idx$Resample1, ]
    sampClass    <- trainClass.df[  samp.idx$Resample1, ]
    
    ## perform the fit using the training data
    tmp.fit      <- try(train(  x=sampDescr[,-1],
                               y=sampClass[,c("label")],
                               method="gbm",
                               trControl=fitControl,
                               verbose=TRUE,
                               tuneGrid=data.frame(.interaction.depth=gbmGrid[1,1], .n.trees=gbmGrid[1,2], .shrinkage=gbmGrid[1,3])
                               ))
   
    ## score the hold-out sample & compute the AMS curve
    tmp.score    <- predict(tmp.fit, newdata=holdDescr[,-1], type="prob")[,c("s")]
    tmp.pred     <- predict(tmp.fit, newdata=holdDescr[,-1])
    tmp.breaks   <- quantile(tmp.score, probs=seq(0,1,0.01))
    tmp.ams      <- sapply(tmp.breaks, calcAmsCutoff, tmp.score, holdClass$label, holdClass$weight)

    ## save the results
    save(tmp.fit, samp.idx, tmp.score, tmp.ams, file=tmp.filename)

}




