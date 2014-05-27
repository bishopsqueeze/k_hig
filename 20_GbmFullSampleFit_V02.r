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
##  1. Use 10-fold cross validation on the entire dataset
##  2. Score each hold-out fold
##
## Since we are manually doing the folds, fit all datapoints
##------------------------------------------------------------------

##------------------------------------------------------------------
## use common naming for the train data.frame
##------------------------------------------------------------------
trainDescr  <- as.data.frame(trainDescr)
trainClass  <- as.data.frame(trainClass)

##------------------------------------------------------------------
## define the fit control parameters (note the use of "none")
##------------------------------------------------------------------
fitControl  <- trainControl(method="none", savePredictions=FALSE)

##------------------------------------------------------------------
## define the number of folds to use
##------------------------------------------------------------------
num.folds   <- 10

##------------------------------------------------------------------
## create a set of folds & verify fold proportions
##------------------------------------------------------------------
fold.idx    <- createFolds(trainClass$label, k=num.folds, list=TRUE, returnTrain=FALSE)
## do.call("rbind",lapply(fold.idx, function(x){prop.table(table(trainClass[x,"label"]))}))

##------------------------------------------------------------------
## for full fits use the best model from the sweeps
##------------------------------------------------------------------
gbmGrid <- expand.grid(.interaction.depth=c(9), .n.trees=c(450), .shrinkage=c(0.05))
nGrid   <- dim(gbmGrid)[1]

##------------------------------------------------------------------
## loop over each fold and do a fit
##------------------------------------------------------------------
for (i in 1:num.folds) {

    ## define an output filename
    tmp.filename <- paste("gbm_full_depth",gbmGrid[1,1],"_trees",gbmGrid[1,2],"_shrink",gbmGrid[1,3],"_fold",i,"_V02.Rdata",sep="")

    ## load the fold index
    tmp.idx      <- fold.idx[[i]]
    
    ## perform the fit using the training data
    tmp.fit      <- try(train(  x=trainDescr[-tmp.idx,-1],
                                y=trainClass[-tmp.idx,c("label")],
                                method="gbm",
                                trControl=fitControl,
                                verbose=TRUE,
                                tuneGrid=data.frame(.interaction.depth=gbmGrid[1,1], .n.trees=gbmGrid[1,2], .shrinkage=gbmGrid[1,3])
                                ))

    hold.score      <- predict(tmp.fit, newdata=trainDescr[tmp.idx,-1], type="prob")[,c("s")]
    hold.pred       <- predict(tmp.fit, newdata=trainDescr[tmp.idx,-1])
    hold.breaks     <- quantile(hold.score, probs=seq(0,1,0.01))
    hold.ams        <- sapply(hold.breaks, calcAmsCutoff, hold.score, trainClass[tmp.idx,"label"], trainClass[tmp.idx,"weight"])

    ## save the results
    save(tmp.fit, fold.idx, tmp.idx, hold.score, hold.ams, file=tmp.filename)

}



