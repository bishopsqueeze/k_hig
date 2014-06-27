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
registerDoMC(12)

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
loadfile <- "02_HiggsPreProcTrain.Rdata"
load(loadfile)

if (loadfile == c("02_HiggsPreProcTrain.Rdata"))
{
    trainDescr.dt  <- train.pp
    trainClass.dt  <- train.eval
}

##******************************************************************
## constants
##******************************************************************
s_val   <- 1
b_val   <- 0
b_tau   <- 10
vsize   <- .10
na_val  <- -999
n_trees <- 500

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
## use common naming for the train data.frame
##------------------------------------------------------------------
trainDescr  <- as.data.frame(trainDescr.dt)
trainClass  <- as.data.frame(trainClass.dt)

##------------------------------------------------------------------
## Create a numeric label
##------------------------------------------------------------------
trainClass$label.num <- as.numeric(ifelse(trainClass$label=="s", s_val, b_val))

##------------------------------------------------------------------
## Compute the sum of signal and background weights
##------------------------------------------------------------------
s_total <- sum(trainClass$weight[trainClass$label.num == s_val])
b_total <- sum(trainClass$weight[trainClass$label.num == b_val])

## compute max_AMS
max_AMS <- AMS(trainClass$label.num, trainClass$label.num, trainClass$weight)

##------------------------------------------------------------------
## Replace NAs (already done in the preprocessing steps)
##------------------------------------------------------------------
#trainDescr[trainDescr==na_val] <- NA

##------------------------------------------------------------------
## Create the hold out data
##------------------------------------------------------------------

## hold-out percentage
ho.pct      <- 0.10

## define a partition index using the full dataset
set.seed(1234)
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
## Renormalize weights
##------------------------------------------------------------------
ttClass$weight <- normalize(ttClass$weight,ttClass$label.num,s_total,b_total)
hoClass$weight <- normalize(hoClass$weight,hoClass$label.num,s_total,b_total)

## confirm normalization
AMS(ttClass$label.num,ttClass$label.num,ttClass$weight)
AMS(hoClass$label.num,hoClass$label.num,hoClass$weight)

s_train <- sum(ttClass$weight[ttClass$label.num==s_val])
b_train <- sum(ttClass$weight[ttClass$label.num==b_val])

w = ifelse(ttClass$label.num==s_val, ttClass$weight/s_train, ttClass$weight/b_train)


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
                    repeats=num.repeat,
                    savePredictions=FALSE)

##------------------------------------------------------------------
## for full fits use the best model from the sweeps
##------------------------------------------------------------------3
gbmGrid <- expand.grid(.interaction.depth=c(9), .n.trees=c(500), .shrinkage=c(0.20))
nGrid   <- dim(gbmGrid)[1]


##------------------------------------------------------------------
## define the number of "iterations"
##------------------------------------------------------------------
set.seed(43214321)
num.iter   <- 50
seed.vec   <- sample.int(1000000,num.iter,replace=FALSE)

##------------------------------------------------------------------
## loop over each fold and do a fit
##------------------------------------------------------------------
for (i in 1:num.iter) {

    ## define an output filename
    tmp.filename <- paste("gbm_full_depth",gbmGrid[1,1],"_trees",gbmGrid[1,2],"_shrink",gbmGrid[1,3],"_iter",i,"_V11.Rdata",sep="")

    ## load the fold index
    ##tmp.idx      <- fold.idx[[i]]
    
    ## perform the fit using the training data
    set.seed(seed.vec[i])
    tmp.fit      <- try(train(  x=ttDescr[,-1],
                                y=ttClass[,c("label")],
                                method="gbm",
                                weights=w,
                                trControl=fitControl,
                                tuneGrid=data.frame(.interaction.depth=gbmGrid[1,1], .n.trees=gbmGrid[1,2], .shrinkage=gbmGrid[1,3]),
                                n.minobsinnode=1,
                                distribution="adaboost",
                                bag.fraction=0.9
                                ))

    hold.score      <- predict(tmp.fit, newdata=hoDescr[,-1], type="prob")[,c("s")]
    hoClass$scores  <- hold.score
    
    hold.res     <- getAMS(hoClass)
    hold.pred    <- ifelse(hoClass$scores >= hold.res$threshold, s_val, b_val)
    AMS(hold.pred, hoClass$label.num, hoClass$weight)
    cat("Iteration = ", i ," AMS Ouput = ", as.numeric(hold.res$amsMax), "\n")

    ## save the results
    save(tmp.fit, samp.idx, hold.score, hold.res, hold.pred, seed.vec, file=tmp.filename)

}




