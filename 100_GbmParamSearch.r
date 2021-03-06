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
loadfile <- "04_HiggsTrainAll.Rdata"
load(loadfile)

if (loadfile == c("04_HiggsTrainAll.Rdata"))
{
    trainDescr  <- train.all
    trainClass  <- train.eval
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
## functions
##******************************************************************


##******************************************************************
## Main
##******************************************************************

##------------------------------------------------------------------
## Transform data to data.frame for the fitting procedure
##------------------------------------------------------------------
trainClass.df   <- as.data.frame(trainClass)
trainDescr.df   <- as.data.frame(trainDescr)

##------------------------------------------------------------------
## Create a numeric label
##------------------------------------------------------------------
trainClass.df$label.num <- as.numeric(ifelse(trainClass.df$label=="s",s_val,b_val))

##------------------------------------------------------------------
## Compute the sum of signal and background weights
##------------------------------------------------------------------
s_total <- sum(trainClass.df$weight[trainClass.df$label.num == s_val])
b_total <- sum(trainClass.df$weight[trainClass.df$label.num == b_val])

## compute max_AMS
max_AMS <- AMS(trainClass.df$label.num,trainClass.df$label.num,trainClass.df$weight)

##------------------------------------------------------------------
## Replace NA
##------------------------------------------------------------------
trainDescr.df[trainDescr.df==na_val] <- NA

##------------------------------------------------------------------
## Use a subset of the available data for the parameter search phase
##------------------------------------------------------------------

## define the fraction to use as a hold-out sample
p_ho    <- 0.10     ## use large fraction for sweeps

## define a partition index
set.seed(7777777)
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
## Renormalize weights
##------------------------------------------------------------------
sampClass$weight <- normalize(sampClass$weight,sampClass$label.num,s_total,b_total)
holdClass$weight <- normalize(holdClass$weight,holdClass$label.num,s_total,b_total)

## confirm normalization
AMS(sampClass$label.num,sampClass$label.num,sampClass$weight)
AMS(holdClass$label.num,holdClass$label.num,holdClass$weight)

s_train <- sum(sampClass$weight[sampClass$label.num==s_val])
b_train <- sum(sampClass$weight[sampClass$label.num==b_val])

w = ifelse(sampClass$label.num==s_val, sampClass$weight/s_train, sampClass$weight/b_train)


##------------------------------------------------------------------
## set-up the fit parameters using the pre-selected (stratified) samples
##------------------------------------------------------------------
num.cv      <- 5
num.repeat  <- 1
num.total   <- num.cv * num.repeat

## define the fit parameters
fitControl <- trainControl(
                    method="cv",
                    number=num.cv,
                    savePredictions=FALSE)

##------------------------------------------------------------------
## for sweeps create a parameter grid and then step through each one,
## saving interim results so it doesn;t crap out on you and you can
## monitor progress
##------------------------------------------------------------------

## [1] Tested after identifying 9/450 via an early sweep
## gbmGrid <- expand.grid(.interaction.depth=9, .n.trees=c(450), .shrinkage=c(.01,.05,.1,.2))
## [2] Tested after 0.05 yielded the best accuracy/ams
## gbmGrid <- expand.grid(.interaction.depth=c(9,10), .n.trees=c(400,450,500), .shrinkage=c(.05))
## [3] Tested after 0.05/450/9 yielded the best accuracy/ams
gbmGrid <- expand.grid(.interaction.depth=c(9), .n.trees=c(500,1000,2000), .shrinkage=c(0.05))


##------------------------------------------------------------------
## perform the fit #### CHANGE LOOP BACK TO 1 ###
##------------------------------------------------------------------
nGrid <- dim(gbmGrid)[1]
for (i in 1:nGrid) {
    
    ## define a filename
    tmp.filename <- paste("gbm_weighted_depth",gbmGrid[i,1],"_trees",gbmGrid[i,2],"_shrink",gsub("\\.","",as.character(gbmGrid[i,3])),".Rdata",sep="")
    
    ## perform the fit
    tmp.fit      <- try(train(  x=sampDescr[,-1],
                                y=sampClass[,c("label")],
                                method="gbm",
                                weights=w,
                                trControl=fitControl,
                                tuneGrid=data.frame(.interaction.depth=gbmGrid[i,1], .n.trees=gbmGrid[i,2], .shrinkage=gbmGrid[i,3]),
                                n.minobsinnode=1,
                                bag.fraction=0.9,
                                ))

    ## score the hold-out sample & compute the AMS curve
    tmp.score    <- predict(tmp.fit, newdata=holdDescr[,-1], type="prob")[,c("s")]
    #tmp.pred     <- predict(tmp.fit, newdata=holdDescr[,-1])
    #tmp.breaks   <- quantile(tmp.score, probs=seq(0,1,0.01))
    #tmp.ams      <- sapply(tmp.breaks, calcAmsCutoff, tmp.score, holdClass$label, holdClass$weight)
    
    test         <- holdClass
    test$scores  <- tmp.score
    
    r = getAMS(test)
    p = ifelse( test$scores >= r$threshold, s_val,b_val)
    AMS(p,test$label.num,test$weight)
    
    ## save the results
    save(tmp.fit, samp.idx, tmp.score, tmp.ams, file=tmp.filename)
}




