##------------------------------------------------------------------
## Create several different sets of training data:
##
##------------------------------------------------------------------

##------------------------------------------------------------------
## Load libraries
##------------------------------------------------------------------
library(data.table)
library(caret)

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
## Load the scaled data
##------------------------------------------------------------------
load("03_HiggsScaledTrain.Rdata")
load("03_HiggsScaledTest.Rdata")


##******************************************************************
## Main (following caret procedure)
##******************************************************************

## Note:  Tested for near-zero-vars & none found

##------------------------------------------------------------------
## Case 0:  Excluded ...
##  - None
##------------------------------------------------------------------
train.all       <- train.sc
test.all        <- test.sc


##------------------------------------------------------------------
## Case 1:  Excluded ...
##  - Redundant box-cox variables (i.e., the unscaled ones)
##------------------------------------------------------------------
redundant.idx   <- which( colnames(train.sc) %in% gsub(".bc","",colnames(train.sc)[grep(".bc$",colnames(train.sc))]) )
train.ex.rbc    <- train.sc[,-redundant.idx,with=FALSE]


##------------------------------------------------------------------
## Case 2:  Excluded ...
##  - Redundant box-cox variables (i.e., the unscaled ones)
##  - Columns with NA(s)
##------------------------------------------------------------------
na.cols             <- unlist(apply(train.ex.rbc, 2, function(x){sum(is.na(x))}))
na.cols             <- names(na.cols[na.cols > 0])
train.ex.rbc.nas    <- train.ex.rbc[,-which(colnames(train.ex.rbc) %in% na.cols), with=FALSE]


##------------------------------------------------------------------
## Case 3:  Excluded ...
##  - Redundant box-cox variables (i.e., the unscaled ones)
##  - Linear combos of the flag variables
##------------------------------------------------------------------

## identify columns you *wont* use in the linear combos calc
flags.idx           <- c( grep(".fl$", colnames(train.ex.rbc)) )

## train cols to remove --> 3  4  5  7  8  9 10 11
train.lc            <- findLinearCombos(train.ex.rbc[,flags.idx,with=FALSE])

## Further reduce the data
train.ex.rbc.lc     <- cbind(   ## data w/out flag vars
                                train.ex.rbc[,-flags.idx,with=FALSE],
                                ## flag vars purged of linear combos
                                train.ex.rbc[,flags.idx,with=FALSE][,-train.lc$remove,with=FALSE])


##------------------------------------------------------------------
## Case 4:  Excluded ...
##  - Redundant box-cox variables (i.e., the unscaled ones)
##  - Linear combos of the flag variables
##  - Highly correlated variables
##------------------------------------------------------------------

## identify columns you *wont* use in the correlation calc
noscale.idx <- c(
                    grep("id", colnames(train.ex.rbc.lc)),
                    grep(".fl$", colnames(train.ex.rbc.lc)),
                    grep(".[0-9]$", colnames(train.ex.rbc.lc))
                )

## train cols to remove --> {}
cor.train       <- cor(train.ex.rbc.lc[,-noscale.idx,with=FALSE], use="complete.obs")
highCor.train   <- findCorrelation(cor.train, cutoff = 0.90)

if (length(highCor.train) > 0) {
    train.ex.rbc.lc.cor <- train.ex.rbc.lc[,-highCor.train,with=FALSE]
} else {
    train.ex.rbc.lc.cor <- train.ex.rbc.lc
}


##------------------------------------------------------------------
## Save the results to individual files
##------------------------------------------------------------------

## Train -> Case 0
save(train.eval, train.all, file="04_HiggsTrainAll.Rdata")

## Train -> Case 1
save(train.eval, train.ex.rbc, file="04_HiggsTrainExRbc.Rdata")

## Train -> Case 2
save(train.eval, train.ex.rbc.nas, file="04_HiggsTrainExRbcNas.Rdata")

## Train -> Case 3
save(train.eval, train.ex.rbc.lc, file="04_HiggsTrainExRbcLc.Rdata")

## Train -> Case 4
save(train.eval, train.ex.rbc.lc.cor, file="04_HiggsTrainExRbcLcCorr.Rdata")

## Test -- > Include all columns and filter upon evaluation)
save(test.all, file="04_HiggsTestAll.Rdata")







