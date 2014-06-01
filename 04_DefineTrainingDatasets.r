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
## train
#redundant.idx   <- which( colnames(train.sc) %in% gsub(".bc","",colnames(train.sc)[grep(".bc$",colnames(train.sc))]) )
#train.ex.rbc    <- train.sc[,-redundant.idx,with=FALSE]

## test
redundant.te    <- which( colnames(test.sc) %in% gsub(".bc","",colnames(test.sc)[grep(".bc$",colnames(test.sc))]) )
test.ex.rbc     <- test.sc[,-redundant.te,with=FALSE]


##------------------------------------------------------------------
## Case 2:  Excluded ...
##  - Redundant box-cox variables (i.e., the unscaled ones)
##  - Columns with NA(s)
##------------------------------------------------------------------
## train
#na.cols             <- unlist(apply(train.ex.rbc, 2, function(x){sum(is.na(x))}))
#na.cols             <- names(na.cols[na.cols > 0])
#train.ex.rbc.nas    <- train.ex.rbc[,-which(colnames(train.ex.rbc) %in% na.cols), with=FALSE]

## test
na.cols             <- unlist(apply(test.ex.rbc, 2, function(x){sum(is.na(x))}))
na.cols             <- names(na.cols[na.cols > 0])
test.ex.rbc.nas     <- test.ex.rbc[,-which(colnames(test.ex.rbc) %in% na.cols), with=FALSE]


##------------------------------------------------------------------
## Case 3:  Excluded ...
##  - Redundant box-cox variables (i.e., the unscaled ones)
##  - Linear combos of the flag variables
##------------------------------------------------------------------
## train
## identify columns you *wont* use in the linear combos calc
#flags.idx           <- c( grep(".fl$", colnames(train.ex.rbc)) )
#
## train cols to remove --> 3  4  5  7  8  9 10 11
#train.lc            <- findLinearCombos(train.ex.rbc[,flags.idx,with=FALSE])
#
## Further reduce the data
#train.ex.rbc.lc     <- cbind(   ## data w/out flag vars
#                                train.ex.rbc[,-flags.idx,with=FALSE],
#                                ## flag vars purged of linear combos
#                                train.ex.rbc[,flags.idx,with=FALSE][,-train.lc$remove,with=FALSE])

## test
## identify columns you *wont* use in the linear combos calc
flags.idx           <- c( grep(".fl$", colnames(test.ex.rbc)) )

## train cols to remove --> 3  4  5  7  8  9 10 11
test.lc             <- findLinearCombos(test.ex.rbc[,flags.idx,with=FALSE])

## Further reduce the data
test.ex.rbc.lc      <- cbind(   ## data w/out flag vars
                                test.ex.rbc[,-flags.idx,with=FALSE],
                                ## flag vars purged of linear combos
                                test.ex.rbc[,flags.idx,with=FALSE][,-test.lc$remove,with=FALSE])


##------------------------------------------------------------------
## Case 4:  Excluded ...
##  - Redundant box-cox variables (i.e., the unscaled ones)
##  - Linear combos of the flag variables
##  - Highly correlated variables
##------------------------------------------------------------------
## train
## identify columns you *wont* use in the correlation calc
#noscale.idx <- c(
#                    grep("id", colnames(train.ex.rbc.lc)),
#                    grep(".fl$", colnames(train.ex.rbc.lc)),
#                    grep(".[0-9]$", colnames(train.ex.rbc.lc))
#                )
#
## train cols to remove --> {}
#cor.train       <- cor(train.ex.rbc.lc[,-noscale.idx,with=FALSE], use="complete.obs")
#highCor.train   <- findCorrelation(cor.train, cutoff = 0.90)
#
#if (length(highCor.train) > 0) {
#    train.ex.rbc.lc.cor <- train.ex.rbc.lc[,-highCor.train,with=FALSE]
#} else {
#    train.ex.rbc.lc.cor <- train.ex.rbc.lc
#}

## test
## identify columns you *wont* use in the correlation calc
noscale.idx <- c(
                    grep("id", colnames(test.ex.rbc.lc)),
                    grep(".fl$", colnames(test.ex.rbc.lc)),
                    grep(".[0-9]$", colnames(test.ex.rbc.lc))
                )

## train cols to remove --> {}
cor.test       <- cor(test.ex.rbc.lc[,-noscale.idx,with=FALSE], use="complete.obs")
highCor.test   <- findCorrelation(cor.test, cutoff = 0.90)

if (length(highCor.test) > 0) {
    test.ex.rbc.lc.cor <- test.ex.rbc.lc[,-highCor.test,with=FALSE]
} else {
    test.ex.rbc.lc.cor <- test.ex.rbc.lc
}



##------------------------------------------------------------------
## Case 5:  Excluded ...
##  - Redundant box-cox variables (i.e., the unscaled ones)
##  - Linear combos of the flag variables
##  - Exchange NA for -9.0 (an outlier given the ranges but not extreme)
##------------------------------------------------------------------
## train
#train.ex.rbc.lc.numna   <- train.ex.rbc.lc[,lapply(.SD, function(x){ifelse(is.na(x),-9.000,x)})]

## test
test.ex.rbc.lc.numna    <- test.ex.rbc.lc[,lapply(.SD, function(x){ifelse(is.na(x),-9.000,x)})]

##------------------------------------------------------------------
## Save the results to individual files
##------------------------------------------------------------------

## Train -> Case 0
#save(train.eval, train.all, file="04_HiggsTrainAll.Rdata")
save(test.all, file="04_HiggsTestAll.Rdata")

## Train -> Case 1
#save(train.eval, train.ex.rbc, file="04_HiggsTrainExRbc.Rdata")
save(test.ex.rbc, file="04_HiggsTestExRbc.Rdata")

## Train -> Case 2
#save(train.eval, train.ex.rbc.nas, file="04_HiggsTrainExRbcNas.Rdata")
save(test.ex.rbc.nas, file="04_HiggsTestExRbcNas.Rdata")

## Train -> Case 3
#save(train.eval, train.ex.rbc.lc, file="04_HiggsTrainExRbcLc.Rdata")
save(test.ex.rbc.lc, file="04_HiggsTestExRbcLc.Rdata")

## Train -> Case 4
#save(train.eval, train.ex.rbc.lc.cor, file="04_HiggsTrainExRbcLcCorr.Rdata")
save(est.ex.rbc.lc.cor, file="04_HiggsTestExRbcLcCorr.Rdata")

## Train -> Case 5
#save(train.eval, train.ex.rbc.lc.numna, file="04_HiggsTrainExRbcLcNumNa.Rdata")
save(test.ex.rbc.lc.numna, file="04_HiggsTestExRbcLcNumNa.Rdata")

## Test -- > Include all columns and filter upon evaluation)
#save(test.all, file="04_HiggsTestAll.Rdata")







