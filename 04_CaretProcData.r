##------------------------------------------------------------------
## From the kaggle website:
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
## Load the training data
##------------------------------------------------------------------
load("03_HiggsScaledTrain.Rdata")
load("03_HiggsScaledTest.Rdata")


##******************************************************************
## Main
##******************************************************************


##------------------------------------------------------------------
## In some situations, the data generating mechanism can create predictors
## that only have a single unique value (i.e. a "zero-variance predictor").
## For many models (excluding tree-based models), this may cause the model
## to crash or the fit to be unstable.
##------------------------------------------------------------------

## train cols to remove --> {}
nzv.train <- nearZeroVar(train.sc, saveMetrics=TRUE)


##------------------------------------------------------------------
##  While there are some models that thrive on correlated predictors
## (such as pls), other models may benefit from reducing the level of
## correlation between the predictors. Given a correlation matrix, the
## findCorrelation function uses the following algorithm to flag
## predictors for removal.
##------------------------------------------------------------------

## identify columns you *wont* use in the correlation calc
noscale.idx <- c(
                    grep("id", colnames(train.sc)),
                    grep(".fl$", colnames(train.sc)),
                    grep(".[0-9]$", colnames(train.sc))
                )

## train cols to remove --> {}
cor.train       <- cor(train.sc[,-noscale.idx,with=FALSE], use="complete.obs")
highCor.train   <- findCorrelation(cor.train, cutoff = 0.90)

## test cols to remove --> {}
cor.test        <- cor(test.sc[,-noscale.idx,with=FALSE], use="complete.obs")
highCor.test    <- findCorrelation(cor.test, cutoff = 0.90)

## Create "reduced" size data tables
train.red    <- train.sc
test.red     <- test.sc
#train.red   <- cbind(train.sc[,noscale.idx,with=FALSE], train.sc[,-noscale.idx,with=FALSE][,-highCor.train,with=FALSE])
#test.red    <- cbind(test.sc[,noscale.idx,with=FALSE], test.sc[,-noscale.idx,with=FALSE][,-highCor.train,with=FALSE])   ## train highCor vars


##------------------------------------------------------------------
## The function findLinearCombos uses the QR decomposition of a matrix
## to enumerate sets of linear combinations (if they exist). For example,
## consider the following matrix that is could have been produced by a
## less-than-full-rank parameterizations of a two-way experimental layout.
##------------------------------------------------------------------

## identify columns you *wont* use in the correlation calc
flags.idx <- c( grep(".fl$", colnames(train.red)) )

## train cols to remove --> 3  4  5  7  8  9 10 11
train.lc <- findLinearCombos(train.red[,flags.idx,with=FALSE])

## test cols to remove --> 3  4  5  7  8  9 10 11
test.lc  <- findLinearCombos(test.red[,flags.idx,with=FALSE])

## Further reduce the data
train.red   <- cbind(train.red[,-flags.idx,with=FALSE], train.red[,flags.idx,with=FALSE][,-train.lc$remove,with=FALSE] )
test.red    <- cbind(test.red[,-flags.idx,with=FALSE], test.red[,flags.idx,with=FALSE][,-test.lc$remove,with=FALSE] )


##------------------------------------------------------------------
## Save the results
##------------------------------------------------------------------
save(train.eval, train.sc, train.red, file="04_HiggsCaretProcTrain.Rdata")
save(test.sc, test.red, file="04_HiggsCaretProcTest.Rdata")







