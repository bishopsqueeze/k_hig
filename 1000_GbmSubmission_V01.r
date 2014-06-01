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
trainFile   <- "04_HiggsTrainExRbcLcNumNa.Rdata"
testFile    <- "04_HiggsTestExRbcLcNumNa.Rdata"

load(trainFile)
if (trainFile == c("04_HiggsTrainExRbcLcNumNa.Rdata")) {
    trainDescr  <- train.ex.rbc.lc.numna
    trainClass  <- train.eval
}

load(testFile)
if (testFile == c("04_HiggsTestExRbcLcNumNa.Rdata")) {
    testDescr  <- test.ex.rbc.lc.numna
    #testClass  <- test.eval
}



##******************************************************************
## Main
##******************************************************************

## Version 1
v01.dir <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/gbm_full_depth9_trees450_shrink0.05_V01"
v03.dir <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/gbm_full_depth9_trees450_shrink0.05_V03"
v04.dir <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/gbm_full_depth9_trees450_shrink0.05_V04"



(loadfile == c("04_HiggsTrainExRbcLcNumNa.Rdata")) {
    trainDescr  <- train.ex.rbc.lc.numna
    trainClass  <- train.eval
}