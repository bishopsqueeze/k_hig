##------------------------------------------------------------------
## The purpose of this test is to see what happens to the AMS if
## we randomize the lables in various ways.  The Kaggle scoreboard
## presents a random submission benchmark score = 0.58477.  I want
## to see if I can reproduce that.
##------------------------------------------------------------------

##------------------------------------------------------------------
## Load libraries
##------------------------------------------------------------------
library(data.table)
library(caret)
library(foreach)
library(doMC)
library(xtable)

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
loadfile <- "04_HiggsTrainExRbcLc.Rdata"
load(loadfile)

if (loadfile == c("04_HiggsTrainExRbcLc.Rdata")) {
    trainDescr  <- train.ex.rbc.lc
    trainClass  <- train.eval
} else if (loadfile == c("04_HiggsTrainExRbcNas.Rdata")) {
    trainDescr  <- train.ex.rbc.nas
    trainClass  <- train.eval
}

##******************************************************************
## Main
##******************************************************************

## isolate the training labels, weights
debugClass  <- as.factor(trainClass[,label])
debugWeight <- as.vector(trainClass[,weight])

## compute the expected values Nb, Ns ~ (411000, 692)
xtabs(debugWeight ~ debugClass)


##------------------------------------------------------------------
## Test 1:  A completely random (and mis-balanced) benchmark
##------------------------------------------------------------------
debugRand   <- as.factor(sample(c("b","s"), length(debugClass), replace=TRUE))
xtabs(debugWeight ~ debugRand)

## step through the AMS calc (ams ~ 0.76 with matched sample size)
w       <- debugWeight
y_true  <- debugClass
y_pred  <- debugRand

tmp <- calcAms(y_pred=y_pred, y_true=y_true, w=w)


##------------------------------------------------------------------
## Test 2:  Same as Test 1, but with half the population
##------------------------------------------------------------------
idx_50  <- unlist(createDataPartition(y=debugClass, p=0.50))

## step through the AMS calc (ams ~ 0.76 with 50% sample size)
w       <- debugWeight[idx_50]
y_true  <- debugClass[idx_50]
y_pred  <- debugRand[idx_50]

tmp <- calcAms(y_pred=y_pred, y_true=y_true, w=w)


##------------------------------------------------------------------
## Test 3:  Same as Test 1, but with 10% of the population
##------------------------------------------------------------------
idx_10  <- unlist(createDataPartition(y=debugClass, p=0.10))

## step through the AMS calc (ams ~ 0.76 with 10% sample size)
w       <- debugWeight[idx_10]
y_true  <- debugClass[idx_10]
y_pred  <- debugRand[idx_10]

tmp <- calcAms(y_pred=y_pred, y_true=y_true, w=w)


##------------------------------------------------------------------
## Test 4:  Shuffle the actual weights
##------------------------------------------------------------------
shuffle.idx <- sample.int(length(debugClass), length(debugClass), replace=FALSE)

## step through the AMS calc (ams ~ 0.63 with matched sample size)
w       <- debugWeight
y_true  <- debugClass
y_pred  <- debugClass[shuffle.idx]

tmp <- calcAms(y_pred=y_pred, y_true=y_true, w=w)







