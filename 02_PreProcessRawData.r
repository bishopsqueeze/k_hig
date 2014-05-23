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
load("01_HiggsRawTrain.Rdata")
load("01_HiggsRawTest.Rdata")


##******************************************************************
## Main
##******************************************************************

## identify columns used for scoring the training data
eval.cols   <- c("eventid", "label", "weight")

## isolate the data columns
train.cols  <- colnames(train.dt)[ which(!(colnames(train.dt) %in% eval.cols))]
test.cols   <- colnames(test.dt)[ which(!(colnames(test.dt) %in% eval.cols))]

## isolate evaluation data
train.eval  <- train.dt[,eval.cols,with=FALSE]
setkey(train.eval, eventid)

## isolate the training data
train.dt    <- train.dt[, -which((colnames(train.dt) %in% c("label", "weight"))), with=FALSE]

## preprocess the training and test data
train.pp    <- preProcessData(mydt=train.dt, mycols=train.cols)
setkey(train.pp, eventid)

test.pp     <- preProcessData(mydt=test.dt, mycols=test.cols)
setkey(test.pp, eventid)


##------------------------------------------------------------------
## Save the pre-processed files
##------------------------------------------------------------------
save(train.pp, train.eval, file="02_HiggsPreProcTrain.Rdata")
save(test.pp, file="02_HiggsPreProcTest.Rdata")



