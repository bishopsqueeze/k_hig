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
load("02_HiggsPreProcTrain.Rdata")
load("02_HiggsPreProcTest.Rdata")


##******************************************************************
## Main
##******************************************************************

## an id column to each subset of data
train.pp[,id:=c("tr")]
test.pp[,id:=c("te")]

## combine the datasets
comb.pp <- rbind(train.pp, test.pp)
setkey(comb.pp, eventid)

## identify columns you *wont* scale
noscale.idx <- c(
                    grep("id", colnames(comb.pp)),
                    grep(".fl$", colnames(comb.pp)),
                    grep(".[0-9]$", colnames(comb.pp))
                )

## join the scaled data to the original after dropping orig
comb.sc <- cbind(
                    comb.pp[,noscale.idx,with=FALSE],
                    data.table(scale(comb.pp[,-noscale.idx,with=FALSE]))
                )
setkey(comb.sc,"id")

## break back into train and test datasets
train.sc    <- comb.sc["tr"]
test.sc     <- comb.sc["te"]

## drop the ids
train.sc[,id:=NULL]
test.sc[,id:=NULL]

## save the results
save(train.eval, train.sc, file="03_HiggsScaledTrain.Rdata")
save(test.sc, file="03_HiggsScaledTest.Rdata")



