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
load("01_HiggsRawTrain.Rdata")


##------------------------------------------------------------------
## Compute a score using only the one mass variable (which has -999 values)
##------------------------------------------------------------------
train.dt[,myscore:=-abs(der_mass_mmc-125.0)]

## histogram of scores
##ggplot(train.dt, aes(x=myscore, fill=label)) + geom_histogram(binwidth=1, alpha=.5, position="identity")

## chosen threshold in the example
thresh <- -22

## compute the predicted value
y.pred  <- as.factor(ifelse( as.vector(train.dt[,myscore]) > thresh, "s", "b"))
y.true  <- train.dt[,label]
wgt     <- as.vector(train.dt[,weight])

## compute AMS
ams     <- calcAMS(y.pred, y.true, wgt)

## Python results:
## Loop again to determine the AMS, using threshold: -22
## AMS computed from training file : 1.52890675501 ( signal= 457.279138287  bkg= 89291.9121254 )

## My results:
## calcAMS :: scalar= 1  signal= 457.2791383  background= 89291.91213  ams= 1.528906755


##------------------------------------------------------------------
## Compute again using a range of thresholds
##------------------------------------------------------------------
thresh.range    <- quantile(as.vector(train.dt[,myscore]), probs=seq(0,1,0.01))
thresh.ams      <- sapply(thresh.range, calcAmsCutoff, as.vector(train.dt[,myscore]), y.true, wgt)



