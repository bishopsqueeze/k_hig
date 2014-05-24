##------------------------------------------------------------------
## Load the raw data files from the Kaggle website into Rdata files
## http://www.kaggle.com/c/higgs-boson/data
##------------------------------------------------------------------

##------------------------------------------------------------------
## Load libraries
##------------------------------------------------------------------
library(data.table)

##------------------------------------------------------------------
## Clear the workspace
##------------------------------------------------------------------
rm(list=ls())

##------------------------------------------------------------------
## Set the working directory
##------------------------------------------------------------------
setwd("/Users/alexstephens/Development/kaggle/higgs/data/raw")

##------------------------------------------------------------------
## Read the test and training datasets
##------------------------------------------------------------------

## read the raw data
test.raw    <- read.csv("test.csv", header=TRUE)
train.raw   <- read.csv("training.csv", header=TRUE)

## lowercase the headers
colnames(test.raw)  <- tolower(colnames(test.raw))
colnames(train.raw) <- tolower(colnames(train.raw))

##------------------------------------------------------------------
## Save results as a data.table
##------------------------------------------------------------------

## create data tables
test.dt     <- data.table(test.raw)
train.dt    <- data.table(train.raw)

## change to a new data directory
setwd("/Users/alexstephens/Development/kaggle/higgs/data/proc")

## save results to separate files given the size
save(test.dt,  file="01_HiggsRawTest.Rdata")
save(train.dt, file="01_HiggsRawTrain.Rdata")

