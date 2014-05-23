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

##------------------------------------------------------------------
## Step 0: Combine training and test data
##------------------------------------------------------------------

## an id column to each subset of data
train.pp[,id:=c("tr")]
test.pp[,id:=c("te")]

## combine the datasets
comb.pp <- rbind(train.pp, test.pp)
setkey(comb.pp, eventid)

## identify columns you plot
noplot.idx <- c(
                    grep("id", colnames(comb.pp)),
                    grep(".fl$", colnames(comb.pp)),
                    grep(".[0-9]$", colnames(comb.pp))
                )


##------------------------------------------------------------------
## Step 1:  Plot histograms of the data
##------------------------------------------------------------------

## loop over the variables and plot histograms of the data
plotvars    <- colnames(comb.pp)[-noplot.idx]
plotdir     <- "/Users/alexstephens/Development/kaggle/higgs/figs/"

    for (i in 1:length(plotvars)) {
        tmp.plotvar     <- plotvars[i]
        tmp.plotfile    <- paste(plotdir,paste(tmp.plotvar,".pdf",sep=""),sep="")
        pdf(file=tmp.plotfile)
            hist(unlist(comb.pp[,tmp.plotvar,with=FALSE]), breaks=100, col="steelblue", xlab=tmp.plotvar, main="")
        dev.off()
    }


##------------------------------------------------------------------
## Step 2:  Test box-cox transforms on visually-inspected variables
##------------------------------------------------------------------

## variables to test
boxcox.cols  <- c(  "der_deltaeta_jet_jet", "der_mass_jet_jet",
                    "der_mass_mmc", "der_mass_transverse_met_lep",
                    "der_mass_vis", "der_pt_h", "der_pt_ratio_lep_tau",
                    "der_pt_tot", "der_sum_pt", "pri_jet_all_pt",
                    "pri_jet_leading_pt", "pri_jet_subleading_pt","pri_lep_pt",
                    "pri_met_sumet", "pri_met", "pri_tau_pt")

## do the transform
boxcox.pp  <- preProcess(comb.pp[,boxcox.cols,with=FALSE], method = "BoxCox")

## perform the transformation
boxcox.df           <- predict(boxcox.pp, comb.pp[,boxcox.cols,with=FALSE])
colnames(boxcox.df) <- paste(colnames(boxcox.df),".bc",sep="")

## drop the originals and append the results
comb.bc <- cbind(comb.pp[,-which(colnames(comb.pp) %in% boxcox.cols),with=FALSE], boxcox.df)

## loop over the variables and plot histograms of the box-cox transformed data
plotvars    <- colnames(comb.bc)[grep(".bc$",colnames(comb.bc))]
plotdir     <- "/Users/alexstephens/Development/kaggle/higgs/figs/"

    for (i in 1:length(plotvars)) {
        tmp.plotvar     <- plotvars[i]
        tmp.plotfile    <- paste(plotdir,paste(tmp.plotvar,".pdf",sep=""),sep="")
        pdf(file=tmp.plotfile)
        hist(unlist(comb.bc[,tmp.plotvar,with=FALSE]), breaks=100, col="orange", xlab=tmp.plotvar, main="")
        dev.off()
    }



##------------------------------------------------------------------
## Step 3:  Scale the data
##------------------------------------------------------------------

## re-identify the columns to scale in the .bc matrix
noscale.idx <- c(
                    grep("id", colnames(comb.bc)),
                    grep(".fl$", colnames(comb.bc)),
                    grep(".[0-9]$", colnames(comb.bc))
                )

## join the scaled data to the original after dropping orig
comb.sc <- cbind(
                    comb.bc[,noscale.idx,with=FALSE],
                    data.table(scale(comb.bc[,-noscale.idx,with=FALSE]))
                )
setkey(comb.sc,"id")

##------------------------------------------------------------------
## Step 5:  Mode replace the missings [???]
##------------------------------------------------------------------

## identify columns with missings
#na.cols <- unlist(apply(as.matrix(comb.bc[reg.idx[[1]],]), 2, function(x){sum(is.na(x))}))

#for (i in 1:length(na.cols)) {
#
#}

##------------------------------------------------------------------
## Step 4:  Break back into train and test datasets
##------------------------------------------------------------------

## use the id key to split
train.sc    <- comb.sc["tr"]
test.sc     <- comb.sc["te"]

## drop the ids
train.sc[,id:=NULL]
test.sc[,id:=NULL]


##------------------------------------------------------------------
## Save the results
##------------------------------------------------------------------
save(train.eval, train.sc, file="03_HiggsScaledTrain.Rdata")
save(test.sc, file="03_HiggsScaledTest.Rdata")



