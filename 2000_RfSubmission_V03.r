##------------------------------------------------------------------
## From the kaggle website:
##------------------------------------------------------------------

##------------------------------------------------------------------
## Load libraries
##------------------------------------------------------------------
library(data.table)
library(caret)
library(gbm)
library(plyr)

##******************************************************************
## gbm_full_depth9_trees450_shrink0.05_v03
##******************************************************************

##------------------------------------------------------------------
## Note that CV was properly conducted on an identical hold-out
## sample, so the AMS computed using the hold-out data is
## comparable across all of the folds.
##------------------------------------------------------------------

## Clear the workspace
rm(list=ls())

## Set the working directory
setwd("/Users/alexstephens/Development/kaggle/higgs/data/proc")

## Source the utilities file
source("/Users/alexstephens/Development/kaggle/higgs/k_hig/00_Utilities.r")

## Load the data used in the training
trainFile   <- "04_HiggsTrainExRbcLcNumNa.Rdata"
testFile    <- "04_HiggsTestExRbcLcNumNa.Rdata"

load(trainFile)
if (trainFile == c("04_HiggsTrainExRbcLcNumNa.Rdata")) {
    trainDescr  <- as.data.frame(train.ex.rbc.lc.numna)
    trainClass  <- as.data.frame(train.eval)
}
load(testFile)
if (testFile == c("04_HiggsTestExRbcLcNumNa.Rdata")) {
    testDescr  <- as.data.frame(test.ex.rbc.lc.numna)
}

## Version 3 -- Get files
v03.dir     <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/rf_full_mtry7_V03"
v03.files   <- dir(v03.dir)[grep("[0-9].Rdata$", dir(v03.dir))]
v03.num     <- length(v03.files)
v03.seq     <- seq(0,1,0.001)
v03.ams     <- matrix(0,nrow=length(v03.seq),ncol=v03.num)
v03.list    <- list()
v03.mat     <- matrix(0,nrow=550000,ncol=length(v03.files)+1)

## loop over each file and compute AMS (order is based on alphabetic filename)
for (i in 1:v03.num) {
    
    ## load the file
    fit.filename  <- v03.files[i]
    load(paste0(v03.dir,"/",fit.filename))
    cat("Using file :", fit.filename,"\n")
    
    ## create addition filenames
    fit.id        <- gsub(".Rdata","",fit.filename)
    out.csv       <- paste0(fit.id,"_v03_Submission.csv")
    out.rdata     <- paste0(fit.id,"_v03_Submission.Rdata")
    
    ## in this case, the sample index represents the *testing* data
    fit.idx     <- samp.idx$Resample1
    holdDescr   <- trainDescr[ -fit.idx,]
    holdClass   <- trainClass[ -fit.idx,]
    
    ## score the hold-out sample & compute the AMS curve
    tmp.score    <- predict(tmp.fit, newdata=holdDescr[,-1], type="prob")[,c("s")]
    tmp.pred     <- predict(tmp.fit, newdata=holdDescr[,-1])
    tmp.breaks   <- quantile(tmp.score, probs=v03.seq)
    tmp.ams      <- sapply(tmp.breaks, calcAmsCutoff, tmp.score, holdClass$label, holdClass$weight)
    tmp.thresh   <- v03.seq[which(tmp.ams == max(tmp.ams))]
    
    ## aggregate the ams results
    v03.ams[,i]  <- tmp.ams
    
    ## plot the ams data based on the hold-out sample
    if (i == 1) {
        plot(v03.seq, v03.ams[,i], type="b", col=i, ylim=c(1,3.9))
    } else {
        points(v03.seq, v03.ams[,i], type="b", col=i)
    }
    abline(h=3.5)
}

## identify the best threshold from all the AMS scores
v03.ams_final <- cbind(v03.seq, apply(v03.ams, 1, mean))
v03.thresh    <- v03.ams_final[which(v03.ams_final[,2] == max(v03.ams_final[,2])), 1]


## loop again and extract fits
for (i in 1:v03.num) {
    
    ## load the file
    fit.filename  <- v03.files[i]
    load(paste0(v03.dir,"/",fit.filename))
    cat("Using file :", fit.filename,"\n")
    
    ## create addition filenames
    fit.id        <- gsub(".Rdata","",fit.filename)
    out.csv       <- paste0(fit.id,"_v03_Submission.csv")
    out.rdata     <- paste0(fit.id,"_v03_Submission.Rdata")

    ## score the test data
    test.score    <- predict(tmp.fit, newdata=testDescr[,-1], type="prob")[,c("s")]
    test.id       <- testDescr[,1]
    test.pred     <- ifelse(test.score >= v03.thresh, "s", "b")

    ## load the combined results into a list
    v03.list[[fit.id]]$test_score   <- data.frame(EventId=test.id, Prob=test.score, Class=test.pred)
    v03.list[[fit.id]]$thresh       <- tmp.thresh
    v03.list[[fit.id]]$ams          <- tmp.ams
    v03.list[[fit.id]]$breaks       <- tmp.breaks
}


## load a matrix with all of the test probabilities
#if (i == 1) {
#    v03.mat[,i]     <- test.id
#    v03.mat[,i+1]   <- test.score
#} else {
#    v03.mat[,i+1]   <- test.score
#}


## save the intermediate results
save(v03.list, v03.mat, v03.ams, file="rf_full_mtry7_V03_AllScored.Rdata")



##******************************************************************
## Intermediate Results
##******************************************************************

##------------------------------------------------------------------
## identify the maximum AMS scores
##------------------------------------------------------------------

# apply(v03.ams, 2, max)
# 3.564364 3.604836 3.547349 3.571381 3.541123 3.612231 3.623436 3.565879 3.627421 3.607781

# order(apply(v03.ams, 2, max))
# 5  3  1  8  4  2 10  6  7  9

# apply(v03.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, max)
# 3.565879 3.571381 3.604836 3.607781 3.612231 3.623436 3.627421

##------------------------------------------------------------------
## get an estimate of the AMS thresholds
##------------------------------------------------------------------

# apply(v03.ams, 2, function(x) {v03.seq[which(x == max(x))]})
# 0.815 0.860 0.845 0.855 0.850 0.820 0.840 0.830 0.860 0.860

# apply(v03.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, function(x) {v03.seq[which(x == max(x))]})
# 0.830 0.855 0.860 0.860 0.820 0.840 0.860

##------------------------------------------------------------------
## AMS threshold extrema
##------------------------------------------------------------------

# mean(apply(v03.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, function(x) {v03.seq[which(x == max(x))]}))
# [1] 0.8464286

## 0.860 and 0.820 are the two peaks



##******************************************************************
## First iteration -- Use an average score based on the top 7 models
## and use three different thresholds (0.820, 0.8464286, 0.86)
##******************************************************************

## the subset of fits to use
#subset <- c(8, 4, 2, 10, 6, 7, 9)
subset  <- 1:13

## a matrix to hold each scored probability
v03.submission01.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- v03.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
  
    ## get the single fit scores
    v03.submission01.probs[,i]  <- v03.list[[fit.id]]$test_score$Prob
}

## compute the mean probabilities
v03.submission01.mean       <- apply(v03.submission01.probs, 1, mean)
v03.submission01.med        <- apply(v03.submission01.probs, 1, median)
v03.submission01.id         <- v03.list[[1]]$test_score$EventId


##------------------------------------------------------------------
## Create three submission files based on the above threholds
##------------------------------------------------------------------


##------------------------------------------------------------------
## S011
##------------------------------------------------------------------

## reorder the data
v03.submission01.order      <- order(v03.submission01.mean)
v03.submission01.mean.order <- v03.submission01.mean[v03.submission01.order]
v03.submission01.id.order   <- v03.submission01.id[v03.submission01.order]
v03.submission01.rank.order <- 1:550000


## output filenames
fit.id        <- "rf_full_mtry7_V03_S011"
out.csv       <- paste0(fit.id,"_v03_Submission.csv")
out.rdata     <- paste0(fit.id,"_v03_Submission.Rdata")

## score -> class
v03.submission01.class <- ifelse(v03.submission01.mean.order >= v03.thresh, "s", "b")
v03.submission01.S011  <- data.frame(
                            EventId=v03.submission01.id.order,
                            RankOrder=v03.submission01.rank.order,
                            Class=v03.submission01.class)

## save the results
save(v03.submission01.S011, file=paste0(v03.dir,"/",out.rdata) )
write.csv(v03.submission01.S011, file=paste0(v03.dir,"/", out.csv), row.names=FALSE)


##------------------------------------------------------------------
## S012
##------------------------------------------------------------------

## reorder the data
v03.submission01.order      <- order(v03.submission01.med)
v03.submission01.med.order  <- v03.submission01.med[v03.submission01.order]
v03.submission01.id.order   <- v03.submission01.id[v03.submission01.order]
v03.submission01.rank.order <- 1:550000


## output filenames
fit.id        <- "rf_full_mtry7_V03_S012"
out.csv       <- paste0(fit.id,"_v03_Submission.csv")
out.rdata     <- paste0(fit.id,"_v03_Submission.Rdata")

## score -> class
v03.submission01.class <- ifelse(v03.submission01.med.order >= v03.thresh, "s", "b")
v03.submission01.S012  <- data.frame(
                            EventId=v03.submission01.id.order,
                            RankOrder=v03.submission01.rank.order,
                            Class=v03.submission01.class)

## save the results
save(v03.submission01.S012, file=paste0(v03.dir,"/",out.rdata) )
write.csv(v03.submission01.S012, file=paste0(v03.dir,"/", out.csv), row.names=FALSE)





