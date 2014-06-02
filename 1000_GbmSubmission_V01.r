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
## Main
##******************************************************************


##******************************************************************
## Review Version_01 fits
##******************************************************************
## Note that CV was performed on varying subsets, so I can't
## compare results across models
##******************************************************************

## Clear the workspace
rm(list=ls())

## Set the working directory
setwd("/Users/alexstephens/Development/kaggle/higgs/data/proc")

## Source the utilities file
source("/Users/alexstephens/Development/kaggle/higgs/k_hig/00_Utilities.r")

## Load the data
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

## Version 1 -- Get files
v01.dir     <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/gbm_full_depth9_trees450_shrink0.05_V01"
v01.files   <- dir(v01.dir)[grep("[0-9].Rdata$", dir(v01.dir))]
v01.num     <- length(v01.files)
v01.seq     <- seq(0,1,0.005)
v01.ams     <- matrix(0,nrow=length(v01.seq),ncol=v01.num)
v01.list    <- list()

## loop over each file and compute AMS (order is based on alphabetic filename)
for (i in 1:v01.num) {
    
    ## load the file
    fit.filename  <- v01.files[i]
    load(paste0(v01.dir,"/",fit.filename))
    cat("Using file :", fit.filename,"\n")
    
    ## create addition filenames
    fit.id        <- gsub(".Rdata","",fit.filename)
    out.csv       <- paste0(fit.id,"_V01_Submission.csv")
    out.rdata     <- paste0(fit.id,"_V01_Submission.Rdata")

    ## in this case, samp.idx epresents the training data ...
    ## and it changes for each file (b/c I screwed-up)
    fit.idx     <- samp.idx$Resample1
    holdDescr   <- trainDescr[-fit.idx,]
    holdClass   <- trainClass[-fit.idx,]
    
    ## score the hold-out sample & compute the AMS curve
    tmp.score    <- predict(tmp.fit, newdata=holdDescr[,-1], type="prob")[,c("s")]
    tmp.pred     <- predict(tmp.fit, newdata=holdDescr[,-1])
    tmp.breaks   <- quantile(tmp.score, probs=v01.seq)
    tmp.ams      <- sapply(tmp.breaks, calcAmsCutoff, tmp.score, holdClass$label, holdClass$weight)
    tmp.thresh   <- v01.seq[which(tmp.ams == max(tmp.ams))]
    
    ## aggregate the ams results
    v01.ams[,i]  <- tmp.ams
    
    ## plot the ams results
    if (i == 1) {
        plot(v01.seq, v01.ams[,i], type="b", col=i, ylim=c(1,3.9))
    } else {
        points(v01.seq, v01.ams[,i], type="b", col=i)
    }
    abline(h=3.5)
    
    ## score the test data
    test.score    <- predict(tmp.fit, newdata=testDescr[,-1], type="prob")[,c("s")]
    test.id       <- testDescr[,1]
    test.pred     <- ifelse(test.score >= tmp.thresh, "s", "b")

    v01.list[[fit.id]]$test_score  <- data.frame(EventId=test.id, Prob=test.score, Class=test.pred)
    v01.list[[fit.id]]$thresh <- tmp.thresh
    v01.list[[fit.id]]$ams    <- tmp.ams
    v01.list[[fit.id]]$breaks <- tmp.breaks
}

save(v01.list, file="gbm_full_depth9_trees450_shrink0.05_V01_AllScored.Rdata")



##------------------------------------------------------------------
## V01 Results
##------------------------------------------------------------------

#> apply(v01.ams, 2, max)
#[1] 3.564364 (3.764494*) 4.246568* 3.480278 (3.590924*) 3.449687 (3.575499*) (3.786709*) 3.714341* (3.750571*)

## Early candidates = c(2, 3, 8, 9, 10), but (3, 8, 9) have late spikes c(2, 10)
#subset <- c(2, 10)
#subset <- c(3, 7, 8)
subset <- c(2,7,10)

for (i in subset) {
    if (i == subset[1]) {
        plot(v01.seq, v01.ams[,i], type="b", col="steelblue", pch=i, ylim=c(1,3.8))
    } else {
        points(v01.seq, v01.ams[,i], type="b", col="steelblue", pch=i)
    }
    abline(h=3.5)
}

## loop over the good files and create an output data set for each, and the average
for (i in subset) {
    
    ## load the file
    fit.filename  <- v01.files[i]
    cat("Using file :", fit.filename,"\n")
    
    fit.id        <- gsub(".Rdata","",fit.filename)
    out.csv       <- paste0(fit.id,"_V01_Submission.csv")
    out.rdata     <- paste0(fit.id,"_V01_Submission.Rdata")
    
    ## get the single fit scores
    test.score    <- v01.list[[fit.id]]$test_score
    
    ## reorder the data
    sub.order   <- order(test.score$Prob)
    
    sub.id      <- test.score$EventId[sub.order]
    sub.pred    <- test.score$Class[sub.order]
    sub.rank    <- 1:550000
    
    ## create a datafame with the submission data AND probs
    tmp.sub <- data.frame(EventId=sub.id, RankOrder=sub.rank, Class=sub.pred)
    
    ## save the results
    save(tmp.sub, file=paste0(v01.dir,"/",out.rdata))
    write.csv(tmp.sub[,c("EventId","RankOrder","Class")], file=paste0(v01.dir,"/", out.csv), row.names=FALSE)

}

















##----


##******************************************************************
## Review Version_03 fits
##******************************************************************
## Note that CV was performed on varying subsets, so I can't
## compare results across models
##------------------------------------------------------------------

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

## Load the data
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

## Version 3
v03.dir     <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/gbm_full_depth9_trees450_shrink0.05_V03"
v03.files   <- dir(v03.dir)[grep(".Rdata$", dir(v03.dir))]
v03.num     <- length(v03.files)
v03.seq     <- seq(0,1,0.005)
v03.ams     <- matrix(0,nrow=length(v03.seq),ncol=v03.num)

for (i in 1:v03.num) {
    
    tmp.file    <- paste0(v03.dir,"/",v03.files[i])
    load(tmp.file)
    
    ## in this case, the sample index represents the *testing* data
    fit.idx     <- samp.idx$Resample1
    holdDescr   <- trainDescr[ -fit.idx,]
    holdClass   <- trainClass[ -fit.idx,]
    
    ## score the hold-out sample & compute the AMS curve
    tmp.score    <- predict(tmp.fit, newdata=holdDescr[,-1], type="prob")[,c("s")]
    tmp.pred     <- predict(tmp.fit, newdata=holdDescr[,-1])
    tmp.breaks   <- quantile(tmp.score, probs=v03.seq)
    tmp.ams      <- sapply(tmp.breaks, calcAmsCutoff, tmp.score, holdClass$label, holdClass$weight)
    v03.ams[,i]  <- tmp.ams
    
    if (i == 1) {
        plot(v03.seq, v03.ams[,i], type="b", col=i, ylim=c(1,3.8))
    } else {
        points(v03.seq, v03.ams[,i], type="b", col=i)
    }
    abline(h=3.5)
}



## V03 Results

#> apply(v03.ams, 2, max)
#[1] 3.564364 3.604836* 3.547349 3.571381 3.541123 3.612231* 3.623436* 3.565879 3.627421* 3.607781*

## Early candidates = c(2, 6, 7, 9, 10)
subset <- c(2, 6, 7, 9, 10)
for (i in subset) {
    if (i == subset[1]) {
        plot(v03.seq, v03.ams[,i], type="b", col=i, pch=i, ylim=c(1,3.8))
    } else {
        points(v03.seq, v03.ams[,i], type="b", col=i, pch=i)
    }
    abline(h=3.5)
}

## plot the average of the two best (not ideal)
avg.ams <- apply(v03.ams[,subset],1,mean)
plot(v03.seq, avg.ams, ylim=c(1,3.8))

## identify the threshold ... but a combined threhsold doesn't make
## sense in this case because we have different subsets against
## which we've scored the data ... nevertheless 0.845 is the result
v03.threshold   <- v03.seq[which(avg.ams == max(avg.ams))]

cbind(v03.seq, avg.ams)
0.860 3.5849804


##------------------------------------------------------------------
## Review Version_04 fits
##------------------------------------------------------------------
## Note that CV was performed on varying subsets, so I can't
## compare results across models
##------------------------------------------------------------------

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

## Load the data
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

## Version 4
v04.dir <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/gbm_full_depth9_trees450_shrink0.05_V04"
v04.files   <- dir(v04.dir)[grep(".Rdata$", dir(v04.dir))]
v04.num     <- length(v04.files)
v04.seq     <- seq(0,1,0.005)
v04.ams     <- matrix(0,nrow=length(v04.seq),ncol=v04.num)

for (i in 1:v04.num) {
    
    tmp.file    <- paste0(v04.dir,"/",v04.files[i])
    load(tmp.file)
    
    ## in this case, the sample index represents the testing data ... since I scored
    ## folds, the hold-out sample is not comparable across iterations
    fit.idx     <- tmp.idx
    holdDescr   <- trainDescr[ fit.idx,]
    holdClass   <- trainClass[ fit.idx,]
    
    ## score the hold-out sample & compute the AMS curve
    tmp.score    <- predict(tmp.fit, newdata=holdDescr[,-1], type="prob")[,c("s")]
    tmp.pred     <- predict(tmp.fit, newdata=holdDescr[,-1])
    tmp.breaks   <- quantile(tmp.score, probs=v04.seq)
    tmp.ams      <- sapply(tmp.breaks, calcAmsCutoff, tmp.score, holdClass$label, holdClass$weight)
    v04.ams[,i]  <- tmp.ams
    
    if (i == 1) {
        plot(v04.seq, v04.ams[,i], type="b", col=i, ylim=c(1,3.8))
    } else {
        points(v04.seq, v04.ams[,i], type="b", col=i)
    }
    abline(h=3.5)
}

#> apply(v04.ams, 2, max)
#[1] 3.601138* 3.543628 3.524940 3.488023 3.777732* 3.747995* 3.568224 3.512311

subset <- c(1,5,6)
for (i in subset) {
    if (i == subset[1]) {
        plot(v04.seq, v04.ams[,i], type="b", col=i, pch=i, ylim=c(1,3.8))
    } else {
        points(v04.seq, v04.ams[,i], type="b", col=i, pch=i)
    }
    abline(h=3.5)
}

## plot the average of the two best (not ideal)
avg.ams <- apply(v04.ams[,subset],1,mean)
plot(v04.seq, avg.ams, ylim=c(1,3.8)); abline(h=3.5)

## identify the threshold ... but a combined threhsold doesn't make
## sense in this case because we have different subsets against
## which we've scored the data ... nevertheless 0.845 is the result
v04.threshold   <- v04.seq[which(avg.ams == max(avg.ams))]
abline(v=v04.threshold)

cbind(v04.seq, avg.ams)




