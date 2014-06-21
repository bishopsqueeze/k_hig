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
## gbm_full_depth9_trees450_shrink0.05_v05
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
v05.dir     <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/gbm_full_depth9_trees450_shrink0.01_V05"
v05.files   <- dir(v05.dir)[grep("[0-9].Rdata$", dir(v05.dir))]
v05.num     <- length(v05.files)
v05.seq     <- seq(0,1,0.01)
v05.ams     <- matrix(0,nrow=length(v05.seq),ncol=v05.num)
v05.list    <- list()
v05.mat     <- matrix(0,nrow=550000,ncol=length(v05.files)+1)

## loop over each file and compute AMS (order is based on alphabetic filename)
for (i in 1:v05.num) {
    
    ## load the file
    fit.filename  <- v05.files[i]
    load(paste0(v05.dir,"/",fit.filename))
    cat("Using file :", fit.filename,"\n")
    
    ## create addition filenames
    fit.id        <- gsub(".Rdata","",fit.filename)
    out.csv       <- paste0(fit.id,"_v05_Submission.csv")
    out.rdata     <- paste0(fit.id,"_v05_Submission.Rdata")
    
    ## in this case, the sample index represents the *testing* data
    fit.idx     <- samp.idx$Resample1
    holdDescr   <- trainDescr[ -fit.idx,]
    holdClass   <- trainClass[ -fit.idx,]
    
    ## score the hold-out sample & compute the AMS curve
    #tmp.score    <- predict(tmp.fit, newdata=holdDescr[,-1], type="prob")[,c("s")]
    #tmp.pred     <- predict(tmp.fit, newdata=holdDescr[,-1])
    tmp.score    <- predict(tmp.fit, newdata=trainDescr[fit.idx,-1], type="prob")[,c("s")]
    tmp.pred     <- predict(tmp.fit, newdata=trainDescr[fit.idx,-1])
    tmp.breaks   <- quantile(tmp.score, probs=v05.seq)
    #tmp.ams      <- sapply(tmp.breaks, calcAmsCutoff, tmp.score, holdClass$label, holdClass$weight)
    tmp.ams      <- sapply(tmp.breaks, calcAmsCutoff, tmp.score, trainClass$label[fit.idx], trainClass$weight[fit.idx])
    tmp.thresh   <- v05.seq[which(tmp.ams == max(tmp.ams))]
    
    ## aggregate the ams results
    v05.ams[,i]  <- tmp.ams
    
    ## plot the ams data based on the hold-out sample
    if (i == 1) {
        plot(v05.seq, v05.ams[,i], type="b", col=i, ylim=c(1,3.9))
    } else {
        points(v05.seq, v05.ams[,i], type="b", col=i)
    }
    abline(h=3.5)
    
    ## score the test data
    test.score    <- predict(tmp.fit, newdata=testDescr[,-1], type="prob")[,c("s")]
    test.id       <- testDescr[,1]
    test.pred     <- ifelse(test.score >= tmp.thresh, "s", "b")

    ## load the combined results into a list
    v05.list[[fit.id]]$test_score   <- data.frame(EventId=test.id, Prob=test.score, Class=test.pred)
    v05.list[[fit.id]]$thresh       <- tmp.thresh
    v05.list[[fit.id]]$ams          <- tmp.ams
    v05.list[[fit.id]]$breaks       <- tmp.breaks
    
    ## load a matrix with all of the test probabilities
    if (i == 1) {
        v05.mat[,i]     <- test.id
        v05.mat[,i+1]   <- test.score
    } else {
        v05.mat[,i+1]   <- test.score
    }
}


## save the intermediate results
save(v05.list, v05.mat, v05.ams, file="gbm_full_depth9_trees450_shrink0.01_v05_AllScored.Rdata")



##******************************************************************
## Intermediate Results
##******************************************************************

##------------------------------------------------------------------
## identify the maximum AMS scores
##------------------------------------------------------------------

# apply(v05.ams, 2, max)
# 3.564364 3.604836 3.547349 3.571381 3.541123 3.612231 3.623436 3.565879 3.627421 3.607781

# order(apply(v05.ams, 2, max))
# 5  3  1  8  4  2 10  6  7  9

# apply(v05.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, max)
# 3.565879 3.571381 3.604836 3.607781 3.612231 3.623436 3.627421

##------------------------------------------------------------------
## get an estimate of the AMS thresholds
##------------------------------------------------------------------

# apply(v05.ams, 2, function(x) {v05.seq[which(x == max(x))]})
# 0.815 0.860 0.845 0.855 0.850 0.820 0.840 0.830 0.860 0.860

# apply(v05.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, function(x) {v05.seq[which(x == max(x))]})
# 0.830 0.855 0.860 0.860 0.820 0.840 0.860

##------------------------------------------------------------------
## AMS threshold extrema
##------------------------------------------------------------------

# mean(apply(v05.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, function(x) {v05.seq[which(x == max(x))]}))
# [1] 0.8464286

## 0.860 and 0.820 are the two peaks



##******************************************************************
## First iteration -- Use an average score based on the top 7 models
## and use three different thresholds (0.820, 0.8464286, 0.86)
##******************************************************************

## the subset of fits to use
subset <- c(8, 4, 2, 10, 6, 7, 9)

## a matrix to hold each scored probability
v05.submission01.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- v05.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
  
    ## get the single fit scores
    v05.submission01.probs[,i]  <- v05.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
v05.submission01.mean   <- apply(v05.submission01.probs, 1, mean)
v05.submission01.id     <- v05.list[[1]]$test_score$EventId

## reorder the data
v05.submission01.order      <- order(v05.submission01.mean)
v05.submission01.mean.order <- v05.submission01.mean[v05.submission01.order]
v05.submission01.id.order   <- v05.submission01.id[v05.submission01.order]
v05.submission01.rank.order <- 1:550000

##------------------------------------------------------------------
## Create three submission files based on the above threholds
##------------------------------------------------------------------

##------------------------------------------------------------------
## S006
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v05_S006"
out.csv       <- paste0(fit.id,"_v05_Submission.csv")
out.rdata     <- paste0(fit.id,"_v05_Submission.Rdata")

## score -> class
thresh.S006            <- 0.820
v05.submission01.class <- ifelse(v05.submission01.mean.order >= thresh.S006, "s", "b")
v05.submission01.S006  <- data.frame(
                            EventId=v05.submission01.id.order,
                            RankOrder=v05.submission01.rank.order,
                            Class=v05.submission01.class)

## save the results
save(v05.submission01.S006, file=paste0(v05.dir,"/",out.rdata) )
write.csv(v05.submission01.S006, file=paste0(v05.dir,"/", out.csv), row.names=FALSE)


##------------------------------------------------------------------
## S007
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v05_S007"
out.csv       <- paste0(fit.id,"_v05_Submission.csv")
out.rdata     <- paste0(fit.id,"_v05_Submission.Rdata")

## score -> class
thresh.S007            <- 0.8464286
v05.submission01.class <- ifelse(v05.submission01.mean.order >= thresh.S007, "s", "b")
v05.submission01.S007  <- data.frame(
EventId=v05.submission01.id.order,
RankOrder=v05.submission01.rank.order,
Class=v05.submission01.class)

## save the results
save(v05.submission01.S007, file=paste0(v05.dir,"/",out.rdata) )
write.csv(v05.submission01.S007, file=paste0(v05.dir,"/", out.csv), row.names=FALSE)


##------------------------------------------------------------------
## S008
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v05_S008"
out.csv       <- paste0(fit.id,"_v05_Submission.csv")
out.rdata     <- paste0(fit.id,"_v05_Submission.Rdata")

## score -> class
thresh.S008            <- 0.860
v05.submission01.class <- ifelse(v05.submission01.mean.order >= thresh.S008, "s", "b")
v05.submission01.S008  <- data.frame(
EventId=v05.submission01.id.order,
RankOrder=v05.submission01.rank.order,
Class=v05.submission01.class)

## save the results
save(v05.submission01.S008, file=paste0(v05.dir,"/",out.rdata) )
write.csv(v05.submission01.S008, file=paste0(v05.dir,"/", out.csv), row.names=FALSE)


##------------------------------------------------------------------
## S009
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v05_S009"
out.csv       <- paste0(fit.id,"_v05_Submission.csv")
out.rdata     <- paste0(fit.id,"_v05_Submission.Rdata")

## score -> class
thresh.S009            <- 0.830
v05.submission01.class <- ifelse(v05.submission01.mean.order >= thresh.S009, "s", "b")
v05.submission01.S009  <- data.frame(
EventId=v05.submission01.id.order,
RankOrder=v05.submission01.rank.order,
Class=v05.submission01.class)

## save the results
save(v05.submission01.S009, file=paste0(v05.dir,"/",out.rdata) )
write.csv(v05.submission01.S009, file=paste0(v05.dir,"/", out.csv), row.names=FALSE)



##------------------------------------------------------------------
## S010
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v05_S010"
out.csv       <- paste0(fit.id,"_v05_Submission.csv")
out.rdata     <- paste0(fit.id,"_v05_Submission.Rdata")

## score -> class
thresh.S010            <- 0.810
v05.submission01.class <- ifelse(v05.submission01.mean.order >= thresh.S010, "s", "b")
v05.submission01.S010  <- data.frame(
EventId=v05.submission01.id.order,
RankOrder=v05.submission01.rank.order,
Class=v05.submission01.class)

## save the results
save(v05.submission01.S010, file=paste0(v05.dir,"/",out.rdata) )
write.csv(v05.submission01.S010, file=paste0(v05.dir,"/", out.csv), row.names=FALSE)







