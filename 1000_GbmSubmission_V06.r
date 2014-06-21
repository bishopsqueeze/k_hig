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
## gbm_full_depth9_trees450_shrink0.05_v06
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
trainFile   <- "04_HiggsTrainAll.Rdata"
testFile    <- "04_HiggsTestAll.Rdata"

load(trainFile)
if (trainFile == c("04_HiggsTrainAll.Rdata")) {
    trainDescr  <- as.data.frame(train.all)
    trainClass  <- as.data.frame(train.eval)
}
load(testFile)
if (testFile == c("04_HiggsTestAll.Rdata")) {
    testDescr  <- as.data.frame(test.all)
}


## Version 3 -- Get files
v06.dir     <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/gbm_full_depth9_trees3000_shrink0.025_V06"
v06.files   <- dir(v06.dir)[grep("[0-9].Rdata$", dir(v06.dir))]
v06.num     <- length(v06.files)
v06.seq     <- seq(0,1,0.01)
v06.ams     <- matrix(0,nrow=length(v06.seq),ncol=v06.num)
v06.list    <- list()
v06.mat     <- matrix(0,nrow=550000,ncol=length(v06.files)+1)

## loop over each file and compute AMS (order is based on alphabetic filename)
for (i in 1:v06.num) {
    
    ## load the file
    fit.filename  <- v06.files[i]
    load(paste0(v06.dir,"/",fit.filename))
    cat("Using file :", fit.filename,"\n")
    
    ## create addition filenames
    fit.id        <- gsub(".Rdata","",fit.filename)
    out.csv       <- paste0(fit.id,"_v06_Submission.csv")
    out.rdata     <- paste0(fit.id,"_v06_Submission.Rdata")
    
    ## in this case, the sample index represents the *testing* data
    fit.idx     <- samp.idx$Resample1
    holdDescr   <- trainDescr[ -fit.idx,]
    holdClass   <- trainClass[ -fit.idx,]
    
    ## score the hold-out sample & compute the AMS curve
    tmp.score    <- predict(tmp.fit, newdata=trainDescr[fit.idx,-1], type="prob")[,c("s")]
    tmp.pred     <- predict(tmp.fit, newdata=trainDescr[fit.idx,-1])
    tmp.breaks   <- quantile(tmp.score, probs=v06.seq)
    tmp.ams      <- sapply(tmp.breaks, calcAmsCutoff, tmp.score, trainClass$label[fit.idx], trainClass$weight[fit.idx])
    tmp.thresh   <- v06.seq[which(tmp.ams == max(tmp.ams))]
    
    ## aggregate the ams results
    v06.ams[,i]  <- tmp.ams
    
    ## plot the ams data based on the hold-out sample
    if (i == 1) {
        plot(v06.seq, v06.ams[,i], type="b", col=i, ylim=c(1,3.9))
    } else {
        points(v06.seq, v06.ams[,i], type="b", col=i)
    }
    abline(h=3.5)
    
    ## score the test data
    test.score    <- predict(tmp.fit, newdata=testDescr[,-1], type="prob")[,c("s")]
    test.id       <- testDescr[,1]
    test.pred     <- ifelse(test.score >= tmp.thresh, "s", "b")

    ## load the combined results into a list
    v06.list[[fit.id]]$test_score   <- data.frame(EventId=test.id, Prob=test.score, Class=test.pred)
    v06.list[[fit.id]]$thresh       <- tmp.thresh
    v06.list[[fit.id]]$ams          <- tmp.ams
    v06.list[[fit.id]]$breaks       <- tmp.breaks
    
    ## load a matrix with all of the test probabilities
    if (i == 1) {
        v06.mat[,i]     <- test.id
        v06.mat[,i+1]   <- test.score
    } else {
        v06.mat[,i+1]   <- test.score
    }
}


## save the intermediate results
save(v06.list, v06.mat, v06.ams, file="gbm_full_depth9_trees450_shrink0.01_v06_AllScored.Rdata")



##******************************************************************
## Intermediate Results
##******************************************************************

##------------------------------------------------------------------
## identify the maximum AMS scores
##------------------------------------------------------------------

# apply(v06.ams, 2, max)
# 3.564364 3.604836 3.547349 3.571381 3.541123 3.612231 3.623436 3.565879 3.627421 3.607781

# order(apply(v06.ams, 2, max))
# 5  3  1  8  4  2 10  6  7  9

# apply(v06.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, max)
# 3.565879 3.571381 3.604836 3.607781 3.612231 3.623436 3.627421

##------------------------------------------------------------------
## get an estimate of the AMS thresholds
##------------------------------------------------------------------

# apply(v06.ams, 2, function(x) {v06.seq[which(x == max(x))]})
# 0.815 0.860 0.845 0.855 0.850 0.820 0.840 0.830 0.860 0.860

# apply(v06.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, function(x) {v06.seq[which(x == max(x))]})
# 0.830 0.855 0.860 0.860 0.820 0.840 0.860

##------------------------------------------------------------------
## AMS threshold extrema
##------------------------------------------------------------------

# mean(apply(v06.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, function(x) {v06.seq[which(x == max(x))]}))
# [1] 0.8464286

## 0.860 and 0.820 are the two peaks



##******************************************************************
## First iteration -- Use an average score based on the top 7 models
## and use three different thresholds (0.820, 0.8464286, 0.86)
##******************************************************************

## the subset of fits to use
subset <- c(1:7)

## a matrix to hold each scored probability
v06.submission01.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- v06.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
  
    ## get the single fit scores
    v06.submission01.probs[,i]  <- v06.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
v06.submission01.mean   <- apply(v06.submission01.probs, 1, mean)
v06.submission01.id     <- v06.list[[1]]$test_score$EventId

## reorder the data
v06.submission01.order      <- order(v06.submission01.mean)
v06.submission01.mean.order <- v06.submission01.mean[v06.submission01.order]
v06.submission01.id.order   <- v06.submission01.id[v06.submission01.order]
v06.submission01.rank.order <- 1:550000

##------------------------------------------------------------------
## Create three submission files based on the above threholds
##------------------------------------------------------------------

##------------------------------------------------------------------
## S011
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v06_S011"
out.csv       <- paste0(fit.id,"_v06_Submission.csv")
out.rdata     <- paste0(fit.id,"_v06_Submission.Rdata")

## score -> class
thresh.S011            <- 0.850
v06.submission01.class <- ifelse(v06.submission01.mean.order >= thresh.S011, "s", "b")
v06.submission01.S011  <- data.frame(
                            EventId=v06.submission01.id.order,
                            RankOrder=v06.submission01.rank.order,
                            Class=v06.submission01.class)

## save the results
save(v06.submission01.S011, file=paste0(v06.dir,"/",out.rdata) )
write.csv(v06.submission01.S011, file=paste0(v06.dir,"/", out.csv), row.names=FALSE)


##------------------------------------------------------------------
## S012
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v06_S012"
out.csv       <- paste0(fit.id,"_v06_Submission.csv")
out.rdata     <- paste0(fit.id,"_v06_Submission.Rdata")

## score -> class
thresh.S012            <- 0.86
v06.submission01.class <- ifelse(v06.submission01.mean.order >= thresh.S012, "s", "b")
v06.submission01.S012  <- data.frame(
EventId=v06.submission01.id.order,
RankOrder=v06.submission01.rank.order,
Class=v06.submission01.class)

## save the results
save(v06.submission01.S012, file=paste0(v06.dir,"/",out.rdata) )
write.csv(v06.submission01.S012, file=paste0(v06.dir,"/", out.csv), row.names=FALSE)


##------------------------------------------------------------------
## the subset of fits to use
##------------------------------------------------------------------
subset <- c(3)

## a matrix to hold each scored probability
v06.submission02.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- v06.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
    
    ## get the single fit scores
    v06.submission02.probs[,i]  <- v06.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
v06.submission02.mean   <- apply(v06.submission02.probs, 1, mean)
v06.submission02.id     <- v06.list[[1]]$test_score$EventId

## reorder the data
v06.submission02.order      <- order(v06.submission02.mean)
v06.submission02.mean.order <- v06.submission02.mean[v06.submission02.order]
v06.submission02.id.order   <- v06.submission02.id[v06.submission02.order]
v06.submission02.rank.order <- 1:550000

##------------------------------------------------------------------
## S013
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v06_S013"
out.csv       <- paste0(fit.id,"_v06_Submission.csv")
out.rdata     <- paste0(fit.id,"_v06_Submission.Rdata")

## score -> class
thresh.S013            <- 0.850
v06.submission02.class <- ifelse(v06.submission02.mean.order >= thresh.S013, "s", "b")
v06.submission02.S013  <- data.frame(
EventId=v06.submission02.id.order,
RankOrder=v06.submission02.rank.order,
Class=v06.submission02.class)

## save the results
save(v06.submission02.S013, file=paste0(v06.dir,"/",out.rdata) )
write.csv(v06.submission02.S013, file=paste0(v06.dir,"/", out.csv), row.names=FALSE)




##------------------------------------------------------------------
## the subset of fits to use
##------------------------------------------------------------------
subset <- c(6,2,1,5,3)

## a matrix to hold each scored probability
v06.submission03.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- v06.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
    
    ## get the single fit scores
    v06.submission03.probs[,i]  <- v06.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
v06.submission03.mean   <- apply(v06.submission03.probs, 1, mean)
v06.submission03.id     <- v06.list[[1]]$test_score$EventId

## reorder the data
v06.submission03.order      <- order(v06.submission03.mean)
v06.submission03.mean.order <- v06.submission03.mean[v06.submission03.order]
v06.submission03.id.order   <- v06.submission03.id[v06.submission03.order]
v06.submission03.rank.order <- 1:550000


##------------------------------------------------------------------
## S014
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v06_S014"
out.csv       <- paste0(fit.id,"_v06_Submission.csv")
out.rdata     <- paste0(fit.id,"_v06_Submission.Rdata")

## score -> class
thresh.S014            <- 0.85
v06.submission03.class <- ifelse(v06.submission03.mean.order >= thresh.S014, "s", "b")
v06.submission03.S014  <- data.frame(
EventId=v06.submission03.id.order,
RankOrder=v06.submission03.rank.order,
Class=v06.submission03.class)

## save the results
save(v06.submission03.S014, file=paste0(v06.dir,"/",out.rdata) )
write.csv(v06.submission03.S014, file=paste0(v06.dir,"/", out.csv), row.names=FALSE)



##------------------------------------------------------------------
## S015
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v06_S015"
out.csv       <- paste0(fit.id,"_v06_Submission.csv")
out.rdata     <- paste0(fit.id,"_v06_Submission.Rdata")

## score -> class
thresh.S015            <- 0.855
v06.submission01.class <- ifelse(v06.submission01.mean.order >= thresh.S015, "s", "b")
v06.submission01.S015  <- data.frame(
EventId=v06.submission01.id.order,
RankOrder=v06.submission01.rank.order,
Class=v06.submission01.class)

## save the results
save(v06.submission01.S015, file=paste0(v06.dir,"/",out.rdata) )
write.csv(v06.submission01.S015, file=paste0(v06.dir,"/", out.csv), row.names=FALSE)




##------------------------------------------------------------------
## S016
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v06_S016"
out.csv       <- paste0(fit.id,"_v06_Submission.csv")
out.rdata     <- paste0(fit.id,"_v06_Submission.Rdata")

## score -> class
thresh.S016            <- 0.860
v06.submission01.class <- ifelse(v06.submission01.mean.order >= thresh.S016, "s", "b")
v06.submission01.S016  <- data.frame(
EventId=v06.submission01.id.order,
RankOrder=v06.submission01.rank.order,
Class=v06.submission01.class)

## save the results
save(v06.submission01.S016, file=paste0(v06.dir,"/",out.rdata) )
write.csv(v06.submission01.S016, file=paste0(v06.dir,"/", out.csv), row.names=FALSE)







