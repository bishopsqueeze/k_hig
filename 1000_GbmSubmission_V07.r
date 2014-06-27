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
## gbm_full_depth9_trees450_shrink0.05_v07
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
trainFile   <- "02_HiggsPreProcTrain.Rdata"
testFile    <- "02_HiggsPreProcTest.Rdata"

load(trainFile)
if (trainFile == c("02_HiggsPreProcTrain.Rdata")) {
    trainDescr  <- as.data.frame(train.pp)
    trainClass  <- as.data.frame(train.eval)
}
load(testFile)
if (testFile == c("02_HiggsPreProcTest.Rdata")) {
    testDescr  <- as.data.frame(test.pp)
}

## Define constants used below
s_val   <- 1
b_val   <- 0
b_tau   <- 10
vsize   <- .10
na_val  <- -999
n_trees <- 500

## Create a numeric label for the class
trainClass$label.num <- as.numeric(ifelse(trainClass$label=="s", s_val, b_val))

## Compute the sum of signal and background weights
s_total <- sum(trainClass$weight[trainClass$label.num == s_val])
b_total <- sum(trainClass$weight[trainClass$label.num == b_val])

## compute max_AMS
max_AMS <- AMS(trainClass$label.num, trainClass$label.num, trainClass$weight)


## Version 3 -- Get files
v07.dir     <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/gbm_full_depth9_trees500_shrink0.05_V07"
v07.files   <- dir(v07.dir)[grep("[0-9].Rdata$", dir(v07.dir))]
v07.num     <- length(v07.files)
v07.seq     <- seq(0,1,0.01)
v07.ams     <- matrix(0,nrow=length(v07.seq),ncol=v07.num)
v07.list    <- list()
v07.mat     <- matrix(0,nrow=550000,ncol=length(v07.files)+1)

## loop over each file and compute AMS (order is based on alphabetic filename)
for (i in 1:v07.num) {
    
    ## load the file
    fit.filename  <- v07.files[i]
    load(paste0(v07.dir,"/",fit.filename))
    cat("Using file :", fit.filename,"\n")
    
    ## create addition filenames
    fit.id        <- gsub(".Rdata","",fit.filename)
    out.csv       <- paste0(fit.id,"_v07_Submission.csv")
    out.rdata     <- paste0(fit.id,"_v07_Submission.Rdata")
    
    ## in this case, the sample index represents the *testing* data
    fit.idx     <- samp.idx$Resample1
    holdDescr   <- trainDescr[ -fit.idx,]
    holdClass   <- trainClass[ -fit.idx,]
    
    ## renormalize the weights for the holdout sample
    s_before <- sum(holdClass$weight[holdClass$label.num == s_val])
    b_before <- sum(holdClass$weight[holdClass$label.num == b_val])
    
    holdClass$weight <- normalize(holdClass$weight, holdClass$label.num, s_total, b_total)
    
    ## renormalize the weights for the holdout sample
    s_after <- sum(holdClass$weight[holdClass$label.num == s_val])
    b_after <- sum(holdClass$weight[holdClass$label.num == b_val])
   
    ## confirm normalization
    AMS(holdClass$label.num, holdClass$label.num, holdClass$weight)

    ## score the hold-out sample & compute the AMS curve
    tmp.score           <- predict(tmp.fit, newdata=holdDescr[,-1], type="prob")[,c("s")]
    holdClass$scores    <- tmp.score
    
    tmp.res             <- getAMS(holdClass)
    tmp.thresh          <- tmp.res$threshold
    tmp.pred            <- ifelse(tmp.score >= tmp.thresh, s_val, b_val)
    
    #AMS(tmp.pred, holdClass$label.num, holdClass$weight)
    
    #tmp.res     <- predict(tmp.fit, newdata=holdDescr[,-1])
    #tmp.breaks   <- quantile(tmp.score, probs=v07.seq)
    #tmp.ams      <- sapply(tmp.breaks, calcAmsCutoff, tmp.score, trainClass$label[fit.idx], trainClass$weight[fit.idx])
    #tmp.thresh   <- v07.seq[which(tmp.ams == max(tmp.ams))]
    
    ## aggregate the ams results
    v07.ams[,i]  <- AMS(tmp.pred, holdClass$label.num, holdClass$weight)
    
    ## plot the ams data based on the hold-out sample
    #if (i == 1) {
    #    plot(v07.seq, v07.ams[,i], type="b", col=i, ylim=c(1,3.9))
    #} else {
    #    points(v07.seq, v07.ams[,i], type="b", col=i)
    #}
    #abline(h=3.5)
    
    ## score the test data
    test.score    <- predict(tmp.fit, newdata=testDescr[,-1], type="prob")[,c("s")]
    test.id       <- testDescr[,1]
    test.pred     <- ifelse(test.score >= tmp.thresh, "s", "b")

    ## load the combined results into a list
    v07.list[[fit.id]]$test_score   <- data.frame(EventId=test.id, Prob=test.score, Class=test.pred)
    v07.list[[fit.id]]$thresh       <- tmp.thresh
    # v07.list[[fit.id]]$ams          <- tmp.ams
    #v07.list[[fit.id]]$breaks       <- tmp.breaks
    
    ## load a matrix with all of the test probabilities
    if (i == 1) {
        v07.mat[,i]     <- test.id
        v07.mat[,i+1]   <- test.score
    } else {
        v07.mat[,i+1]   <- test.score
    }
}


## save the intermediate results
save(v07.list, v07.mat, v07.ams, file="gbm_full_depth9_trees450_shrink0.01_v07_AllScored.Rdata")



##******************************************************************
## Intermediate Results
##******************************************************************

##------------------------------------------------------------------
## identify the maximum AMS scores
##------------------------------------------------------------------

# apply(v07.ams, 2, max)
# 3.564364 3.604836 3.547349 3.571381 3.541123 3.612231 3.623436 3.565879 3.627421 3.607781

# order(apply(v07.ams, 2, max))
# 5  3  1  8  4  2 10  6  7  9

# apply(v07.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, max)
# 3.565879 3.571381 3.604836 3.607781 3.612231 3.623436 3.627421

##------------------------------------------------------------------
## get an estimate of the AMS thresholds
##------------------------------------------------------------------

# apply(v07.ams, 2, function(x) {v07.seq[which(x == max(x))]})
# 0.815 0.860 0.845 0.855 0.850 0.820 0.840 0.830 0.860 0.860

# apply(v07.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, function(x) {v07.seq[which(x == max(x))]})
# 0.830 0.855 0.860 0.860 0.820 0.840 0.860

##------------------------------------------------------------------
## AMS threshold extrema
##------------------------------------------------------------------

# mean(apply(v07.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, function(x) {v07.seq[which(x == max(x))]}))
# [1] 0.8464286

## 0.860 and 0.820 are the two peaks



##******************************************************************
## First iteration -- Use an average score based on the top 7 models
## and use three different thresholds (0.820, 0.8464286, 0.86)
##******************************************************************

## the subset of fits to use
subset <- c(1:v07.num)

## a matrix to hold each scored probability
v07.submission01.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- v07.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
  
    ## get the single fit scores
    v07.submission01.probs[,i]  <- v07.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
v07.submission01.mean   <- apply(v07.submission01.probs, 1, mean)
v07.submission01.id     <- v07.list[[1]]$test_score$EventId

## reorder the data
v07.submission01.order      <- order(v07.submission01.mean)
v07.submission01.mean.order <- v07.submission01.mean[v07.submission01.order]
v07.submission01.id.order   <- v07.submission01.id[v07.submission01.order]
v07.submission01.rank.order <- 1:550000

##------------------------------------------------------------------
## Create three submission files based on the above threholds
##------------------------------------------------------------------

##------------------------------------------------------------------
## S017
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v07_S017"
out.csv       <- paste0(fit.id,"_v07_Submission.csv")
out.rdata     <- paste0(fit.id,"_v07_Submission.Rdata")

## score -> class
thresh.S017            <- 0.93
v07.submission01.class <- ifelse(v07.submission01.mean.order >= thresh.S017, "s", "b")
v07.submission01.S017  <- data.frame(
                            EventId=v07.submission01.id.order,
                            RankOrder=v07.submission01.rank.order,
                            Class=v07.submission01.class)

## save the results
save(v07.submission01.S017, file=paste0(v07.dir,"/",out.rdata) )
write.csv(v07.submission01.S017, file=paste0(v07.dir,"/", out.csv), row.names=FALSE)



##------------------------------------------------------------------
## S018
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v07_S018"
out.csv       <- paste0(fit.id,"_v07_Submission.csv")
out.rdata     <- paste0(fit.id,"_v07_Submission.Rdata")

## score -> class
thresh.S018            <- 0.925
v07.submission01.class <- ifelse(v07.submission01.mean.order >= thresh.S018, "s", "b")
v07.submission01.S018  <- data.frame(
EventId=v07.submission01.id.order,
RankOrder=v07.submission01.rank.order,
Class=v07.submission01.class)

## save the results
save(v07.submission01.S018, file=paste0(v07.dir,"/",out.rdata) )
write.csv(v07.submission01.S018, file=paste0(v07.dir,"/", out.csv), row.names=FALSE)



##******************************************************************
## Second iteration -- Take top 5 models
##******************************************************************

## the subset of fits to use
subset <- c(16,20,12,10,8)

## a matrix to hold each scored probability
v07.submission02.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- v07.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
    
    ## get the single fit scores
    v07.submission02.probs[,i]  <- v07.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
v07.submission02.mean   <- apply(v07.submission02.probs, 1, mean)
v07.submission02.id     <- v07.list[[1]]$test_score$EventId

## reorder the data
v07.submission02.order      <- order(v07.submission02.mean)
v07.submission02.mean.order <- v07.submission02.mean[v07.submission02.order]
v07.submission02.id.order   <- v07.submission02.id[v07.submission02.order]
v07.submission02.rank.order <- 1:550000


##------------------------------------------------------------------
## S019
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v07_S019"
out.csv       <- paste0(fit.id,"_v07_Submission.csv")
out.rdata     <- paste0(fit.id,"_v07_Submission.Rdata")

## score -> class
thresh.S019            <- 0.930
v07.submission02.class <- ifelse(v07.submission02.mean.order >= thresh.S019, "s", "b")
v07.submission02.S019  <- data.frame(
EventId=v07.submission02.id.order,
RankOrder=v07.submission02.rank.order,
Class=v07.submission02.class)

## save the results
save(v07.submission02.S019, file=paste0(v07.dir,"/",out.rdata) )
write.csv(v07.submission02.S019, file=paste0(v07.dir,"/", out.csv), row.names=FALSE)





##******************************************************************
## Third iteration -- Take best
##******************************************************************

## the subset of fits to use
subset <- c(8)

## a matrix to hold each scored probability
v07.submission03.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- v07.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
    
    ## get the single fit scores
    v07.submission03.probs[,i]  <- v07.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
v07.submission03.mean   <- apply(v07.submission03.probs, 1, mean)
v07.submission03.id     <- v07.list[[1]]$test_score$EventId

## reorder the data
v07.submission03.order      <- order(v07.submission03.mean)
v07.submission03.mean.order <- v07.submission03.mean[v07.submission03.order]
v07.submission03.id.order   <- v07.submission03.id[v07.submission03.order]
v07.submission03.rank.order <- 1:550000


##------------------------------------------------------------------
## S020
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_v07_S020"
out.csv       <- paste0(fit.id,"_v07_Submission.csv")
out.rdata     <- paste0(fit.id,"_v07_Submission.Rdata")

## score -> class
thresh.S020            <- 0.940456
v07.submission03.class <- ifelse(v07.submission03.mean.order >= thresh.S020, "s", "b")
v07.submission03.S020  <- data.frame(
EventId=v07.submission03.id.order,
RankOrder=v07.submission03.rank.order,
Class=v07.submission03.class)

## save the results
save(v07.submission03.S020, file=paste0(v07.dir,"/",out.rdata) )
write.csv(v07.submission03.S020, file=paste0(v07.dir,"/", out.csv), row.names=FALSE)



