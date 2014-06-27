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
## gbm_full_depth9_trees450_shrink0.05_V08
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
V08.dir     <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/gbm_full_depth9_trees500_shrink0.2_V08"
V08.files   <- dir(V08.dir)[grep("[0-9].Rdata$", dir(V08.dir))]
V08.num     <- length(V08.files)
V08.seq     <- seq(0,1,0.01)
V08.ams     <- matrix(0,nrow=length(V08.seq),ncol=V08.num)
V08.list    <- list()
V08.mat     <- matrix(0,nrow=550000,ncol=length(V08.files)+1)

## loop over each file and compute AMS (order is based on alphabetic filename)
for (i in 1:V08.num) {
    
    ## load the file
    fit.filename  <- V08.files[i]
    load(paste0(V08.dir,"/",fit.filename))
    cat("Using file :", fit.filename,"\n")
    
    ## create addition filenames
    fit.id        <- gsub(".Rdata","",fit.filename)
    out.csv       <- paste0(fit.id,"_V08_Submission.csv")
    out.rdata     <- paste0(fit.id,"_V08_Submission.Rdata")
    
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
    #tmp.breaks   <- quantile(tmp.score, probs=V08.seq)
    #tmp.ams      <- sapply(tmp.breaks, calcAmsCutoff, tmp.score, trainClass$label[fit.idx], trainClass$weight[fit.idx])
    #tmp.thresh   <- V08.seq[which(tmp.ams == max(tmp.ams))]
    
    ## aggregate the ams results
    V08.ams[,i]  <- AMS(tmp.pred, holdClass$label.num, holdClass$weight)
    
    ## plot the ams data based on the hold-out sample
    #if (i == 1) {
    #    plot(V08.seq, V08.ams[,i], type="b", col=i, ylim=c(1,3.9))
    #} else {
    #    points(V08.seq, V08.ams[,i], type="b", col=i)
    #}
    #abline(h=3.5)
    
    ## score the test data
    test.score    <- predict(tmp.fit, newdata=testDescr[,-1], type="prob")[,c("s")]
    test.id       <- testDescr[,1]
    test.pred     <- ifelse(test.score >= tmp.thresh, "s", "b")

    ## load the combined results into a list
    V08.list[[fit.id]]$test_score   <- data.frame(EventId=test.id, Prob=test.score, Class=test.pred)
    V08.list[[fit.id]]$thresh       <- tmp.thresh
    # V08.list[[fit.id]]$ams          <- tmp.ams
    #V08.list[[fit.id]]$breaks       <- tmp.breaks
    
    ## load a matrix with all of the test probabilities
    if (i == 1) {
        V08.mat[,i]     <- test.id
        V08.mat[,i+1]   <- test.score
    } else {
        V08.mat[,i+1]   <- test.score
    }
}


## save the intermediate results
save(V08.list, V08.mat, V08.ams, file="gbm_full_depth9_trees450_shrink0.01_V08_AllScored.Rdata")



##******************************************************************
## Intermediate Results
##******************************************************************

##------------------------------------------------------------------
## identify the maximum AMS scores
##------------------------------------------------------------------

# apply(V08.ams, 2, max)
# 3.564364 3.604836 3.547349 3.571381 3.541123 3.612231 3.623436 3.565879 3.627421 3.607781

# order(apply(V08.ams, 2, max))
# 5  3  1  8  4  2 10  6  7  9

# apply(V08.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, max)
# 3.565879 3.571381 3.604836 3.607781 3.612231 3.623436 3.627421

##------------------------------------------------------------------
## get an estimate of the AMS thresholds
##------------------------------------------------------------------

# apply(V08.ams, 2, function(x) {V08.seq[which(x == max(x))]})
# 0.815 0.860 0.845 0.855 0.850 0.820 0.840 0.830 0.860 0.860

# apply(V08.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, function(x) {V08.seq[which(x == max(x))]})
# 0.830 0.855 0.860 0.860 0.820 0.840 0.860

##------------------------------------------------------------------
## AMS threshold extrema
##------------------------------------------------------------------

# mean(apply(V08.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, function(x) {V08.seq[which(x == max(x))]}))
# [1] 0.8464286

## 0.860 and 0.820 are the two peaks



##******************************************************************
## First iteration -- Use an average score based on the top 7 models
## and use three different thresholds (0.820, 0.8464286, 0.86)
##******************************************************************

## the subset of fits to use
subset <- c(1:V08.num)

## a matrix to hold each scored probability
V08.submission01.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- V08.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
  
    ## get the single fit scores
    V08.submission01.probs[,i]  <- V08.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
V08.submission01.mean   <- apply(V08.submission01.probs, 1, mean)
V08.submission01.id     <- V08.list[[1]]$test_score$EventId

## reorder the data
V08.submission01.order      <- order(V08.submission01.mean)
V08.submission01.mean.order <- V08.submission01.mean[V08.submission01.order]
V08.submission01.id.order   <- V08.submission01.id[V08.submission01.order]
V08.submission01.rank.order <- 1:550000

##------------------------------------------------------------------
## Create three submission files based on the above threholds
##------------------------------------------------------------------

##------------------------------------------------------------------
## S020
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V08_S020"
out.csv       <- paste0(fit.id,"_V08_Submission.csv")
out.rdata     <- paste0(fit.id,"_V08_Submission.Rdata")

## score -> class
thresh.S020            <- 0.944
V08.submission01.class <- ifelse(V08.submission01.mean.order >= thresh.S020, "s", "b")
V08.submission01.S020  <- data.frame(
                            EventId=V08.submission01.id.order,
                            RankOrder=V08.submission01.rank.order,
                            Class=V08.submission01.class)

## save the results
save(V08.submission01.S020, file=paste0(V08.dir,"/",out.rdata) )
write.csv(V08.submission01.S020, file=paste0(V08.dir,"/", out.csv), row.names=FALSE)




##******************************************************************
## Second iteration -- Take top 11 models
##******************************************************************

## sort by AMS
tmp.order   <- order(apply(V08.ams, 2, max))


## now try just the top 11 models
subset <- tmp.order[10:20]

## a matrix to hold each scored probability
V08.submission02.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- V08.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
    
    ## get the single fit scores
    V08.submission02.probs[,i]  <- V08.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
V08.submission02.mean   <- apply(V08.submission02.probs, 1, mean)
V08.submission02.id     <- V08.list[[1]]$test_score$EventId

## reorder the data
V08.submission02.order      <- order(V08.submission02.mean)
V08.submission02.mean.order <- V08.submission02.mean[V08.submission02.order]
V08.submission02.id.order   <- V08.submission02.id[V08.submission02.order]
V08.submission02.rank.order <- 1:550000


##------------------------------------------------------------------
## S021
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V08_S021"
out.csv       <- paste0(fit.id,"_V08_Submission.csv")
out.rdata     <- paste0(fit.id,"_V08_Submission.Rdata")

## score -> class
thresh.S021            <- 0.946
V08.submission02.class <- ifelse(V08.submission02.mean.order >= thresh.S021, "s", "b")
V08.submission02.S021  <- data.frame(
EventId=V08.submission02.id.order,
RankOrder=V08.submission02.rank.order,
Class=V08.submission02.class)

## save the results
save(V08.submission02.S021, file=paste0(V08.dir,"/",out.rdata) )
write.csv(V08.submission02.S021, file=paste0(V08.dir,"/", out.csv), row.names=FALSE)





##******************************************************************
## Third iteration -- Take top 9
##******************************************************************

## now try just the top 9 models
subset <- tmp.order[12:20]

## a matrix to hold each scored probability
V08.submission03.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- V08.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
    
    ## get the single fit scores
    V08.submission03.probs[,i]  <- V08.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
V08.submission03.mean   <- apply(V08.submission03.probs, 1, mean)
V08.submission03.id     <- V08.list[[1]]$test_score$EventId

## reorder the data
V08.submission03.order      <- order(V08.submission03.mean)
V08.submission03.mean.order <- V08.submission03.mean[V08.submission03.order]
V08.submission03.id.order   <- V08.submission03.id[V08.submission03.order]
V08.submission03.rank.order <- 1:550000


##------------------------------------------------------------------
## S022
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V08_S022"
out.csv       <- paste0(fit.id,"_V08_Submission.csv")
out.rdata     <- paste0(fit.id,"_V08_Submission.Rdata")

## score -> class
thresh.S022            <- 0.946
V08.submission03.class <- ifelse(V08.submission03.mean.order >= thresh.S022, "s", "b")
V08.submission03.S022  <- data.frame(
EventId=V08.submission03.id.order,
RankOrder=V08.submission03.rank.order,
Class=V08.submission03.class)

## save the results
save(V08.submission03.S022, file=paste0(V08.dir,"/",out.rdata) )
write.csv(V08.submission03.S022, file=paste0(V08.dir,"/", out.csv), row.names=FALSE)






##******************************************************************
## Fourth iteration -- Take top 7
##******************************************************************

## now try just the top 7 models
subset <- tmp.order[14:20]

## a matrix to hold each scored probability
V08.submission04.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- V08.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
    
    ## get the single fit scores
    V08.submission04.probs[,i]  <- V08.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
V08.submission04.mean   <- apply(V08.submission04.probs, 1, mean)
V08.submission04.id     <- V08.list[[1]]$test_score$EventId

## reorder the data
V08.submission04.order      <- order(V08.submission04.mean)
V08.submission04.mean.order <- V08.submission04.mean[V08.submission04.order]
V08.submission04.id.order   <- V08.submission04.id[V08.submission04.order]
V08.submission04.rank.order <- 1:550000


##------------------------------------------------------------------
## S023
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V08_S023"
out.csv       <- paste0(fit.id,"_V08_Submission.csv")
out.rdata     <- paste0(fit.id,"_V08_Submission.Rdata")

## score -> class
thresh.S023            <- 0.946
V08.submission04.class <- ifelse(V08.submission04.mean.order >= thresh.S023, "s", "b")
V08.submission04.S023  <- data.frame(
EventId=V08.submission04.id.order,
RankOrder=V08.submission04.rank.order,
Class=V08.submission04.class)

## save the results
save(V08.submission04.S023, file=paste0(V08.dir,"/",out.rdata) )
write.csv(V08.submission04.S023, file=paste0(V08.dir,"/", out.csv), row.names=FALSE)




##------------------------------------------------------------------
## S024
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V08_S024"
out.csv       <- paste0(fit.id,"_V08_Submission.csv")
out.rdata     <- paste0(fit.id,"_V08_Submission.Rdata")

## score -> class
thresh.S024            <- 0.947
V08.submission04.class <- ifelse(V08.submission04.mean.order >= thresh.S024, "s", "b")
V08.submission04.S024  <- data.frame(
EventId=V08.submission04.id.order,
RankOrder=V08.submission04.rank.order,
Class=V08.submission04.class)

## save the results
save(V08.submission04.S024, file=paste0(V08.dir,"/",out.rdata) )
write.csv(V08.submission04.S024, file=paste0(V08.dir,"/", out.csv), row.names=FALSE)



