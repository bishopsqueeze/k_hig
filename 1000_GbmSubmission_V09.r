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
## gbm_full_depth9_trees450_shrink0.05_V09
##******************************************************************

##------------------------------------------------------------------
## Note that CV was properly conducted on an identical hold-out
## sample, so the AMS computed using the hold-out data is
## comparable across all of the folds.
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

##------------------------------------------------------------------
## Load the data used to train the model
##------------------------------------------------------------------
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

##------------------------------------------------------------------
## Define constants used below
##------------------------------------------------------------------
s_val   <- 1
b_val   <- 0
b_tau   <- 10
vsize   <- .10
na_val  <- -999
n_trees <- 500

##------------------------------------------------------------------
## Create a numeric label for the dependent variable
##------------------------------------------------------------------
trainClass$label.num <- as.numeric(ifelse(trainClass$label=="s", s_val, b_val))

##------------------------------------------------------------------
## Compute the sum of signal and background weights
##------------------------------------------------------------------
s_total <- sum(trainClass$weight[trainClass$label.num == s_val])
b_total <- sum(trainClass$weight[trainClass$label.num == b_val])

##------------------------------------------------------------------
## compute max_AMS
##------------------------------------------------------------------
max_AMS <- AMS(trainClass$label.num, trainClass$label.num, trainClass$weight)
cat("Max AMS == ", max_AMS, "\n")


##------------------------------------------------------------------
## Grab the candidate fits
##------------------------------------------------------------------
V09.dir     <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/candidate_fits"
V09.files   <- dir(V09.dir)[grep("[0-9].Rdata$", dir(V09.dir))]
V09.num     <- length(V09.files)
V09.seq     <- seq(0,1,0.01)
V09.ams     <- matrix(0,nrow=length(V09.seq),ncol=V09.num)
V09.list    <- list()
V09.mat     <- matrix(0,nrow=550000,ncol=length(V09.files)+1)

##------------------------------------------------------------------
## loop over each fit and compute fit statistics
##------------------------------------------------------------------
for (i in 1:V09.num) {
    
    ## load the file
    fit.filename  <- V09.files[i]
    load(paste0(V09.dir,"/",fit.filename))
    cat("Using file :", fit.filename,"\n")
    
    ## create additional filenames
    fit.id        <- gsub(".Rdata","",fit.filename)
    out.csv       <- paste0(fit.id,"_V09_Submission.csv")
    out.rdata     <- paste0(fit.id,"_V09_Submission.Rdata")
    
    ## in this case, the sample index represents the *testing* data
    fit.idx     <- samp.idx$Resample1
    holdDescr   <- trainDescr[ -fit.idx,]
    holdClass   <- trainClass[ -fit.idx,]
    
    ## x-check:  compute weights before renormalization
    s_before <- sum(holdClass$weight[holdClass$label.num == s_val])
    b_before <- sum(holdClass$weight[holdClass$label.num == b_val])
    
    holdClass$weight <- normalize(holdClass$weight, holdClass$label.num, s_total, b_total)
    
    ## renormalize the weights for the holdout sample
    s_after <- sum(holdClass$weight[holdClass$label.num == s_val])
    b_after <- sum(holdClass$weight[holdClass$label.num == b_val])
    
    ## x-check: compare before/after
    cat("s_before=",s_before," s_after=",s_after," s_total=",s_total,"\n")
    cat("b_before=",b_before," b_after=",b_after," b_total=",b_total,"\n")
    
    ## score the hold-out sample & compute the AMS curve
    tmp.score           <- predict(tmp.fit, newdata=holdDescr[,-1], type="prob")[,c("s")]
    holdClass$scores    <- tmp.score
    
    ## tmp.res is a list {ams, amsMax, threshold}
    tmp.res             <- getAMS(holdClass)
    tmp.thresh          <- tmp.res$threshold
    tmp.pred            <- ifelse(tmp.score >= tmp.thresh, s_val, b_val)
    cat("AMS=",tmp.res$amsMax,"\n")
    
    ## aggregate the ams results
    V09.ams[,i]         <- AMS(tmp.pred, holdClass$label.num, holdClass$weight)
    
    ## plot the ams data based on the hold-out sample
    if (i == 1) {
        plot(tmp.res$ams, type="l", col=i, ylim=c(1,4))
    } else {
        points(tmp.res$ams, type="l", col=i)
    }
    abline(h=3.50, col="red")
    abline(h=3.75, col="green")
    
    ## score the test data
    test.score    <- predict(tmp.fit, newdata=testDescr[,-1], type="prob")[,c("s")]
    test.id       <- testDescr[,1]
    test.pred     <- ifelse(test.score >= tmp.thresh, "s", "b")

    ## load the combined results into a list
    V09.list[[fit.id]]$test_score   <- data.frame(EventId=test.id, Prob=test.score, Class=test.pred)
    V09.list[[fit.id]]$thresh       <- tmp.thresh
    
    ## load a matrix with all of the test probabilities
    if (i == 1) {
        V09.mat[,i]     <- test.id
        V09.mat[,i+1]   <- test.score
    } else {
        V09.mat[,i+1]   <- test.score
    }
}


## save the intermediate results
save(V09.list, V09.mat, V09.ams, file="gbm_full_depth9_trees450_shrink0.01_V09_AllScored.Rdata")


##******************************************************************
## Review initial resuls
##******************************************************************
ams.vec     <- apply(V09.ams, 2, max)
thresh.vec  <- unlist(lapply(V09.list, function(x){x$thresh}))
order.vec   <- order(apply(V09.ams, 2, max))

order.best  <- order.vec[floor(length(order.vec)/2):length(order.vec)]
best.thresh <- thresh.vec[order.best]
best.ams    <- ams.vec[order.best]

##******************************************************************
## Intermediate Results
##******************************************************************

##------------------------------------------------------------------
## identify the maximum AMS scores
##------------------------------------------------------------------

# apply(V09.ams, 2, max)
# 3.564364 3.604836 3.547349 3.571381 3.541123 3.612231 3.623436 3.565879 3.627421 3.607781

# order(apply(V09.ams, 2, max))
# 5  3  1  8  4  2 10  6  7  9

# apply(V09.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, max)
# 3.565879 3.571381 3.604836 3.607781 3.612231 3.623436 3.627421

##------------------------------------------------------------------
## get an estimate of the AMS thresholds
##------------------------------------------------------------------

# apply(V09.ams, 2, function(x) {V09.seq[which(x == max(x))]})
# 0.815 0.860 0.845 0.855 0.850 0.820 0.840 0.830 0.860 0.860

# apply(V09.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, function(x) {V09.seq[which(x == max(x))]})
# 0.830 0.855 0.860 0.860 0.820 0.840 0.860

##------------------------------------------------------------------
## AMS threshold extrema
##------------------------------------------------------------------

# mean(apply(V09.ams[,c(8, 4, 2, 10, 6, 7, 9)], 2, function(x) {V09.seq[which(x == max(x))]}))
# [1] 0.8464286

## 0.860 and 0.820 are the two peaks



##******************************************************************
## First iteration -- Use top half
##******************************************************************

## the subset of fits to use
subset <- c(order.best)

## a matrix to hold each scored probability
V09.submission01.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- V09.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
  
    ## get the single fit scores
    V09.submission01.probs[,i]  <- V09.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
V09.submission01.mean   <- apply(V09.submission01.probs, 1, mean)
V09.submission01.id     <- V09.list[[1]]$test_score$EventId

## reorder the data
V09.submission01.order      <- order(V09.submission01.mean)
V09.submission01.mean.order <- V09.submission01.mean[V09.submission01.order]
V09.submission01.id.order   <- V09.submission01.id[V09.submission01.order]
V09.submission01.rank.order <- 1:550000

##------------------------------------------------------------------
## Create three submission files based on the above threholds
##------------------------------------------------------------------

##------------------------------------------------------------------
## S025
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V09_S025"
out.csv       <- paste0(fit.id,"_V09_Submission.csv")
out.rdata     <- paste0(fit.id,"_V09_Submission.Rdata")

## score -> class
thresh.S025            <- 0.944
V09.submission01.class <- ifelse(V09.submission01.mean.order >= thresh.S025, "s", "b")
V09.submission01.S025  <- data.frame(
                            EventId=V09.submission01.id.order,
                            RankOrder=V09.submission01.rank.order,
                            Class=V09.submission01.class)

## save the results
save(V09.submission01.S025, file=paste0(V09.dir,"/",out.rdata) )
write.csv(V09.submission01.S025, file=paste0(V09.dir,"/", out.csv), row.names=FALSE)


##------------------------------------------------------------------
## S026
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V09_S026"
out.csv       <- paste0(fit.id,"_V09_Submission.csv")
out.rdata     <- paste0(fit.id,"_V09_Submission.Rdata")

## score -> class
thresh.S026            <- 0.9484765 ## Threhsold optimal @0.9484765 ??

V09.submission01.class <- ifelse(V09.submission01.mean.order >= thresh.S026, "s", "b")
V09.submission01.S026  <- data.frame(
EventId=V09.submission01.id.order,
RankOrder=V09.submission01.rank.order,
Class=V09.submission01.class)

## save the results
save(V09.submission01.S026, file=paste0(V09.dir,"/",out.rdata) )
write.csv(V09.submission01.S026, file=paste0(V09.dir,"/", out.csv), row.names=FALSE)



##******************************************************************
## Second iteration -- Take top 11 models
##******************************************************************

order.best  <- (1:length(ams.vec))[intersect(which(ams.vec >= 3.8),which(ams.vec <= 4.0))]
best.thresh <- thresh.vec[order.best]
best.ams    <- ams.vec[order.best]

## the subset of fits to use
subset <- c(order.best)

## a matrix to hold each scored probability
V09.submission02.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- V09.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
    
    ## get the single fit scores
    V09.submission02.probs[,i]  <- V09.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
V09.submission02.mean   <- apply(V09.submission02.probs, 1, mean)
V09.submission02.id     <- V09.list[[1]]$test_score$EventId

## reorder the data
V09.submission02.order      <- order(V09.submission02.mean)
V09.submission02.mean.order <- V09.submission02.mean[V09.submission02.order]
V09.submission02.id.order   <- V09.submission02.id[V09.submission02.order]
V09.submission02.rank.order <- 1:550000


##------------------------------------------------------------------
## S027
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V09_S027"
out.csv       <- paste0(fit.id,"_V09_Submission.csv")
out.rdata     <- paste0(fit.id,"_V09_Submission.Rdata")

## score -> class
thresh.S027            <- 0.946
V09.submission02.class <- ifelse(V09.submission02.mean.order >= thresh.S027, "s", "b")
V09.submission02.S027  <- data.frame(
EventId=V09.submission02.id.order,
RankOrder=V09.submission02.rank.order,
Class=V09.submission02.class)

## save the results
save(V09.submission02.S027, file=paste0(V09.dir,"/",out.rdata) )
write.csv(V09.submission02.S027, file=paste0(V09.dir,"/", out.csv), row.names=FALSE)





##******************************************************************
## Third iteration -- Take top 11
##******************************************************************

order.best  <- order(ams.vec)[(length(ams.vec)-10):length(ams.vec)]
best.thresh <- thresh.vec[order.best]
best.ams    <- ams.vec[order.best]

## the subset of fits to use
subset <- c(order.best)
## a matrix to hold each scored probability
V09.submission03.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- V09.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
    
    ## get the single fit scores
    V09.submission03.probs[,i]  <- V09.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
V09.submission03.mean   <- apply(V09.submission03.probs, 1, mean)
V09.submission03.id     <- V09.list[[1]]$test_score$EventId

## reorder the data
V09.submission03.order      <- order(V09.submission03.mean)
V09.submission03.mean.order <- V09.submission03.mean[V09.submission03.order]
V09.submission03.id.order   <- V09.submission03.id[V09.submission03.order]
V09.submission03.rank.order <- 1:550000


##------------------------------------------------------------------
## S028
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V09_S028"
out.csv       <- paste0(fit.id,"_V09_Submission.csv")
out.rdata     <- paste0(fit.id,"_V09_Submission.Rdata")

## score -> class
thresh.S028            <- 0.946
V09.submission03.class <- ifelse(V09.submission03.mean.order >= thresh.S028, "s", "b")
V09.submission03.S028  <- data.frame(
EventId=V09.submission03.id.order,
RankOrder=V09.submission03.rank.order,
Class=V09.submission03.class)

## save the results
save(V09.submission03.S028, file=paste0(V09.dir,"/",out.rdata) )
write.csv(V09.submission03.S028, file=paste0(V09.dir,"/", out.csv), row.names=FALSE)




##------------------------------------------------------------------
## S029
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V09_S029"
out.csv       <- paste0(fit.id,"_V09_Submission.csv")
out.rdata     <- paste0(fit.id,"_V09_Submission.Rdata")

## score -> class
thresh.S029            <- 0.9478266
V09.submission03.class <- ifelse(V09.submission03.mean.order >= thresh.S029, "s", "b")
V09.submission03.S029  <- data.frame(
EventId=V09.submission03.id.order,
RankOrder=V09.submission03.rank.order,
Class=V09.submission03.class)

## save the results
save(V09.submission03.S029, file=paste0(V09.dir,"/",out.rdata) )
write.csv(V09.submission03.S029, file=paste0(V09.dir,"/", out.csv), row.names=FALSE)


