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
## gbm_full_depth9_trees450_shrink0.05_V10
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
V10.dir     <- "/Users/alexstephens/Development/kaggle/higgs/data/proc/candidate_fits"
V10.files   <- dir(V10.dir)[grep("[0-9].Rdata$", dir(V10.dir))]
V10.num     <- length(V10.files)
V10.seq     <- seq(0,1,0.01)
V10.ams     <- matrix(0,nrow=length(V10.seq),ncol=V10.num)
V10.list    <- list()
V10.mat     <- matrix(0,nrow=550000,ncol=length(V10.files)+1)
V10.idx     <- matrix(0,nrow=225001,ncol=length(V10.files))
V10.pred    <- matrix(0,nrow=24999,ncol=length(V10.files))

##------------------------------------------------------------------
## loop over each fit and compute fit statistics
##------------------------------------------------------------------
for (i in 1:V10.num) {
    
    ## load the file
    fit.filename  <- V10.files[i]
    load(paste0(V10.dir,"/",fit.filename))
    cat("Using file :", fit.filename,"\n")
    
    ## create additional filenames
    fit.id        <- gsub(".Rdata","",fit.filename)
    out.csv       <- paste0(fit.id,"_V10_Submission.csv")
    out.rdata     <- paste0(fit.id,"_V10_Submission.Rdata")
    
    ## in this case, the sample index represents the *testing* data
    fit.idx     <- samp.idx$Resample1
    holdDescr   <- trainDescr[ -fit.idx,]
    holdClass   <- trainClass[ -fit.idx,]
    
    ## x-check the fit index used
    V10.idx[,i] <- fit.idx
    
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
    V10.pred[,i]            <- tmp.score
    
    ## tmp.res is a list {ams, amsMax, threshold}
    tmp.res             <- getAMS(holdClass, doplot=FALSE)
    tmp.thresh          <- tmp.res$threshold
    tmp.pred            <- ifelse(tmp.score >= tmp.thresh, s_val, b_val)
    cat("AMS=",tmp.res$amsMax,"\n")
    
    ## aggregate the ams results
    V10.ams[,i]         <- AMS(tmp.pred, holdClass$label.num, holdClass$weight)
    
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
    V10.list[[fit.id]]$test_score   <- data.frame(EventId=test.id, Prob=test.score, Class=test.pred)
    V10.list[[fit.id]]$thresh       <- tmp.thresh
    
    ## load a matrix with all of the test probabilities
    if (i == 1) {
        V10.mat[,i]     <- test.id
        V10.mat[,i+1]   <- test.score
    } else {
        V10.mat[,i+1]   <- test.score
    }
}



##------------------------------------------------------------------
## combine probabilities (assuming you used identical hold-out
## samples ... to see if there is a point that maximizes the estimated
## max_ams
##------------------------------------------------------------------
ams.vec     <- apply(V10.ams, 2, max)
thresh.vec  <- unlist(lapply(V10.list, function(x){x$thresh}))
order.vec   <- order(apply(V10.ams, 2, max), decreasing=TRUE)
comb.mat    <- matrix(0,nrow=length(ams.vec),ncol=3)

for (i in 1:length(order.vec)) {
    
    comb.idx            <- order.vec[1:i]
    if (i == 1) {
        comb.score          <- V10.pred[,comb.idx]
    } else {
        comb.score          <- apply(V10.pred[,comb.idx],1,mean)
    }
    
    holdClass$scores    <- comb.score
    comb.res            <- getAMS(holdClass, doplot=FALSE)
    comb.mat[i,]        <- c(i, comb.res$amsMax, comb.res$threshold)
    
}


##******************************************************************
## Review initial resuls
##******************************************************************
ams.vec     <- apply(V10.ams, 2, max)
thresh.vec  <- unlist(lapply(V10.list, function(x){x$thresh}))
order.vec   <- order(apply(V10.ams, 2, max), decreasing=TRUE)

##******************************************************************
## First iteration -- Use 50%+
##******************************************************************

## order the data
order.best  <- order.vec[1:ceiling(length(order.vec)/2)]
best.thresh <- thresh.vec[order.best]
best.ams    <- ams.vec[order.best]

## the subset of fits to use
subset <- c(order.best)

## a matrix to hold each scored probability
V10.submission01.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- V10.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
    
    ## get the single fit scores
    V10.submission01.probs[,i]  <- V10.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
V10.submission01.mean   <- apply(V10.submission01.probs, 1, mean)
V10.submission01.id     <- V10.list[[1]]$test_score$EventId

## reorder the data
V10.submission01.order      <- order(V10.submission01.mean)
V10.submission01.mean.order <- V10.submission01.mean[V10.submission01.order]
V10.submission01.id.order   <- V10.submission01.id[V10.submission01.order]
V10.submission01.rank.order <- 1:550000

##------------------------------------------------------------------
## Create three submission files based on the above threholds
##------------------------------------------------------------------

##------------------------------------------------------------------
## S030
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V10_S030"
out.csv       <- paste0(fit.id,"_V10_Submission.csv")
out.rdata     <- paste0(fit.id,"_V10_Submission.Rdata")

## score -> class
thresh.S030            <- 0.945
V10.submission01.class <- ifelse(V10.submission01.mean.order >= thresh.S030, "s", "b")
V10.submission01.S030  <- data.frame(
EventId=V10.submission01.id.order,
RankOrder=V10.submission01.rank.order,
Class=V10.submission01.class)

## save the results
save(V10.submission01.S030, file=paste0(V10.dir,"/",out.rdata) )
write.csv(V10.submission01.S030, file=paste0(V10.dir,"/", out.csv), row.names=FALSE)


##------------------------------------------------------------------
## S031
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V10_S031"
out.csv       <- paste0(fit.id,"_V10_Submission.csv")
out.rdata     <- paste0(fit.id,"_V10_Submission.Rdata")

## score -> class
thresh.S031            <- 0.9455

V10.submission01.class <- ifelse(V10.submission01.mean.order >= thresh.S031, "s", "b")
V10.submission01.S031  <- data.frame(
EventId=V10.submission01.id.order,
RankOrder=V10.submission01.rank.order,
Class=V10.submission01.class)

## save the results
save(V10.submission01.S031, file=paste0(V10.dir,"/",out.rdata) )
write.csv(V10.submission01.S031, file=paste0(V10.dir,"/", out.csv), row.names=FALSE)



##------------------------------------------------------------------
## S034
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V10_S034"
out.csv       <- paste0(fit.id,"_V10_Submission.csv")
out.rdata     <- paste0(fit.id,"_V10_Submission.Rdata")

## score -> class
thresh.S034            <- 0.946

V10.submission01.class <- ifelse(V10.submission01.mean.order >= thresh.S034, "s", "b")
V10.submission01.S034  <- data.frame(
EventId=V10.submission01.id.order,
RankOrder=V10.submission01.rank.order,
Class=V10.submission01.class)

## save the results
save(V10.submission01.S034, file=paste0(V10.dir,"/",out.rdata) )
write.csv(V10.submission01.S034, file=paste0(V10.dir,"/", out.csv), row.names=FALSE)


##******************************************************************
## Second iteration -- Use top 75%
##******************************************************************

order.best  <- order.vec[1:ceiling(length(order.vec)*(3/4))]
best.thresh <- thresh.vec[order.best]
best.ams    <- ams.vec[order.best]

## the subset of fits to use
subset <- c(order.best)

## a matrix to hold each scored probability
V10.submission02.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- V10.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
    
    ## get the single fit scores
    V10.submission02.probs[,i]  <- V10.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
V10.submission02.mean   <- apply(V10.submission02.probs, 1, mean)
V10.submission02.id     <- V10.list[[1]]$test_score$EventId

## reorder the data
V10.submission02.order      <- order(V10.submission02.mean)
V10.submission02.mean.order <- V10.submission02.mean[V10.submission02.order]
V10.submission02.id.order   <- V10.submission02.id[V10.submission02.order]
V10.submission02.rank.order <- 1:550000


##------------------------------------------------------------------
## S032
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V10_S032"
out.csv       <- paste0(fit.id,"_V10_Submission.csv")
out.rdata     <- paste0(fit.id,"_V10_Submission.Rdata")

## score -> class
thresh.S032            <- 0.945
V10.submission02.class <- ifelse(V10.submission02.mean.order >= thresh.S032, "s", "b")
V10.submission02.S032  <- data.frame(
EventId=V10.submission02.id.order,
RankOrder=V10.submission02.rank.order,
Class=V10.submission02.class)

## save the results
save(V10.submission02.S032, file=paste0(V10.dir,"/",out.rdata) )
write.csv(V10.submission02.S032, file=paste0(V10.dir,"/", out.csv), row.names=FALSE)


##******************************************************************
## Third iteration -- Use 100%
##******************************************************************

order.best  <- order.vec
best.thresh <- thresh.vec[order.best]
best.ams    <- ams.vec[order.best]

## the subset of fits to use
subset <- c(order.best)

## a matrix to hold each scored probability
V10.submission03.probs   <- matrix(0,nrow=550000, ncol=length(subset))

## loop over the chosen fits and load a matrix of probabilities
for (i in 1:length(subset)) {
    
    ## index of the list to use
    fit.idx       <- subset[i]
    
    ## load the file
    fit.filename  <- V10.files[fit.idx]
    fit.id        <- gsub(".Rdata","",fit.filename)
    cat("Using file :", fit.filename,"\n")
    
    ## get the single fit scores
    V10.submission03.probs[,i]  <- V10.list[[fit.id]]$test_score$Prob
    
}

## compute the mean probabilities
V10.submission03.mean   <- apply(V10.submission03.probs, 1, mean)
V10.submission03.id     <- V10.list[[1]]$test_score$EventId

## reorder the data
V10.submission03.order      <- order(V10.submission03.mean)
V10.submission03.mean.order <- V10.submission03.mean[V10.submission03.order]
V10.submission03.id.order   <- V10.submission03.id[V10.submission03.order]
V10.submission03.rank.order <- 1:550000


##------------------------------------------------------------------
## S033
##------------------------------------------------------------------

## output filenames
fit.id        <- "gbm_full_depth9_trees450_shrink0.05_V10_S033"
out.csv       <- paste0(fit.id,"_V10_Submission.csv")
out.rdata     <- paste0(fit.id,"_V10_Submission.Rdata")

## score -> class
thresh.S033            <- 0.944
V10.submission03.class <- ifelse(V10.submission03.mean.order >= thresh.S033, "s", "b")
V10.submission03.S033  <- data.frame(
EventId=V10.submission03.id.order,
RankOrder=V10.submission03.rank.order,
Class=V10.submission03.class)

## save the results
save(V10.submission03.S033, file=paste0(V10.dir,"/",out.rdata) )
write.csv(V10.submission03.S033, file=paste0(V10.dir,"/", out.csv), row.names=FALSE)




## save the intermediate results
#save(V10.list, V10.mat, V10.ams, file="gbm_full_depth9_trees450_shrink0.01_V10_AllScored.Rdata")



