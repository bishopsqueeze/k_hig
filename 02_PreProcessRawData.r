##------------------------------------------------------------------
## Do some basic preprocess of the raw data:
##  1. Create binary masks for each variable that contains
##     uninformative data
##  2. Split factors into sets of binary flags
##  3. Swap the uninformative values (-999.000) for NA
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
load("01_HiggsRawTrain.Rdata")
load("01_HiggsRawTest.Rdata")

##------------------------------------------------------------------
## <function> preProcessData
##------------------------------------------------------------------
## The purpose of this function is to do some simple preprocessing
## of the raw data file.  Steps taken include:
##  1. Identify columns containing uninfomative variables, and
##     create a companion variable with a binary flag indicating
##     the presence of an uninformative variable
##  2. Take the one quasi-factor variable pri_jet_num, and create
##     a set of binary flags for the jets
##  3. Exchange the "uninformative" -999 for the equally
##     uninformative NA
##------------------------------------------------------------------
preProcessData <- function(mydt, mycols)
{
    ## number of columns
    nc <- length(mycols)
    
    ##------------------------------------------------------------------
    ## Step 1:  The value -999.000 is known to be "uninformative".  For
    ## each column that contains this value, create an additional column
    ## to hold a binary indicator for the "uninformative" value
    ##------------------------------------------------------------------
    
    ## create a "boolean" data.table for all variables indicating "uninformative"
    dt.bool <- 1*mydt[,lapply(.SD, function(x){(x==-999.000)})]
    
    ## loop over the passed columns
    for (i in 1:nc) {
        
        ## new variable is <old_variable_name>.<fl>
        tmp.col <- mycols[i]
        new.col <- paste(tmp.col,".fl",sep="")
        
        ## add the binary flag if uninformatives exist
        if ( sum(dt.bool[,tmp.col,with=FALSE]) > 0 ) {
            mydt[,eval(new.col):=dt.bool[,tmp.col,with=FALSE]]
        }
    }
    
    ##------------------------------------------------------------------
    ## Step 2:  Split "pri_jet_num" into equivalent binary flags
    ##------------------------------------------------------------------
    
    ## grab the quasi-factor columns
    factor.cols <- c("pri_jet_num")
    
    ## loop over the columns
    for (i in 1:length(factor.cols)) {
        
        ## explode the factor variables
        tmp.col <- factor.cols[i]
        tmp.mat <- expandFactors( as.factor(unlist(mydt[,eval(tmp.col),with=FALSE])), v=tmp.col )
        
        ## load the exploded factors back into the data.table
        for (j in 1:ncol(tmp.mat)) {
            mydt <- mydt[,colnames(tmp.mat)[j]:=tmp.mat[,j],with=FALSE]
        }
        
        ## drop the original column
        mydt[,eval(tmp.col):=NULL]
    }
    
    ##------------------------------------------------------------------
    ## Step 3:  Swap -999 for NA in the data
    ##------------------------------------------------------------------
    mydt    <- mydt[,lapply(.SD, function(x){ifelse((x==-999),NA,x)})]  ## probably a better way to do this
    
    ##------------------------------------------------------------------
    ## return the cleansed data.table
    ##------------------------------------------------------------------
    return(mydt)
}



##******************************************************************
## Main
##******************************************************************

## identify columns used for scoring the training data
eval.cols   <- c("eventid", "label", "weight")

## isolate the data columns
train.cols  <- colnames(train.dt)[ which(!(colnames(train.dt) %in% eval.cols)) ]
test.cols   <- colnames(test.dt)[ which(!(colnames(test.dt) %in% eval.cols)) ]

## isolate evaluation data
train.eval  <- train.dt[,eval.cols,with=FALSE]

## isolate the training data
train.dt    <- train.dt[, -which((colnames(train.dt) %in% c("label", "weight"))), with=FALSE]

## preprocess the training and test data
train.pp    <- preProcessData(mydt=train.dt, mycols=train.cols)
test.pp     <- preProcessData(mydt=test.dt, mycols=test.cols)


##------------------------------------------------------------------
## Save the pre-processed files
##------------------------------------------------------------------
save(train.pp, train.eval, file="02_HiggsPreProcTrain.Rdata")
save(test.pp, file="02_HiggsPreProcTest.Rdata")



