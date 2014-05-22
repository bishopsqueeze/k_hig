##------------------------------------------------------------------
## From the kaggle website:
##------------------------------------------------------------------

##------------------------------------------------------------------
## Load libraries
##------------------------------------------------------------------
library(data.table)

##------------------------------------------------------------------
## Clear the workspace
##------------------------------------------------------------------
rm(list=ls())

##------------------------------------------------------------------
## Set the working directory
##------------------------------------------------------------------
setwd("/Users/alexstephens/Development/kaggle/higgs/data/proc")

##------------------------------------------------------------------
## Load the training data
##------------------------------------------------------------------
load("01_HiggsRawTrain.Rdata")
load("01_HiggsRawTest.Rdata")

##------------------------------------------------------------------
## <function> addMissingFlags
##------------------------------------------------------------------
addUninformativeFlags <- function(mydt, mycols)
{
    ## the value -999.000 is known to be "uninformative"
    nc      <- length(mycols)
    dt.bool <- 1*mydt[,lapply(.SD, function(x){(x==-999.000)})]
    
    ## loop over the columns and add a flag if uninformatives exist
    for (i in 1:nc) {
        
        tmp.col <- mycols[i]
        if ( sum(dt.bool[,tmp.col,with=FALSE]) > 0 ) {
            
            new.col <- paste(tmp.col,".fl",sep="")
            mydt[,eval(new.col):=dt.bool[,tmp.col,with=FALSE]]
            
        }
    }
    return(mydt)
}


##------------------------------------------------------------------
## <function> checkUninformative
##------------------------------------------------------------------

checkUninformative  <- function(x)
{
    if (class(x) %in% c("integer")) {
        s <- sum(x == -999)
    } else if (class(x) %in% c("numeric")) {
        s <- sum(x == -999.000)
    } else {
        s <- -1
    }
    return(s)
}

##------------------------------------------------------------------
## Load the training data
##------------------------------------------------------------------

## training data == train.dt

## isolate the data columns
train.cols  <- colnames(train.dt)[ which(!(colnames(train.dt) %in% c("eventid","label","weight")))]
test.cols   <- colnames(test.dt)[ which(!(colnames(test.dt) %in% c("eventid","label","weight")))]

train.fl <- addUninformativeFlags(mydt=train.dt, mycols=train.cols)
test.fl <- addUninformativeFlags(mydt=test.dt, mycols=test.cols)



