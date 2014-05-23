##------------------------------------------------------------------
## Utility functions for the kaggle "social circle" competition:
##------------------------------------------------------------------
## Reference:
## http://www.kaggle.com/c/higgs-boson
##------------------------------------------------------------------

##------------------------------------------------------------------
## <function> :: trim
##------------------------------------------------------------------
## Remove leading/trailing whitespace from a character string
##------------------------------------------------------------------
trim <- function (x)
{
    return(gsub("^\\s+|\\s+$", "", x))
}


##------------------------------------------------------------------
## <function> :: expandFactors
##------------------------------------------------------------------
expandFactors   <- function(x, v="v")
{
    n       <- nlevels(x)
    lvl     <- levels(x)
    mat     <- matrix(, nrow=length(x), ncol=n)
    tmp.v   <- c()
    
    for (i in 1:n) {
        tmp.lvl <- lvl[i]
        tmp.v   <- c(tmp.v, paste(v,".",tmp.lvl,sep=""))
        mat[,i] <- as.integer((x == tmp.lvl))
    }
    colnames(mat) <- tmp.v
    
    return(mat)
}


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



