##------------------------------------------------------------------
## Utility functions for the kaggle "social circle" competition:
##------------------------------------------------------------------
## Reference: http://www.kaggle.com/c/higgs-boson
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
## Explode factors into vectors of dummy variables, one for each factor
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
## <function> :: calcAMS
##------------------------------------------------------------------
calcAMS <- function(y_pred, y_true, w, numw=250000)
{
    wfac    <- numw/length(w)
    s       <- wfac*(w %*% ((as.character(y_true)=="s")*(as.character(y_pred)=="s")))
    b       <- wfac*(w %*% ((as.character(y_true)=="b")*(as.character(y_pred)=="s")))
    br      <- 10
    return(qrt(2*(((s+b+br)*log(1+(s/(b+br))))-s)))
}

