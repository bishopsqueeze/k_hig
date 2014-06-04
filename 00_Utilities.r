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
## <function> :: calcAms
##------------------------------------------------------------------
calcAms <- function(y_pred, y_true, w, numw=250000)
{
    wfac    <- numw/length(w)
    s       <- wfac*(w %*% ((as.character(y_true)=="s")*(as.character(y_pred)=="s")))
    b       <- wfac*(w %*% ((as.character(y_true)=="b")*(as.character(y_pred)=="s")))
    ams     <- sqrt(2*((s+b+10)*log(1+s/(b+10))-s))
    
    cat("calcAMS :: wfac=",wfac," sig=",s," bkg=",b," ams=",ams,"\n")
    return(ams)
}


##------------------------------------------------------------------
## <function> :: calcAmsCutoff
##------------------------------------------------------------------
calcAmsCutoff <- function(mycutoff, myscore, y, w)
{
    ## presumes that, on average, score(b) < score(s)
    yhat    <- as.factor(ifelse(myscore > mycutoff, "s", "b"))
    ams     <- calcAms(yhat, y, w)
    return(ams)
}



##------------------------------------------------------------------
## <function> :: amsSummary
##------------------------------------------------------------------
amsSummary  <- function (data, lev = NULL, model = NULL)
{
    cat("Data=", data, "\n")
    wfac    <- 250000/nrow(data)
    
    nb      <- sum(data[, "obs"]== "b")
    ns      <- sum(data[, "obs"]== "s")
    ws      <- rep(692/nrow(data), nrow(data))
    wb      <- rep(411000/nrow(data), nrow(data))
    
    cat("Num s =", ns, "Num =", nb, "\n")
    
    s       <- wfac*(ws %*% ((data[, "obs"]=="s")*(data[, "pred"]=="s")))
    b       <- wfac*(wb %*% ((data[, "obs"]=="b")*(data[, "pred"]=="s")))
    ams     <- sqrt(2*((s+b+10)*log(1+s/(b+10))-s))
    
    out <- c(ams)
    names(out) <- c("AMS")
    out
}


