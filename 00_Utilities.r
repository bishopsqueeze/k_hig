##------------------------------------------------------------------
## Utility functions for the kaggle "social circle" competition:
##------------------------------------------------------------------
## Reference: http://www.kaggle.com/c/higgs-boson
##------------------------------------------------------------------



##------------------------------------------------------------------
## <function> :: normalize
##------------------------------------------------------------------
## normalize training data weights
##------------------------------------------------------------------
normalize <- function(weights, labels, s, b, n)
{
    s_norm = s/sum(weights[labels==s_val])
    b_norm = b/sum(weights[labels==b_val])
    return(ifelse(labels==s_val, s_norm*weights, b_norm*weights))
}

##------------------------------------------------------------------
## <function> :: AMS
##------------------------------------------------------------------
## Compute the AMS score
##------------------------------------------------------------------
AMS <-function(pred,real,weight )
{
    pred_s_ind = which(pred==s_val)
    real_s_ind = which(real==s_val)
    real_b_ind = which(real==b_val)
    s = sum(weight[intersect(pred_s_ind,real_s_ind)])
    b = sum(weight[intersect(pred_s_ind,real_b_ind)])
    
    ans = sqrt(2*((s+b+b_tau)*log(1+s/(b+b_tau))-s))
    return(ans)
}

##------------------------------------------------------------------
## <function> :: trim
##------------------------------------------------------------------
## Remove leading/trailing whitespace from a character string
##------------------------------------------------------------------
getAMS <- function(test)
{
    test = test[order(test$scores),]
    
    s <- sum(test$weight[test$label.num==s_val])
    b <- sum(test$weight[test$label.num==b_val])
    
    ams = rep(0,floor(.9*nrow(test)))
    amsMax = 0
    threshold = 0
    for (i in 1:floor(.9*nrow(test))) {
        s = max(0,s)
        b = max(0,b)
        ams[i] = sqrt(2*((s+b+b_tau)*log(1+s/(b+b_tau))-s))
        if (ams[i] > amsMax) {
            amsMax = ams[i]
            threshold = test$scores[i]
        }
        if (test$label.num[i] == s_val)
        s= s-test$weight[i]
        else
        b= b-test$weight[i]
        
    }
    plot(ams,type="l")
    return(data.frame(ams=amsMax,threshold=threshold))
}

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
    #cat("Data=", as.matrix(data), "\n")
    wfac    <- 250000/nrow(data)
    
    nb      <- sum(data[, "obs"]== "b")
    ns      <- sum(data[, "obs"]== "s")
    ws      <- rep(692/250000, nrow(data))
    wb      <- rep(411000/250000, nrow(data))
    
    cat("Num s =", ns, "Num =", nb, "\n")
    
    s       <- wfac*(ws %*% ((data[, "obs"]=="s")*(data[, "pred"]=="s")))
    b       <- wfac*(wb %*% ((data[, "obs"]=="b")*(data[, "pred"]=="s")))
    ams     <- sqrt(2*((s+b+10)*log(1+s/(b+10))-s))
    
    out <- c(ams)
    names(out) <- c("AMS")
    out
}

