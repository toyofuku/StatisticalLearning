######
######     Learning by a Tensor Machine
######
########################## Preparation ####################
rm(list=ls())
###################### Excersize #######################
n <- 100      ### n=100, 1000, Number of Training samples
CASE <- 2     ### CASE= 1, 2, Case of a true function
#####################################################
NOISE <- 0.1  ### Output noise
Hyperparameter <- 0.0001
##################### Definition of a True function ################
if(CASE==1){
  g <- function(x1,x2){return(x1*x2)}
}
if(CASE==2){
  g <- function(x1,x2){return(2*exp(-3*(x1^2+x2^2)))}
}

par(mfrow=c(2,2))
########################## Figure 1 : True function ######################
D <- seq(-1,1,0.05)
#X1 <- sapply(1:41, function(x){return(D)})
#X2 <- t(X1)
Y <- outer(D,D,g)
persp(D,D,Y,main='True function',theta=0,phi=30,lwd=0.5)
##################### Figure2 : Training Samples #################
x1 <- 2*runif(n) - 1
x2 <- 2*runif(n) - 1
y <- g(x1,x2) + NOISE * runif(n)

library(scatterplot3d)
scatterplot3d(x1,x2,y,main='Training samples',angle=55,grid=T,type='h',color='blue',pch=16)

############################ Definition of a Tensor Learning Machine  ##############
ee <- function(m,x1,x2){return(x1^((m-1) %% 4) * x2^((m-1) %/% 4))}
wee <- function(m,w,x1,x2){return(w[m]*ee(m,x1,x2))}
f <- function(x1,x2,w){
  ret <- matrix(0,length(x1),length(x2))
  for(i in 1:16){
    f_wee <- function(x1,x2){
      return(wee(i,w,x1,x2))
    }
    ret <- ret + outer(x1,x2,f_wee)
  }
  return(ret)
}
############################### Learning Mathematics ###########
### f        =         sum_k w(k)*e_k(x)
### E        = sum_i { sum_k w(k)*e_k(x(i)) - y(i)  }^2
### dE/dw(j) = sum_i ( sum_k w(k)*e_k(x(i)) - y(i)  )*e_j(x(i)) = 0
### sum_k w(k)*{ sum_i e_k(x(i))*e_j(x(i)) } = sum_i y(i)*e_j(x(i))
### A(k,j)   =  sum_i e_k(x(i))*e_j(x(i))
### b(j)     =                                 sum_i y(i)*e_j(x(i))
### w=A^{-1}*b
###################### Learning Process #######################
A <- matrix(0,16,16)
B <- matrix(0,16,1)
for(j in 1:16){
  for(k in 1:16){
    A[j,k] <- sum(ee(j,x1,x2) * ee(k,x1,x2))
  }
}
for(j in 1:16){
  B[j,1] <- sum(y * ee(j,x1,x2))
}
ww <- solve(A+Hyperparameter) %*% B

#################### Figure 3: Output of Trained Learning Machine ################################
Yt <- f(D,D,ww)
persp(D,D,Yt,main='Trained Learning Machine',theta=0,phi=30,lwd=0.5)
################### Figure 4: Generalization Error ####################
Z <- abs(f(D,D,ww) - outer(D,D,g))
persp(D,D,Z,main='Generalization error for each point',theta=0,phi=30,lwd=0.5)
cat('CASE =', CASE, ', n =',n, '\n')
cat('Training Error = ', mean((f(x1,x2,ww)-y)^2), '\n')
cat('Generalization Error =', NOISE^2+sum(sum(Z^2))/41^2, '\n')
################################################################
