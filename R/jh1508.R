rm(list=ls())
##################################################################
################ K Means ############################################
#####################################################################
EPSILON <- 0.0001 ### If 0/0, choose average vector
n <- 200          ### Training samples
K <- 8            ### Learning Components
CYCLE <- 20       ### Repeated Time
#################################################################
######################### Choose SAMPLE Distributions ###########
TYPESET <- 6      ### Training Sample Type
#################################################################
#################################################################
if(TYPESET==1){   ############### (1) Uniform ####
  x <- matrix(runif(2*n),2,n)
}
end
if(TYPESET==2){   ############### (2) Gaussian
  x <- matrix(c(0.1,0.12,0.12,0.1),nrow=2)  %*% matrix(rnorm(2*n,mean=0.5),nrow=2)
}
if(TYPESET==3){   ############### (3) Gaussian mixtue #####
  x <- matrix(0,2,n)
  x0 <- matrix(c(0, 1, 0, 1, 0.5, 0, 0, 1, 1, 0.5), ncol=5, byrow=T)
  for(i in 1:n){
    j <- 1 + floor(5*runif(1))
    x[,i] <- x0[,j]
  }
  x <- x + 0.05 * rnorm(2*n)
}

if(TYPESET==4){   ############### (4) line %%%
  x <- matrix(0,2,n)
  x[1,] <- runif(n)
  x[2,] <- 4*(0.5-x[1,])^2
}
if(TYPESET==5){   #### (5) circle ###
  x <- matrix(0,2,n)
  x[1,] <- runif(n)
  x[2,] <- sign(2*runif(n)-1) * sqrt(0.25-(x[1,]-0.5)^2)+0.5
}

if(TYPESET==6){
  x <- matrix(0,2,n)
  x[1,] <- runif(n)
  x[2,] <- 0.3*sin(2*pi*x[1,])+0.05*rnorm(n)
}
############################################################
######################## Matrix definition #########
recordy <- array(rep(0,(CYCLE+1)*2*K),dim=c(CYCLE+1,2,K))   ### Recording
ID1 <- matrix(1,1,K)
ID2 <- matrix(1,n,1)
###################### Learning Machine and Record ##########
y <- matrix(runif(2*K),2,K)
recordy[1,,] <- y # array(y,dim=c(2,K))
###################### Learning Begin ###############
for(j in 2:(CYCLE+1)){
  c <- matrix(0,n,K)
  E <- 0
  for(i in 1:n){
    yy <- y - x[,i] %*% ID1
    dist <- apply(yy*yy, 2, sum)
    minval <- min(dist)
    k <- which.min(dist)
    E <- E + t(x[,i]-y[,k]) %*% (x[,i]-y[,k])
    c[i,k] <- 1
  }
  cat('E[', j-1, '] =', E, '\n')
  d <- (c + EPSILON * ID2%*%ID1) / (ID2 %*% apply(c,2,sum)+EPSILON*n)
  y <- x %*% d
  recordy[j,,] <- y # array(y,dim=c(2,K))
}
###################### Learning End ####################
###################### Patition Numbers ################
cat('Partition Numbers: ')
for(k in 1:K){
  cat(sum(c[,k]), ' ')
}
cat('\n')

###################### Draw graph ###################

par(mfrow=c(2,2))
plot(x[1,],x[2,],col='blue',pch=4, main='x: sample')

plot(x[1,],x[2,],col='blue',pch=4, main='K-Mean. x: sample, o: start, square: goal')

for(k in 1:K){
  points(recordy[1,1,k],recordy[1,2,k],col='red',pch=1)
  lines(recordy[,1,k],recordy[,2,k],col='red')
  if(sum(c[,k])==0)
    points(recordy[CYCLE+1,1,k],recordy[CYCLE+1,2,k],col='green', pch=0)
  else
    points(recordy[CYCLE+1,1,k],recordy[CYCLE+1,2,k],col='green', pch=15)
}
############################################
plot(x[1,],x[2,],col='blue', pch=4, main='K-Mean and Voronoi',xlim=c(0,1))
for(k in 1:K){
  if(sum(c[,k])==0)
    points(recordy[CYCLE+1,1,k],recordy[CYCLE+1,2,k],col='green',pch=0)
  else
    points(recordy[CYCLE+1,1,k],recordy[CYCLE+1,2,k],col='red',pch=15)
}

# https://cran.r-project.org/web/packages/deldir/index.html
library(deldir)
vtess <- deldir(y[1,],y[2,])
plot(vtess,wlines="tess", wpoints="none", number=FALSE, add=TRUE, lty=1)
