#############################################################
################ Kohonen's Self Orginizing Map (SOM)
################ One dimension
############################################################

rm(list=ls())

############################################################

n <- 100
K <- 20
TTT <- 500
epsilon1 <- 0.002

DATACASE <- 1  ##### 1 2 3

###############################

x <- matrix(0,n,2)
y <- matrix(0,K,2)
z <- matrix(0,K,2)

################# Training Data ###############

if(DATACASE==1){
   x[,1] <- 4 * runif(n) - 2
   x[,2] <- x[,1]^2 + 0.5 * rnorm(n) + 2
} else if(DATACASE==2){
   RRR <- runif(n)
   x[,1] <- 3*cos(2*pi*RRR) + 0.2 * rnorm(n)
   x[,2] <- 3*sin(2*pi*RRR) + 0.2 * rnorm(n)
} else if(DATACASE==3){
   for(i in 1:(n%/%2)){
      x[i,1] <- 4*( -(n/4)+i )/(n/2) + 0.1 * rnorm(1)
      x[i,2] <- 0.1 * rnorm(1)
   }
   for(i in 1:(n%/%2)){
      x[i+n%/%2,1] <- 0.1 * rnorm(1)
      x[i+n%/%2,2] <- 4*( -(n/4)+i )/(n/2) + 0.1 * rnorm(1)
   }
}

#################### Initialization ###########

for(k in 1:K){
  y[k,1] <- (-k+K/2)/K
  y[k,2] <- 0.5
}

#################### Distance ##############################

dist <- function(i,x,y){
  return((x[i,1]-y[,1])^2 + (x[i,2]-y[,2])^2)
}

##################### Figure 1 ##################################
par(mfrow=c(1,2))
plot(x[,1],x[,2],col='blue',pch=20,ylim=c(0,7))
lines(y[,1],y[,2],col='black')
title('SOM: Training Process')
####################################################

for(kur in 1:TTT){

  ######## SOM Learning
  for(i in 1:n){
     dd <- dist(i,x,y)
     minval <- min(dd)
     mink   <- which.min(dd)
     y[mink,] <- y[mink,] + 2*epsilon1*(x[i,] - y[mink,])
     if(mink>1){
       y[mink-1,] <- y[mink-1,] + epsilon1*(x[i,] - y[mink-1,])
     }
     if(mink>2){
       y[mink-2,] <- y[mink-2,] + epsilon1*(x[i,] - y[mink-2,])
     }
     if(mink<K){
       y[mink+1,] <- y[mink+1,] + epsilon1*(x[i,] - y[mink+1,])
     }
     if(mink<K-1){
       y[mink+2,] <- y[mink+2,] + epsilon1*(x[i,] - y[mink+2,])
     }
  }

  ########## smoothing

  ttt <- kur/2
  for(k in 2:(K-1)){
    z[k,] <- (y[k-1,] + ttt*y[k,] + y[k+1,]) / (ttt+2)
  }
  z[1,] <- (ttt*y[1,] + y[2,]) / (ttt+1)
  z[K,] <- (y[K-1,] + ttt*y[K,]) / (ttt+1)
  y <- z

  ######## Total Distance

  ######## graph

#  pause(0.01)
  Sys.sleep(0.01)

  if(floor(sqrt(kur))==sqrt(kur)){
        lines(y[,1],y[,2],col='red')
        Total <- 0
        for(i in 1:n){
            dd <- dist(i,x,y)
            Total <- Total + min(dd)
        }
        cat(sprintf("[%4g]=%f\n",kur,Total))
  }
}

#################### Figure 2 #########################

plot(x[,1],x[,2],col='blue',pch=20)
lines(y[,1],y[,2],col='red',type='b')
title('SOM: Training Result')
