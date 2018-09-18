rm(list=ls())
library("pracma")
###
##   Steepest Descent Dynamics
###
######################## Constants #############################
AAA <- 5.0
BBB <- 0.02  
CCC <- 2 
VECT <- 0.3
################################################################
KURIKAESHI <- 100
##################################################
GRADFIELD <- 1       ### gradient vector 
OPTIMIZE <- 1        ### optimization
##################################################
#########################################################
ETA <- 0.1           ##### homework     0.01   0.1   0.2
ALPHA <- 0.0         ##### homework     0.0    0.3   0.8
LANGEVIN <- 0.0      ##### homework     0.0    0.5   1.0
#########################################################
X0 <- 1.4            ### initial point
Y0 <- -2.5           ### initial point 
##################################################
V1 <- seq(-9,9,0.5)
NNN <- LANGEVIN * sqrt(2.0*ETA)
X1 <- meshgrid(V1,V1)$X
Y1 <- meshgrid(V1,V1)$Y
### E(x,y) = (5sin(x)-y)^2+0.02x^4-2x^2
Z1 <- AAA * sin(X1) - Y1
Z2 <- Z1 * Z1 + BBB * X1 ^ 4 - CCC * X1 ^ 2
PX1 <- gradient(Z2)$X
PY1 <- gradient(Z2)$Y

par(mfrow=c(1,2))

plot(X0,Y0,xlim=c(-9,9),ylim=c(-9,9), col='red', main='Steepest Descent')
contour(V1, V1, t(Z2), add=TRUE, col='black')
if(GRADFIELD==1){
  quiver(X1,Y1,-VECT*PX1,-VECT*PY1,col='darkgray')
}

###########################################

if(OPTIMIZE==1){
  X2 <- 0.0
  Y2 <- 0.0
  III <- matrix(0,1,KURIKAESHI)
  FFF <- matrix(0,1,KURIKAESHI)
  for(i in 1:KURIKAESHI){
    Z0 <- AAA * sin(X0) - Y0
    DX <- AAA * Z0 * cos(X0) + 4 * BBB * X0 ^ 3 - 2.0 * CCC * X0
    DY <- -2.0 * Z0
    X1 <- X0 - ETA * DX + ALPHA * X2 + NNN * rnorm(1)
    Y1 <- Y0 - ETA * DY + ALPHA * Y2 + NNN * rnorm(1)
    X2 <- X1 - X0
    Y2 <- Y1 - Y0
    Z1 <- AAA * sin(X1) - Y1
    III[i] <- i
    FFF[i] <- Z1 * Z1 + BBB * X1 ^ 4 - CCC * X1 ^ 2
    points(X1,Y1,col='red')
    segments(X0,Y0,X1,Y1,col='red')
    ###  drawnow
    X0 <- X1
    Y0 <- Y1
  }
  plot(III,FFF,type='b',col='blue',pch=15,cex=0.5, xlab='Cycle',ylab='Evaluation')
}
