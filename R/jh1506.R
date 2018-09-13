rm(list=ls())

# constants
NNN <- 30        ### 10 20 30
NOISEL <- 0.3    ### 0.0 0.1 0.2 0.3
CCC <- 20000     ### PARAMETR for soft margin
SUPPORT <- 0.1   ### 0.1 1 10 100
CYCLE <- 20000   ### Cycle of optimization process
DISPLAY <- 20
INTERVAL <- CYCLE/DISPLAY
ETA <- 0.05      ### Optimization coefficient
LLL <- 100       ### Parameter for Condition, dot(YYY,Alpha)=0
# true parameter
A10 <- 1
A20 <- 1
B00 <- -1
# make inputs and outputs
W00 <- matrix(c(A10,A20), nrow=1, ncol=2)
XXX <- matrix(runif(2*NNN),nrow=2,ncol=NNN)
HHH <- W00 %*% XXX + B00 + NOISEL * (1-2*runif(1*NNN))
YYY <- sign(HHH)
ZZZ <- (YYY+1)/2
#
par(mfrow=c(2,2))
plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), xlab = "x", ylab = "y", main='Blue -1 : Red : +1')
points(XXX[1,YYY==1], XXX[2,YYY==1], col = "red",  pch = 16)
points(XXX[1,YYY==-1], XXX[2,YYY==-1], col = "blue", pch = 16)

# kernel
KXX <- t(XXX) %*% XXX
KYY <- t(YYY) %*% YYY
KYYXX <- KYY * KXX

# initialize optimization
CCCS <- CCC * matrix(1,1,NNN)
MATYX <- KYYXX + diag(1,NNN) + LLL * KYY
AT <- solve(MATYX, matrix(1,NNN,1))
Alpha <- t(AT)
AA <- matrix(0,1,NNN)
# optimization
### Dual Parameters are optimized by steepest descent
III <- matrix(0,1,DISPLAY)
EEE <- matrix(0,1,DISPLAY)
COND <- matrix(0,1,DISPLAY)
for(i in 1:CYCLE){
  ETA2 <- ETA*(1-i/CYCLE)
  TTT <- YYY %*% t(Alpha)
  AB <- Alpha+ETA2*(1-Alpha %*% t(KYYXX) - LLL * TTT[1] * YYY)  ### Dual Parameters are optimized by steepest descent
  AB <- AB - (YYY %*% t(AB))[1]*YYY/NNN
  AB <- (AB+abs(AB))/2              ### make (AB >= 0)
  AB <- (AB+CCCS-abs(AB-CCCS))/2    ### make (AB <= CCCS )
  Alpha <- AB
  if(i %% INTERVAL==0){
    III[i %/% INTERVAL] <- i
    EEE[i %/% INTERVAL] <- Alpha %*% t(matrix(1,1,NNN)) - 0.5 * Alpha %*% KYYXX %*% t(Alpha)
    COND[i %/% INTERVAL] <- 0.5*LLL*TTT * TTT;
  }
}
plot(III,EEE, col='red', main='Horizontal: cycle, Red: Dual Loss')

# WWW and BBB are calculated
cat('Optimization Completed \n')
TTT <- YYY %*% t(Alpha)
CONDITION <- TTT * TTT
cat('Condition =',CONDITION, 'sufficiently small.\n')
#
ALY <- Alpha * YYY
WWW <- ALY %*% t(XXX)
COUNTER <- 0
BBB <- 0.0
plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), xlab = "x", ylab = "y", main='Red: True,  Green: Estimated,  o: Support Vectors')
for(i in 1:NNN){
  if(Alpha[i] > SUPPORT){
    points(XXX[1,i], XXX[2,i], col = "black",  pch = 1)
    BBB <- BBB + YYY[1,i] - WWW %*% XXX[,i]
    COUNTER <- COUNTER+1
  }
}
BBB <- BBB/COUNTER
# display results
points(XXX[1,YYY==1], XXX[2,YYY==1], col = "red",  pch = 4)
points(XXX[1,YYY==-1], XXX[2,YYY==-1], col = "blue", pch = 4)
#
X00 <- seq(0,1,0.05)
Y00 <- seq(0,1,0.05)
X000 <- sapply(1:21, function(x){return(X00)})
Y000 <- t(X000)
Z00 <- A10*X000+A20*Y000+B00
ZZ <- WWW[1]*X000+WWW[2]*Y000+BBB[1]

contour(X00,Y00,ZZ, levels=c(0), col='green', drawlabels=FALSE, add=TRUE)
contour(X00,Y00,Z00, levels=c(0),  col='red', drawlabels=FALSE, add=TRUE)

barplot(Alpha, main='Optimized Values of Dual Variables')

# Generalization Error
NTEST <- 1000
Xtest <- runif(NTEST)
Ytest <- runif(NTEST)
Ztesttrue <- A10*Xtest+A20*Ytest+B00
Ztest <- WWW[1]*Xtest+WWW[2]*Ytest + BBB[1,1]
SUM <- sum( (1+sign(Ztesttrue * Ztest))/2 )
cat("Generalization: Recognition Rate =", SUM/NTEST)
