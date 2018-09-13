rm(list=ls())

############ Constants ####################
NNN <- 50             ### Training Sample number
BETA <- 5             ### 1 5 25 125 Parameter of Gaussian Kernel
################################################################
NOISEY <- 0.0         ### NOISE LEVEL in Training Samples
CCC <- 1000           ### PARAMETR for soft margin
CYCLE <- 20000        ### Cycle of optimization process
DISPLAY <- 20
INTERVAL <- CYCLE/DISPLAY
ETA <- 0.01           ### Optimization coefficient
LLL <- 1              ### Parameter for Condition, dot(YYY,Alpha)=0
SUPPORT <- 0.01       ### Judgement of support vector
RAND_0 <- 100         ### random seed
Complexity <- 3 * pi  ### complexity of the true distribution
##################### True parameter ############################
A10 <- 1
A20 <- 1
B00 <- -1
######################### make inputs and outputs ###############
XXX <- matrix(0,2,NNN)
XXX[1,] <- runif(NNN)
XXX[2,] <- runif(NNN)
HHH <- 0.5 * A10 * (matrix(1,1,NNN) + sin(Complexity * XXX[1,])) + A20 * XXX[2,] + B00 + NOISEY * (1 - 2 * runif(1*NNN))
YYY <- sign(HHH)
ZZZ <- (YYY+1)/2

par(mfrow=c(2,2))

plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), xlab = "x", ylab = "y", main='Blue -1 : Red : +1')
points(XXX[1,YYY==1], XXX[2,YYY==1], col = "red",  pch = 16)
points(XXX[1,YYY==-1], XXX[2,YYY==-1], col = "blue", pch = 16)

##########################################Kernel##########################
KXX <- matrix(0,NNN,NNN)
for(i in 1:NNN){
  for(j in 1:NNN){
    DXX <- XXX[,i] - XXX[,j]
    KXX[i,j] <- t(DXX) %*% DXX
  }
}
KXX <- exp(-BETA * KXX)
KYY <- t(YYY) %*% YYY
KYYXX <- KYY * KXX
#########################################Initialize Optimization############################
CCCS <- CCC * matrix(1,1,NNN)
MATYX <- KYYXX + diag(1,NNN) + LLL * KYY
AT <- solve(MATYX, matrix(1,NNN,1))
Alpha <- t(AT)
AlphaA <- matrix(0,1,NNN)
##########################################Optimization########################
####### Dual Parameters are optimized by steepest descent ##############
III <- matrix(0,1,DISPLAY)
EEE <- matrix(0,1,DISPLAY)
COND <- matrix(0,1,DISPLAY)
for(i in 1:CYCLE){
  ETA2 <- ETA*(1-i/CYCLE)
  TTT <- YYY %*% t(Alpha)
  AB <- Alpha+ETA2*(1-Alpha %*% t(KYYXX) - LLL * TTT[1] * YYY)  ### Optimization
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
############################################ WWW and BBB are calculated ####################
cat('Optimization Completed \n')
TTT <- YYY %*% t(Alpha)
CONDITION <- TTT * TTT
cat('Condition =',CONDITION, 'should be sufficiently small.\n')
################################### Result Drawing ######
COUNTER <- 0
BBB <- 0.0
plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), xlab = "x", ylab = "y", main='Red: True,  Green: Estimated,  o: Support Vectors')
for(i in 1:NNN){
  if(Alpha[i] > SUPPORT){
    points(XXX[1,i], XXX[2,i], col = "black",  pch = 1)
    UUU <- 0
    for(j in 1:NNN){
      if(Alpha[j] > SUPPORT){
        UUU <- UUU + Alpha[j] * YYY[j] * KXX[j,i]
      }
    }
    BBB <- BBB + YYY[1,i] - UUU
    COUNTER <- COUNTER+1
  }
}
BBB <- BBB/COUNTER

points(XXX[1,YYY==1], XXX[2,YYY==1], col = "red",  pch = 4)
points(XXX[1,YYY==-1], XXX[2,YYY==-1], col = "blue", pch = 4)
####################################
X10 <- seq(0,1,0.05)
X20 <- seq(0,1,0.05)
X100 <- sapply(1:21, function(x){return(X10)})
X200 <- t(X100)
Z00 <- 0.5*A10*(matrix(1,21,21)+sin(Complexity*X100))+A20*X200+B00
####################
ZZZ <- matrix(0,21,21)
for(p in 1:21){
  for(q in 1:21){
    for(i in 1:NNN){
      if(Alpha[i] > SUPPORT){
        D10 <- XXX[1,i] - X100[p,q]
        D20 <- XXX[2,i] - X200[p,q]
        ZZZ[p,q] <- ZZZ[p,q] + Alpha[i] * YYY[i] * exp(-BETA*(D10*D10+D20*D20))
      }
    }
    ZZZ[p,q] <- ZZZ[p,q] + BBB
  }
}

contour(X10,X20,Z00, levels=c(0), col='red',   drawlabels=FALSE, add=TRUE)
contour(X10,X20,ZZZ, levels=c(0), col='green', drawlabels=FALSE, add=TRUE)

barplot(Alpha, main='Optimized Values of Dual Variables')

########################################## Generalization Error
Correctans <- 0
for(p in 1:21){
  for(q in 1:21){
    H00 <- 0.5 * A10 * (1 + sin(Complexity*X100[p,q])) + A20 * X200[p,q] + B00
    Y00 <- sign(H00)
    if(Y00 * ZZZ[p,q] > 0){
      Correctans <- Correctans + 1
    }
  }
}

cat("Number of Support Vectors =", COUNTER,"\n")
cat("Generalization: Recognition Rate =", Correctans / 441, "\n")
