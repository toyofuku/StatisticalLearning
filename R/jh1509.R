rm(list=ls())
library(jpeg)
library(scatterplot3d)
######################################################
### Image compression by K-Means
######################################################
MATLAB <- 1
#################################
K <- 4          ### K-means Components K <- 4, 8, 12, 16, 20
CYCLE <- 20      ### Repeated Time
REDUCE <- 3     ### Reduce Image
########################################
A <- readJPEG('wtnb/suzukake.jpg')
TATE <- dim(A)[1]
YOKO <- dim(A)[2]
IRO  <- dim(A)[3]
TT <- TATE %/% REDUCE
YY <- YOKO %/% REDUCE
EPSILON <- 0.0001 ### If 0/0, choose average vector
n <- TT*YY        ### Training samples
cat(sprintf('pixels reduced: (%g,%g)-->(%g,%g)\n',TATE,YOKO,TT,YY))
#############################################
B <- array(rep(0,TT*YY*IRO),dim=c(TT,YY,IRO))
for(i in 1:TT){
 for(j in 1:YY){
   B[i,j,] <- as.double(A[REDUCE*i,REDUCE*j,])   # /256
 }
}
x <- matrix(0,IRO,TT*YY)
for(i in 1:TT){
 for(j in 1:YY){
   x[,YY*(i-1)+j] <- B[i,j,]
 }
}
################ K Means ###########
recordy <- array(rep(0,CYCLE*IRO*K), dim=c(CYCLE,IRO,K))
ID1 <- matrix(1, 1,K)
ID2 <- matrix(1, n,1)
###################### Learning Machine and Record ##########
y <- rowMeans(x) %*% ID1 + 0.3 * matrix(rnorm(IRO*K),IRO,K)
recordy[1,,] <- y
###################### Learning Begin ###############
for(j in 2:CYCLE){
 cc <- matrix(0,n,K)
 Err <- 0
 for(i in 1:n){
  yy <- y - x[,i] %*% ID1
  dist <- colSums(yy*yy)
  minval <- min(dist)
  k      <- which.min(dist)
  Err <- Err + t(x[,i]-y[,k]) %*% (x[,i]-y[,k])
  cc[i,k] <- 1
 }
 cat(sprintf('Err[%2g] <- %.2f\n',j-1,Err))
 d <- (cc+EPSILON*ID2 %*% ID1) / ((ID2 %*% colSums(cc))+EPSILON*n)
 y <- x %*% d
 recordy[j,,] <- y
}
###################### Learning End ####################
################################# Draw Image ###################
par(mfrow=c(3,2))

if(exists("rasterImage")) {
  plot(c(0,YOKO),c(0,TATE), type='n')
  rasterImage(A, 0,0,YOKO,TATE)
}
if(exists("rasterImage")) {
  plot(c(0,YY),c(0,TT), type='n')
  rasterImage(B, 0,0,YY,TT)
}

################################# Region judge
for(i in 1:n){
  yy        <- y - x[,i] %*% ID1
  dist      <- apply(yy*yy, 2,sum)
  minval    <- min(dist)
  k         <- which.min(dist)
  ii        <- (i-1) %/% YY + 1
  jj        <- (i-1) %%  YY + 1
  B[ii,jj,] <- y[,k]
}
########################################################
########################################
if(exists("rasterImage")) {
  plot(c(0,YY),c(0,TT), type='n')
  rasterImage(B, 0,0,YY,TT)
}
###################### Patition Numbers ################
cat('Partition Numbers: ')
for(k in 1:K){
  cat(sprintf('%g ',sum(cc[,k])))
}
cat('\n')


barplot(colSums(cc))
title('Partition Numbers')

###################### Draw graph ###################
scatterplot3d(x[1,],x[2,],x[3,],pch='.')
title('RED, GREEN, BLUE')

scatterplot3d(y[1,],y[2,],y[3,],pch=0,color='red')
title('RED, GREEN, BLUE')
####################################################
Rate <- (TT*YY*log2(K)+K*log2(3))/(TT*YY*3*8)
cat(sprintf('Error=%f, Compression Rate=%f\n',Err,Rate))
