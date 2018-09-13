### 政府統計の総合窓口（ｅ－Ｓｔａｔ）のデータを使わせて頂きました。
### 「ご利用にあたって」のページをお読みの上で著作権に十分に注意してください。
### http://www.e-stat.go.jp/SG1/estat/eStatTopPortal.do
#####################################################
rm(list=ls())
library('pracma')
#####################################################
data <- data.matrix( read.csv('wtnb/hakusai_negi.txt', sep="\t", header=F) )
#####################################################
NUM <- dim(data)[1]    ### 年月数
DDD <- data[,2]        ### 白菜の値段
EEE <- data[,3]        ### ねぎの値段
DIM <- 5               ### 比較するモデルの個数
######################################################
NNN <- NUM              ### 20 25 30 60 120

par(mfrow=c(1,2))
#####################################################
X <- t(DDD[1:NNN])       ### 白菜
Y <- t(EEE[1:NNN])       ### ねぎ
plot(X,Y,col='black') #pch=20
title('X:Hakusai, Y:Negi')
#####################################################
xlim <- c(min(X)-5,max(X)+5)
ylim <- c(min(Y)-5,max(Y)+5)

plot(X,Y,col='black',xlim=xlim,ylim=ylim)

X0 <- seq(min(X)-5, max(X)+5, 5)

TER <- rep(0,DIM)
AIC <- rep(0,DIM)
BIC <- rep(0,DIM)

for(dim in 1:DIM){
  pp  <-  polyfit(X,Y,dim-1) ### Least Square for training sample
  YY  <-  polyval(pp,X)       ### Optimized Output for trained sample

  TER[dim]  <-  sum((Y-YY)^2)
  AIC[dim]  <-  TER[dim] * (1 + 2 * dim/NNN)        ### AIC
  BIC[dim]  <-  TER[dim] * (1 + log(NNN) * dim/NNN) ### BIC

  if(dim == 1)
    Y1  <-  polyval(pp,X0)       ### Estimated Output for test sample
  if(dim == 2)
    Y2  <-  polyval(pp,X0)       ### Estimated Output for test sample
  if(dim == 3)
    Y3  <-  polyval(pp,X0)       ### Estimated Output for test sample
  if(dim == 4)
    Y4  <-  polyval(pp,X0)       ### Estimated Output for test sample
  if(dim == 5)
    Y5  <-  polyval(pp,X0)       ### Estimated Output for test sample
}

par(new=T)
plot(X0,Y1,col='black',type='l',xlim=xlim,ylim=ylim,xlab="",ylab="")
par(new=T)
plot(X0,Y2,col='black',type='l',lty="dashed",xlim=xlim,ylim=ylim,xlab="",ylab="")
par(new=T)
plot(X0,Y3,col='blue',type='l',xlim=xlim,ylim=ylim,xlab="",ylab="")
par(new=T)
plot(X0,Y4,col='green',type='l',xlim=xlim,ylim=ylim,xlab="",ylab="")
par(new=T)
plot(X0,Y5,col='red',type='l',xlim=xlim,ylim=ylim,xlab="",ylab="")


for (dim in 1:DIM) {
  cat(sprintf('DIM[%g]: AIC=%f, BIC=%f\n',dim,AIC[dim],BIC[dim]))
}
title('Hakusai-Negi')
