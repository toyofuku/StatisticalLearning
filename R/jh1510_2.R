##
## 自己組織化写像を用いた日本の市区町村の考察
##
## 謝辞：独立行政法人統計センターのデータを用いた。
## http://www.e-stat.go.jp/SG1/estat/eStatTopPortal.do
## データの著作権は独立行政法人統計センターのページをご覧ください。
##　http://www.e-stat.go.jp/estat/html/spec.html
##　このデータは２０１２年の市区町村の人口である。
## 欠損データを含む市区町村は（利用者により）除外されています。
## 全人口　１５歳未満　１５－６４歳　６５以上　出生　死亡　転入　転出　昼間人口　結婚　離婚


rm(list=ls())

D <- data.matrix( read.csv('wtnb/jh1510_2dat1.csv',header=F) )

################ データの正規化############
NNN <- dim(D)[1]
DIM <- dim(D)[2]          ### NNN 市の数、DIM 特徴量の次元
############# 人口でないデータを人口比で正規化
for(i in 1:NNN){
  for(j in 2:DIM){
    D[i,j] <- D[i,j] / D[i,1]
  }
}
#############  データの次元ごとに平均と標準偏差で正規化
for(d in 1:DIM){
  sum1 <- sum(D[,d])
  sum2 <- sum(D[,d]^2)
  average <- sum1 / NNN
  stddev <- sqrt(sum2 / NNN - average * average)
  D[,d] <- (D[,d] - average) / stddev
}
#################################################
#################################################
################## 自己組織化写像
################## EPSILON2=0 のとき競合学習
###############################################
KURIKAESHI <- 500
KURIKAESHIDISP <- 20
EPSILON1 <- 0.005
EPSILON2 <- 0.003
TATE <- 4
YOKO <- 4
#################################################
trans <- function(m,n){return((m-1) * YOKO + n)}
SOM <- 0.02 * matrix(rnorm(TATE*YOKO*DIM), TATE*YOKO, DIM)
ID <- matrix(1, TATE*YOKO, 1)
####################################　SOM 学習 ############################
for(k in 1:KURIKAESHI){
  mindistsum <- 0
  for(i in 1:NNN){
    ####### find the nearest reference vector
    refv <- apply((SOM - ID %*% D[i,])^2, 1, sum)
    minval <- min(refv)
    index <- which.min(refv)
    miniy <- (index-1) %/% YOKO + 1
    minix <- (index-1) %% YOKO + 1
    mindistsum <- mindistsum + minval
    ############ update SOM ###############################
    SOM[trans(miniy,minix),] <- SOM[trans(miniy,minix),]+EPSILON1*(D[i,]-SOM[trans(miniy,minix),])
    if(miniy>1){
      SOM[trans(miniy-1,minix),] <- SOM[trans(miniy-1,minix),]+EPSILON2*(D[i,]-SOM[trans(miniy-1,minix),])
    }
    if(miniy<TATE){
      SOM[trans(miniy+1,minix),] <- SOM[trans(miniy+1,minix),]+EPSILON2*(D[i,]-SOM[trans(miniy+1,minix),])
    }
    if(minix>1){
      SOM[trans(miniy,minix-1),] <- SOM[trans(miniy,minix-1),]+EPSILON2*(D[i,]-SOM[trans(miniy,minix-1),])
    }
    if(minix<YOKO){
      SOM[trans(miniy,minix+1),] <- SOM[trans(miniy,minix+1),]+EPSILON2*(D[i,]-SOM[trans(miniy,minix+1),])
    }
  }
  if(k %% KURIKAESHIDISP==0){
    cat(sprintf("Total Distance[%d]=%f\n", k, mindistsum))
  }
}
################################## グラフ作成 ############################
par(mfrow=c(TATE,YOKO))
Y <- matrix(0,1,DIM)
for(tatey in 1:TATE){
  for(yokox in 1:YOKO){
#    subplot(TATE,YOKO,(tatey-1)*YOKO+yokox)
    for(d in 1:DIM){
      Y[d] <- SOM[(tatey-1)*YOKO+yokox, d]
    }
    colnames(Y) <- c(1,2,3,4,5,6,7,8,9,10,11)
    barplot(Y, xlim=c(0,length(Y)+1),ylim=c(-1.5,1.5))
  }
}
############################### 具体例 ###############################
### 1 全人口　2子供　3労働者 4老人 5出生　6死亡　7転入　8転出　9昼間人口　10結婚　11離婚
que <- c(
  27,
  70,
  14,
  154,
  171
)
#que(1)=27 ### 東松山市
#que(2)=70 ### 長瀞町
#que(3)=14; ### 中央区
#que(4)=154 ### 杉並区
#que(5)=171 ### 町田市
for(qqq in 1:5){
  refv <- apply((SOM - ID %*% D[que[qqq],])^2, 1, sum)
  index <- which.min(refv)
  miniy <- (index-1) %/% YOKO + 1
  minix <- (index-1) %%  YOKO + 1
  cat(sprintf('City[%g]=(TATE=%g,YOKO=%g)\n', que[qqq], miniy, minix))
}
########################## 代表点に近いデータの個数 #####################
Chosen_times <- matrix(0,TATE,YOKO)
for(qqq in 1:NNN){
  refv <- apply((SOM - ID %*% D[qqq,])^2, 1, sum)
  index <- which.min(refv)
  miniy <- (index-1) %/% YOKO + 1
  minix <- (index-1) %%  YOKO + 1
  Chosen_times[miniy,minix] <- Chosen_times[miniy,minix] + 1
}
print(Chosen_times)
#################################################################
