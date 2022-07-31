### Generate two time series
library(fBasics)
library(rattle)
library("car")

n <- 100 # number of data points
t <- seq(0,10,length.out = 100)
a1 <- 3
a2 <- 4
b1 <- pi
b2 <- 2*pi
c1 <- rnorm(n)
c2 <- rnorm(n)
amp1 <- 1
amp2 <- 1


set.seed(5)
y1 <- a1*sin(b1*t)+c1*amp1 + 0 # time series 1
y2 <- a2*sin(b2*t)+c2*amp2 + 0 # time series 2


# plot results
plot(t, y1, t="l", ylim=range(y1,y2)*c(1,1.2),ylab="y1,y2")
lines(t, y2, col=2)
legend("top", legend=c("y1", "y2"), col=1:2, lty=1, ncol=2, bty="n")

e = y1 - y2
# plot results
plot(t, y1-y2, t="l", ylim=range(e,e)*c(1,1.2),ylab="y1 - y2")


#calculate cross correlation
ccf(y1, y2)

library(urca)

y1 %>% ur.kpss() %>% summary()

y2 %>% ur.kpss() %>% summary()

e %>% ur.kpss() %>% summary()

# Auto-ARIMA models

fit1 <- arima(y1, order=c(1,0,1))
fit2 <- arima(y2, order=c(1,0,1))

fit1$coef
fit2$coef