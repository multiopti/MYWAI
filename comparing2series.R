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

qqPlot(y1)
qqPlot(y2)

# plot results
plot(t, y1, t="l", ylim=range(y1,y2)*c(1,1.2),ylab="y1,y2")
lines(t, y2, col=2)
legend("top", legend=c("y1", "y2"), col=1:2, lty=1, ncol=2, bty="n")

hist(y1)
hist(y2)

df1 <- data.frame(y1,y2)

lapply(df1, basicStats)

df1$ser1 <- 'ser1'
df1$ser2 <- 'ser2'


ks2Test(na.omit(df1[, "y1"]), na.omit(df1[, "y2"]))

locationTest(na.omit(df1[, "y1"]), na.omit(df1[, "y2"]))

varianceTest(na.omit(df1[, "y1"]), na.omit(df1[, "y2"]))

y <- c(y1,y2)
ser <- c(df1[,'ser1'],df1[,'ser2'])
df2 <- data.frame(y,ser)

p01 <- crs %>%
  with(df2) %>%
  dplyr::mutate(ser=as.factor(ser)) %>%
  dplyr::select(y, ser) %>%
  ggplot2::ggplot(ggplot2::aes(x=y)) +
  ggplot2::geom_density(lty=3) +
  ggplot2::geom_density(ggplot2::aes(fill=ser, colour=ser), alpha=0.55) +
  ggplot2::xlab("y\n\n 2022-Jul-24 12:43:21 ") +
  ggplot2::ggtitle("Distribution of y\nby ser") +
  ggplot2::labs(fill="ser", y="Density")

# Display the plots.

gridExtra::grid.arrange(p01)


# Frequency analysis

Fn <- n/(20) # frequency Nyquist

y_sq_sum1 <- sum(y1^2)
y_sq_sum2 <- sum(y2^2)

fft_y1 <- fft(y1)
fft_y_sq_sum1 <- sum(abs(fft(y1))^2)/n

fft_y2 <- fft(y2)
fft_y_sq_sum2 <- sum(abs(fft(y2))^2)/n

print(paste("Check for Parseval's theorem: y_sq_sum1 = ", y_sq_sum1, "; fft_y_sq_sum1 = ", fft_y_sq_sum1, sep=""))

print(paste("Check for Parseval's theorem: y_sq_sum2 = ", y_sq_sum2, "; fft_y_sq_sum2 = ", fft_y_sq_sum2, sep=""))



plot.frequency.spectrum <- function(Xk, xlimits=c(0,length(Xk))) {
  plot.data  <- cbind(0:(length(Xk)-1), Mod(Xk))
  
  plot(plot.data, t="h", lwd=2, main="", 
       xlab="Frequency", ylab="Strength", 
       xlim=xlimits, ylim=c(0,max(Mod(plot.data[,2]))))
}

plot.frequency.spectrum(fft_y1, xlimits=c(0,n/2))
plot.frequency.spectrum(fft_y2, xlimits=c(0,n/2))

