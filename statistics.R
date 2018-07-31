gazestatistics <- read_excel("projects/engagement-l2tor/gazestatistics.xlsx", sheet="gaze2stats", col_names = FALSE)
x <- gazestatistics[[3]]
y <- gazestatistics[[6]]
cor.test(x,y)
plot(x,y, main = "NUMBER OF GAZES AT ROBOT VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col="blue")

x <-gazestatistics[[4]]
cor.test(x,y)
plot(x,y, main = "NUMBER OF GAZES AT TABLET VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col="blue")

x <-gazestatistics[[5]]
cor.test(x,y)
plot(x,y, main = "NUMBER OF GAZES AT OTHER VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col="blue")

#first subset, excluding yellow videos and participant with an asian background
gazestatistics <- read_excel("projects/engagement-l2tor/gazestatistics.xlsx", sheet="withoutyellowandasians", col_names = TRUE)
x <- gazestatistics[[3]]
y <- gazestatistics[[6]]
cor.test(x,y)
plot(x,y, main = "NUMBER OF GAZES AT ROBOT VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col = "blue")

x <- gazestatistics[[4]]
y <- gazestatistics[[6]]
cor.test(x,y)
plot(x,y, main = "NUMBER OF GAZES AT TABLET VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col = "blue")

x <- gazestatistics[[5]]
y <- gazestatistics[[6]]
cor.test(x,y)
plot(x,y, main = "NUMBER OF GAZES AT OTHER VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col = "blue")

#second subset, with two scores >0
gazestatistics <- read_excel("projects/engagement-l2tor/gazestatistics.xlsx", sheet="minstens2groterdan1", col_names = TRUE)
x <- gazestatistics[[3]]
y <- gazestatistics[[6]]
cor.test(x,y)
plot(x,y, main = "NUMBER OF GAZES AT ROBOT VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col = "blue")

x <- gazestatistics[[4]]
y <- gazestatistics[[6]]
cor.test(x,y)
plot(x,y, main = "NUMBER OF GAZES AT TABLET VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col = "blue")

x <- gazestatistics[[5]]
y <- gazestatistics[[6]]
cor.test(x,y)
plot(x,y, main = "NUMBER OF GAZES AT OTHER VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col = "blue")

#comparing annotated stuff
gazestatistics <- read_excel("projects/engagement-l2tor/gazestatistics.xlsx", sheet="annotated", col_names = TRUE)
x <- gazestatistics[[3]]
y <- gazestatistics[[6]]
z <- gazestatistics[[7]]
cor.test(x,y)
plot(x,y, main = "NUMBER OF GAZES AT ROBOT VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col = "blue")
plot(z,y, main = "NUMBER OF ANNOTATED GAZES AT ROBOT VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col = "blue")


x <- gazestatistics[[4]]
y <- gazestatistics[[6]]
z <- gazestatistics[[8]]
cor.test(x,y)
plot(x,y, main = "NUMBER OF GAZES AT TABLET VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col = "blue")
plot(z,y, main = "NUMBER OF ANNOTATED GAZES AT TABLET VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col = "blue")
cor.test(z,y)


x <- gazestatistics[[5]]
y <- gazestatistics[[6]]
cor.test(x,y)
plot(x,y, main = "NUMBER OF GAZES AT OTHER VERSUS ENGAGEMENT RATINGS", xlab = "number of gazes", ylab = "engagement", col = "blue")

#whether smiling is correlated with high engagement
smiling <- read_excel("projects/engagement-l2tor/gazestatistics.xlsx", sheet="smiling", col_names=FALSE)
x <- smiling[[2]]
y <- smiling[[3]]
cor.test(x,y)

plot(x,y, main = "PREDICTION OF SMILING VERSUS ENGAGEMENT", xlab = "smiling", ylab = "engagement", col = "blue")

#does movement correlate with engagement?
smiling <- read_excel("projects/engagement-l2tor/gazestatistics.xlsx", sheet="smiling", col_names=TRUE)
x <- smiling[[3]]
y <- smiling[[5]]
cor.test(x,y)
plot(x,y, main = "BODY MOVEMENT VERSUS ENGAGEMENT", ylab = "movement", xlab = "engagement", col = "blue")


z <- smiling[[6]]
cor.test(x,z)
plot(x,z, main = "HEAD MOVEMENT VERSUS ENGAGEMENT", ylab = "movement", xlab = "engagement", col="blue")
