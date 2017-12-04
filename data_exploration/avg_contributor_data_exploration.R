setwd("C:/Users/Taras/Dropbox/Workspaces/RStudio/data");
data = read.csv("avg-contributor-100000.csv", header = TRUE);
data$contributorID = NULL

# Repository Creation Activity
boxplot(data[, c(1, 2)], outline=FALSE,
        main="Contributor Repository Creation Activity", ylab="Avg. Times Per Week")
boxplot(data[, c(1, 2)], outline=TRUE,
        main="Contributor Repository Creation Activity (with Outliers)", ylab="Avg. Times Per Week")

# Code Pushes And Pull Requests Activity
boxplot(data[, c(3, 4, 5, 6)], outline=FALSE, las=2, par(mar=c(15,4,4,2)),
        main="Contributor Code Pushes And Pull Requests Activity", ylab="Avg. Times Per Week")
boxplot(data[, c(3, 4, 5, 6)], outline=TRUE, las=2, par(mar=c(15,4,4,2)),
        main="Contributor Code Pushes And Pull Requests Activity (with Outliers)", ylab="Avg. Times Per Week")

# Issue Related Activity
boxplot(data[, c(7, 8, 9, 10)], outline=FALSE, las=2, par(mar=c(15,4,4,2)),
        main="Contributor Repository Issues Activity", ylab="Avg. Times Per Week")
boxplot(data[, c(7, 8, 9, 10)], outline=TRUE, las=2, par(mar=c(15,4,4,2)),
        main="Contributor Repository Issues Activity (with Outliers)", ylab="Avg. Times Per Week")