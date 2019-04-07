#Capstone Project

#Dataset: [Biomechanical Features of Orthopedic Patients]
#(https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients)

setwd("/Users/Raven/Documents/R/Projects/Orthopedic_Patients")
ortho_data <- read.csv("column_2C_weka.csv")

library(tidyverse)
library(caret)
library(dslabs)

#Create a train and test set - the test set will be 25% of the ortho_data dataset
set.seed(1)
ind <- createDataPartition(ortho_data$class, times=1, p=0.25, list=FALSE)
train <- ortho_data[-ind, ]
test <- ortho_data[ind, ]

#Exploratory Data Analysis
dim(train)
dim(test)

head(ortho_data)
dim(ortho_data)

#How many patients are considered abnormal in our dataset?
ortho_data %>% filter(class=="Abnormal") %>% tally

#Let's determine the proportion of patients in our dataset categorized as abmnormal
ggplot() + geom_boxplot(aes(ortho_data$class))

#Let's examine if we see any initial correlation between pelvic_incidence and pelivic_tilt.numeric with class
ortho_data %>% ggplot(aes(pelvic_incidence, pelvic_tilt.numeric, color=class)) + geom_point()

#Now let's view all of these correlations on one plot
pairs(ortho_data[,1:6], col=c("green","blue")[ortho_data$class], pch=5, upper.panel=NULL)

#The corrplot package is used to display these correlations.
#Note that positive correlations are displayed in blue and negative correlations are in red
#Color intensity and the size of the circle are proportional to the correlation coefficients
library(corrplot)
M <- cor(ortho_data[,1:6])
corrplot(M, method="shade", type="lower")
        
#Results

#Begin by using k-nearest neighbors approach
#K-nearest neighbors
knn_fit <- knn3(class~., data=train, k=5)
y_hat_knn <- predict(knn_fit, test, type="class")
confusionMatrix(data=y_hat_knn, reference=test$class)$overall["Accuracy"]
#Using k set to be 5, knn achieves an accuracy of 91%

#Let's determine a k that maximizes accuracy
library(purrr)
set.seed(2019)
ks <- seq(1,125,1)
accuracy <- map_df(ks, function(k){
  fit <- knn3(class ~ ., data = train, k = k)
  
  y_hat <- predict(fit, test, type = "class")
  cm_train <- confusionMatrix(data = y_hat, reference = test$class)
  test_error <- cm_train$overall["Accuracy"]
  
  tibble(test = test_error)
})

ggplot() + geom_line(aes(x = ks, y = accuracy$test), color="blue", size=1)

ks[which.max(accuracy$test)]
max(accuracy$test)
#We see that k=9 produces the maxiumum accuracy of 93.6%

#Method 2: Logistic Regression
#The next method used to explore the dataset is Logistic Regression
library(gplots)
library(ROCR)
glm <- glm(class~., data=train, family="binomial")
pred_glm <- predict(glm, newdata=test, type="response")
pred <- prediction (pred_glm,test$class)
accuracy <- as.numeric(performance(pred,"auc")@y.values)
accuracy

#Let's look at a ROC plot for logistic regression
roc.plot <- performance(pred,"tpr","fpr")
plot(roc.plot, colorize=TRUE)

#Method 3 - Random Forests
#Let's start with the Classification Tree Method
library(rpart)
library(rpart.plot)
rpart_1 <- rpart(class~., data=train, method="class")
#Let's take a look at the classification tree
prp(rpart_1)

#Determine the accuracy of this method
predict_rpart_1 <- predict(rpart_1, newdata = test, type="prob")
roc_tree <- prediction(predict_rpart_1[,2],test$class)
accuracy <- as.numeric(performance(roc_tree,"auc")@y.values)
accuracy
#This is the lowest produced accuracy - only 86.8%

#Let's consider the ROC plot for the Classification Tree Method
roc.plot_tree <- performance(roc_tree,"tpr","fpr")
plot(roc.plot_tree, colorize=TRUE)


library(randomForest)
library(Rborist)
#The estimate will be smoothed by changing the parameter that controls the minimum number of data points in the nodes of the tree. 
#Note that this code may take a few minutes to run
train_rf <- train(class ~ .,
                    method = "Rborist",
                    tuneGrid = data.frame(predFixed = 2, minNode = c(3, 125)),
                    data = train)
confusionMatrix(predict(train_rf, test), test$class)$overall["Accuracy"]
#Here, we see that Random Forest produces an accuracy of 91.0%, still lower than both knn and logistic regression

#Logistic Regression has the highest accuracy, and best ability to distinguish between classes, so it is chosen as the best machine learning algorithm

