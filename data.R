#Load data
train <- read.csv("archive/train.csv")
test <- read.csv("archive/test.csv")

#shape of the data
dim(train)
dim(test)

#display the first 5 rows of the dataset
head(train, n = 5)
head(test, n = 5)

#print the structure of the dataset
str(train)
str(test)

#print the summary of the dataset
summary(train)
summary(test)


# install the dplyr package
#install.packages("dplyr")

# load the dplyr package
library(dplyr)

glimpse(train)
glimpse(test)

colSums(is.na(train))
colSums(is.na(test))

colMeans(is.na(train)) * 100
colMeans(is.na(test)) * 100

# calculate the mean 
Arrival.Delay.in.Minutes_mean_train <- mean(train$ Arrival.Delay.in.Minutes, na.rm = TRUE)

Arrival.Delay.in.Minutes_mean_test <- mean(test$ Arrival.Delay.in.Minutes, na.rm = TRUE)


print(Arrival.Delay.in.Minutes_mean_train)

print(Arrival.Delay.in.Minutes_mean_test)

# replace missing values  with mean
train$ Arrival.Delay.in.Minutes <- ifelse(is.na(train$ Arrival.Delay.in.Minutes), Arrival.Delay.in.Minutes_mean_train, train$ Arrival.Delay.in.Minutes)
test$ Arrival.Delay.in.Minutes <- ifelse(is.na(test$ Arrival.Delay.in.Minutes), Arrival.Delay.in.Minutes_mean_test, test$ Arrival.Delay.in.Minutes)

sum(is.na(train))
sum(is.na(test))

# drop the columns "cyl" and "am"
train <- select(train, -X, -id)
test <- select(test, -X, -id)

# create a barplot 
barplot(table(train$Class), main = "Customer class", xlab = "Class", ylab = "Count")

# create a barplot 
barplot(table(train$satisfaction), main = "Satisfaction", xlab = "Class", ylab = "Count")


# create a histogram of the "age" column
hist(train$Age, main = "Histogram of Age", xlab = "Age", ylab = "Frequency")

barplot(table(train$Customer.Type), main = "Customer Type", xlab = "Class", ylab = "Count")

barplot(table(train$Gender), main = "Gender", xlab = "Class", ylab = "Count")

unique(train$satisfaction)
unique(test$satisfaction)

train$satisfaction <- ifelse(train$satisfaction == "satisfied", 1, 0)
test$satisfaction <- ifelse(test$satisfaction == "satisfied", 1, 0)

train$Male <- ifelse(train$Gender == "Male", 1, 0)
test$Male <- ifelse(test$Gender == "Male", 1, 0)

train$Loyal.Customer <- ifelse(train$Customer.Type == "Loyal Customer", 1, 0)
test$Loyal.Customer <- ifelse(test$Customer.Type == "Loyal Customer", 1, 0)

train$Business.Travel <- ifelse(train$Type.of.Travel == "Business travel", 1, 0)
test$Business.Travel <- ifelse(test$Type.of.Travel == "Business travel", 1, 0)

# create a new column "class_code" based on the values in the "class" column
train$Class <- ifelse(train$Class == "Eco", 1, ifelse(train$Class == "Eco Plus", 2, 3))

test$Class <- ifelse(test$Class == "Eco", 1, ifelse(test$Class == "Eco Plus", 2, 3))

# drop the columns "cyl" and "am"
train <- select(train, -Gender, -Customer.Type,-Type.of.Travel)
test <- select(test, -Gender, -Customer.Type,-Type.of.Travel)

#print the structure of the dataset
str(train)
str(test)


# install and load the corrplot package
#install.packages("corrplot")
library(corrplot)


# calculate the correlation matrix
corr_matrix <- cor(train)

# create a correlation heatmap
corrplot(corr_matrix, method = "color")


#print the structure of the dataset
str(train)
str(test)

library(rpart)
library(caret)

#levels(test$satisfaction) <- levels(train$satisfaction)

dt_model <- rpart(satisfaction ~ ., data = train, method = "class")

test_predictions <- predict(dt_model, newdata = test, type = "class")

confusionMatrix(test_predictions, as.factor(test$satisfaction))

confusion_matrix <- confusionMatrix(test_predictions, as.factor(test$satisfaction))
# Compute the accuracy, precision, recall, F1-score, etc.
accuracy <- confusion_matrix$overall['Accuracy']
precision <- confusion_matrix$byClass['Precision']
recall <- confusion_matrix$byClass['Recall']
f1_score <- confusion_matrix$byClass['F1']


print(confusion_matrix)
print(accuracy)
print(precision)
print(recall)
print(f1_score)


levels(test_predictions)
levels(as.factor(test$satisfaction))

#install.packages("rpart.plot")

library(rpart.plot)
plot(dt_model)
text(dt_model)
rpart.plot(dt_model, extra = 102, cex = 0.8,main = "Decision Tree for Airline Passenger Satisfication")
prp(dt_model)



library(rpart)
library(caret)

library(randomForest)

# Separate the response variable from the predictors in the training data
train_response <- train$satisfaction
train_predictors <- train[, -which(names(train) == "satisfaction")]

# Separate the response variable from the predictors in the testing data
test_response <- test$satisfaction
test_predictors <- test[, -which(names(test) == "satisfaction")]


str(test_response)
str(test_predictors)



# Set the number of trees to grow in the forest
ntrees <- 50
# Set the number of variables to select at each split
mtry <- sqrt(ncol(train_predictors))
# Set the size of the subsample to use for each tree
sampsize <- floor(nrow(train) * 0.8)
# Train the random forest model
rf_model <- randomForest(x = train_predictors,
                         y = as.factor(train_response),
                         ntree = ntrees,
                         mtry = mtry,
                         sampsize = sampsize, times=10)

# Predict the response variable for the testing data
test_predictions2 <- predict(rf_model, test_predictors)

library(caret)
# Compute the confusion matrix
confusion_matrix <- confusionMatrix(test_predictions2, as.factor(test_response))
# Compute the accuracy, precision, recall, F1-score, etc.
accuracy <- confusion_matrix$overall['Accuracy']
precision <- confusion_matrix$byClass['Precision']
recall <- confusion_matrix$byClass['Recall']
f1_score <- confusion_matrix$byClass['F1']


print(confusion_matrix)
print(accuracy)
print(precision)
print(recall)
print(f1_score)




# plot the random forest model
varImpPlot(rf_model)


barplot(table(train$Online.boarding), main = "Online boarding", xlab = "Class", ylab = "Count")
barplot(table(train$Inflight.wifi.service), main = "Inflight wifi service", xlab = "Class", ylab = "Count")
barplot(table(train$Business.Travel), main = "Business Travel", xlab = "Class", ylab = "Count")
summary(dt_model)


# Assess variable importance
var_imp <- varImp(dt_model)
print(var_imp)
plot(var_imp)


# Example using the randomForestSRC package
library(randomForestSRC)

# Build a Random Forest model
rf_model1 <- rfsrc(satisfaction ~ ., data = train,ntree=50)
# Plot the Random Forest model
plot(rf_model1)

library(partykit)

# Extract a single decision tree from the Random Forest model (e.g., first tree)
tree <- getTree(rf_model, k = 1, labelVar = TRUE)
# Plot the decision tree
plot(tree)

text(tree)


install.packages('party')
library("party")
x <- ctree(satisfaction ~ ., data=train)
plot(x, type="simple")



