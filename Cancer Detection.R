library(readr)
library(DescTools)
library(dplyr)
library(ggplot2)
library(Boruta)
library(caret)
library(lubridate)
library(sampling)
library(RWeka)
library(FSelector)
library(rsample)
library(rpart)
library(xgboost)
library(e1071)
library(smotefamily)
library(mltools)
library(pROC)
library(ROSE)


compute_metrics <- function(predictions, actuals, positive_class = "Y") {
  # Create confusion matrix
  conf_matrix <- confusionMatrix(predictions, actuals, positive = positive_class)
  
  # Extract confusion matrix values
  tp <- conf_matrix$table[2, 2]  # True Positives
  tn <- conf_matrix$table[1, 1]  # True Negatives
  fp <- conf_matrix$table[2, 1]  # False Positives
  fn <- conf_matrix$table[1, 2]  # False Negatives
  
  # Compute metrics for positive class (Class 1)
  tpr_pos <- tp / (tp + fn)  # True Positive Rate (Sensitivity, Recall)
  fpr_pos <- fp / (fp + tn)  # False Positive Rate
  precision_pos <- tp / (tp + fp)  # Precision
  recall_pos <- tpr_pos  # Recall
  f_measure_pos <- 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)  # F1 Score
  roc_auc_pos <- AUC(ifelse(actuals == 'Y', 1, 0),ifelse(predictions == 'Y', 1, 0))
  mcc_pos <- mltools::mcc(ifelse(predictions == 'Y', 1, 0), ifelse(actuals == 'Y', 1, 0))
  
  # Compute metrics for negative class (Class 0)
  tpr_neg <- tn / (tn + fp)  # True Negative Rate (Specificity)
  fpr_neg <- fn / (fn + tp)  # False Negative Rate
  precision_neg <- tn / (tn + fn)  # Precision
  recall_neg <- tpr_neg  # Recall
  f_measure_neg <- 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg)  # F1 Score
  roc_auc_neg <- AUC(ifelse(actuals == 'N', 1, 0),ifelse(predictions == 'N', 1, 0))
  mcc_neg <- mltools::mcc(ifelse(predictions == 'N', 1, 0), ifelse(actuals == 'N', 1, 0))
  
  # Weighted averages
  total <- tp + tn + fp + fn
  weight_pos <- (tp + fn) / total
  weight_neg <- (tn + fp) / total
  
  weighted_tpr <- (tpr_pos * weight_pos) + (tpr_neg * weight_neg)
  weighted_fpr <- (fpr_pos * weight_pos) + (fpr_neg * weight_neg)
  weighted_precision <- (precision_pos * weight_pos) + (precision_neg * weight_neg)
  weighted_recall <- (recall_pos * weight_pos) + (recall_neg * weight_neg)
  weighted_f_measure <- (f_measure_pos * weight_pos) + (f_measure_neg * weight_neg)
  weighted_auc <- (roc_auc_pos * weight_pos) + (roc_auc_neg * weight_neg)
  weighted_mcc <- (mcc_pos * weight_pos) + (mcc_neg * weight_neg)
  
  kappa <- conf_matrix$overall['Kappa']
  
  # Create data frames for metrics and confusion matrix
  metrics_df <- data.frame(
    TPR = c(tpr_neg, tpr_pos, weighted_tpr),
    FPR = c(fpr_neg, fpr_pos, weighted_fpr),
    Precision = c(precision_neg, precision_pos, weighted_precision),
    Recall = c(recall_neg, recall_pos, weighted_recall),
    F_measure = c(f_measure_neg, f_measure_pos, weighted_f_measure),
    ROC = c(roc_auc_neg, roc_auc_pos, weighted_auc),
    MCC = c(mcc_neg, mcc_pos, weighted_mcc),
    Kappa = c(kappa, kappa, kappa),
    row.names = c("Class N", "Class Y", "Wt. Average")
  )
  
    metrics_df
}

#Loading the project_data.csv file
pd <- read_csv("project_data.csv")

#View file
View(pd)

#Know more properties about file
dim(pd)
summary(pd)
colnames(pd)
sapply(pd,class)

#check for missing data
cat("Total missing values:")
sum(is.na(pd))
cat("Total missing values in each column:")
colSums(is.na(pd))


#Data Cleaning


#Resolve Inconsistencies 


process_data <- function(pd) {
  
  #IDATE
  pd$IDATE <- mdy(pd$IDATE)
  
  #GENHLTH
  pd$GENHLTH[pd$GENHLTH == 7 | pd$GENHLTH == 9] <- NA
  
  # PHYSHLTH
  pd$PHYSHLTH[pd$PHYSHLTH == 88] <- 0
  pd$PHYSHLTH[pd$PHYSHLTH == 77 | pd$PHYSHLTH == 99] <- NA
 
  
  # MENTHLTH
  pd$MENTHLTH[pd$MENTHLTH == 88] <- 0
  pd$MENTHLTH[pd$MENTHLTH == 77 | pd$MENTHLTH == 99] <- NA
  
  
  #CHILDREN
  pd$CHILDREN[pd$CHILDREN == 88] <- 0
  pd$CHILDREN[pd$CHILDREN == 99] <- NA
  
  #WTKG3
  pd$WTKG3 <- pd$WTKG3/100
  
  
  # ALCDAY5
  pd$ALCDAY5[pd$ALCDAY5 == 777 | pd$ALCDAY5 == 999] <- NA
  pd$ALCDAY5[pd$ALCDAY5 == 888] <- 0
  pd$ALCDAY5 <- ifelse(pd$ALCDAY5 > 100 & pd$ALCDAY5 < 108, (pd$ALCDAY5 - 100) * 4, pd$ALCDAY5)
  pd$ALCDAY5 <- ifelse(pd$ALCDAY5 > 200 & pd$ALCDAY5 < 231, pd$ALCDAY5 - 200, pd$ALCDAY5)
  
  
  # STRENGTH
  
  pd$STRENGTH[pd$STRENGTH == 777 | pd$STRENGTH == 999] <- NA
  pd$STRENGTH[pd$STRENGTH == 888] <- 0
  #Convert weeks to month. Resolve inconsistencies.
  pd$STRENGTH <- ifelse(pd$STRENGTH > 100 & pd$STRENGTH < 200,(pd$STRENGTH-100)*4, pd$STRENGTH)
  pd$STRENGTH <- ifelse(pd$STRENGTH > 200 & pd$STRENGTH < 300, pd$STRENGTH-200 , pd$STRENGTH)
  
  
  
  #Transform attributes in month ( convert from days, weeks to months)
  
  transform_attributes <- function(data, attributes) {
    for (attr in attributes) {
      data[[attr]][data[[attr]] == 777 | data[[attr]] == 999] <- NA
      data[[attr]][data[[attr]] == 555] <- 0
      data[[attr]] <- ifelse(data[[attr]] > 100 & data[[attr]] < 200, (data[[attr]] - 100) * 30, data[[attr]])
      data[[attr]] <- ifelse(data[[attr]] > 200 & data[[attr]] < 300, (data[[attr]] - 200) * 4, data[[attr]])
      data[[attr]] <- ifelse(data[[attr]] > 300 & data[[attr]] < 400, (data[[attr]] - 300), data[[attr]])
      data[[attr]][data[[attr]] == 300] <- 1
    }
    return(data)
  }
  
  # Attributes selected
  pd <- transform_attributes(pd, c("FRUIT2", "FRUITJU2", "FVGREEN1","FRENCHF1","POTATOE1","VEGETAB2"))
  
  
  #Remove values other than 1 and 2 from binary attributes.
  
  coerce_binary <- function(data, attributes) {
    for (attr in attributes) {
      data[[attr]] <- ifelse(data[[attr]] %in% c(1, 2), data[[attr]], NA)
    }
    return(data)
  }
  
  #Attributes selected.
  pd <- coerce_binary(pd, c("HLTHPLN1","MEDCOST","TOLDHI2","CVDINFR4","CVDCRHD4","CVDSTRK3","ASTHMA3","CHCCOPD2","ADDEPEV3","CHCKDNY2","HAVARTH4","VETERAN3","DEAF", "BLIND", "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON", "SMOKE100", "EXERANY2", "FLUSHOT7", "PNEUVAC4", "HIVTST7", "DRNKANY5"))
  
  #Assign NA to data not sure/ don't know or refused.
  
  pd$MARITAL[pd$MARITAL == 9] <- NA
  pd$INCOME2[pd$INCOME2 == 77 | pd$INCOME2 == 99] <- NA
  pd$EMPLOY1[pd$EMPLOY1 == 9] <- NA
  

  return(pd)
}

pd <- process_data(pd)
View(pd)


#check for missing data
sum(is.na(pd))
colSums(is.na(pd))

#Delete Redundant attributes

# Columns to delete
columns_to_delete <- c("IMONTH","IDAY","IYEAR",'HTM4')

pd[columns_to_delete] <- NULL



#Handling missing data. 

#Replace Categorical data with mode

Mode <- function(x) {
  ux <- unique(x)
  tab <- tabulate(match(x, ux))
  ux[tab == max(tab)]
}

#Categories to replace NA with mode.

pd <- pd %>%
  mutate(across(c("SEXVAR","GENHLTH","HLTHPLN1","PERSDOC2","MEDCOST","CHECKUP1","BPHIGH4","CHOLCHK2","TOLDHI2","CVDINFR4","CVDCRHD4","CVDSTRK3","ASTHMA3","CHCCOPD2","ADDEPEV3","CHCKDNY2","DIABETE4","HAVARTH4","MARITAL","EDUCA","RENTHOM1","CPDEMO1B","VETERAN3","EMPLOY1","INCOME2","DEAF","BLIND","DECIDE","DIFFWALK","DIFFDRES","DIFFALON","SMOKE100", "USENOW3", "EXERANY2", "FLUSHOT7", "TETANUS1", "PNEUVAC4", "HIVTST7", "HIVRISK5", "QSTVER", "QSTLANG", "DRNKANY5", "Class"), ~ifelse(is.na(.), Mode(.), .)))



#Replacing numeric data with median (Attributes with skewed distribution).

replace_na_with_median <- function(df, columns) {
  for (column in columns) {
    median_value <- median(df[[column]], na.rm = TRUE)
    df[[column]][is.na(df[[column]])] <- median_value
  }
  return(df)
}

# Columns to replace NA with median
columns_to_replace <- c("PHYSHLTH", "MENTHLTH", "CHILDREN", "ALCDAY5", "STRENGTH", "FRUIT2", "FRUITJU2", "FVGREEN1", "FRENCHF1", "POTATOE1","VEGETAB2")

pd <- replace_na_with_median(pd, columns_to_replace)


#Replacing numeric data with mean (Attributes with symmetrical distribution). 

replace_na_with_rounded_mean <- function(df, columns) {
  for (column in columns) {
    mean_value <- mean(df[[column]], na.rm = TRUE)
    rounded_mean <- round(mean_value)
    df[[column]][is.na(df[[column]])] <- rounded_mean
  }
  return(df)
}

# Columns to replace NA with mean
columns_to_replace <- c("HTIN4","WTKG3","WEIGHT2","HEIGHT3")


pd <- replace_na_with_rounded_mean(pd, columns_to_replace)


View(pd)

#check for missing data
sum(is.na(pd))
colSums(is.na(pd))

#Converting target category into factor


pd$Class <- factor(pd$Class)
sapply(pd, class)

#Data Reduction

## near zero variance

nearZeroVar(pd, names = TRUE)

# Attributes with near zero variance : CVDSTRK3,CHCKDNY2,DIFFDRES,USENOW3,HIVRISK5,QSTLANG. 



## Collinearity

selected_columns <- c( "HTIN4","WTKG3","PHYSHLTH","MENTHLTH","CHILDREN","ALCDAY5","STRENGTH","FRUIT2","FRUITJU2","FVGREEN1","FRENCHF1", "POTATOE1","VEGETAB2")
selected_data <- pd[selected_columns]
corr <- cor(selected_data)
highCorr <- findCorrelation(corr, cutoff = 0.7, names = TRUE)
length(highCorr)
highCorr
#No numeric attributes with more than 70% collinearity.




#Split into training and test dataset

set.seed(31)
split <- initial_split(pd, prop = 0.66, strata = Class)
train <- training(split)
train2<-train
test <- testing(split)



#Create balanced training dataset method 1

#Random under sampling method

table(train$Class)
set.seed(31)
st.1 <- strata(train, stratanames = c("Class"),method = "srswor",size = rep(553, 553))
st <- getdata(train, st.1)
st <- st[, !(names(st) %in% c("ID_unit", "Prob", "Stratum"))]
table(st$Class)
View(st)


#----------------------------------------------------------------------------------

# Information gain
info.gain <- information.gain(Class ~ ., st)
info.gain <- cbind(rownames(info.gain), data.frame(info.gain, row.names=NULL))
names(info.gain) <- c("Attribute", "Info Gain")
sorted.info.gain <- info.gain[order(-info.gain$"Info Gain"), ]
sorted.info.gain
# top 10
ig<-sorted.info.gain[1:10, ]
ig<-ig$Attribute
print("Top 10 Information gain features:")
ig

columns_to_include <- c(ig, "Class")

nt <- st[, columns_to_include]
test_xgboost1<-test[,columns_to_include]
View(nt)

#Models


#m1
#Decision Tree , random undersampling, information gain

modelLookup("J48")
# repeat 10-fold cross-validation 5 times
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)

J48Grid <-  expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))

set.seed(31)
m1 <- train(Class ~ ., data = nt, method = "J48", trControl = train_control,
               tuneGrid = J48Grid)
m1
plot(m1)
m1$bestTune

pred <- predict(m1, test)
print("Confusion matrix of m1 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m2 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m2
#Knn, random undersampling, information gain
modelLookup("knn")
set.seed(31)
m2 <- train(Class ~., data = nt, method = "knn",
                  trControl=train_control,
                  preProcess = c("center", "scale"),
                  tuneLength = 200)

m2
plot(m2)
m2$bestTune

pred <- predict(m2, test)
print("Confusion matrix of m2 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m2 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m3

#nnet, random undersampling, information gain 

modelLookup("nnet")
ctrl <- trainControl(method = "repeatedcv",number=10,repeats=5,summaryFunction = twoClassSummary,classProbs = TRUE,savePredictions = TRUE)
nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1))
set.seed(31)
m3 <- train(x = nt[,-11], 
                 y = nt$Class,
                 method = "nnet",
                 metric = "ROC",                 
                 preProc = c("center", "scale"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 50,
                 MaxNWts = 1000,
                 trControl = ctrl)

m3
plot(m3)
m3$bestTune

pred <- predict(m3, test)

print("Confusion matrix of m3 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m3 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m4
#SVM, random undersampling, information gain 
modelLookup("svmRadial")

svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.1), C = seq(1.0, 2.0, by = 0.2))
set.seed(31)
m4 <- caret::train(Class ~ ., data = nt, method = "svmRadial",
                      preProc = c("center", "scale"),
                      trControl = train_control, tuneGrid = svmGrid)
m4
plot(m4)

m4$bestTune

pred <- predict(m4, test)
print("Confusion matrix of m4 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m4 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m5
# random forest, random undersampling, information gain 
modelLookup("rf")
mtryValues <- seq(2, ncol(nt)-1, by = 1)
set.seed(31)
m5 <- caret::train(x = nt[,-11], 
                      y = nt$Class,
                      method = "rf",
                      ntree = 100,
                      tuneGrid = data.frame(mtry = mtryValues),
                      importance = TRUE,
                      metric = "ROC",
                      trControl = ctrl)
m5
plot(m5)


imp <- varImp(rfFit)
imp

pred <- predict(m5, test)
print("Confusion matrix of m5 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m5 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m6
# XGBoost, random undersampling, information gain 
xgb_control = trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                           summaryFunction = defaultSummary)

xgbGrid <- expand.grid(
  nrounds = seq(from= 100, to= 300, by= 100),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)
set.seed(31)
m6 <- caret::train(x = nt[, -11], 
                         y = nt$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         verbose = FALSE,
                         trControl = xgb_control)
m6
plot(m6)
m6$bestTune
pred <- predict(m6, test_xgboost1)
print("Confusion matrix of m6 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m6 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#-----------------------------------------------------------------------------------------

#Correlation-based Feature Selection
# cfs
subset <- cfs(Class ~., st)
subset
columns_to_include <- c(subset, "Class")
cfssize<-length(subset)+1
cfssize

nf <- st[, columns_to_include]
test_xgboost1<-test[,columns_to_include]
View(nf)



#Models

#m7
#Decision Tree, random undersampling, cfs 

modelLookup("J48")
# repeat 10-fold cross-validation 5 times
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)

J48Grid <-  expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
set.seed(31)
m7 <- train(Class ~ ., data = nf, method = "J48", trControl = train_control,
            tuneGrid = J48Grid)
m7
plot(m7)
m7$bestTune

pred <- predict(m7, test)
print("Confusion matrix of m7 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m7 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m8
#Knn, random undersampling, cfs
modelLookup("knn")
set.seed(31)
m8 <- train(Class ~., data = nf, method = "knn",
            trControl=train_control,
            preProcess = c("center", "scale"),
            tuneLength = 200)

m8
plot(m8)
m8$bestTune

pred <- predict(m8, test)
print("Confusion matrix of m8 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m8 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m9
#nnet , random undersampling, cfs

modelLookup("nnet")

ctrl <- trainControl(method = "repeatedcv",number=10,repeats=5,summaryFunction = twoClassSummary,classProbs = TRUE,savePredictions = TRUE)
nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1))
set.seed(31)
m9 <- train(x = nf[,-(cfssize)], 
            y = nf$Class,
            method = "nnet",
            metric = "ROC",                 
            preProc = c("center", "scale"),
            tuneGrid = nnetGrid,
            trace = FALSE,
            maxit = 50,
            MaxNWts = 1000,
            trControl = ctrl)

m9
plot(m9)
m9$bestTune


pred <- predict(m9, test)
print("Confusion matrix of m9 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m9 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m10
#SVM, random undersampling, cfs
modelLookup("svmRadial")
svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.1), C = seq(1.0, 2.0, by = 0.2))
set.seed(31)
m10 <- caret::train(Class ~ ., data = nf, method = "svmRadial",
                   preProc = c("center", "scale"),
                   trControl = train_control, tuneGrid = svmGrid)
m10
plot(m10)

m10$bestTune

pred <- predict(m10, test)
print("Confusion matrix of m10 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m10 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m11
# random forest, random undersampling, cfs
modelLookup("rf")

mtryValues <- seq(2, ncol(nt)-1, by = 1)
set.seed(31)
m11 <- caret::train(x = nf[,-(cfssize)], 
                   y = nf$Class,
                   method = "rf",
                   ntree = 100,
                   tuneGrid = data.frame(mtry = mtryValues),
                   importance = TRUE,
                   metric = "ROC",
                   trControl = ctrl)
m11
plot(m11)


imp <- varImp(rfFit)
imp


pred <- predict(m11, test)
print("Confusion matrix of m11 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m11 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
View(matreturn)

#m12
# XGBoost, random undersampling, cfs

xgb_control = trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                           summaryFunction = defaultSummary)
set.seed(31)

xgbGrid <- expand.grid(
  nrounds = seq(from= 100, to= 300, by= 100),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)

m12 <- caret::train(x = nf[,-(cfssize)], 
                   y = nf$Class,
                   method = "xgbTree",
                   tuneGrid = xgbGrid,
                   verbose = FALSE,
                   trControl = xgb_control)
m12
plot(m12)
m12$bestTune

pred <- predict(m12, test_xgboost1)
print("Confusion matrix of m12 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m12 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#---------------------------------------------------------------------------------------------

# Boruta
cancer.boruta <- Boruta(Class~.,data=st)
cancer.boruta
cb<-getSelectedAttributes(cancer.boruta)
cb
cbsize<-length(cb)+1
cbsize

columns_to_include <- c(cb, "Class")

test_xgboost1<-test[,columns_to_include]
nb <- st[, columns_to_include]
View(nb)
dim(nb)

#Models

#m13
#Decision Tree , random undersampling, Boruta

modelLookup("J48")
# repeat 10-fold cross-validation 5 times
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)

J48Grid <-  expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
set.seed(31)
m13 <- train(Class ~ ., data = nb, method = "J48", trControl = train_control,
            tuneGrid = J48Grid)
m13
plot(m13)
m1$bestTune

pred <- predict(m13, test)
print("Confusion matrix of m13 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m13 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m14
#Knn, random undersampling, Boruta
modelLookup("knn")
set.seed(31)
m14 <- train(Class ~., data = nb, method = "knn",
            trControl=train_control,
            preProcess = c("center", "scale"),
            tuneLength = 200)

m14
plot(m14)
m14$bestTune

pred <- predict(m14, test)
print("Confusion matrix of m14 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m14 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn


#m15
#nnet,random undersampling, Boruta

modelLookup("nnet")

ctrl <- trainControl(method = "repeatedcv",number=10,repeats=5,summaryFunction = twoClassSummary,classProbs = TRUE,savePredictions = TRUE)
nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1))
set.seed(31)
m15 <- train(x = nb[,-(cbsize)], 
            y = nb$Class,
            method = "nnet",
            metric = "ROC",                 
            preProc = c("center", "scale"),
            tuneGrid = nnetGrid,
            trace = FALSE,
            maxit = 50,
            MaxNWts = 1000,
            trControl = ctrl)

m15
plot(m15)
m3$bestTune

pred <- predict(m15, test)
print("Confusion matrix of m15 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m15 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m16
#SVM, random undersampling, Boruta
modelLookup("svmRadial")

svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.1), C = seq(1.0, 2.0, by = 0.2))
set.seed(31)
m16 <- caret::train(Class ~ ., data = nb, method = "svmRadial",
                   preProc = c("center", "scale"),
                   trControl = train_control, tuneGrid = svmGrid)
m16
plot(m16)

m16$bestTune

pred <- predict(m16, test)
print("Confusion matrix of m16 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m16 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m17
# random forest, random undersampling, Boruta
modelLookup("rf")
set.seed(31)
mtryValues <- seq(2, ncol(nb)-1, by = 1)
m17 <- caret::train(x = nb[,-(cbsize)], 
                   y = nb$Class,
                   method = "rf",
                   ntree = 100,
                   tuneGrid = data.frame(mtry = mtryValues),
                   importance = TRUE,
                   metric = "ROC",
                   trControl = ctrl)
m17
plot(m17)


imp <- varImp(rfFit)
imp

pred <- predict(m17, test)
print("Confusion matrix of m17 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m17 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m18
# XGBoost, random undersampling, Boruta

xgb_control = trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                           summaryFunction = defaultSummary)
set.seed(31)

xgbGrid <- expand.grid(
  nrounds = seq(from= 100, to= 300, by= 100),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)

m18 <- caret::train(x = nb[, -(cbsize)], 
                   y = nb$Class,
                   method = "xgbTree",
                   tuneGrid = xgbGrid,
                   verbose = FALSE,
                   trControl = xgb_control)
m18
plot(m18)
m18$bestTune

pred <- predict(m18, test_xgboost1)
print("Confusion matrix of m18 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m18 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn



#Create balanced training dataset method 2

#ovun.sample 


train2<-train
sapply(train2,class)
dim(train2)
table(train2$Class)
set.seed(31)
csd <- ovun.sample(Class ~ ., data=train2, method="under",p=0.5,seed=1)$data
table(csd$Class)
View(csd)
dim(csd)


#----------------------------------------------------------------------------------

# Information gain
info.gain <- information.gain(Class ~ ., csd)
info.gain <- cbind(rownames(info.gain), data.frame(info.gain, row.names=NULL))
names(info.gain) <- c("Attribute", "Info Gain")
sorted.info.gain <- info.gain[order(-info.gain$"Info Gain"), ]
sorted.info.gain
# top 10
ig<-sorted.info.gain[1:10, ]
ig<-ig$Attribute
ig

columns_to_include <- c(ig, "Class")

nt <- csd[, columns_to_include]
nrow(nt)
test_xgboost1<-test[,columns_to_include]
View(nt)


#Models

#m19
#Decision Tree , ovun.sample, Information gain

modelLookup("J48")
set.seed(31)
# repeat 10-fold cross-validation 5 times
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)

J48Grid <-  expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
m19 <- train(Class ~ ., data = nt, method = "J48", trControl = train_control,
            tuneGrid = J48Grid)
m19
plot(m19)
m19$bestTune

pred <- predict(m19, test)
print("Confusion matrix of m19 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m19 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn


#m20
#Knn, ovun.sample, Information gain
modelLookup("knn")
set.seed(31)
m20 <- train(Class ~., data = nt, method = "knn",
            trControl=train_control,
            preProcess = c("center", "scale"),
            tuneLength = 200)

m20
plot(m20)
m20$bestTune

pred <- predict(m20, test)
print("Confusion matrix of m20 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m20 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn


#m21
#nnet , ovun.sample, Information gain

modelLookup("nnet")
ctrl <- trainControl(method = "repeatedcv",number=10,repeats=5,summaryFunction = twoClassSummary,classProbs = TRUE,savePredictions = TRUE)
nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1))
set.seed(31)
m21 <- train(x = nt[,-11], 
            y = nt$Class,
            method = "nnet",
            metric = "ROC",                 
            preProc = c("center", "scale"),
            tuneGrid = nnetGrid,
            trace = FALSE,
            maxit = 50,
            MaxNWts = 1000,
            trControl = ctrl)

m21
plot(m21)
m21$bestTune

pred <- predict(m21, test)
print("Confusion matrix of m21 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m21 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn

#m22
#SVM, ovun.sample, Information gain
modelLookup("svmRadial")
svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.1), C = seq(1.0, 2.0, by = 0.2))
set.seed(31)
m22 <- caret::train(Class ~ ., data = nt, method = "svmRadial",
                   preProc = c("center", "scale"),
                   trControl = train_control, tuneGrid = svmGrid)
m22
plot(m22)

m22$bestTune

pred <- predict(m22, test)
print("Confusion matrix of m22 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m22 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn


#m23
# random forest, ovun.sample, Information gain
modelLookup("rf")
mtryValues <- seq(2, ncol(nt)-1, by = 1)
set.seed(31)
m23 <- caret::train(x = nt[,-11], 
                   y = nt$Class,
                   method = "rf",
                   ntree = 100,
                   tuneGrid = data.frame(mtry = mtryValues),
                   importance = TRUE,
                   metric = "ROC",
                   trControl = ctrl)
m23
plot(m23)


imp <- varImp(rfFit)
imp

pred <- predict(m23, test)
print("Confusion matrix of m23 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m23 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn


#m24
# XGBoost, ovun.sample, Information gain

xgb_control = trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                           summaryFunction = defaultSummary)

xgbGrid <- expand.grid(
  nrounds = seq(from= 100, to= 300, by= 100),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)
set.seed(31)
m24 <- caret::train(x = nt[, -11], 
                   y = nt$Class,
                   method = "xgbTree",
                   tuneGrid = xgbGrid,
                   verbose = FALSE,
                   trControl = xgb_control)
m24
plot(m24)
m6$bestTune

pred <- predict(m24, test_xgboost1)
print("Confusion matrix of m24 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m24 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn


#-----------------------------------------------------------------------------------------

#Correlation-based Feature Selection
# cfs
subset <- cfs(Class ~., csd)
subset
columns_to_include <- c(subset, "Class")
cfssize<-length(subset)+1
cfssize

nf <- csd[, columns_to_include]
test_xgboost1<-test[,columns_to_include]
View(nf)

#Models

#m25
#Decision Tree , ovun.sample, cfs

modelLookup("J48")
# repeat 10-fold cross-validation 5 times
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)

J48Grid <-  expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
set.seed(31)
m25 <- train(Class ~ ., data = nf, method = "J48", trControl = train_control,
            tuneGrid = J48Grid)
m25
plot(m25)
m25$bestTune

pred <- predict(m25, test)
print("Confusion matrix of m25 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m25 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn


#m26
#Knn, ovun.sample, cfs
modelLookup("knn")
set.seed(31)
m26 <- train(Class ~., data = nf, method = "knn",
            trControl=train_control,
            preProcess = c("center", "scale"),
            tuneLength = 200)

m26
plot(m26)
m8$bestTune

pred <- predict(m26, test)
print("Confusion matrix of m26 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m26 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn


#m27
#nnet , ovun.sample, cfs

modelLookup("nnet")
ctrl <- trainControl(method = "repeatedcv",number=10,repeats=5,summaryFunction = twoClassSummary,classProbs = TRUE,savePredictions = TRUE)
nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1))
set.seed(31)
m27 <- train(x = nf[,-(cfssize)], 
            y = nf$Class,
            method = "nnet",
            metric = "ROC",                 
            preProc = c("center", "scale"),
            tuneGrid = nnetGrid,
            trace = FALSE,
            maxit = 50,
            MaxNWts = 1000,
            trControl = ctrl)

m27
plot(m27)
m27$bestTune

pred <- predict(m27, test)
print("Confusion matrix of m27 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m27 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn


#m28
#SVM, ovun.sample, cfs
modelLookup("svmRadial")
svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.1), C = seq(1.0, 2.0, by = 0.2))
set.seed(31)
m28 <- caret::train(Class ~ ., data = nf, method = "svmRadial",
                    preProc = c("center", "scale"),
                    trControl = train_control, tuneGrid = svmGrid)
m28
plot(m28)

m28$bestTune

pred <- predict(m28, test)
print("Confusion matrix of m28 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m28 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn


#m29
# random forest, ovun.sample, cfs
modelLookup("rf")
mtryValues <- seq(2, ncol(nt)-1, by = 1)
set.seed(31)
m29 <- caret::train(x = nf[,-(cfssize)], 
                    y = nf$Class,
                    method = "rf",
                    ntree = 100,
                    tuneGrid = data.frame(mtry = mtryValues),
                    importance = TRUE,
                    metric = "ROC",
                    trControl = ctrl)
m29
plot(m29)


imp <- varImp(rfFit)
imp


pred <- predict(m29, test)
print("Confusion matrix of m29 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m29 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')


matreturn<-compute_metrics(pred, test$Class)
matreturn


#m30
# XGBoost, ovun.sample, cfs

xgb_control = trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                           summaryFunction = defaultSummary)

xgbGrid <- expand.grid(
  nrounds = seq(from= 100, to= 300, by= 100),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)
set.seed(31)
m30 <- caret::train(x = nf[,-(cfssize)], 
                    y = nf$Class,
                    method = "xgbTree",
                    tuneGrid = xgbGrid,
                    verbose = FALSE,
                    trControl = xgb_control)
m30
plot(m30)
m30$bestTune

pred <- predict(m30, test_xgboost1)
print("Confusion matrix of m30 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m30 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn


#---------------------------------------------------------------------------------------------

# Boruta
cancer.boruta <- Boruta(Class~.,data=csd)
cancer.boruta
cb<-getSelectedAttributes(cancer.boruta)
cb
cbsize<-length(cb)+1
cbsize

columns_to_include <- c(cb, "Class")

nb <- csd[, columns_to_include]
test_xgboost1<-test[,columns_to_include]
View(nb)
dim(nb)

#Models

#m31
#Decision Tree , ovun.sample, Boruta

modelLookup("J48")
# repeat 10-fold cross-validation 5 times
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)

J48Grid <-  expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
set.seed(31)
m31 <- train(Class ~ ., data = nb, method = "J48", trControl = train_control,
             tuneGrid = J48Grid)
m31
plot(m31)
m31$bestTune

pred <- predict(m31, test)
print("Confusion matrix of m31 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m31 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn


#m32
#Knn, ovun.sample, Boruta
modelLookup("knn")
set.seed(31)
m32 <- train(Class ~., data = nb, method = "knn",
             trControl=train_control,
             preProcess = c("center", "scale"),
             tuneLength = 200)

m32
plot(m32)
m32$bestTune


pred <- predict(m32, test)
print("Confusion matrix of m32 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m32 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
matreturn


#m33
#nnet , ovun.sample, Boruta

modelLookup("nnet")
ctrl <- trainControl(method = "repeatedcv",number=10,repeats=5,summaryFunction = twoClassSummary,classProbs = TRUE,savePredictions = TRUE)
nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1))
set.seed(31)
m33 <- train(x = nb[,-(cbsize)], 
             y = nb$Class,
             method = "nnet",
             metric = "ROC",                 
             preProc = c("center", "scale"),
             tuneGrid = nnetGrid,
             trace = FALSE,
             maxit = 50,
             MaxNWts = 1000,
             trControl = ctrl)

m33
plot(m33)
m33$bestTune

pred <- predict(m33, test)
print("Confusion matrix of m33 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m33 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
View(matreturn)


#m34
#SVM, ovun.sample, Boruta
modelLookup("svmRadial")
svmGrid <-  expand.grid(sigma = seq(0.1, 0.4, by = 0.1), C = seq(1.0, 2.0, by = 0.2))
set.seed(31)
m34 <- caret::train(Class ~ ., data = nb, method = "svmRadial",
                    preProc = c("center", "scale"),
                    trControl = train_control, tuneGrid = svmGrid)
m34
plot(m34)

m34$bestTune


pred <- predict(m34, test)
print("Confusion matrix of m34 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m34 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
View(matreturn)


#m35
# random forest, ovun.sample, Boruta
modelLookup("rf")
mtryValues <- seq(2, ncol(nb)-1, by = 1)
set.seed(31)
m35 <- caret::train(x = nb[,-(cbsize)], 
                    y = nb$Class,
                    method = "rf",
                    ntree = 100,
                    tuneGrid = data.frame(mtry = mtryValues),
                    importance = TRUE,
                    metric = "ROC",
                    trControl = ctrl)
m35
plot(m35)


imp <- varImp(rfFit)
imp

pred <- predict(m35, test)
print("Confusion matrix of m35 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m35 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')

matreturn<-compute_metrics(pred, test$Class)
View(matreturn)


#m36
# XGBoost, ovun.sample, Boruta

xgb_control = trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                           summaryFunction = defaultSummary)

xgbGrid <- expand.grid(
  nrounds = seq(from= 100, to= 300, by= 100),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)
set.seed(31)
m36 <- caret::train(x = nb[, -(cbsize)], 
                    y = nb$Class,
                    method = "xgbTree",
                    tuneGrid = xgbGrid,
                    verbose = FALSE,
                    trControl = xgb_control)
m36
plot(m36)
m36$bestTune


pred <- predict(m36, test_xgboost1)
print("Confusion matrix of m36 when class Y is positive")
confusionMatrix(pred, test$Class,positive = 'Y')

print("Confusion matrix of m36 when class N is positive")
confusionMatrix(pred, test$Class,positive = 'N')


matreturn<-compute_metrics(pred, test$Class)
View(matreturn)






