#load in libraries
library(data.table)
library(caret)
library(Metrics)
library(Rtsne)
library(ggplot2)
library(ClusterR)
library(glmnet)
library(xgboost)
library(randomForest)

#read in data
#setwd("./desktop/R/kaggle/titanic/")
train<-fread("./project/volume/data/interim/train.csv")
test<-fread("./project/volume/data/interim/test.csv")
submission<-fread("./project/volume/data/raw/gender_submission.csv")

# make train and test have the same columns
test$Survived<-0


#####################################################################
###################        XGBOOST PREDICT        ###################
#####################################################################
#Save response value for XGboost
y.train<-train$Survived

#build Dmatrix
train<-as.matrix(train)
test<-as.matrix(test)

d.train <- xgb.DMatrix(train,label=y.train)
d.test <- xgb.DMatrix(test)

########################
# Use cross validation #
########################

param <- list(  objective           = "binary:logistic",
                gamma               = 0,
                eval_metric         = "error",
                eta                 = 0.0001, #how different each tree should be >> range [0,1]
                max_depth           = 7, #how many branches
                subsample           = 0.9, 
                colsample_bytree    = 0.2,
                min_child_weight    = 1
)

# use CV to get some good nrounds
testing <- xgb.cv(data=d.train, params = param,
                  nfold=200, nrounds=2000,
                  verbose = T, maximize=FALSE, early_stopping_rounds = 100)

####################################
# fit the model to all of the data #
####################################
watchlist <- list(train = d.train)

#now fit the full model
XGBm<-xgboost(data=d.train, params=param, nrounds=71, watchlist, print_every_n=1)

pred <- predict(XGBm, d.test)

#frame the table
final = data.frame(PassengerId = submission$PassengerId, Survived = pred)

final$Survived = ifelse(final$Survived > 0.5, 1,0)

#now we can write out a submission
fwrite(final,"./project/volume/data/processed/submit5.csv")


#####################################################################
################        Random Forest PREDICT        ################
#####################################################################
set.seed(1234)
rf_model <- randomForest(factor(Survived) ~ ., data = train)

rf.fitted = predict(rf_model)
ans_rf = rep(NA,891)
for(i in 1:891){
  ans_rf[i] = as.integer(rf.fitted[[i]]) - 1
}

# Result
table(ans_rf)

prediction <- predict(rf_model, test)

# Solution 2 columns (prediction)
solution <- data.frame(PassengerID = submission$PassengerId, Survived = prediction)

#now we can write out a submission
fwrite(solution,"./project/volume/data/processed/submit_RF1.csv")

# Error
plot(rf_model, ylim=c(0,0.36), main = 'RF_MODEL')
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)


#####################################################################
################        Naive Bayesian PREDICT        ###############
#####################################################################
fitControl <- trainControl(method = "cv", number = 10, savePredictions = TRUE)

set.seed(1234) # Set random seed

nb_model <- train(as.factor(Survived) ~ ., 
                  data = train,
                  method = "nb", # Specify navie bayes model
                  trControl = fitControl)

confusionMatrix(nb_model)

prediction_nb <- predict(nb_model, test)

solution_nb <- data.frame(PassengerID = test$PassengerId, Survived = prediction_nb)

#now we can write out a submission
fwrite(solution_nb,"./project/volume/data/processed/submit_NB.csv")


#####################################################################
#############        Logistic Regression PREDICT        #############
#####################################################################
set.seed(1234) # Set random seed

lr_model <- train(factor(Survived) ~ .,
                  data = train,
                  method = 'glm', 
                  family = binomial(),
                  trControl = fitControl)

confusionMatrix(lr_model)

prediction_lr <- predict(lr_model, test)
solution_lr <- data.frame(PassengerID = test$PassengerId, Survived = prediction_lr)

#now we can write out a submission
fwrite(solution_lr,"./project/volume/data/processed/submit_LR.csv")



##Rerference_Random Forest, Naive Bayesian, Logistic Regression.
#https://www.kaggle.com/vincentlugat/titanic-data-analysis-rf-prediction-0-81818
#https://www.kaggle.com/tavoosi/predicting-survival-on-the-titanic-with-rf-lr-nb#machine-learning



