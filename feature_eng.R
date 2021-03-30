#load in libraries
library(data.table)
library(caret)
library(Metrics)
library(ggplot2)


#read in data
setwd("./desktop/R/kaggle/titanic/")
train<-fread("./project/volume/data/raw/train.csv")
test<-fread("./project/volume/data/raw/test.csv")

#check out the columns and missing values
#head(train,10) #NAs in Age variable, missing values in Cabin variable
#head(test,10) #NAs in Age variable, missing values in Cabin variable

#combine dataset for time-saving
master<-rbind(train,test,fill=TRUE)

########################################################
##draw bar chart to check how important the values are##
########################################################

#0. remove 'Ticket' variable since it doesn`t look like important
master$Ticket<-NULL

#1-1. check out Sex variable.
(ggplot(master,aes(master$Survived,fill=master$Sex))
  +geom_bar()
  +xlab("Survived")
  +ylab("number of people")
  +guides(fill=guide_legend(title="Sex")))
#looks like women did more survived than men and this is plausible result
#because usually men let women and children take a boat first.

#1-2. change variable to numeric value for the modeling.
master[master$Sex=='male']$Sex='1'
master[master$Sex=='female']$Sex='0'


#2-1. check out Name variable.
#let`s sort out titles
master$Title<-0
titles<-gsub("^.*, (.*?)\\..*$", "\\1", master$Name)
master$Title<-titles
table(titles) #looks like most titles are Mr., Mrs., Miss., and Master.
main.title<-c('Mr','Mrs','Miss','Master') 
master$Title[!master$Title %in% main.title]='Others'

(ggplot(master,aes(master$Survived,fill=master$Title))
  +geom_bar()
  +xlab("Survived")
  +ylab("number of people")
  +guides(fill=guide_legend(title="Title")))
#looks like women(Mrs. and Miss) did more survived than men and this is plausible result
#because usually men let women and children take a boat first.

#2-2. change variable to numeric value for the modeling.
master$Title[master$Title=="Mr"]=0
master$Title[master$Title=="Mrs"]=1
master$Title[master$Title=="Miss"]=2
master$Title[master$Title=="Master"]=3
master$Title[master$Title=="Others"]=4

master$Name<-NULL #don`t need name anymore.


#3-1. check out Name variable.
#fill out NAs from the median age of each title. 
title_Age<-aggregate(master$Age,by = list(master$Title), FUN = function(x) median(x, na.rm = T))
master[is.na(master$Age), "Age"] <- apply(master[is.na(master$Age), ] , 1, function(x) title_Age[title_Age[, 1]==x["Title"], 2])

#3-2. categorize age and change to numeric value.
master[master$Age < 20]$Age -> 'Child'
master[20 <= master$Age & master$Age < 40]$Age -> 'Adult'
master[40 <= master$Age & master$Age < 60]$Age -> 'Med_Age'
master[60 <= master$Age]$Age -> 'Senior'

master$Age[master$Age %in% Child]=0
master$Age[master$Age %in% Adult]=1
master$Age[master$Age %in% Med_Age]=2
master$Age[master$Age %in% Senior]=3

#check NAs 
sum(is.na(master$Age))

(ggplot(master,aes(master$Survived, fill=master$Age))
  +geom_bar()
  +xlab("Survived")
  +ylab("number of people")
  +guides(fill=guide_legend(title="Age")))


#4-1. check out Pclass variable.
(ggplot(master,aes(master$Survived,fill=master$Pclass))
  +geom_bar()
  +xlab("Survived")
  +ylab("number of people")
  +guides(fill=guide_legend(title="Passenger class")))


#5-1. check out Fare variable.
#fill out NA from the median Pclass which is 3 in this case.
master[order(master$Fare)] #NA from Pclass '3'.
Fare_NA<-aggregate(master$Fare,by = list(master$Pclass), FUN = function(x) median(x, na.rm = T))
master[is.na(master$Fare), "Fare"] <- Fare_NA[3,2]

(ggplot(master,aes(master$Fare))
  +geom_density(fill = 'blue', alpha=0.4)
  +geom_vline(aes(xintercept=median(Fare, na.rm=T)),colour='green', linetype='dashed', lwd=1)
  +xlab("Fare")
  +ylab("number of people")
  +guides(fill=guide_legend(title="Fare")))

#5-2. categorize Fare values.
master[master$Fare < 15]$Fare = 0
master[15 <= master$Fare & master$Fare < 35]$Fare = 1
master[35 <= master$Fare & master$Fare < 100]$Fare = 2
master[100 <= master$Fare]$Fare = 3


#6-1. check out Embarked variable.
unique(master$Embarked) #found two missing Embarked info.

(ggplot(master,aes(master$Pclass,fill=master$Embarked))
  +geom_bar()
  +xlab("Class")
  +ylab("number of people")
  +guides(fill=guide_legend(title="Embarked")))
#most passengers are from 'S', so we replace missing value to 'S'.
master[order(master$Embarked)]
master[master$PassengerId==62]$Embarked <- 'S'
master[master$PassengerId==830]$Embarked <- 'S'

#6-2. convert to numeric values
master[master$Embarked=='S']$Embarked=0
master[master$Embarked=='Q']$Embarked=1
master[master$Embarked=='C']$Embarked=2


#7-1. check out Cabin variable
unique(master$Cabin)

#ignore this due to the sparseness.
master$Cabin<-NULL


#8-1. check out SibSp variable
(ggplot(master,aes(master$SibSp))
  +geom_density(fill = 'blue', alpha=0.4)
  +geom_vline(aes(xintercept=median(Fare, na.rm=T)),colour='green', linetype='dashed', lwd=1)
  +xlab("FamilySize")
  +ylab("number of people")
  +guides(fill=guide_legend(title="FamilySize")))

table(master$SibSp)

#8-2. categorize SibSp values.
master[master$SibSp ==0]$SibSp = 0
master[1 <= master$SibSp & master$SibSp < 3]$SibSp = 1
master[3 <= master$SibSp]$SibSp = 2

master$FamilySize<-master$SibSp
master$SibSp<-NULL

#9. now we split train and test set
t.train<-master[1:891,]
a<-t.train$Survived
t.train$Survived<-NULL
t.train$Survived<-a

t.test<-master[892:1309,]
t.test$Survived<-NULL

#now save feature engineering part for the modeling script.
fwrite(t.train,"./project/volume/data/interim/train.csv")
fwrite(t.test,"./project/volume/data/interim/test.csv")


