# Machine Learning Project
Stella Li  
##Introduction

###Background cited from project description
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

###Summary
1. Load original testing and training data;
2. Use a simple holdout method for cross validation: 70% of the original testing data were used to build models(myTrain data) and the rest of original testing data were used to test the models;
3. Clean data: Since there are so many variables, we removed some columns;
4. Build and test models: We first tried Decision Tree model, however the accuracy was rather low. We then tried Random Forest model and the accuracy was 0.99. Since there were still many variables, we also applied PCA to simplify model, however the accuracy of new model was not as good;
5. Apply Random Forest model (without PCA) to the original testing data (20 samples) to estimate their classes.

##Analysis
###1. Packages and Seed
We set the overall pseudo-random number generator seed as 1017. In order to reproduce the same results as below, the same seed should be used. Different packages were downloaded and installed

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(rpart)
set.seed(1017)
```

###2. Load and split data

```r
trainURL <- c("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testURL <- c("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
training <- read.csv(url(trainURL), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testURL), na.strings=c("NA","#DIV/0!",""))
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
myTrain <- training[inTrain, ]
myTest <- training[-inTrain, ]
dim(myTrain)
```

```
## [1] 13737   160
```

###3. Clean data
There are 160 columns in myTrain data. We did some initial exploration to understand the dataset, and decide to remove some of the columns to clean data. To save some space, I will not print the result of summary(myTrain).

```r
## delete the columns that cannot be used for prediction
myTrain <- myTrain[ , -(1:7)]
## exclude the columns with little variance
nzv <- nearZeroVar(myTrain, saveMetrics = TRUE)
myTrain <- myTrain[ , !nzv$nzv]
## exclude the columns with doo many NAs; there are 67 columns have more than 95% NAs
NAcol <- apply(myTrain, 2, function(x) sum(is.na(x))/nrow(myTrain))
myTrain <- myTrain[ , !NAcol>0.95]
dim(myTrain)
```

```
## [1] 13737    53
```

###4. Prediction Models
Model 1. Decision Tree

```r
myfit.dt <- rpart(classe ~., data=myTrain, method="class")
## clean myTest data before testing our prediction
myTest <- myTest[ , -(1:7)]
myTest <- myTest[ , !nzv$nzv]
myTest <- myTest[ , !NAcol>0.95]
confusionMatrix(myTest$classe, predict(myfit.dt, myTest,type="class"))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1479   57   46   57   35
##          B  156  655  181   80   67
##          C   16   78  842   65   25
##          D   56   89  146  593   80
##          E   18   87  138   51  788
## 
## Overall Statistics
##                                          
##                Accuracy : 0.7404         
##                  95% CI : (0.729, 0.7515)
##     No Information Rate : 0.2931         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6714         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8574   0.6781   0.6223   0.7009   0.7920
## Specificity            0.9531   0.9016   0.9594   0.9264   0.9399
## Pos Pred Value         0.8835   0.5751   0.8207   0.6151   0.7283
## Neg Pred Value         0.9416   0.9345   0.8948   0.9486   0.9569
## Prevalence             0.2931   0.1641   0.2299   0.1438   0.1691
## Detection Rate         0.2513   0.1113   0.1431   0.1008   0.1339
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9053   0.7898   0.7909   0.8137   0.8659
```

Model 2. Random Forest

```r
myfit.rf1 <- randomForest(classe ~., data=myTrain)
confusionMatrix(myTest$classe, predict(myfit.rf1, myTest))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    0    0    0    1
##          B    7 1130    2    0    0
##          C    0    6 1018    2    0
##          D    0    0   19  945    0
##          E    0    0    3    4 1075
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9925        
##                  95% CI : (0.99, 0.9946)
##     No Information Rate : 0.2855        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9905        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9958   0.9947   0.9770   0.9937   0.9991
## Specificity            0.9998   0.9981   0.9983   0.9961   0.9985
## Pos Pred Value         0.9994   0.9921   0.9922   0.9803   0.9935
## Neg Pred Value         0.9983   0.9987   0.9951   0.9988   0.9998
## Prevalence             0.2855   0.1930   0.1771   0.1616   0.1828
## Detection Rate         0.2843   0.1920   0.1730   0.1606   0.1827
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9978   0.9964   0.9877   0.9949   0.9988
```

Model 3. Random Forest with PCA
Since there are still more than 50 variables in myTrain data, we applied PCA to removing variables that have high correlations with themselves

```r
## pre-process data to cover 95% of variance 
PreProc <- preProcess(myTrain[ ,-53], method="pca", thresh=0.95)
myTrainPre <- predict(PreProc, myTrain[ ,-53])
myfit.rf2 <- randomForest(myTrain$classe ~., data=myTrainPre)
myTestPre <- predict(PreProc, myTest[ ,-53])
confusionMatrix(myTest$classe, predict(myfit.rf2, myTestPre))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1658    4    6    4    2
##          B   20 1107   10    0    2
##          C    3   19  998    5    1
##          D    1    0   39  922    2
##          E    0   12   11   11 1048
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9742          
##                  95% CI : (0.9698, 0.9781)
##     No Information Rate : 0.2858          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9673          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9857   0.9694   0.9380   0.9788   0.9934
## Specificity            0.9962   0.9933   0.9942   0.9915   0.9930
## Pos Pred Value         0.9904   0.9719   0.9727   0.9564   0.9686
## Neg Pred Value         0.9943   0.9926   0.9864   0.9959   0.9985
## Prevalence             0.2858   0.1941   0.1808   0.1601   0.1793
## Detection Rate         0.2817   0.1881   0.1696   0.1567   0.1781
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9910   0.9813   0.9661   0.9851   0.9932
```

##Discussion
Random Forest models yielded better results than Decision Tree model. The accuracy for the two Random Forest models were 0.97 and 0.99 respectively. Sensitivity for all classes were between 93% to 99% and specificity were all higher than 99%.
I decided to use the Random Forest without PCA model to predict the classes of samples in the original testing dataset, as this model had a higher accuracy. However, I am aware that this model could be overfitted with so many variables.

```r
predict(myfit.rf1, testing)
```
