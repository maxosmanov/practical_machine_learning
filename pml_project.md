Background
==========

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

Goal of the project
===================

We want to predict whether people do exercises correctly or not based on
the data from the electronic equipment such as Jawbone UP. Moreover, we
want to classify the pattern to understand to which class the
performance belongs. In future it allows to check whether people do
exercises corrrectly and help them to improve.

Training dataset
================

Firstly, we load the dataser

    library(readr)
    initial_dataset<- read.csv("C:/Users/maks/OneDrive/coursera/Data Science/Practical Mahine Learning/pml-training.csv")

Dataset contains 160 variables and about 20 000 observations which is
really a lot and requires carefull feature selection.

We realise that many variables contains NA, meaning that measurements
were not performed. Even though we can use different techniques to fill
these lacking values we use this fact to delete variables which contain
missing values. This is our process of feature elimination. In addition,
we expect that names of features are related to belt, arm of dumbbell
since we speak about physical exercises. The following code determines
the new set of predictors which is reduced to 52 variables

    names_without_na = sapply(initial_dataset, function (x) any(is.na(x)))
    names_filled = sapply(initial_dataset, function (x)  any(x == ""))


    names_belt = grepl("belt", names(initial_dataset))
    names_arm = grepl("arm", names(initial_dataset))
    names_dumbbell = grepl("dumbbell", names(initial_dataset))

    predictors = subset(names(initial_dataset), (!names_without_na& !names_filled)& (names_belt|names_arm|names_dumbbell))

Thus we create the new dataset with the new predictors and the target
variable Classe

    fitness_dataset = initial_dataset[, c(predictors, "classe")]

To avoid overfitting we need to divide our dataset into the training and
testing part (it can be considered as 1-fold cross-validation).
Following standard procedures we obtain the training and testing
datasets

    inTrainingSet = createDataPartition(fitness_dataset$classe, p = 0.7, list = FALSE)

    fitnessTrain = fitness_dataset[inTrainingSet,]
    fitnessTest = fitness_dataset[-inTrainingSet,]

and normalize them (by applying scaling and centering)

    preProcValues = preProcess(fitnessTest[, c(predictors, "classe")], method = c("center", "scale"))

    trainScaled = predict(preProcValues, fitnessTrain[, c(predictors, "classe")])
    testScaled = predict(preProcValues, fitnessTest[, c(predictors, "classe")])

We apply the Gradient Boosted Machine (GBM) to create model for
predictions. This method is very good for classification with a good
overall accuracy. Taking into account the huge amount of training data
we do not expect overfit with the method. The only problem is time of
training which is extremelly long.

    gbmModel = train(x = trainScaled[,predictors], y = trainScaled$classe, method ="gbm", verbose = FALSE)

    ## Loading required package: gbm

    ## Loading required package: survival

    ## 
    ## Attaching package: 'survival'

    ## The following object is masked from 'package:caret':
    ## 
    ##     cluster

    ## Loading required package: splines

    ## Loading required package: parallel

    ## Loaded gbm 2.1.3

    ## Loading required package: plyr

After training the model we firstly compute the intrain accuracy equal
to 0.974 which is a very good precision

    gbmPred = predict(gbmModel, trainScaled)
    confusionMatrix(gbmPred, trainScaled$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 3874   49    0    3    3
    ##          B   20 2562   57    4   19
    ##          C    8   45 2314   47   13
    ##          D    4    2   20 2189   29
    ##          E    0    0    5    9 2461
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9755         
    ##                  95% CI : (0.9727, 0.978)
    ##     No Information Rate : 0.2843         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.969          
    ##  Mcnemar's Test P-Value : 5.86e-11       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9918   0.9639   0.9658   0.9720   0.9747
    ## Specificity            0.9944   0.9910   0.9900   0.9952   0.9988
    ## Pos Pred Value         0.9860   0.9624   0.9534   0.9755   0.9943
    ## Neg Pred Value         0.9967   0.9913   0.9927   0.9945   0.9943
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2820   0.1865   0.1685   0.1594   0.1792
    ## Detection Prevalence   0.2860   0.1938   0.1767   0.1634   0.1802
    ## Balanced Accuracy      0.9931   0.9774   0.9779   0.9836   0.9867

Clearly, the accuracy is bit lower for the validation set and is equal
to 0.9616 which is still an impressive value larger than 95%.

    gbmPredTest = predict(gbmModel, testScaled)
    confusionMatrix(gbmPredTest, testScaled$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1655   37    0    1    0
    ##          B   13 1067   26    5   11
    ##          C    4   31  987   32    9
    ##          D    0    3   13  919   16
    ##          E    2    1    0    7 1046
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9641          
    ##                  95% CI : (0.9591, 0.9687)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9546          
    ##  Mcnemar's Test P-Value : 5.389e-07       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9886   0.9368   0.9620   0.9533   0.9667
    ## Specificity            0.9910   0.9884   0.9844   0.9935   0.9979
    ## Pos Pred Value         0.9776   0.9510   0.9285   0.9664   0.9905
    ## Neg Pred Value         0.9955   0.9849   0.9919   0.9909   0.9925
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2812   0.1813   0.1677   0.1562   0.1777
    ## Detection Prevalence   0.2877   0.1907   0.1806   0.1616   0.1794
    ## Balanced Accuracy      0.9898   0.9626   0.9732   0.9734   0.9823

Therefore we expect the error of prediction to be smaller that 5%.

Finally, we apply the model to make a prediction for the unlabeled
dataset.
