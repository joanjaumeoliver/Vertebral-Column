######## INTRODUCTION ########
#I highly recommend reading the Report.pdf first since all can be found there with more detail.
#The nomenclature is quite strange since this is the code file I've been working on for days,
#and I didn't want to have trouble with old working variables and methods.

#Remeber, this is just the R script were I tested all the methods present on the Report.

#(Loading libraries)
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(plotly)) install.packages("plotly", repos = "http://cran.us.r-project.org")
if(!require(pamr)) install.packages("pamr", repos = "http://cran.us.r-project.org")
if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org")
if(!require(neuralnet)) install.packages("neuralnet", repos = "http://cran.us.r-project.org")

#(Downloading data)
dl <- tempfile()
download.file(paste("https://raw.githubusercontent.com/joanjaumeoliver/",
                    "Vertebral-Column/master/Vertebral-Column%20Data.csv",sep=""), dl)
data <- read_csv(dl)
rm(dl)

any(is.na(data)) #No NA values

summary(data) %>% knitr::kable() #Seems to be an outlier at the Spondylolisthesis class
summary(data)[,6]

which.max(data$degree_spondylolisthesis)
#Line 116

data %>% arrange(degree_spondylolisthesis) %>% tail()
#As we see the 418 value, is an outlier and it would possibly be a typing mistake.

data <- data[-116,] #Remove line
summary(data)

######## PREPARING THE DATA ########
# validation set of 10%
set.seed(1)
main_index <- createDataPartition(y = data$class, times = 1, p = 0.1, list = FALSE)
main_set <- data[-main_index,]
validation_set <- data[main_index,]
rm(main_index)
main_set_2<-main_set %>% mutate(class=ifelse(class!="Normal",1,0))
#Here we don't have main_set_3 as in the Report.pdf

#Cross Validation of main_set
set.seed(1997)
test_index <- createDataPartition(y = main_set$class, times = 5, p = 0.1, list = FALSE)
trains_set <- list(data[-test_index[,1],],data[-test_index[,2],],data[-test_index[,3],],data[-test_index[,4],],data[-test_index[,5],])
tests_set <- list(data[test_index[,1],],data[test_index[,2],],data[test_index[,3],],data[test_index[,4],],data[test_index[,5],])
rm(test_index)

######## ANALYSING THE DATA ########

#PCA main_set
x<-main_set_2 %>% select(-class) %>% as.matrix()
cor(x) #Correlations

pca<-prcomp(x)
summary(pca)

data.frame(pca$x[,1:2], Class=main_set_2$class)%>%
  ggplot(aes(PC1,PC2, fill = Class))+
  geom_point(cex=3, pch=21) +
  coord_fixed(ratio = 1)
rm(x)

#The third PC was near 7% it's important?
temp <- data.frame(pca$x[,1:3], Class=main_set_2$class)
plot_ly(temp,x =temp$PC1, y = temp$PC2, z = temp$PC3, color = temp$Class) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'PC1'),
                      yaxis = list(title = 'PC2'),
                      zaxis = list(title = 'PC3')))
rm(temp)
rm(pca)

#COnfusionMatrixPlot function
ConfMatrixPlot <- function(Matrix){
  confusion<-as.data.frame(as.table(scale(as.matrix(Matrix),center=FALSE)))
  colnames(confusion)<-c("Reference","Prediction","Freq")
  ggplot(confusion) + geom_tile(aes(x=Reference, y=Prediction, fill=Freq)) +
    scale_x_discrete(name="Actual Class") + 
    scale_y_discrete(name="Predicted Class") +
    scale_fill_gradient(breaks=seq(from=-.5, to=4, by=.2),low="darkorchid4",high="yellow") + 
    labs(fill="Normalized\nFrequency")
}

ConfMatrixPlot(diag(3)) #IdealResults 100% accuracy

#knn3 for all sets and 50 K
sets<-seq(1:5)
list<-sapply(sets, function(sets){
  train_temp <- trains_set[[sets]]
  test_temp <- tests_set[[sets]]
  k <- seq(1:50)
  accuracy<-sapply(k, function(k){
    knn_fit <- knn3(as.factor(class) ~ ., data = train_temp, k = k)
    y_hat_knn <- predict(knn_fit, test_temp, type = "class")
    confusionMatrix(data = y_hat_knn, reference = as.factor(test_temp$class))$overall["Accuracy"]
  })
  list(tibble(k,accuracy)) 
})
rm(sets)

#Results plot
ggplot()+
  geom_line(data=list[[1]],aes(list[[1]]$k,list[[1]]$accuracy),colour="red")+
  geom_line(data=list[[2]],aes(list[[2]]$k,list[[2]]$accuracy),colour="blue")+
  geom_line(data=list[[3]],aes(list[[3]]$k,list[[3]]$accuracy),colour="green")+
  geom_line(data=list[[4]],aes(list[[4]]$k,list[[4]]$accuracy),colour="purple")+
  geom_line(data=list[[5]],aes(list[[5]]$k,list[[5]]$accuracy))


######## TRYING COMMON CARET METHODS ########
#From now on, all the methods present in the code were tested with the caret 
#train function structure, most of them without tunning parameters.

#knn
set.seed(1997)
train_knn <- train(as.factor(class) ~ ., method = "knn", data = trains_set[[1]])
y_hat_knn <- predict(train_knn, tests_set[[1]], type = "raw")
confusion_matrix<-confusionMatrix(y_hat_knn, as.factor(tests_set[[1]]$class))
confusion_matrix$overall[["Accuracy"]]
#Results confusion matrix plot
ConfMatrixPlot(confusion_matrix$table)
#Parameter evolution plot
ggplot(train_knn, highlight = TRUE)
rm(train_knn,y_hat_knn, confusion_matrix)


#Automatic cross validation with tunning parameters.
set.seed(1997)
train_knn <- train(as.factor(class)~ ., method = "knn",
                   data = main_set,
                   tuneGrid = data.frame(k = seq(9, 71, 2)))
ggplot(train_knn, highlight = TRUE)
train_knn$bestTune
train_knn$finalModel
max(train_knn$results$Accuracy)
train_knn$results$AccuracySD[which.max(train_knn$results$Accuracy)]

#Centroids
set.seed(1997)
train_pam <- train(as.factor(class)~ ., method = "pam",
                   data = main_set,
                   tuneGrid=data.frame(threshold=seq(1,9)))
ggplot(train_pam, highlight = TRUE)
train_pam$bestTune
train_pam$finalModel
max(train_pam$results$Accuracy)
train_pam$results$AccuracySD[which.max(train_pam$results$Accuracy)]

#LogisticRegression
fit_glm <- glm(as.factor(class) ~ pelvic_incidence + pelvic_tilt+
                 lumbar_lordosis_angle+sacral_slope+pelvic_radius+degree_spondylolisthesis, 
               data=trains_set[[1]], family = "binomial")
p_hat_glm <- predict(fit_glm, tests_set[[1]])

set.seed(1997)
temp_set<-main_set %>% mutate(class=ifelse(class!="Normal","Ill","Normal"))
train_glm <- train(as.factor(class)~ ., method = "glm",
                   data = temp_set)
train_glm$finalModel
train_glm$results$Accuracy
train_glm$results$AccuracySD

#QDA - Not working
set.seed(1997)
train_qda <- train(as.factor(class)~ ., method = "qda",
                   data = main_set)

#LDA
set.seed(1997)
train_lda <- train(as.factor(class)~ ., method = "lda",
                   data = main_set)
train_lda$finalModel
train_lda$results$Accuracy
train_lda$results$AccuracySD
accuracy_results <- rbind(accuracy_results,tibble(Method = "lda", 
                                                  Accuracy = train_lda$results$Accuracy,
                                                  SD=train_lda$results$AccuracySD,
                                                  bestTune=""))

#Naive Bayes
set.seed(1997)
train_bayes <- train(as.factor(class)~ ., method = "naive_bayes",
                     data = main_set)
train_bayes$finalModel
train_bayes$results$Accuracy
train_bayes$results$AccuracySD

#Bayes glm
set.seed(1997)
train_bayesglm <- train(as.factor(class)~ ., method = "bayesglm",
                        data = main_set)
train_bayesglm$finalModel
train_bayesglm$results$Accuracy
train_bayesglm$results$AccuracySD

#Rpart
set.seed(1997)
train_rpart <- train(as.factor(class) ~ .,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                     data = main_set)
plot(train_rpart)
train_rpart$bestTune
train_rpart$finalModel
max(train_rpart$results$Accuracy)
train_rpart$results$AccuracySD[which.max(train_rpart$results$Accuracy)]


#Rborist
set.seed(1997)
train_rborist <- train(as.factor(class) ~ .,
                       method = "Rborist",
                       tuneGrid = data.frame(predFixed = 2, minNode = c(3, 50)),
                       data = main_set)
train_rborist$bestTune
train_rborist$finalModel
max(train_rborist$results$Accuracy)
train_rborist$results$AccuracySD[which.max(train_rborist$results$Accuracy)]

#Adaboost
set.seed(1997)
#Binarize data
temp_set<-main_set %>% mutate(class=ifelse(class!="Normal","Ill","Normal"))
train_adaboost <- train(as.factor(class) ~ .,
                        method = "adaboost",
                        data = temp_set)
train_adaboost$bestTune
train_adaboost$finalModel
max(train_adaboost$results$Accuracy)

#Rf
train_rf <- train(as.factor(class) ~ .,
                  method = "rf",
                  tuneGrid = data.frame(mtry = seq(2,100,1)),
                  data = main_set)
train_rf$bestTune
train_rf$finalModel
max(train_rf$results$Accuracy)

######## OTHER CARET METHODS ########
#Other caret-in built methods tried
#Be aware, not all of them did work!

#Binarize the data
temp_set<-main_set %>% mutate(class=ifelse(class!="Normal",1,0))

#(List of methods)
method<-c("svmLinear3","svmLinearWeights2","svmRadialSigma","svmRadialCost",
          "svmRadial","gbm","monmlp","mlp","avNNet","wsrf","ranger","loclda",
          "Mlda","pda","pda2","stepQDA","bagFDA","BstLm","LogitBoost","C5.0",
          "C5.0Cost","xgbLinear","lvq","svmLinear","gamboost","sda","sparseLDA",
          "dwdPoly","gamLoess","kknn")

#Function to apply to all the methods
Others<-sapply(method,function(x){
  set.seed(1997)
  train <- train(as.factor(Y) ~ .,
                 method=x,
                 data = temp_set)
  return(tibble(Accuracy=max(train$results$Accuracy),SD=max(train$results$AccuracySD)))
})

######## DATA MODIFICATIONS ########
main_set2aux <- main_set %>% mutate(Y=ifelse(Y=="Normal",0,1))
validation_set2aux <- validation_set %>% mutate(Y=ifelse(Y=="Normal",0,1))

######## KNN WITH PCA ########

set.seed(1997)
#(PCA all data)
pca_trainset = main_set2aux %>% select( -Y )
dat<-as.matrix(validation_set2aux %>% select(-Y))

pca = prcomp( pca_trainset)
train = data.frame( Y = main_set2aux$Y, pca$x[,1:4] )
test = sweep(dat,2,colMeans(dat)) %*% pca$rotation
test <- test[,1:4]

#(Training data)
train_knn <- train(as.factor(Y)~ ., method = "knn",
                   data = train)
#(Results)
max(train_knn$results$Accuracy)
#Prediction Accuracy
confusionMatrix(predict(train_knn,test),factor(validation_set2aux$Y))$overall["Accuracy"]

######## NEURONAL NETWORKS ########
set.seed(1997)

n = names( main_set2aux )
f = as.formula( paste( "Y ~", paste( n[!n %in% "Y"], collapse = "+" ) ) )

nn = neuralnet(f,main_set2aux,hidden = 4, linear.output = FALSE, threshold = 0.1 )

nn.results = neuralnet::compute( nn, validation_set2aux )
results = data.frame( actual = validation_set2aux$Y, prediction = round( nn.results$net.result ) )
t = table(results)
print(confusionMatrix(t))

set.seed(1997)
pca_trainset = main_set2aux %>% select( -Y )
dat<-as.matrix(validation_set2aux %>% select(-Y))
pca = prcomp( pca_trainset)
train = data.frame( Y = main_set2aux$Y, pca$x[,1:4] )
test = sweep(dat,2,colMeans(dat)) %*% pca$rotation
test <- test[,1:4]

n = names( train )
f = as.formula( paste( "Y ~", paste( n[!n %in% "Y" ], collapse = "+" ) ) )
nn = neuralnet( f, train, hidden =4, linear.output = F, threshold = 0.1)

nn.results = neuralnet::compute( nn, test )

# Results
results = data.frame( actual = validation_set2aux$Y, 
                      prediction = round( nn.results$net.result ) )
test = table( results ) 
print( confusionMatrix( test ) )