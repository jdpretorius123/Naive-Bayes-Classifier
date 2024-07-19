# This is a program that executes a simple Naive Bayes Classifier algorithm.
#
# Functions
#  computeJoint(case, obs)
#    Returns joint probability of data given class
#
#  computePosterior(obs)
#    Returns posterior probability of observation belonging to positive class


# loading dependencies
library(pROC)


# clear memory
rm(list = ls())


# reproducibility
set.seed(1)


# import training data
train = read.table(
  file = "train.txt",
  header = TRUE,
  sep = "\t"
)


# initializing data frame for binned data
binData = data.frame(
  train$class,
  train$drug,
  train$dose,
  train$weight, 
  train$ancestry
)


# building names for binned data
names = c()
for (i in 1:ncol(train)) {
  if (i < 6) {
    colnames(binData)[i] = names(train)[i]
  } else {
    names[i - 5] = names(train)[i] 
  }
}


# building data frame for binned data
for (i in 6:ncol(train)) {
  binData[names[i-5]] = findInterval(
    train[,i],
    unname(quantile(train[,i], c(0,.25,.5,.75)))
  )
}


# creating case distribution
cases = table(binData$class) / nrow(binData)


# selecting negative cases
negativeCase = binData[which(binData$class == 0),]


# selecting positive cases
positiveCase = binData[which(binData$class == 1),]


# maximum likelihood estimation: P(X=x|C=c)
mleList = list()
for (i in 1:(ncol(binData) - 1)) {
  mleList[[i]] = t(
    matrix(
      data = c(
        table(negativeCase[,(i+1)]) / nrow(negativeCase),
        table(positiveCase[,(i+1)]) / nrow(positiveCase)
      ),
      ncol = length(unique(negativeCase[,(i+1)])),
      byrow = TRUE
    )
  )
}


# naming list maximum likelihood estimates
names(mleList) = names(binData)[2:ncol(binData)]


# computing joint probability
computeJoint = function(case, obs) {
  "
  Returns joint probability of data belonging to given class
  
  Parameters:
    case: case for which joint probability of data is to be computed
    obs: observation as vector of features

  Return:
    jointP: joint probability of observation belonging to given class
    
  Assumptions:
    Case is a value of either 1 or 2
    Observation is vector of features
    Number of features for new observation equals number of elements in mleList
    Order of features for new observation matches order of elements in mleList
  "
  jointP = cases[case]
  for (i in 1:length(mleList)) {
    jointP = jointP * as.vector(unlist(mleList[[i]][obs[i], case]))
  }
  
  return(jointP)
}


# computing posterior probability
computePosterior = function(obs) {
  "
  Returns posterior probability of observation belonging to positive class
  
  Parameters:
    obs: observation as vector of features

  Return:
    posterior: posterior probability of observation belonging to positive class
    
  Assumptions:
    Observation is vector of features
    Observation's first value represents its true class
    Number of features for new observation equals number of elements in mleList
    Order of features for new observation matches order of elements in mleList
  "
  obs = as.vector(unlist(obs[2:length(obs)]))
  negativeJoint = computeJoint(1, obs)
  positiveJoint = computeJoint(2, obs)
  posterior = positiveJoint / (positiveJoint + negativeJoint)
  
  return(posterior)
}


# import test data
test = read.table(
  file = "test.txt",
  header = TRUE,
  sep = "\t"
)


# initializing data frame for binned data
binTest = data.frame(
  test$class,
  test$drug,
  test$dose,
  test$weight, 
  test$ancestry
)


# building names for binned test data
names = c()
for (i in 1:ncol(test)) {
  if (i < 6) {
    colnames(binTest)[i] = names(test)[i]
  } else {
    names[i - 5] = names(test)[i] 
  }
}


# building data frame for binned test data
for (i in 6:ncol(test)) {
  binTest[names[i-5]] = findInterval(
    test[,i],
    unname(quantile(test[,i], c(0,.25,.5,.75)))
  )
}


# testing model
predictions = data.frame()
for (i in 1:nrow(binTest)) {
  obs = binTest[i,]
  posteriorP = computePosterior(obs)
  predictions = rbind(
    predictions, 
    data.frame(Class = obs$class, PostP = posteriorP)
  )
}


# calculating accuracy, sensitivity, specificity
numCorrectAcc = 0
truePositive = 0
falseNegative = 0
falsePositive = 0

cutoff = 0.51
for (i in 1:nrow(predictions)) {
  trueClass = predictions$Class[i]
  prob = predictions$PostP[i]
  
  if ((prob >= cutoff && trueClass == 1) || (prob < cutoff && trueClass == 0)) {
    numCorrectAcc = numCorrectAcc + 1
  }
  
  if (prob >= cutoff && trueClass == 1) {
    truePositive = truePositive + 1
  }
  
  if (prob < cutoff && trueClass == 1) {
    falseNegative = falseNegative + 1
  }
  
  if (prob >= cutoff && trueClass == 0) {
    falsePositive = falsePositive + 1
  }
}

accuracy = numCorrectAcc / nrow(test) #Acc = 0.805
sensitivity = truePositive / (truePositive + falseNegative) #Sn = 0.732
specificity = truePositive / (truePositive + falsePositive) #Sp = 0.801


# creating ROC curve
pROC = roc(
  as.vector(predictions$Class), 
  as.vector(predictions$PostP)
)


# plotting ROC curve
plot(
  tpr ~ fpr, 
  coords(
    pROC,
    "all",
    ret=c("tpr","fpr"),
    transpose=FALSE,
    auc = TRUE
  ), 
  type="l",
  main = "ROC Curve",
  xlab = "FPR",
  ylab = "TPR"
)
values = seq(from = 0.0, to = 1.0, by = 0.1)
lines(
  x = values,
  y = values,
  lty = 2,
  col = 'gray',
  lwd = 1
)
auc = auc(pROC) #AUC = 0.8985


# plotting posterior probabilities
plot(
  density(predictions[which(predictions$Class == 0),]$PostP),
  main = "Posterior Probabilities",
  xlab = "Probability of Class",
  col = 'red'
)
lines(
  density(predictions[which(predictions$Class == 1),]$PostP),
  col = 'blue'
)
legend(
  x = 0.4,
  y = 3,
  legend = c("True Negative", "True Positive"),
  col = c('red', 'blue'),
  pch = 15
)