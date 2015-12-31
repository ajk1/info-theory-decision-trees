library(arules)
library(rpart)
library(ROCR)
library(party)
library(reshape2)
library(ggplot2)

import.csv <- function(filename) {
  return(read.csv(filename, sep = ",", header = TRUE))
}

write.csv <- function(ob, filename) {
  write.table(ob, filename, quote = FALSE, sep = ",", row.names = FALSE)
}

my.data <- import.csv('cars.csv')


# Entropy, Information gain, Feature selection

# Compute entropy for a vector of symbolic values
# Inputs: Vector of symbolic values
# Output: Entropy (in bits)
entropy <- function(x) {
  H = 0 #entropy of x
  for (level in levels(as.factor(x))) {
    p = length(which(x==level)) / length(x) #probability of x
    if (p != 0) {
      I = -log2(p) #information content of x
      H = H + p*I
    }
  }
  return(H)
}

# Unit test for function entropy
x <- c(rep('A', 3),rep('B', 2),rep('C', 5))
print(ifelse(abs(entropy(x) - 1.485475 ) < 1.0e-05, 'entropy function has passed this test!', 'entropy function has failed this test'))

# Compute information gain IG(x,y)
# Inputs: x, y: vectors of symbolic values of equal length
# Output: information gain IG(x,y)
info.gain <- function(x,y){
  Hxy = 0 #conditional entropy of x given y
  for (level in (levels(as.factor(y)))) {
    p = length(which(y==level))/length(y) #probability of y
    Hxy = Hxy + p*entropy(x[which(y==level)])
  }
  IG = entropy(x) - Hxy
  return(IG)
}

# IG <-function(x,y){
#   H= 0
#   for (level in levels(as.factor(x))){
#     H= H+(length(which(x==level & y ==0))/length(which(y==0))-
#           length(which(x==level & y ==1))/length(which(y==1)))*
#       (log2(length(which(x==level & y ==0))/length(which(y==0)))-
#          log2(length(which(x==level & y ==1))/length(which(y==1))))
#   }
#   return(H)
# }

# Unit test for function info.gain
x <- c(rep('A',3),rep('B',2),rep('C',5))
y <- c(rep('X',4),rep('Y',6))
print(ifelse(abs(info.gain(x,y) - 0.7709506 ) < 1.0e-05, 'Info.gain function has passed this test!', 'info.gain function has failed this test'))

# Information-gain-based feature selection: exhaustive search
# Input: df is a data frame with last column being the output attribute
#        m: size of feature set, default is 1
# Output: data frame with name(s) of selected feature(s), information gain, relative information gain, sorted by the value of information gain
features <- function(df, m = 1){
  nf <- ncol(df) -1 # number of input features
  idx <- 1: nf  # column indices of input features
  output <- df[, ncol(df)]  # output column
  outputH <- entropy(output) # entropy for output
  idx.list <- combn(idx, m) # matrix storing all combinations of size m from idx
  IG.res <-NULL # output data frame
  # iterate through all combinations of index 
  for (ii in 1:ncol(idx.list)){
    this.idx <- idx.list[, ii]  
    input.df <- data.frame(df[,this.idx]) 
    # create a vector where each element is a concatenation of all values of a row of a data frame
    this.input <- apply(input.df, 1, paste, collapse='') 
    # create a new feature name which is a concatenation of feature names in a feature set
    this.input.names <- paste(names(df)[this.idx], collapse=' ')    
    this.IG <-info.gain(this.input,output) # information gain
    this.RIG <- this.IG / outputH # relative information gain
    this.res <- data.frame(feature = this.input.names, IG = this.IG, RIG = this.RIG) #assemble a df
    IG.res <- rbind(IG.res, this.res) # concatenate the results    
  }
  sorted <- IG.res[order(-IG.res$IG), ] # sort the result by information gain in a descending order
  return (sorted)
}

class.data <- my.data[,c(1:4,6,5)] #reorder columns so class is output
dtree <- tree(class.data) 

# calculates area under the ROC
# input vector of predicted probabilities, vector of actual (binary) classes
auc <- function(predicted, actual) {
  df <- data.frame(predicted, actual=factor(actual))
  df <- df[order(-df$predicted),]
  num.pos <- sum(df$actual==levels(df$actual)[2])
  num.neg <- sum(df$actual==levels(df$actual)[1])
  tpr <- 0
  npr <- 0
  area <- 0
  for (i in 1:nrow(df)) {
    if (df$actual[i]==levels(df$actual)[1]) {
      tpr <- tpr + 1/num.pos
    }
    else {
      npr <- npr + 1/num.neg
      area <- area + tpr/num.neg
    }
  }
  return(area)
}

cat("\nEntropy of 'class':\n")
print(entropy(my.data$class))
#(b)
cat("\nInformation Gain of 'class' vs 'price':\n")
print(info.gain(my.data$class, my.data$price))
cat("\nRelative Information Gain of 'class' vs 'price':\n")
print(info.gain(my.data$class, my.data$price)/entropy(my.data$class))

cat("\nFeature selection for features of size 1:\n")
print(features(class.data, m=1))
cat("\nFeature selection for features of size 2:\n")
print(features(class.data, m=2))
cat("\nFeature selection for features of size 3:\n")
print(features(class.data, m=3))

feat1 <- melt(features(class.data, m=1)[,c(1,3,2)])
feat1 <- transform(feat1, feature = reorder(feature, value))
a <- ggplot(feat1, aes(feature, value)) +
  geom_bar(aes(fill = variable), width=0.6, position="dodge", stat="identity") +
  ggtitle("Information Gain of individual features on Car Class") + coord_flip()
print(a)

# Association Rules

rules <- apriori(my.data, control = list(verbose=FALSE),
                 parameter = list(minlen=2, supp=0.1, conf=0.5),
                 appearance = list(rhs=c("price=high"), default="lhs"))
cat("\nAll non-empty rules predicting price=high,
    with min support 0.1 and min confidence 0.5:\n")
inspect(rules) #it appears minlen=1 gives me empty rules, so I set it to 2
cat("\nThere were", length(rules), "rules that met these criteria\n")
cat("\nTop 5 rules by support:\n")
inspect(sort(rules, by="support")[1:5,])
cat("\nTop 5 rules by confidence:\n")
inspect(sort(rules, by="confidence")[1:5,])

# Inputs: rules is the object returned by the apriori function
#         df is a data frame from which the rules are learned
# Output: a rule object with extra metrics (95% CI of score)
expand_rule_metrics <- function(rules,df){
  rule.df <- interestMeasure(rules, c('support', 'confidence'), df) # extract metrics into a data frame
  nn <- nrow(df)
  c = rule.df$confidence
  s = rule.df$support
  ci.low <- c - 1.96 * sqrt( c*(1-c) / (s*nn) )
  ci.high <- c + 1.96 * sqrt( c*(1-c) / (s*nn) )
  quality(rules) <-cbind(quality(rules), ci.low, ci.high) # update the quality slot of the rules object
  return(rules)
}
cols <- c("channel","numberofborrowers1","haspmi","loan.purpose")
for (c in cols) {
  segmenters[[i]] <- segmenters[[i]][,-which(rownames(segmenters[[i]])==c)]
}

cat("\nTop 5 rules by conservative confidence (lower bound):\n")
inspect(sort(expand_rule_metrics(rules, my.data), by="ci.low")[1:5,])

# Decision Trees

tree <- rpart(price ~ ., data = my.data, method = "class")
# par(mar=c(1,0,4,0))
plot(tree, uniform=T, main="Classification Tree for Car Price")
text(tree, use.n=T, all=T, cex=0.8, pretty=TRUE)
# post(tree, file = "C:/tree2.ps", title = "Classification Tree for Car Price")
pred <- prediction(predict(tree, my.data)[,1], ifelse(my.data$price=="high",1,0))
plot(performance(pred, 'tpr', 'fpr'), main="Car price ROC curve with rpart")
abline(0,1)
cat("\nrpart AUC:\n")
print(performance(pred, measure='auc'))

tree <- ctree(price ~ ., data = my.data)
plot(tree, uniform=T, main="Classification Tree for Car Price")
pred <- prediction(t(as.data.frame(predict(tree, my.data, type='prob', simply=FALSE)))[,1], 
                   ifelse(my.data$price=="high",1,0))
plot(performance(pred, 'tpr', 'fpr'), main="Car price ROC curve with ctree")
abline(0,1)
cat("\nctree AUC:\n")
print(performance(pred, measure='auc'))

##### homebrewed decision tree generator
# # creates a decision tree by splitting on variable with highest IG at each node
# # stops when a minimum IG threshold is reached.
# # input data frame of factors with output variable in last column
# # output decision tree in the form of a list of lists of lists, each level being a decision
# tree <- function(df, min_IG = 0.2) {
#   sorted <- features(df)
#   if(sorted$IG[1] < min_IG | ncol(df) == 2) {
#     return(paste("P(", colnames(df)[ncol(df)], "==", levels(df[,ncol(df)])[2], ") = ", 
#                  length(which(df[,ncol(df)]==levels(df[,ncol(df)])[2]))/length(df[,ncol(df)]),
#                  sep = ""))
#   }
#   root <- df[,as.integer(rownames(sorted)[1])]
#   partitions <- list()
#   for (i in 1:length(levels(root))) {
#     part <- data.frame(df[which(root == levels(root)[i]),])
#     partitions[[i]] <- part[,-as.integer(rownames(sorted)[1])]
#   }
#   names(partitions) <- paste(sorted$feature[1], "==", levels(root), sep="")
#   t <- list()
#   for (i in 1:length(partitions)) {
#     t[[i]] <- list(names(partitions)[i], tree(partitions[[i]]))
#   }
#   return(t)
# }
# 
# # takes an output from tree() and returns a list of all decisions and their results
# DFS <- function(t) {
#   if (length(t) == 1) {
#     return(as.character(t[[1]]))
#   }
#   l <- list()
#   for (i in 1:length(t)) {
#     list[[i]] <- DFS(t[[i]])
#   }
#   return 
# }