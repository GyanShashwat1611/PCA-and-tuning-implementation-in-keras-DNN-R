#Author: Gyan Shashwat
#PCA-and-tuning-implementation-in-keras-DNN-R

#************************************* Begin Code *********************************
# load packages and data0
library(keras)
load("data_activity_recognition.RData") #loading dataset
dim (x_train) #checking the dimensions of the features 
## [1] 7600 125 45
# we convert these into 125x45 vectors
x_train <- array_reshape(x_train, c(nrow(x_train), 125*45))
x_test <- array_reshape(x_test, c(nrow(x_test), 125*45))
#checking the reshaped data 
dim (x_train)
## [1] 7600 5625
# range normalization
range_norm <- function(x, a = 0, b = 1) {
  ( (x - min(x)) / (max(x) - min(x)) )*(b - a) + a
} 
x_train <- apply(x_train, 2,range_norm)
x_test <- apply(x_test, 2,range_norm)

library(MASS)
pca <- prcomp(x_train) # pca

#save(pca, file="pca.RData") # save pca data into a file so that delays can be avoided
#load("pca.RData") run only when you have saved a file
prop <- cumsum(pca$sdev^2)/sum(pca$sdev^2) # compute cumulative proportion of variance
Q <- length( prop[prop < 0.99] ) # only a handful is retained
xz_train <- pca$x[,1:Q] # extract first Q principal components
# visualize
cols <- as.numeric(as.factor(y_train)) #setting color as per category of daily activities
plot(xz_train, col = adjustcolor(cols, 0.7), pch= 19, main="Distribution of classes in PC1 and PC2") #plot

# map original test data points on to the learned subspace
xz_test <- predict(pca, x_test)[,1:Q]

# one-hot encoding of target variable
y_train<-factor(y_train) #factor response variable train data
y_train <- to_categorical(as.numeric(y_train)-1, num_classes = 19) #one-hot encoding
y_test<-factor(y_test) #factor response variable test data
y_test<-to_categorical(as.numeric(y_test)-1, num_classes = 19) #one-hot encoding

V <- ncol(xz_train)
N<-nrow(xz_train)
singlemodel <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = V) %>%
  layer_dense(units = 19, activation = "softmax")
singlemodel %>%compile(
  loss = "categorical_crossentropy", metrics = "accuracy",
  optimizer = optimizer_sgd(),
)
#train the model
fitsingle <- singlemodel %>% fit(
  x = xz_train, y = y_train,
  validation_data = list(xz_test, y_test),
  epochs = 100,
  verbose = 0
)

# to add a smooth line to points
smooth_line <- function(y) {
  x <- 1:length(y)
  out <- predict( loess(y ~ x) )
  return(out)
}
# some colors will be used later
cols <- c("black", "dodgerblue3", "gray50", "deepskyblue2")
# check performance ---> error
outsingle <- 1 - cbind(fitsingle$metrics$accuracy,
                       fitsingle$metrics$val_accuracy)
matplot(outsingle, pch = 19, ylab = "Error", xlab = "Epochs",col = adjustcolor(cols[1:2], 0.3),
        log = "y")
# on log scale to visualize better differences
matlines(apply(outsingle, 2, smooth_line), lty = 1, col = cols[1:2], lwd = 2)
legend("topright", legend = c("Training", "Test"),
       fill = cols[1:2], bty = "n")
singlemodel %>% evaluate(xz_test, y_test, verbose = 0)

V <- ncol(xz_train) #total number of columns in training input variable dataset
N<-nrow(xz_train)
#keras sequential model for multilayer neural network
modeldouble <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = V) %>% #First hidden layer considering 256 units  and relu as activation function and input shape as value of V
  layer_dense(units = 128, activation = "relu") %>% #Second hidden layer considering 128 units and relu as activation function
  layer_dense(units = 19, activation = "softmax") %>% #Output layer considering 19 output units and softmax as activation function
  #compling the above model by taking cross entropy as error function, accuracy as performance measure and stochastic gradient descent for optimization.
  compile(
    loss = "categorical_crossentropy", metrics = "accuracy",
    optimizer = optimizer_sgd(),
  )
# # Model fit for multiple neural netwrok with 2 hidden layers.
fit <- modeldouble %>% fit(
  x = xz_train, y = y_train, #traning data for the model xz_train is from PCA
  validation_data = list(xz_test, y_test), #testing data for validation
  epochs = 100, #no of epoch
  verbose = 0
)

# to add a smooth line to points
smooth_line <- function(y) {
  x <- 1:length(y)
  out <- predict( loess(y ~ x) )
  return(out)
}
# some colors will be used later
cols <- c("black", "dodgerblue3", "gray50", "deepskyblue2")
# check performance ---> error
outdouble <- 1 - cbind(fitsingle$metrics$accuracy,fitsingle$metrics$val_accuracy,fit$metrics$accuracy,fit$metrics$val_accuracy)
matplot(outdouble, pch = 19, ylab = "Error", xlab = "Epochs",
        col = adjustcolor(cols, 0.3),
        log = "y")
# on log scale to visualize better differences
matlines(apply(outdouble, 2, smooth_line), lty = 1, col = cols[1:2], lwd = 2)
legend("topright", legend = c("Training single", "Test single", "Train_double", "Test_double"), fill = cols, bty = "n")

apply (outdouble, 2, min)#minimun value of each error colum for training and testing datset for both single and double hidden layers.

modeldouble %>% evaluate(xz_test, y_test, verbose = 0)

#keras sequential model for multilayer neural network with one extra layer
#First hidden layer considering 256 units  and relu as activation function and input shape as value of V, kernel_regularizer as regularizer_l2() 
#for wieght decay regularisation , l is the hyperparameter, here value is 0.004 (selected via hit and trail) 
model_regularised <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape =V, kernel_regularizer = regularizer_l2(l = 0.004))%>% 
  
  layer_dense(units = 128, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.004))%>%
  #First hidden layer considering 128 units  and relu as activation function , kernel_regularizer as regularizer_l2() for wieght decay regularisation , and l is the hyperparameter, here value is 0.004 (selected via hit and trail) 
  
  layer_dense(units = 19, activation = "softmax") %>% #Output layer considering 10 output units and softmax as activation function
  #compling the above model by taking cross entropy as error function, accuracy as performance measure and stochastic gradient descent for optimization.
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(),
    metrics = "accuracy"
  )
# count parameters of regularised model
# Model fit for regularised multiple neural netwrok with 2 hidden layers.
# train and evaluate on test data at each epoch
fit_reg <- model_regularised %>% fit(
  x = xz_train, y = y_train,#traning data for the model
  validation_data = list(xz_test, y_test),#testing data for validation
  epochs = 100,#no of epoch
  verbose = 0
)
out_reg <- 1 - cbind(fit$metrics$accuracy,
                     fit$metrics$val_accuracy,
                     fit_reg$metrics$accuracy,
                     fit_reg$metrics$val_accuracy)#data with performance parameters for regularised model
# check performance
matplot(out_reg, pch = 19, ylab = "Error", xlab = "Epochs",
        col = adjustcolor(cols, 0.3),
        log = "y")#matrix plot of performance at each epoch
matlines(apply(out_reg, 2, smooth_line), lty = 1, col = cols, lwd = 2)#matrix lines connecting each points
legend("topright", legend = c("Training", "Test", "Train_reg", "Test_reg"),
       fill = cols, bty = "n")#legend for the plot.
apply(out_reg, 2, min)#minimun value of each error colum for training and testing datset for both regularised and non regularised models.

model_regularised %>% evaluate(xz_test, y_test, verbose = 0)

# get all weights
w_all <- get_weights(modeldouble) #all weights for 2 hidden layer model
w_all_reg <- get_weights(model_regularised) #all weights for regularised 2 hidden layer model
# weights of first hidden layer
# one input --> 64 units
w <- w_all[[3]][1,]
w_reg <- w_all_reg[[3]][1,]
# compare visually the magnitudes
par(mfrow = c(2,1), mar = c(2,2,0.5,0.5))#non regularised plot
r <- range(w)
n <- length(w)
plot(w, ylim = r, pch = 19, col = adjustcolor(1, 0.5))#adding lines in non regularised plot
abline(h = 0, lty = 2, col = "red")
segments(1:n, 0, 1:n, w)
#
plot(w_reg, ylim = r, pch = 19, col = adjustcolor(1, 0.5))# regularised plot
abline(h = 0, lty = 2, col = "red")#adding lines in regularised plot
segments(1:n, 0, 1:n, w_reg)

V<-ncol(xz_train)
N<-nrow(xz_train)
library(tfruns)
# split the test data in two halves: one for validation
# and the other for actual testing
set.seed(19200276)
# there are 2007 images in x_test
val <- sample(1:nrow(xz_test),760) #smaple rows for validation data
test <- setdiff(1:nrow(xz_test), val) # smaple rows for test data
x_val <- xz_test[val,] # predictor variable for validation data
y_val <- y_test[val,] # response variable for validation data
xtun_test <- xz_test[test,] #predictor variable for test data
ytun_test <- y_test[test,] #response variable for test data
# flags grid of nodes for 3 hidden layer and dropout
size1_set=c(256,128,64)
size2_set=c(256,128,64)
size3_set=c(256,128,64)
dropout_set=c(0,0.3,0.5,0.6)

lambda_set <- c(0, exp( seq(-6, -4, length = 9) ))
#running the model                                
runs <- tuning_run("tuning.R",
                   runs_dir = "tuning_pca",
                   flags = list(
                     dropout = dropout_set,
                     unit1 = size1_set,
                     unit2 = size2_set,
                     unit3 = size3_set,
                     lambda=lambda_set
                   ),sample = 0.3)
library(jsonlite) #importing jsonlite pakage
library(doParallel)
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel

library(tfruns)
read_metrics <- function(path, files = NULL)
  # 'path' is where the runs are --> e.g. "path/to/runs"
{
  path <- paste0(path, "/")
  if ( is.null(files) ) files <- list.files(path)
  n <- length(files)
  out <- vector("list", n)
  for ( i in 1:n ) {
    dir <- paste0(path, files[i], "/tfruns.d/")
    out[[i]] <- jsonlite::fromJSON(paste0(dir, "metrics.json"))
    out[[i]]$flags <- jsonlite::fromJSON(paste0(dir, "flags.json"))
    out[[i]]$evaluation <- jsonlite::fromJSON(paste0(dir, "evaluation.json"))
  }
  return(out)
}
plot_learning_curve <- function(x, ylab = NULL, cols = NULL, top = 3, span = 0.4, ...)
{
  # to add a smooth line to points
  smooth_line <- function(y) {
    x <- 1:length(y)
    out <- predict( loess(y ~ x, span = span) )
    return(out)
  }
  matplot(x, ylab = ylab, xlab = "Epochs", type = "n", ...)
  grid()
  matplot(x, pch = 19, col = adjustcolor(cols, 0.3), add = TRUE)
  tmp <- apply(x, 2, smooth_line)
  tmp <- sapply( tmp, "length<-", max(lengths(tmp)) )
  set <- order(apply(tmp, 2, max, na.rm = TRUE), decreasing = TRUE)[1:top]
  cl <- rep(cols, ncol(tmp))
  cl[set] <- "deepskyblue2"
  matlines(tmp, lty = 1, col = cl, lwd = 2)
}
out <- read_metrics("tuning_pca")
# extract validation accuracy and plot learning curve
acc <- sapply(out, "[[", "val_accuracy")
plot_learning_curve(acc, col = adjustcolor("black", 0.3), ylim = c(0.3, 1),
                    ylab = "Val accuracy", top = 3)
# all flag value result object
res <- ls_runs(metric_val_accuracy > 0.92,
               runs_dir = "tuning_pca", order = metric_val_accuracy)

res <- res[,c(2,4,8:14)]
res[1:10,]

#********************************************* End Code ***********************************************
