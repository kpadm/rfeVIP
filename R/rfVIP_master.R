#' rfeTrainingSplit
#' @description This function does the training-test split of data
#' @description The split is 80 percent training and 20 percent test data
#' @param x dataframe containing x values
#' @param y vector containing y values
#' @param nSeed integer containing seed value, defaults to 3456
#' @return a list containing values in the following order: training y, training x,
#' test-train index, full y, full x
#' @examples
#' splitList <- rfeTrainingSplit(x,y,nSeed=3456)
#' @import randomForest
#' @import caret
rfeTrainingSplit <- function (x, y, nSeed=3456){


  # check for missing data
  # remove index of missing data
  y.index<-is.na(y)
  new.y<-y[!y.index]
  new.x<- x[!y.index,]

  # use na.roughfix function from random forests package to impute missing x data
  new.x<-na.roughfix(new.x)

  # split test data from the master data set using caret
  set.seed(nSeed)
  testTrainIndex <- createDataPartition(new.y, p = .8 ,list = FALSE, times = 1)

  # split training data from the master data set using caret
  train.y <- new.y[testTrainIndex,drop=FALSE]
  train.x <- new.x[testTrainIndex,]


  # return the splits of test and training data
  splitList <- list(train.y,train.x,testTrainIndex,new.y,new.x)
  names(splitList) <- c("trainy", "trainx", "testTrainIndex", "newy", "newx")
  return(splitList)

}

#' rfeTuningFunction
#' @description This function does one round of random forest modeling and finds the best performing mtry
#' @param train.x dataframe containing x values
#' @param train.y vector containing y values
#' @param nSeed integer containing seed value, defaults to 3456
#' @param nCores integer designating the number of cores for parallel processing, defaults to 1
#' @return an integer containing the mtry value with least error
#' @examples
#' m.try <- rfeTuningFunction(train.x,train.y,nSeed=3456,nCores=4)
#' @import randomForest
#' @import caret
#' @import doParallel
rfeTuningFunction <- function(train.x,train.y,nSeed=3456, nCores=1){

  #RF model for selected variables
  registerDoParallel(cores = nCores)

  #set-up cross-validation controls
  set.seed(nSeed)
  fitControl <- trainControl(
    method = "repeatedcv",
    number = 10,
    ## repeated ten times
    repeats = 10)



  rfgrid <-  expand.grid(mtry = c(2:10, 25, 50, 100))

  #create basic RF model

  rf_model<-train(train.y~.,data=train.x,method="rf",
                  verbose=TRUE,
                  tuneGrid = rfgrid,
                  preProc = c("center", "scale"),
                  importance=TRUE,allowParallel=TRUE)

  # isolate the best performing m.try value from the model

  return(rf_model$bestTune[,1])

}

#' createRFmodel
#' @description This function does the random forest modeling for a given x and y
#' @param train.x dataframe containing x values
#' @param train.y vector containing y values
#' @param m.try tuned m.try value
#' @param nSeed integer containing seed value, defaults to 3456
#' @param nCores integer indicating number of cores used for parallal processing
#' @return random forest model of class "train"
#' @examples
#' rf_model <- createRFmodel(train.x,train.y,nSeed=3456,nCores=4)
#' @import randomForest
#' @import caret
#' @import doParallel
createRFmodel <- function(train.x, train.y, m.try, nSeed = 3456, nCores=1){

  registerDoParallel(cores = nCores)
  set.seed(nSeed)

  # construct a fitness control function
  fitControl <- trainControl(
    method = "repeatedcv",
    number = 10,
    ## repeated ten times
    repeats = 10)


  # construct a random forest grid for the chosen m.try value
  rfgrid <-  expand.grid(mtry = c(m.try))

  # create a basic RF model
  rf_model<- train(train.y~.,data=train.x,method="rf",
                   verbose=TRUE,
                   tuneGrid = rfgrid,
                   trControl = fitControl,
                   preProc = c("center", "scale"),
                   importance=TRUE,
                   allowParallel=TRUE)

  # return the rf_model
  return(rf_model)

}


#' rfVIP
#' @description This function performs recursive feature elimination based on the random forest variable importance parameter
#' @param x dataframe containing the x values
#' @param y vector containing y values
#' @param nSeed integer value of seed, default value is 3456
#' @param nCores integer value for multicore computation, default value is 1
#' @return the final random forest model with no negative variable importance variables
#' @examples
#' rf_model <- rfVIP(x,y,nSeed=3456,nCores=1)
#' @import caret
#' @import dplyr
#' @import randomForest
#' @import doParallel
#' @export

rfVIP <- function(x,y,nSeed=3456,nCores=1){

  registerDoParallel(cores = nCores)

  # get the training and test data split
  splitList <- rfeTrainingSplit(x, y, nSeed)

  # get the results out of the list
  train.y <- splitList$trainy
  train.x <- splitList$trainx
  testTrainIndex <- splitList$testTrainIndex
  new.y <- splitList$newy
  new.x <- splitList$newx

  # tune the random forest model
  # get the best m.try value
  m.try <- rfeTuningFunction(train.x,train.y,nSeed,nCores)

  # get the rf_model
  rf_model <- createRFmodel(train.x,train.y,m.try,nSeed,nCores)

  # check for positive VIPs
  vip <- varImp(rf_model, scale=FALSE)
  mat.imp <- as.matrix(vip$importance)
  filt.var <- names(mat.imp[which(mat.imp[,1] > 0),])

  # check if all VIPs are positive
  if (length(filt.var) < nrow(vip$importance)){
    # if all VIPs are not positive
    changed.x <- x[, filt.var]
    rfVIP(changed.x, y,nSeed, nCores)
  } else {

    # if all VIPs are positive
    selectVar <- rf_model$coefnames
    changed.x <- x[,selectVar]

    splitList <- rfeTrainingSplit(changed.x, y, nSeed)

    # get the results out of the list
    testTrainIndex <- splitList$testTrainIndex
    train.y <- splitList$trainy
    train.x <- splitList$trainx
    new.x <- splitList$newx
    new.y <- splitList$newy

    m.try <- rfeTuningFunction(train.x,train.y,nSeed,nCores)

    # do one round of modeling
    rf_model <- createRFmodel(train.x,train.y,m.try,nSeed,nCores)

    # get some basic parameters of the model
    # predict the model using the test split
    pred.model <- predict(rf_model,new.x[-testTrainIndex,])
    R2 <- lm(pred.model ~ new.y[-testTrainIndex]) %>% summary(.) %>% .$r.squared
    performance <-list(RMSE=RMSE(pred.model,new.y[-testTrainIndex]),R2=R2)
    save(performance, rf_model, selectVar , file=paste0("final_model_results"))
    # return the final random forest model
    return(rf_model)


  }

}
