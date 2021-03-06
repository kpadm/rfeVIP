% Generated by roxygen2 (4.1.1): do not edit by hand
% Please edit documentation in R/rfVIP_master.R
\name{createRFmodel}
\alias{createRFmodel}
\title{createRFmodel}
\usage{
createRFmodel(train.x, train.y, m.try, nSeed = 3456, nCores = 1)
}
\arguments{
\item{train.x}{dataframe containing x values}

\item{train.y}{vector containing y values}

\item{m.try}{tuned m.try value}

\item{nSeed}{integer containing seed value, defaults to 3456}

\item{nCores}{integer indicating number of cores used for parallal processing}
}
\value{
random forest model of class "train"
}
\description{
This function does the random forest modeling for a given x and y
}
\examples{
rf_model <- createRFmodel(train.x,train.y,nSeed=3456,nCores=4)
}

