% Generated by roxygen2 (4.1.1): do not edit by hand
% Please edit documentation in R/rfVIP_master.R
\name{rfVIP}
\alias{rfVIP}
\title{rfVIP}
\usage{
rfVIP(x, y, nSeed = 3456, nCores = 1)
}
\arguments{
\item{x}{dataframe containing the x values}

\item{y}{vector containing y values}

\item{nSeed}{integer value of seed, default value is 3456}

\item{nCores}{integer value for multicore computation, default value is 1}
}
\value{
the final random forest model with no negative variable importance variables
}
\description{
This function performs recursive feature elimination based on the random forest variable importance parameter
}
\examples{
rf_model <- rfVIP(x,y,nSeed=3456,nCores=1)
}

