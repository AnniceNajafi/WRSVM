#' Inject majority-centroid outliers
#'
#' Flip labels of majority-class samples that sit farthest from the majority
#' centroid to a uniformly random minority class.
#'
#' @param X Numeric matrix.
#' @param y Class labels.
#' @param outlier_rate Fraction of total samples to flip. Default 0.14.
#' @param seed Optional integer seed.
#' @return Corrupted label vector of the same length as `y`.
#' @export
inject_outliers_majority <- function(X, y, outlier_rate = 0.14, seed = NULL) {
  py <- reticulate::import("wrsvm")
  X_mat <- as.matrix(X); storage.mode(X_mat) <- "double"
  y_vec <- if (is.factor(y)) as.integer(y) else y
  seed_arg <- if (is.null(seed)) NULL else as.integer(seed)
  out <- py$inject_outliers_majority(X_mat, y_vec, as.numeric(outlier_rate),
                                       seed_arg)
  as.vector(out)
}

#' Inject minority-targeted outliers
#'
#' For each minority class, flip a fraction `outlier_rate` of its labels to a
#' uniformly random other class. Majority class is left untouched.
#'
#' @param X Numeric matrix. Unused; kept for API symmetry.
#' @param y Class labels.
#' @param outlier_rate Fraction of each minority class to flip. Default 0.3.
#' @param seed Optional integer seed.
#' @return Corrupted label vector of the same length as `y`.
#' @export
inject_outliers_minority <- function(X, y, outlier_rate = 0.3, seed = NULL) {
  py <- reticulate::import("wrsvm")
  X_mat <- as.matrix(X); storage.mode(X_mat) <- "double"
  y_vec <- if (is.factor(y)) as.integer(y) else y
  seed_arg <- if (is.null(seed)) NULL else as.integer(seed)
  out <- py$inject_outliers_minority(X_mat, y_vec, as.numeric(outlier_rate),
                                       seed_arg)
  as.vector(out)
}
