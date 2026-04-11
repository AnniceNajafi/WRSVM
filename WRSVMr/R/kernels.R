#' RBF kernel matrix
#'
#' Compute K(x, y) = exp(-gamma * ||x - y||^2) for two matrices.
#'
#' @param X1 Numeric matrix of shape (n1, d).
#' @param X2 Numeric matrix of shape (n2, d).
#' @param gamma Bandwidth parameter.
#' @param backend One of `"numpy"` or `"torch"`.
#' @return Kernel matrix of shape (n1, n2).
#' @export
rbf_kernel <- function(X1, X2, gamma = 0.5, backend = "numpy") {
  py <- reticulate::import("wrsvm")
  X1m <- as.matrix(X1); storage.mode(X1m) <- "double"
  X2m <- as.matrix(X2); storage.mode(X2m) <- "double"
  py$rbf_kernel(X1m, X2m, gamma = as.numeric(gamma), backend = backend)
}
