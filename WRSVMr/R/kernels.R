#' Kernel matrix
#'
#' Compute a kernel Gram matrix for two input matrices. Dispatches to the
#' Python `wrsvm.compute_kernel` over reticulate.
#'
#' Supported kernels:
#' \itemize{
#'   \item `rbf`       : `exp(-gamma * ||x - y||^2)` (default)
#'   \item `linear`    : `x . y`
#'   \item `poly`      : `(gamma * x . y + coef0)^degree`
#'   \item `sigmoid`   : `tanh(gamma * x . y + coef0)`
#'   \item `laplacian` : `exp(-gamma * ||x - y||_1)`
#' }
#'
#' @param X1 Numeric matrix of shape (n1, d).
#' @param X2 Numeric matrix of shape (n2, d).
#' @param kernel One of `"rbf"`, `"linear"`, `"poly"`, `"sigmoid"`, `"laplacian"`.
#' @param gamma Kernel bandwidth (rbf, poly, sigmoid, laplacian).
#' @param degree Polynomial degree (only used when `kernel = "poly"`).
#' @param coef0 Independent term for `"poly"` and `"sigmoid"`.
#' @param backend One of `"numpy"` or `"torch"` (torch currently only accelerates rbf).
#' @return Kernel matrix of shape (n1, n2).
#' @export
compute_kernel <- function(X1, X2, kernel = "rbf",
                           gamma = 0.5, degree = 3L, coef0 = 0.0,
                           backend = "numpy") {
  py <- reticulate::import("wrsvm")
  X1m <- as.matrix(X1); storage.mode(X1m) <- "double"
  X2m <- as.matrix(X2); storage.mode(X2m) <- "double"
  py$compute_kernel(
    X1m, X2m,
    kernel = kernel,
    gamma = as.numeric(gamma),
    degree = as.integer(degree),
    coef0 = as.numeric(coef0),
    backend = backend
  )
}

#' RBF (Gaussian) kernel matrix
#'
#' Shortcut for `compute_kernel(..., kernel = "rbf")`.
#'
#' @param X1 Numeric matrix of shape (n1, d).
#' @param X2 Numeric matrix of shape (n2, d).
#' @param gamma Bandwidth parameter.
#' @param backend One of `"numpy"` or `"torch"`.
#' @return Kernel matrix.
#' @export
rbf_kernel <- function(X1, X2, gamma = 0.5, backend = "numpy") {
  compute_kernel(X1, X2, kernel = "rbf", gamma = gamma, backend = backend)
}
