#' Fit a Weighted Relaxed SVM classifier
#'
#' Trains one of four decomposition strategies (Crammer-Singer, SimMSVM,
#' One-vs-One, One-vs-Rest) for the WRSVM formulation on multiclass data.
#' Internally calls the Python `wrsvm` package via reticulate.
#'
#' @param X Numeric matrix of shape (N, d).
#' @param y Class labels (integer or factor). Length must equal `nrow(X)`.
#' @param strategy One of `"cs"`, `"simmsvm"`, `"ovo"`, `"ovr"`. Default `"cs"`.
#' @param C Regularization parameter. Default 100.
#' @param gamma Kernel bandwidth (rbf, poly, sigmoid, laplacian). Default 0.1.
#' @param upsilon Relaxation penalty. Default 0.2.
#' @param kernel One of `"rbf"`, `"linear"`, `"poly"`, `"sigmoid"`, `"laplacian"`. Default `"rbf"`.
#' @param degree Polynomial degree (used when `kernel = "poly"`). Default 3.
#' @param coef0 Independent term for `"poly"` and `"sigmoid"` kernels. Default 0.
#' @param solver CVXPY solver name. Default `"CLARABEL"`. Use `"SCS_GPU"`
#'   for GPU acceleration when `scs` is built with CUDA.
#' @param kernel_backend One of `"numpy"` (CPU) or `"torch"` (GPU if available).
#'
#' @return A `wrsvm_model` object wrapping the Python classifier.
#' @export
#'
#' @examples
#' \dontrun{
#' data(iris)
#' X <- scale(as.matrix(iris[, 1:4]))
#' y <- as.integer(iris$Species)
#' fit <- wrsvm_fit(X, y, strategy = "cs",
#'                  C = 100, gamma = 0.1, upsilon = 0.2)
#' preds <- wrsvm_predict(fit, X)
#' mean(preds == y)
#' }
wrsvm_fit <- function(X, y,
                      strategy = "cs",
                      C = 100.0,
                      gamma = 0.1,
                      upsilon = 0.2,
                      kernel = "rbf",
                      degree = 3L,
                      coef0 = 0.0,
                      solver = "CLARABEL",
                      kernel_backend = "numpy") {
  stopifnot(is.matrix(X) || is.data.frame(X))
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y_vec <- if (is.factor(y)) as.integer(y) else y

  py <- reticulate::import("wrsvm")
  clf <- py$WRSVMClassifier(
    strategy = strategy,
    C = as.numeric(C), gamma = as.numeric(gamma), upsilon = as.numeric(upsilon),
    kernel = kernel,
    degree = as.integer(degree),
    coef0 = as.numeric(coef0),
    solver = solver, kernel_backend = kernel_backend
  )
  clf$fit(X, y_vec)

  structure(list(py_clf = clf, classes = clf$classes_,
                  strategy = strategy,
                  C = C, gamma = gamma, upsilon = upsilon,
                  kernel = kernel, degree = degree, coef0 = coef0),
            class = "wrsvm_model")
}

#' Predict class labels from a fitted WRSVM model
#'
#' @param model A `wrsvm_model` returned by [wrsvm_fit()].
#' @param X_new Numeric matrix of new samples.
#' @return Integer vector of predicted class labels.
#' @export
wrsvm_predict <- function(model, X_new) {
  stopifnot(inherits(model, "wrsvm_model"))
  X_new <- as.matrix(X_new)
  storage.mode(X_new) <- "double"
  preds <- model$py_clf$predict(X_new)
  as.vector(preds)
}

#' @export
print.wrsvm_model <- function(x, ...) {
  cat("WRSVM model\n")
  cat(sprintf("  strategy: %s\n", x$strategy))
  cat(sprintf("  kernel: %s\n", x$kernel))
  cat(sprintf("  C = %g, gamma = %g, upsilon = %g\n",
              x$C, x$gamma, x$upsilon))
  cat(sprintf("  classes: %s\n", paste(x$classes, collapse = ", ")))
  invisible(x)
}
