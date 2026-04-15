test_data <- function() {
  X <- as.matrix(read.csv(system.file("reference_X.csv", package = "WRSVMr"),
                           header = FALSE))
  y <- as.integer(read.csv(system.file("reference_y.csv", package = "WRSVMr"),
                            header = FALSE)[, 1])
  list(X = X, y = y)
}

test_that("fit + predict works for each strategy", {
  skip_if_not_installed("reticulate")
  d <- test_data()
  for (s in c("cs", "simmsvm", "ovo", "ovr")) {
    fit <- wrsvm_fit(d$X, d$y, strategy = s,
                      C = 10, gamma = 0.5, upsilon = 0.2)
    preds <- wrsvm_predict(fit, d$X)
    expect_length(preds, length(d$y))
    expect_true(all(preds %in% sort(unique(d$y))))
    expect_gt(mean(preds == d$y), 0.5)
  }
})

test_that("minority noise preserves majority class", {
  d <- test_data()
  y_noisy <- inject_outliers_minority(d$X, d$y, outlier_rate = 0.3, seed = 0)
  sizes <- table(d$y); maj <- as.integer(names(sizes)[which.max(sizes)])
  expect_gte(sum(y_noisy == maj), sum(d$y == maj))
})
