wrsvm_py <- NULL

.onLoad <- function(libname, pkgname) {
  wrsvm_py <<- reticulate::import("wrsvm", delay_load = TRUE)
}
