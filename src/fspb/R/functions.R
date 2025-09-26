#' Extract data components from the generated monte carlo data.
#'
#' @param simulation_data A list containing the generated monte carlo data for a single
#' iteration.
#' @return A list containing the extracted components:
#'   - y: The y values. Has shape (N, length(time_grid)).
#'   - x: The x values. Has shape (N, 2, length(time_grid)).
#'   - new_y: The new y values. Has shape (length(time_grid),).
#'   - new_x: The new x values. Has shape (2, length(time_grid)).
#'   - time_grid: The time grid. Has shape (length(time_grid),).
extract_data_components = function(simulation_data) {
    new_y = simulation_data[["new_y"]]
    new_x = list(simulation_data[["new_x"]])

    time_grid = simulation_data[["time_grid"]]

    y = lapply(simulation_data[["y"]], function(x) list(x))
    x = simulation_data[["x"]]

    return(list(y=y, x=x, new_y=new_y, new_x=new_x, time_grid=time_grid))
}


#' CONCURRENT LINEAR MODEL (y has shape N x 1 x T)
#' x:     list(N) of list(P) of list(T)
#' y:     list(N) of list(1) of list(T)
#' newx:  list(M) of list(P) of list(T)
#' Returns coeff: P x T
#' @importFrom stats lm.fit
#' @export
concurrent <- function() {

  to_array3 <- function(x) {
    N <- length(x); P <- length(x[[1]]); TT <- length(x[[1]][[1]])
    arr <- array(NA_real_, dim = c(N, P, TT))
    for (i in seq_len(N)) for (p in seq_len(P)) {
      arr[i, p, ] <- as.numeric(unlist(x[[i]][[p]], recursive = TRUE, use.names = FALSE))
    }
    arr
  }

  # y is N x 1 x T -> return N x T matrix
  y_to_mat <- function(y) {
    N <- length(y); stopifnot(length(y[[1]]) == 1L)
    TT <- length(y[[1]][[1]])
    mat <- matrix(NA_real_, nrow = N, ncol = TT)
    for (i in seq_len(N)) {
      mat[i, ] <- as.numeric(unlist(y[[i]][[1]], recursive = TRUE, use.names = FALSE))
    }
    mat
  }

  train.fun <- function(x, t, y) {
    X  <- to_array3(x)   # N x P x T
    Y  <- y_to_mat(y)    # N x T

    if (dim(X)[1] != nrow(Y)) stop("N in x and y differ.")
    if (dim(X)[3] != ncol(Y)) stop("T in x and y differ.")

    N <- dim(X)[1]; P <- dim(X)[2]; TT <- dim(X)[3]
    beta <- matrix(NA_real_, nrow = P, ncol = TT)

    for (j in seq_len(TT)) {
      Xj <- X[, , j, drop = FALSE][, , 1]  # N x P
      yj <- Y[, j]
      fit <- stats::lm.fit(x = Xj, y = yj) # no implicit intercept; include it in x if desired
      bj <- rep(NA_real_, P); bj[seq_along(fit$coefficients)] <- fit$coefficients
      beta[, j] <- bj
    }

    list(coeff = beta) # P x T
  }

  predict.fun <- function(out, newx, t) {
    B    <- out$coeff              # P x T
    Xnew <- to_array3(newx)        # M x P x T

    if (dim(Xnew)[2] != nrow(B)) stop("P in newx and model differ.")
    if (dim(Xnew)[3] != ncol(B)) stop("T in newx and model differ.")

    M <- dim(Xnew)[1]; TT <- dim(Xnew)[3]
    Yhat <- matrix(NA_real_, nrow = M, ncol = TT)

    for (j in seq_len(TT)) {
      Yhat[, j] <- as.numeric(Xnew[, , j, drop = FALSE][, , 1] %*% B[, j])
    }

    to_return = lapply(seq_len(M), function(i) as.list(Yhat[i, ]))
    # list(list(unlist(to_return[[1]])))
    to_return
  }

  list(train.fun = train.fun, predict.fun = predict.fun)
}


#' Fit conformal inference bands.
#'
#' @param data A list containing the extracted data components.
#' @param significance_level The significance level.
#' @param fit_method The method to use for predicting the new function values, given
#'   the training data. Can be "mean" or "linear".
#' @return A result object from conformalInference.fd::conformal.fun.split.
fit_conformal_inference = function(data, significance_level, fit_method) {

  set.seed(0)

  n_samples = length(data[["y"]])

  if (fit_method == "mean") {
    train_and_pred_functions = conformalInference.fd::mean_lists()
  } else if (fit_method == "linear") {
    train_and_pred_functions = concurrent()
  } else {
    stop("Invalid fit method: ", fit_method)
  }

  time_grid = replicate(n_samples, data[["time_grid"]], simplify = FALSE)

  x = data[["x"]]
  x0 = data[["new_x"]]

  result = conformalInference.fd::conformal.fun.split(
    x=x,
    t_x=time_grid,
    y=data[["y"]],
    t_y=time_grid,
    x0=x0,
    train.fun = train_and_pred_functions[["train.fun"]],
    predict.fun = train_and_pred_functions[["predict.fun"]],
    alpha = significance_level,
    split = NULL,
    seed = FALSE,
    randomized = TRUE,
    seed.rand = FALSE,
    verbose = FALSE,
    rho = 0.5,
    s.type = "identity"
  )

  return(result)
}


#' Process the result of conformal inference.
#'
#' @param conformal_inference_result A result object from
#' conformalInference.fd::conformal.fun.split.
#' @return A list containing the processed result. Has the following components:
#'   - estimate: The estimated function.
#'   - lower: The lower bound of the confidence band.
#'   - upper: The upper bound of the confidence band.
process_result = function(conformal_inference_result) {
  processed = list(
    estimate=conformal_inference_result[["pred"]][[1]][[1]],
    lower=conformal_inference_result[["lo"]][[1]][[1]],
    upper=conformal_inference_result[["up"]][[1]][[1]]
  )
  return(processed)
}
