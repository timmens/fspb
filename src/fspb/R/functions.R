#' Extract data components from the generated monte carlo data.
#'
#' @param monte_carlo_data A list containing the generated monte carlo data for a single
#' iteration.
#' @return A list containing the extracted components:
#'   - y: The y values. Has shape (N, length(time_grid)).
#'   - x: The x values. Has shape (N, 2, length(time_grid)).
#'   - new_y: The new y values. Has shape (length(time_grid),).
#'   - new_x: The new x values. Has shape (2, length(time_grid)).
#'   - time_grid: The time grid. Has shape (length(time_grid),).
extract_data_components = function(monte_carlo_data) {
    new_y = monte_carlo_data[["new_y"]]
    new_x = list(monte_carlo_data[["new_x"]])

    time_grid = monte_carlo_data[["time_grid"]]

    y = lapply(monte_carlo_data[["y"]], function(x) list(x))
    x = monte_carlo_data[["x"]]

    return(list(y=y, x=x, new_y=new_y, new_x=new_x, time_grid=time_grid))
}


#' Fit conformal inference bands.
#'
#' @param data A list containing the extracted data components.
#' @return A result object from conformalInference.fd::conformal.fun.split.
fit_conformal_inference = function(data) {

  n_samples = length(data[["y"]])

  train_and_pred_functions = conformalInference.fd::mean_lists()

  time_grid = replicate(30, data[["time_grid"]], simplify = FALSE)

  result = conformalInference.fd::conformal.fun.split(
    x=data[["x"]],
    t_x=time_grid,
    y=data[["y"]],
    t_y=time_grid,
    x0=data[["new_x"]],
    train.fun = train_and_pred_functions[["train.fun"]],
    predict.fun = train_and_pred_functions[["predict.fun"]],
    alpha = 0.1,
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
