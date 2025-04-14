path = "/home/tim/sciebo-thinky/fspb/bld/monte_carlo/R/n=30-d=5-c=non_stationary.json"

monte_carlo_data = rjson::fromJSON(file=path)


iteration = 1

#' Extract data components from the generated monte carlo data.
#'
#' @param data A list containing the generated monte carlo data for a single iteration.
#' @return A list containing the extracted components:
#'   - y: The y values. Has shape (N, length(time_grid)).
#'   - x: The x values. Has shape (N, 2, length(time_grid)).
#'   - new_y: The new y values. Has shape (length(time_grid),).
#'   - new_x: The new x values. Has shape (2, length(time_grid)).
#'   - time_grid: The time grid. Has shape (length(time_grid),).
extract_data_components = function(data) {
    new_y = data[["new_y"]]
    new_x = list(data[["new_x"]])

    time_grid = data[["time_grid"]]

    y = lapply(data[["y"]], function(x) list(x))
    x = data[["x"]]

    return(list(y=y, x=x, new_y=new_y, new_x=new_x, time_grid=time_grid))
}


data = extract_data_components(data=monte_carlo_data[[iteration]])

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



plot(
  data[["time_grid"]],
  result[["pred"]][[1]][[1]],
  type="l",
  col="black",
  ylim=c(-1, 3)
)
lines(
  data[["time_grid"]],
  data[["new_y"]],
  type="l",
  col="red",
)
lines(
  data[["time_grid"]],
  result[["lo"]][[1]][[1]],
  type="l",
  col="blue",
)
lines(
  data[["time_grid"]],
  result[["up"]][[1]][[1]],
  type="l",
  col="blue",
)
