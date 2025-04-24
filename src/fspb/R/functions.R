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


#' Fit the concurrent model.
#'
#' Taken from https://github.com/paolo-vergo/conformal-fd/blob/main/R/concurrent.R
concurrent = function() {

  train.fun = function(x,t,y) {


    yy=lapply(y, rapply, f = c) # Now a list of n components (join the internal p lists)
    xx=lapply(x, rapply, f = c)
    yyy=do.call(rbind, yy) #Convert the previous yy to a matrix
    xxx=do.call(rbind, xx)

    full = ncol(yyy)
    full_x = ncol(xxx)

    if(full!=full_x)
      stop(" The concurrent model requires a value of x for each value of y.
           If the number of dimension is diffent, then use another model.
           For instance the mean_fun model is available.")


    coeff=vapply(1:full, function(i)
      lm(formula = yyy[,i] ~  xxx[,i ])$coefficients,numeric(2))

    return(list(coeff=coeff))
  }

  # Prediction function
  predict.fun = function(out,newx,t) {

    #Redefine structure

    new_xx=lapply(newx, rapply, f = c)
    new_xxx=do.call(rbind, new_xx)
    temp=out$coeff


    l=length(newx)
    ya=temp[1,]
    yaa=t(matrix(replicate(l,ya),nrow=length(ya)))
    sol=new_xxx*temp[2,]+yaa


    list_sol=lapply(seq_len(nrow(sol)), function(i) list(sol[i,]))

    return(list_sol)
  }

  return(list(train.fun=train.fun, predict.fun=predict.fun))
}


#' Fit conformal inference bands.
#'
#' @param data A list containing the extracted data components.
#' @return A result object from conformalInference.fd::conformal.fun.split.
fit_conformal_inference = function(data) {

  n_samples = length(data[["y"]])

  train_and_pred_functions = concurrent()

  time_grid = replicate(30, data[["time_grid"]], simplify = FALSE)

  # Extract only the second dimension (of the covariate dimension) of the x and new_x
  # lists (the first dimension corresponds to the intercept)
  x = lapply(data[["x"]], "[", 2)
  x0 = lapply(data[["new_x"]], "[", 2)

  result = conformalInference.fd::conformal.fun.split(
    x=x,
    t_x=time_grid,
    y=data[["y"]],
    t_y=time_grid,
    x0=x0,
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
