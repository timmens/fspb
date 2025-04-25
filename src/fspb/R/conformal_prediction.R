# ======================================================================================
# Read configuration
# ======================================================================================

args = commandArgs(trailingOnly = TRUE)
path_to_json = args[length(args)]
config = jsonlite::read_json(path_to_json)

functions_script_path = config[["functions_script_path"]]
simulation_data_path = config[["simulation_data_path"]]
significance_level = config[["significance_level"]]
fit_method = config[["fit_method"]]
results_path = config[["results_path"]]

# ======================================================================================
# Load functions
# ======================================================================================

source(functions_script_path)


# ======================================================================================
# Run simulation
# ======================================================================================

simulation_data = jsonlite::read_json(simulation_data_path)

processed_results = list()

for (iteration in seq_along(simulation_data)) {

  data = extract_data_components(simulation_data[[iteration]])

  result = fit_conformal_inference(
    data,
    significance_level = significance_level,
    fit_method = fit_method
  )

  processed = process_result(result)

  processed_results[[iteration]] = processed
}

# ======================================================================================
# Save results
# ======================================================================================

jsonlite::write_json(processed_results, results_path)
