# ======================================================================================
# Read configuration
# ======================================================================================

args = commandArgs(trailingOnly = TRUE)
path_to_json = args[length(args)]
config = jsonlite::read_json(path_to_json)

functions_path = config[["functions_path"]]
data_path = config[["data_path"]]
product_path = config[["product_path"]]

# ======================================================================================
# Load functions
# ======================================================================================

source(functions_path)


# ======================================================================================
# Run simulation
# ======================================================================================

simulation_data = jsonlite::read_json(data_path)

processed_results = list()

for (iteration in seq_along(simulation_data)) {

  data = extract_data_components(simulation_data[[iteration]])

  result = fit_conformal_inference(data)

  processed = process_result(result)

  processed_results[[iteration]] = processed
}

# ======================================================================================
# Save results
# ======================================================================================

jsonlite::write_json(processed_results, product_path)
