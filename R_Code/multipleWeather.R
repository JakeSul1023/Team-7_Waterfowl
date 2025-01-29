# Load required libraries
library(httr)
library(jsonlite)

# Function to retrieve weather forecast
get_weather_forecast <- function(latitude, longitude) {
  # Step 1: Get metadata for the location
  points_url <- sprintf("https://api.weather.gov/points/%.4f,%.4f", latitude, longitude)
  points_response <- GET(points_url, user_agent("DuckFlightPredictor (your_email@example.com)"))
  
  if (http_error(points_response)) {
    stop("Error retrieving points metadata for ", latitude, ", ", longitude, ": ", http_status(points_response)$message)
  }
  
  # Parse the points data
  points_data <- fromJSON(content(points_response, as = "text", encoding = "UTF-8"))
  
  # Extract the forecast URL
  forecast_url <- points_data$properties$forecast
  
  if (is.null(forecast_url)) {
    stop("Could not find forecast URL for ", latitude, ", ", longitude)
  }
  
  # Step 2: Retrieve forecast data
  forecast_response <- GET(forecast_url, user_agent("DuckFlightPredictor (your_email@example.com)"))
  if (http_error(forecast_response)) {
    stop("Error retrieving forecast data for ", latitude, ", ", longitude, ": ", http_status(forecast_response)$message)
  }
  
  # Parse the forecast data
  forecast_data <- fromJSON(content(forecast_response, as = "text", encoding = "UTF-8"))
  
  # Extract forecast periods
  forecast <- forecast_data$properties$periods
  return(forecast)
}

# Define key coordinates for the Mississippi Flyway (sample points)
flyway_coordinates <- list(
  list(lat = 35.0456, lon = -90.0489),  # Memphis, Tennessee
  list(lat = 33.4484, lon = -91.0671),  # Greenville, Mississippi
  list(lat = 31.3113, lon = -92.4451),  # Alexandria, Louisiana
  list(lat = 30.4515, lon = -91.1871),  # Baton Rouge, Louisiana
  list(lat = 40.6331, lon = -89.3985)   # Springfield, Illinois
)

# Collect forecasts for all locations
flyway_forecasts <- lapply(flyway_coordinates, function(coord) {
  get_weather_forecast(coord$lat, coord$lon)
})

# Print the short forecast for the first location
for (i in seq_along(flyway_forecasts)) {
  cat(sprintf("Location %d:\n", i))  # Print a header for each location
  print(flyway_forecasts[[i]][, c("name", "shortForecast", "temperature", "temperatureUnit")])
  cat("\n")  # Add a blank line between locations
}
