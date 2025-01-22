# Load required library
library(httr)
library(jsonlite)

# Function to retrieve weather forecast
get_weather_forecast <- function(latitude, longitude) {
  # Step 1: Get metadata for the location
  points_url <- sprintf("https://api.weather.gov/points/%.4f,%.4f", latitude, longitude)
  points_response <- GET(points_url, user_agent("YourAppName (your_email@example.com)"))
  
  if (http_error(points_response)) {
    stop("Error retrieving points metadata: ", http_status(points_response)$message)
  }
  
  # Manually parse the response as JSON
  points_data <- fromJSON(content(points_response, as = "text", encoding = "UTF-8"))
  
  # Extract forecast URL
  forecast_url <- points_data$properties$forecast
  hourly_forecast_url <- points_data$properties$forecastHourly
  
  if (is.null(forecast_url) || is.null(hourly_forecast_url)) {
    stop("Could not find forecast URLs in the metadata.")
  }
  
  # Step 2: Get the forecast data
  forecast_response <- GET(forecast_url, user_agent("YourAppName (your_email@example.com)"))
  if (http_error(forecast_response)) {
    stop("Error retrieving forecast data: ", http_status(forecast_response)$message)
  }
  
  # Manually parse the response as JSON
  forecast_data <- fromJSON(content(forecast_response, as = "text", encoding = "UTF-8"))
  
  # Step 3: Extract and return the forecast
  forecast <- forecast_data$properties$periods
  return(forecast)
}

# Example usage for Cookeville, Tennessee (latitude: 36.1628, longitude: -85.5016)
latitude <- 36.1628
longitude <- -85.5016
forecast <- get_weather_forecast(latitude, longitude)

# Print the forecast
print(forecast)

