library(httr)     # For API requests
library(jsonlite) # For JSON parsing

# Function to retrieve weather forecast
get_weather_forecast <- function(latitude, longitude) {
  points_url <- sprintf("https://api.weather.gov/points/%.4f,%.4f", latitude, longitude)
  points_response <- GET(points_url, user_agent("YourAppName (your_email@example.com)"))
  
  if (http_error(points_response)) {
    stop("Error retrieving points metadata: ", http_status(points_response)$message)
  }
  
  points_data <- fromJSON(content(points_response, as = "text", encoding = "UTF-8"))
  forecast_url <- points_data$properties$forecast
  
  if (is.null(forecast_url)) {
    stop("Could not find forecast URL in the metadata.")
  }
  
  forecast_response <- GET(forecast_url, user_agent("YourAppName (your_email@example.com)"))
  if (http_error(forecast_response)) {
    stop("Error retrieving forecast data: ", http_status(forecast_response)$message)
  }
  
  forecast_data <- fromJSON(content(forecast_response, as = "text", encoding = "UTF-8"))
  forecast <- forecast_data$properties$periods
  
  # Extract relevant fields
  result <- data.frame(
    TimePeriod = forecast$name,
    Temperature = forecast$temperature,
    TemperatureUnit = forecast$temperatureUnit,
    WindSpeed = forecast$windSpeed,
    WindDirection = forecast$windDirection,
    BarometricPressure = ifelse(is.null(forecast$probabilityOfPrecipitation$value), NA, forecast$probabilityOfPrecipitation$value)
  )
  return(result)
}

# Predefined locations along the Mississippi Flyway
locations <- data.frame(
  Name = c("Minneapolis, MN", "Dubuque, IA", "St. Louis, MO", "Memphis, TN", "New Orleans, LA"),
  Latitude = c(44.9778, 42.5006, 38.6270, 35.1495, 29.9511),
  Longitude = c(-93.2650, -90.6646, -90.1994, -90.0490, -90.0715)
)

# Fetch weather forecasts for all locations and combine into one data frame
flyway_forecasts <- data.frame()

for (i in 1:nrow(locations)) {
  cat(sprintf("Fetching forecast for %s (%f, %f)\n", 
              locations$Name[i], locations$Latitude[i], locations$Longitude[i]))
  forecast <- tryCatch(
    get_weather_forecast(locations$Latitude[i], locations$Longitude[i]),
    error = function(e) {
      cat(sprintf("Error fetching forecast for %s: %s\n", locations$Name[i], e$message))
      return(NULL)
    }
  )
  if (!is.null(forecast)) {
    # Add location name to the forecast data
    forecast$Location <- locations$Name[i]
    # Combine with the main data frame
    flyway_forecasts <- rbind(flyway_forecasts, forecast)
  }
}

# Save the combined forecasts to a CSV file
output_file <- "flyway_forecasts.csv"
write.csv(flyway_forecasts, output_file, row.names = FALSE)
cat(sprintf("Forecasts saved to %s\n", output_file))
