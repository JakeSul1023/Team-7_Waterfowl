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
# Function to print a simplified version of the forecast
print_simplified_forecast <- function(forecast, num_periods = 5) {
  # Ensure the forecast has at least num_periods rows
  if (nrow(forecast) < num_periods) {
    stop("The forecast data has fewer periods than requested.")
  }
  
  # Select the first `num_periods` rows
  limited_forecast <- head(forecast, num_periods)
  
  # Exclude the startTime and endTime columns
  simplified_forecast <- limited_forecast[, !(colnames(limited_forecast) %in% c("startTime", "endTime"))]
  
  # Print each period
  for (i in 1:nrow(simplified_forecast)) {
    cat("Name:", simplified_forecast$name[i], "\n")
    cat("Temperature:", simplified_forecast$temperature[i], simplified_forecast$temperatureUnit[i], "\n")
    cat("Wind:", simplified_forecast$windSpeed[i], simplified_forecast$windDirection[i], "\n")
    cat("Short Forecast:", simplified_forecast$shortForecast[i], "\n")
    cat("===================================\n")
  }
}
# Function to return the first 5 detailedForecast values
get_first_five_detailed_forecasts <- function(forecast) {
  # Extract the first 5 detailedForecast values
  detailed_forecasts <- forecast$detailedForecast[1:5]
  
  # Return the result
  return(detailed_forecasts)
}

# Function to check if snow is forecasted
is_snow_forecasted <- function(forecast) {
  # Check the shortForecast column for "Snow"
  snow_in_short <- grepl("Snow", forecast$shortForecast, ignore.case = TRUE)
  
  # Check the detailedForecast column for "Snow"
  snow_in_detailed <- grepl("Snow", forecast$detailedForecast, ignore.case = TRUE)
  
  # Combine results to see if snow is mentioned in either column
  any_snow <- snow_in_short | snow_in_detailed
  
  # Return rows where snow is forecasted
  return(forecast[any_snow, ])
}

# Prints the forecasted snow
snow_forecast <- is_snow_forecasted(forecast)
print(snow_forecast)


# Print the detailed forecast of the weather
#first_five_detailed_forecasts <- get_first_five_detailed_forecasts(forecast)
#print(first_five_detailed_forecasts)

# Print the first 5 periods of the forecast
#print_simplified_forecast(forecast)

# List of Forecast columns
# Number: of forecast
# Name: Keeps track of day and time such as Saturday Night
# StartTime: Very specific start date and time
# endTime: Very specific end date and time
# isDaytime: Boolean for if the forecast period is day or night
# Temperature: temp in Fahrenheit by default
# temperatureUnit: Unit for temp Fahrenheit by default
# temperatureTrend:Detects if the temperature is expected to rise or fall
# Probality of Preception: Chance of precipitation 
# windSpeed: Forecasted wind speed
# Icon: Url that will get an icon for the weather like a little sun
# ShortForecast: Written description of all the forecast
# DetailedForecast: In depth written description of all the forecast
# 
