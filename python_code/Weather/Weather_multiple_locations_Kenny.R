# Load required libraries
library(httr)
library(jsonlite)
library(dplyr)

# Function to retrieve weather forecast from NWS API
get_weather_forecast <- function(latitude, longitude) {
  # Construct the URL for the points endpoint
  points_url <- sprintf("https://api.weather.gov/points/%.4f,%.4f", latitude, longitude)
  
  # Send GET request to the points endpoint
  points_response <- GET(points_url, user_agent("YourAppName (your_email@example.com)"))
  
  # Check for request errors
  if (http_error(points_response)) {
    stop("Error retrieving points metadata: ", http_status(points_response)$message)
  }
  
  # Parse the response as JSON
  points_data <- fromJSON(content(points_response, as = "text", encoding = "UTF-8"))
  
  # Extract the forecast URL
  forecast_url <- points_data$properties$forecast
  
  # Check if the forecast URL is available
  if (is.null(forecast_url)) {
    stop("Could not find forecast URL in the metadata.")
  }
  
  # Send GET request to the forecast endpoint
  forecast_response <- GET(forecast_url, user_agent("YourAppName (your_email@example.com)"))
  
  # Check for request errors
  if (http_error(forecast_response)) {
    stop("Error retrieving forecast data: ", http_status(forecast_response)$message)
  }
  
  # Parse the forecast data as JSON
  forecast_data <- fromJSON(content(forecast_response, as = "text", encoding = "UTF-8"))
  
  # Extract the forecast periods
  forecast <- forecast_data$properties$periods
  
  # Extract relevant fields, including humidity
  result <- data.frame(
    TimePeriod = forecast$name,
    Temperature = forecast$temperature,
    TemperatureUnit = forecast$temperatureUnit,
    WindSpeed = forecast$windSpeed,
    WindDirection = forecast$windDirection,
    ShortForecast = forecast$shortForecast,
    DetailedForecast = forecast$detailedForecast,
    Humidity = sapply(forecast$relativeHumidity, function(x) ifelse(is.null(x$value), NA, x$value))
  )
  
  return(result)
}

# Function to return predefined locations along the Mississippi Flyway
get_flyway_locations <- function() {
  return(data.frame(
    Name = c("Minneapolis, MN", "Dubuque, IA", "St. Louis, MO", "Memphis, TN", "New Orleans, LA"),
    Latitude = c(44.9778, 42.5006, 38.6270, 35.1495, 29.9511),
    Longitude = c(-93.2650, -90.6646, -90.1994, -90.0490, -90.0715)
  ))
}

# Function to print a simplified version of the forecast
print_simplified_forecast <- function(forecast, num_periods = 5) {
  if (nrow(forecast) < num_periods) {
    stop("The forecast data has fewer periods than requested.")
  }
  
  limited_forecast <- head(forecast, num_periods)
  simplified_forecast <- limited_forecast[, !(colnames(limited_forecast) %in% c("startTime", "endTime"))]
  
  for (i in 1:nrow(simplified_forecast)) {
    cat("Name:", simplified_forecast$TimePeriod[i], "\n")
    cat("Temperature:", simplified_forecast$Temperature[i], simplified_forecast$TemperatureUnit[i], "\n")
    cat("Wind:", simplified_forecast$WindSpeed[i], simplified_forecast$WindDirection[i], "\n")
    cat("Humidity:", simplified_forecast$Humidity[i], "%\n")
    cat("Short Forecast:", simplified_forecast$ShortForecast[i], "\n")
    cat("===================================\n")
  }
}

# Function to return the first 5 detailed forecast values
get_first_five_detailed_forecasts <- function(forecast) {
  detailed_forecasts <- forecast$DetailedForecast[1:5]
  return(detailed_forecasts)
}

# Function to check if snow is forecasted
is_snow_forecasted <- function(forecast) {
  snow_in_short <- grepl("Snow", forecast$ShortForecast, ignore.case = TRUE)
  snow_in_detailed <- grepl("Snow", forecast$DetailedForecast, ignore.case = TRUE)
  any_snow <- snow_in_short | snow_in_detailed
  return(forecast[any_snow, ])
}

is_severe_weather_forecasted <- function(forecast) {
  # Define severe weather keywords
  severe_keywords <- c("Thunderstorm", "Tornado", "Hurricane", "High Winds", 
                       "Severe", "Hail", "Flash Flood", "Blizzard", "Storm", "Damaging Winds")
  
  # Check if any of these words appear in shortForecast or detailedForecast
  severe_in_short <- grepl(paste(severe_keywords, collapse = "|"), forecast$ShortForecast, ignore.case = TRUE)
  severe_in_detailed <- grepl(paste(severe_keywords, collapse = "|"), forecast$DetailedForecast, ignore.case = TRUE)
  
  # Combine results to see if severe weather is mentioned
  any_severe <- severe_in_short | severe_in_detailed
  
  # Get the severe weather periods
  severe_forecast <- forecast[any_severe, ]
  
  # Print results
  if (nrow(severe_forecast) > 0) {
    cat("**Severe Weather Alert!** The following periods have severe weather:\n")
    print(severe_forecast)
  } else {
    cat("*No severe weather detected in the forecast.**\n")
  }
  
  return(severe_forecast)
}
# Function to fetch and save weather forecasts for all flyway locations
fetch_and_save_flyway_forecasts <- function(output_file = "flyway_forecasts.csv") {
  locations <- get_flyway_locations()
  flyway_forecasts <- NULL
  
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
      forecast$Location <- locations$Name[i]  
      forecast <- as_tibble(forecast)
      if (is.null(flyway_forecasts)) {
        flyway_forecasts <- forecast
      } else {
        flyway_forecasts <- bind_rows(flyway_forecasts, forecast)
      }
    }
  }
  
  if (!is.null(flyway_forecasts)) {
    write.csv(flyway_forecasts, output_file, row.names = FALSE)
    cat(sprintf("Forecasts saved to %s\n", output_file))
  } else {
    cat("No forecast data available to save.\n")
  }
}

# Example usage
fetch_and_save_flyway_forecasts()
severe_weather <- is_severe_weather_forecasted(forecast)
print(severe_weather)

# List of Forecast columns
# Number: of forecast
# Name: Keeps track of day and time such as Saturday Night
# StartTime: Very specific start date and time
# EndTime: Very specific end date and time
# isDaytime: Boolean for if the forecast period is day or night
# Temperature: temp in Fahrenheit by default
# TemperatureUnit: Unit for temp Fahrenheit by default
# TemperatureTrend: Detects if the temperature is expected to rise or fall
# ProbabilityOfPrecipitation: Chance of precipitation 
# WindSpeed: Forecasted wind speed
# WindDirection: Wind direction in compass format (e.g., NW)
# Icon: URL that provides an icon for the weather (e.g., sunny, cloudy)
# ShortForecast: Brief written description of the forecast
# DetailedForecast: In-depth written description of the forecast

