library(httr)     # For API requests
library(jsonlite) # For JSON parsing
library(dplyr)    # For data frame 

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

# Function to fetch weather forecasts for all flyway locations and save to CSV
fetch_and_save_flyway_forecasts <- function(output_file = "flyway_forecasts.csv") {
  locations <- get_flyway_locations()
  flyway_forecasts <- NULL  # Start as NULL to avoid rbind() conflicts
  
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
      forecast$Location <- locations$Name[i]  # Add location column
      
      # Convert to tibble (data frame without row names) and ensure clean binding
      forecast <- as_tibble(forecast)
      
      # Safely bind rows while avoiding row name duplication
      if (is.null(flyway_forecasts)) {
        flyway_forecasts <- forecast
      } else {
        flyway_forecasts <- bind_rows(flyway_forecasts, forecast)
      }
    }
  }
  
  # Save to CSV only if data exists
  if (!is.null(flyway_forecasts)) {
    write.csv(flyway_forecasts, output_file, row.names = FALSE)
    cat(sprintf("Forecasts saved to %s\n", output_file))
  } else {
    cat("No forecast data available to save.\n")
  }
}



# Example usage for a single location
latitude <- 36.1628
longitude <- -85.5016
forecast <- get_weather_forecast(latitude, longitude)

# Print the first 5 periods of the forecast
print_simplified_forecast(forecast)

# Print the detailed forecast of the weather
first_five_detailed_forecasts <- get_first_five_detailed_forecasts(forecast)
print(first_five_detailed_forecasts)

# Prints the forecasted snow
snow_forecast <- is_snow_forecasted(forecast)
print(snow_forecast)

# Fetch and save forecasts for all flyway locations
fetch_and_save_flyway_forecasts()

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
