# Load required libraries
library(httr)
library(jsonlite)
library(dplyr)
library(geosphere)  # For calculating distance between coordinates
library(dplyr)      # For data manipulation

# Function to get the NOAA grid ID and coordinates for a latitude/longitude
get_noaa_grid <- function(latitude, longitude) {
  url <- sprintf("https://api.weather.gov/points/%.4f,%.4f", latitude, longitude)
  response <- GET(url, user_agent("MyApp (myemail@example.com)"))
  
  if (http_error(response)) {
    return(data.frame(Latitude = latitude, Longitude = longitude, GridID = NA, GridX = NA, GridY = NA))
  }
  
  data <- fromJSON(content(response, as = "text", encoding = "UTF-8"))
  
  return(data.frame(
    Latitude = latitude,
    Longitude = longitude,
    GridID = data$properties$gridId,
    GridX = data$properties$gridX,
    GridY = data$properties$gridY
  ))
}

# Generate a list of grid points for a region
scan_region_for_grid_points <- function(lat_min, lat_max, lon_min, lon_max, step = 0.1) {
  grid_data <- data.frame()
  
  for (lat in seq(lat_min, lat_max, by = step)) {
    for (lon in seq(lon_min, lon_max, by = step)) {
      grid_info <- get_noaa_grid(lat, lon)
      grid_data <- rbind(grid_data, grid_info)
    }
  }
  
  return(grid_data)
}

# Example: Scan a small region around Cookeville, TN
lat_min <- 36.0
lat_max <- 36.3
lon_min <- -85.7
lon_max <- -85.3

grid_points <- scan_region_for_grid_points(lat_min, lat_max, lon_min, lon_max, step = 0.1)

# Save results
write.csv(grid_points, "noaa_grid_points.csv", row.names = FALSE)

# Print sample results
print(grid_points)

# Load precomputed NOAA grid points table
load_noaa_grid_points <- function(file_path = "noaa_grid_points.csv") {
  if (!file.exists(file_path)) {
    stop("Grid points file not found. Please generate the NOAA grid table first.")
  }
  return(read.csv(file_path))
}

# Function to find the nearest NOAA grid point to a duck's location
find_nearest_noaa_grid <- function(duck_lat, duck_lon, grid_points) {
  # Calculate distance from the duck's location to each NOAA grid point
  distances <- distVincentySphere(
    matrix(c(rep(duck_lon, nrow(grid_points)), rep(duck_lat, nrow(grid_points))), ncol = 2),
    matrix(c(grid_points$Longitude, grid_points$Latitude), ncol = 2)
  )
  
  # Find the closest grid point
  nearest_index <- which.min(distances)
  
  # Return the matched grid point
  return(grid_points[nearest_index, ])
}

# Example usage
grid_points <- load_noaa_grid_points()  # Load the precomputed NOAA grid table

# Example duck location (latitude and longitude)
duck_lat <- 36.1500
duck_lon <- -85.5000

nearest_grid <- find_nearest_noaa_grid(duck_lat, duck_lon, grid_points)
print(nearest_grid)
