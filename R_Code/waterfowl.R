# Install and load required packages
# Define the list of required packages

# Load libraries
library(sp)
library(terra)
library(sf)
library(caret)
library(fields)
library(rnaturalearth)
library(ggplot2)

# Step 1: Load Dataset
# Assuming the dataset is already in your environment
waterfowl_data <- Mallard_Connectivity_Recent_Data

# Step 2: Inspect Dataset Structure
str(waterfowl_data)
head(waterfowl_data)

# Ensure required columns are present
if (!all(c("location.long", "location.lat", "event.id") %in% colnames(waterfowl_data))) {
  stop("Dataset must contain 'location-long', 'location-lat', and 'event-id' columns!")
}

# Step 3: Create Spatial Data
sp_data <- st_as_sf(
  waterfowl_data,
  coords = c("location.long", "location.lat"),
  crs = 4326  # WGS84 CRS
)

# Step 4: Create Raster from Observed Data
r <- rast(ext(sp_data), resolution = 0.1)  # Adjust resolution as needed
observed_raster <- rasterize(
  sp_data,
  r,
  field = "event.id",
  fun = mean
)

# Step 5: Prepare Data for Prediction
# Normalize observed data
waterfowl_data$event_id_normalized <- scale(waterfowl_data$`event.id`, center = TRUE, scale = TRUE)

# Step 6: Train Predictive Model
set.seed(123)
model <- train(
  event_id_normalized ~ `location.long` + `location.lat`,
  data = waterfowl_data,
  method = "rf"
)

# Step 7: Predict Future Locations
predicted_data <- predict(model, newdata = waterfowl_data)
sp_data$predicted <- predicted_data

# Create Predicted Raster
predicted_raster <- rasterize(
  sp_data,
  r,
  field = "predicted",
  fun = mean
)

# Step 8: Overlay Current vs. Predicted Data
overlay_raster <- stack(observed_raster, predicted_raster)

# Step 9: Visualization
# Plot Current vs Predicted
par(mfrow = c(1, 1))
plot(overlay_raster, main = "Observed (Red) vs Predicted (Blue)", col = c("red", "blue"))

# Step 10: Advanced Visualization with ggplot2
observed_df <- as.data.frame(observed_raster, xy = TRUE)
predicted_df <- as.data.frame(predicted_raster, xy = TRUE)

ggplot() +
  geom_tile(data = observed_df, aes(x = x, y = y, fill = layer), alpha = 0.7) +
  geom_tile(data = predicted_df, aes(x = x, y = y, fill = layer), alpha = 0.5) +
  scale_fill_gradientn(colors = terrain.colors(10)) +
  labs(title = "Overlay of Observed vs Predicted Waterfowl Distribution", fill = "Density") +
  theme_minimal()

# Save Final Output as GeoTIFF
writeRaster(observed_raster, "observed_raster.tif", format = "GTiff", overwrite = TRUE)
writeRaster(predicted_raster, "predicted_raster.tif", format = "GTiff", overwrite = TRUE)

#hey