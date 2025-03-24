

from duck_migration_prediction import *

#import duck_migration_prediction as dmp


# Example call
duck_trajectories, duck_gdf = load_duck_data("ShortTermSetData(Aug-Sept).csv")  # Replace with actual file path

# Example call
flyway_bbox = get_flyway_region()

# Example call
all_timestamps = get_all_timestamps(duck_trajectories)


#Call
lat = 39.0
lon = -90.0
token = "pKsYMZQINiCtWEbiUdvJHImQDqYlZmhu"

forecast_json = fetch_noaa_forecast_for_point(lat, lon, token)
print(forecast_json)



# Example call
# Use a small date range for demonstration
#start_date = datetime.now() - timedelta(days=7)
#end_date = datetime.now() - timedelta(days=6)
#weather_data = fetch_weather_data(weather_stations, start_date, end_date)


# Instead of trying to directly use or convert the Stations instance
# We need to fetch the stations data first using the fetch() method

# Modified approach to handle station data
stations_instance = Stations()
stations_data = stations_instance.fetch()

# Filter stations to ensure they have the required fields
valid_stations = []
for station in stations_data.itertuples():
    # Convert namedtuple to dictionary with required fields
    station_dict = {
        'id': getattr(station, 'id', f"station_{len(valid_stations)}"),  # Create ID if missing
        'name': getattr(station, 'name', 'Unknown'),
        'latitude': getattr(station, 'latitude', None),
        'longitude': getattr(station, 'longitude', None),
        'elevation': getattr(station, 'elevation', 0)
    }
    # Only include stations with valid coordinates
    if station_dict['latitude'] is not None and station_dict['longitude'] is not None:
        valid_stations.append(station_dict)

# Use the filtered station list
start_date = datetime.now() - timedelta(days=7)
end_date = datetime.now()
weather_data = fetch_weather_data(valid_stations[:50], start_date, end_date)  # Limit to 50 stations for testing






# Use the modified function with reduced resolution to minimize API calls
future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 3)]  # Reduced to 2 days
forecast_data = fetch_weather_forecast_robust(flyway_bbox, resolution=3.0, forecast_dates=future_dates)

# =====================================================




#now good till here and down
# GOOOD up to here
# -------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Example call
weather_rasters = create_weather_raster_timeseries(weather_data,load_duck_data, flyway_bbox, resolution=0.5)

# Example call
station_tree, station_coords = build_kdtree_for_stations(weather_stations)

# Example call - use a sample location in the flyway region
sample_lat, sample_lon = 40.0, -90.0  # Near St. Louis, MO
nearest_stations = find_nearest_stations(sample_lat, sample_lon, station_tree, weather_stations)

# Example call
point1_lat, point1_lon = 40.0, -90.0  # Near St. Louis, MO
point2_lat, point2_lon = 42.0, -88.0  # Near Chicago, IL
distance = haversine_distance(point1_lat, point1_lon, point2_lat, point2_lon)

# Example call
bearing = calculate_bearing(point1_lat, point1_lon, point2_lat, point2_lon)

# Example call
sample_time = datetime.now() - timedelta(days=6, hours=12)  # A time within our weather data range
point_weather = extract_weather_at_point(weather_rasters, sample_time, sample_lat, sample_lon)

# Example call
weather_gradients = calculate_weather_gradient(
    weather_rasters, sample_time, point1_lat, point1_lon, point2_lat, point2_lon
)

# Example call
sample_wspd = 15.0  # Wind speed in km/h
sample_wdir = 225.0  # Wind direction in degrees (SW wind)
sample_bearing = 180.0  # Duck traveling south
wind_assistance = calculate_wind_assistance(sample_wspd, sample_wdir, sample_bearing)

# Example call
engineered_features = engineer_features(duck_trajectories, weather_rasters, station_tree, weather_stations)