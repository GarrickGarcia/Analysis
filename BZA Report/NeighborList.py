from arcgis.gis import GIS
from arcgis.geocoding import geocode
from arcgis.geometry import Geometry, filters
from arcgis.features import FeatureLayer, FeatureSet
import pandas as pd
import arcpy
import yaml

# Load credentials from CityLogins.yaml
import yaml
with open("../../CityLogins.yaml", "r") as file:
    config = yaml.safe_load(file)

def get_gis(city_name):
    """Get a GIS object for the specified city.

    Args:
        city_name (str): The name of the city.

    Returns:
        GIS: A connected GIS object.
    """
    city_config = config['cities'][city_name]
    url = city_config['url']
    username = city_config['username']
    password = city_config['password']
    gis = GIS(url, username, password)
    return gis

def create_address(street, city, state, zip_code):
    """Concatenate address components into a single address string.

    Args:
        street (str): The street address.
        city (str): The city.
        state (str): The state.
        zip_code (str): The ZIP code.

    Returns:
        str: The full address string.
    """
    return f"{street}, {city}, {state} {zip_code}"

def geocode_address(address, location_type="rooftop"):
    """
    Geocodes an address and returns the geometry of the first match.
    Raises a ValueError if no matches are found.
    
    Args:
        address (str): A single-line address.
        location_type (str): 'rooftop', 'street', or 'none'.
    
    Returns:
        dict: A dictionary representing the geometry of the first geocoded match.
    """
    geocoded_location_fs = geocode(
        address=address,
        location_type=location_type,
        as_featureset=True
    )
    if not geocoded_location_fs or not geocoded_location_fs.features:
        raise ValueError(f"Address '{address}' could not be geocoded.")

    address_point = geocoded_location_fs.features[0].geometry
    if "spatialReference" not in address_point:
        address_point["spatialReference"] = {"wkid": 4326}

    return address_point

def dict_to_arcpy_point_geometry(api_geom_dict):
    """
    Converts a dictionary-based geometry (from the ArcGIS Python API)
    into an arcpy.PointGeometry for use with arcpy geoprocessing.
    """
    x = api_geom_dict["x"]
    y = api_geom_dict["y"]
    srid = api_geom_dict["spatialReference"]["wkid"]
    spatial_ref = arcpy.SpatialReference(srid)
    return arcpy.PointGeometry(arcpy.Point(x, y), spatial_ref)

# Connect to the GIS
source_city = 'Abonmarche'
gis = get_gis(source_city)
print(f"Connected to the source GIS of {source_city}.")

# Address components
street_address = "95 W Main St"
city = "Benton Harbor"
state = "MI"
zip_code = "49022"

# Concatenate address
input_address = create_address(street_address, city, state, zip_code)

# Parameters
parcel_service_url = "https://services6.arcgis.com/o5a9nldztUcivksS/arcgis/rest/services/Benton_Harbor_Parcels/FeatureServer/0"  # Replace with the actual URL of the parcel layer
parcel_layer = FeatureLayer(parcel_service_url)
buffer_distance = '200 Feet'  # Distance in feet

# Geocode the input address
address_point = geocode_address(input_address, location_type="rooftop")
# Convert your point dict into a FeatureSet so it can be displayed
point_fset = FeatureSet([{"attributes": {}, "geometry": address_point}])

# make a map widget and display the geocoded point ands the parcels
map1 = gis.map("Benton Harbor, MI")
map1.content.add(parcel_layer)
map1.content.add(point_fset)
map1

arcpy_pt = dict_to_arcpy_point_geometry(address_point)

# make a feature class from the point
arcpy.management.CreateFeatureclass(r"memory", "pt", "POINT", spatial_reference=arcpy_pt.spatialReference)
with arcpy.da.InsertCursor("pt", "SHAPE@") as cursor:
    cursor.insertRow([arcpy_pt])

workspace = r"memory"
arcpy.env.workspace = workspace
# Get the coordinate system of the parcel layer
coordinate_system = arcpy.SpatialReference(parcel_layer.properties.extent.spatialReference.wkid)
arcpy.env.outputCoordinateSystem = coordinate_system

# Create a buffer around the arcpy point
buffer = "buffer"
arcpy.analysis.PairwiseBuffer(arcpy_pt, buffer, buffer_distance)

# Clip the parcels with the puffer
neighbors = 'neighbors'
arcpy.analysis.Clip(parcel_service_url, buffer, neighbors)

# feature to point
neighbors_points = 'neighbors_points'
arcpy.management.FeatureToPoint(neighbors, neighbors_points, "INSIDE")

# convert the clipped parcels to a pandas dataframe
neighbors_df = pd.DataFrame.spatial.from_featureclass(neighbors_points)
# keep only columns address, owner, and SHAPE
neighbors_df = neighbors_df[['address', 'owner', 'SHAPE']]

# add neighbors_df to the map
neighbors_df.spatial.plot(map_widget=map1)
