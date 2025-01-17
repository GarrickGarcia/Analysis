from arcgis.gis import GIS
from arcgis.geocoding import geocode
from arcgis.geometry import Geometry
from arcgis.features import FeatureLayer
from arcgis.geometry import filters
from arcgis.features import GeoAccessor, GeoSeriesAccessor
import pandas as pd
import arcpy

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

def geocode_address(address):
    """
    Geocodes an address and returns the geometry of the first match.
    Raises a ValueError if no matches are found.
    """
    geocoded_location_fs = geocode(address, as_featureset=True)
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
buffer_distance = '100 Feet'  # Distance in feet

# Geocode the input address
address_point = geocode_address(input_address)

arcpy_pt = dict_to_arcpy_point_geometry(address_point)

# make a feature class from the point
arcpy.management.CreateFeatureclass(r"memory", "pt", "POINT", spatial_reference=arcpy_pt.spatialReference)
with arcpy.da.InsertCursor("pt", "SHAPE@") as cursor:
    cursor.insertRow([arcpy_pt])

point_df = pd.DataFrame.spatial.from_featureclass("pt")
# list the spatial reference of the point
sr = arcpy_pt.spatialReference
print(sr.name)


workspace = r"memory"
arcpy.env.workspace = workspace

# Create a buffer around the arcpy point
buffer = "buffer"
arcpy.analysis.PairwiseBuffer(arcpy_pt, buffer, buffer_distance)

# Clip the parcels with the puffer
neighbors = 'neighbors'
arcpy.analysis.Clip(parcel_service_url, buffer, neighbors)

# convert the clipped parcels to a pandas dataframe
neighbors_df = pd.DataFrame.spatial.from_featureclass(neighbors)