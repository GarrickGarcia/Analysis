from arcgis.gis import GIS
from arcgis.geocoding import geocode
from arcgis.geometry import Geometry, filters
from arcgis.features import FeatureLayer, FeatureSet
from arcgis.map import Map
import pandas as pd
import arcpy
import yaml

with open("../../CityLogins.yaml", "r") as file:
    config = yaml.safe_load(file)

def get_gis(city_name):
    """
    Returns a connected GIS object.
    """
    city_config = config['cities'][city_name]
    url = city_config['url']
    username = city_config['username']
    password = city_config['password']
    return GIS(url, username, password)

def create_address(street, city, state, zip_code):
    """
    Concatenates address components into a single address string.
    """
    return f"{street}, {city}, {state} {zip_code}"

def geocode_address(address, location_type="rooftop"):
    """
    Geocodes an address and returns the geometry dictionary of the first match.
    """
    geocoded_fs = geocode(address=address, location_type=location_type, as_featureset=True)
    if not geocoded_fs or not geocoded_fs.features:
        raise ValueError(f"Address '{address}' could not be geocoded.")
    geom = geocoded_fs.features[0].geometry
    if "spatialReference" not in geom:
        geom["spatialReference"] = {"wkid": 4326}
    return geom

def dict_to_arcpy_point_geometry(api_geom_dict):
    """
    Converts a dictionary-based geometry into an arcpy.PointGeometry.
    """
    x = api_geom_dict["x"]
    y = api_geom_dict["y"]
    srid = api_geom_dict["spatialReference"]["wkid"]
    sr = arcpy.SpatialReference(srid)
    return arcpy.PointGeometry(arcpy.Point(x, y), sr)

def get_parcel_number(arcpy_point, parcel_service_url):
    """
    Finds the intersecting parcel using Select By Location and returns its parcelnumb as a string.
    """
    arcpy.management.MakeFeatureLayer(parcel_service_url, "parcels_lyr")
    arcpy.management.CreateFeatureclass("memory", "temp_pt", "POINT", spatial_reference=arcpy_point.spatialReference)
    with arcpy.da.InsertCursor("temp_pt", "SHAPE@") as cur:
        cur.insertRow([arcpy_point])
    arcpy.management.MakeFeatureLayer("temp_pt", "point_lyr")
    arcpy.management.SelectLayerByLocation("parcels_lyr", "INTERSECT", "point_lyr")
    with arcpy.da.SearchCursor("parcels_lyr", ["parcelnumb"]) as cursor:
        for row in cursor:
            return str(row[0])
    return ""

def process_neighbors(arcpy_point, parcel_service_url, buffer_distance):
    """
    Buffers an arcpy point, clips parcels, converts to a neighbors DataFrame.
    """
    arcpy.management.CreateFeatureclass("memory", "pt", "POINT", spatial_reference=arcpy_point.spatialReference)
    with arcpy.da.InsertCursor("pt", "SHAPE@") as cursor:
        cursor.insertRow([arcpy_point])

    arcpy.analysis.PairwiseBuffer("pt", "buffer", buffer_distance)
    arcpy.analysis.Clip(parcel_service_url, "buffer", "neighbors")
    arcpy.management.FeatureToPoint("neighbors", "neighbors_pts", "INSIDE")
    df = pd.DataFrame.spatial.from_featureclass("neighbors_pts")
    df = df[["address", "owner", "SHAPE"]]
    return df

# Main script
source_city = "Abonmarche"
gis_conn = get_gis(source_city)
print(f"Connected to the source GIS of {source_city}.")

report_map_id = "12603c6980c44a09aae5d41e5baf8a70"
webmap_item = gis_conn.content.get(report_map_id)
report_map = Map(webmap_item)

parcel_service_url = "https://services6.arcgis.com/o5a9nldztUcivksS/arcgis/rest/services/Benton_Harbor_Parcels/FeatureServer/0"
street_address = "95 W Main St"
city = "Benton Harbor"
state = "MI"
zip_code = "49022"
input_address = create_address(street_address, city, state, zip_code)
address_point = geocode_address(input_address)
arcpy_point = dict_to_arcpy_point_geometry(address_point)

arcpy.env.workspace = "memory"
fl = FeatureLayer(parcel_service_url)
srid = fl.properties.extent.spatialReference.wkid
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(srid)
arcpy.env.overwriteOutput = True

parcel_number = get_parcel_number(arcpy_point, parcel_service_url)
neighbors_df = process_neighbors(arcpy_point, parcel_service_url, "200 Feet")

for lyr in report_map.layers:
    if lyr.title.lower() == "parcel of interest":
        lyr.definition_expression = f"parcelnumb = '{parcel_number}'"

report_map
