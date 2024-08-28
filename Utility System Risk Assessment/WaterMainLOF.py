import arcpy
import pandas as pd
from arcgis.features import GeoAccessor, GeoSeriesAccessor
import os
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
from arcgis.gis import GIS
import yaml

# sign into arcgis online with user credentials
# Load credentials from CityLogins.yaml
with open("../CityLogins.yaml", "r") as file:
    config = yaml.safe_load(file)

# Function to get GIS object from city name
def get_gis(city_name):
    city_config = config['cities'][city_name]
    url = city_config['url']
    username = city_config['username']
    password = city_config['password']
    gis = GIS(url, username, password)
    return gis

def classify_kmeans(values, num_bins):
    values = values.values.reshape(-1, 1)  # Reshape for the KMeans function
    kmeans = KMeans(n_clusters=num_bins, random_state=0).fit(values)
    
    # Get the cluster assignments and cluster centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Get the sorted order of the cluster centers
    ordered_centers = np.argsort(centers, axis=0)

    # Create a mapping from old labels to new labels
    label_map = {old_label: new_label for new_label, old_label in enumerate(ordered_centers.flatten())}

    # Map the old labels to the new labels
    bins = np.vectorize(label_map.get)(labels)

    return bins

def calculate_break_density(fc, breaks, results_folder, uniqueid):
    """
    Calculate the break density for a given feature class and breaks feature class

    :param fc: The feature class to calculate break density for
    :param breaks: The breaks feature class
    :param results_folder: The folder to save the density raster
    :param uniqueid: The unique identifier field
    :return: A pandas dataframe with the break density score for each feature in the feature class
    """
    # select breaks that intersect with the main
    arcpy.management.SelectLayerByLocation(in_layer=breaks, overlap_type="INTERSECT", select_features=fc)
    
    # kernel density of the selected breaks
    density = arcpy.sa.KernelDensity(
        in_features=breaks,
        population_field="NONE",
        cell_size=37.1596514022946,
        area_unit_scale_factor="SQUARE_MILES",
        out_cell_values="DENSITIES",
        method="PLANAR",
        in_barriers=None
    )
    
    # save the density raster
    density_raster = f"{fc}_density_raster.tif"
    density_raster_path = os.path.join(results_folder, density_raster)
    density.save(density_raster_path)

    # add surface info to the main feature class
    arcpy.ddd.AddSurfaceInformation(
        in_feature_class=fc,
        in_surface=density_raster_path,
        out_property="Z_MAX",
        method="BILINEAR"
    )

    # make a dataframe from the feature class with only uniqueid and Z_MAX
    fc_df = pd.DataFrame.spatial.from_featureclass(fc, fields=[uniqueid, 'Z_MAX'])
    fc_df = fc_df.drop(columns='SHAPE')
    
    # if z_max is null, set it to 0
    fc_df['Z_MAX'] = fc_df['Z_MAX'].fillna(0)

    # classify the Z_MAX values into 11 bins
    fc_df['Break Density Score'] = classify_kmeans(fc_df['Z_MAX'], 11)
    
    return fc_df

# Define your User
user = 'Decatur'

# Connect to GIS
user_gis = get_gis(user)

# User Variables
workspace = r"memory"
results_folder = r"C:\Users\ggarcia\OneDrive - Abonmarche\Documents\GitHub\Utility-System-Risk\DecaturResults"
service_life_table = r"C:\Users\ggarcia\OneDrive - Abonmarche\Documents\GitHub\Utility-System-Risk\DecaturServiceLife.csv"
water_main_url = "https://services3.arcgis.com/oLBR41j9nVxBv9mh/arcgis/rest/services/Water_Distribution_System/FeatureServer/12"
breaks_url = "https://services3.arcgis.com/oLBR41j9nVxBv9mh/arcgis/rest/services/wMainBreaks/FeatureServer/0"
UniqueID = "FACILITYID"
InstallDate = "INSTALLDATE"
Material = "MATERIAL"

arcpy.env.workspace = workspace
arcpy.env.overwriteOutput = True
arcpy.env.maintainAttachments = False
coordinate_system = arcpy.SpatialReference(102690)
arcpy.env.outputCoordinateSystem = coordinate_system
dir_path = os.getcwd()

# export the water main feature service to a feature class where facilityID install date and material are not null
arcpy.conversion.ExportFeatures(water_main_url, "WaterMainFC", where_clause=f"{UniqueID} IS NOT NULL AND {InstallDate} IS NOT NULL AND {Material} IS NOT NULL")

# export the breaks feature service to a feature class
arcpy.conversion.ExportFeatures(breaks_url, "BreaksFC")

water_main = "WaterMainFC"
breaks = "BreaksFC"
columns = [UniqueID, InstallDate, Material]

# Water main feature class to pandas dataframe
water_main_df = pd.DataFrame.spatial.from_featureclass(water_main)
# keep only columns as specified in list
water_main_df = water_main_df[columns]
water_main_df = water_main_df.replace(r'^\s*$', np.nan, regex=True)
water_main_df = water_main_df.dropna()
# make sure UniqueID and material are strings and InstallDate is a datetime object
water_main_df[UniqueID] = water_main_df[UniqueID].astype(str)
water_main_df[Material] = water_main_df[Material].astype(str)
water_main_df[InstallDate] = pd.to_datetime(water_main_df[InstallDate], errors='coerce')

# # make a dataframe to store service life of different pipe materials
# pipe_materials = ['Cast Iron', 'Ductile Iron', 'Polyvinyl Chloride', 'Asbestos Cement', 'High Density Polyethylene', 'Copper', 'Galvanized Pipe']
# pipe_service_life = [75, 90, 90, 70, 100, 100, 50]
# pipe_service_life_df = pd.DataFrame({Material: pipe_materials, 'Service Life': pipe_service_life})

# read the service life table into a dataframe
pipe_service_life_df = pd.read_csv(service_life_table)

# copy the water main dataframe add rows for age, service life, and lof then calculate lof as age/service life
WM_sl_Calc_df = water_main_df.copy()
WM_sl_Calc_df['Age'] = datetime.now().year - WM_sl_Calc_df[InstallDate].dt.year
WM_sl_Calc_df = WM_sl_Calc_df.merge(pipe_service_life_df, left_on=Material, right_on='Material', how='left')
WM_sl_Calc_df['Service Life Score'] = WM_sl_Calc_df['Age'] / WM_sl_Calc_df['Service Life'] * 10

# round the Service life score and adjusted service life score values to the next whole number
WM_sl_Calc_df['Service Life Score'] = np.ceil(WM_sl_Calc_df['Service Life Score'])

# if the service life score value is greater than 10, set it to 10
WM_sl_Calc_df.loc[WM_sl_Calc_df['Service Life Score'] > 10, 'Service Life Score'] = 10

# if the service life score is less than or equal to 0 set it to 1
WM_sl_Calc_df.loc[WM_sl_Calc_df['Service Life Score'] <= 0, 'Service Life Score'] = 1

# save to csv
# output_path = os.path.join(results_folder, "Service_Life.csv")
# WM_sl_Calc_df.to_csv(output_path, index=False)

try:
    # spatial join water mains to breaks to get the pipe FacilityID into the breaks table
    if breaks and breaks != "":
        breaks_mains_join = "breaks_mains_join"
        arcpy.analysis.SpatialJoin(
            target_features=breaks,
            join_features=water_main,
            out_feature_class=breaks_mains_join,
            join_operation="JOIN_ONE_TO_ONE",
            join_type="KEEP_COMMON",
            match_option="INTERSECT"
        )

        # check the number of features in the spatial join result
        result_count = arcpy.management.GetCount(breaks_mains_join)
        if int(result_count.getOutput(0)) > 0:
            # convert the spatial join result to a pandas dataframe
            breaks_mains_join_df = pd.DataFrame.spatial.from_featureclass(breaks_mains_join)
            breaks_mains_join_df = breaks_mains_join_df[['OBJECTID', 'Join_Count', UniqueID]]
            breaks_mains_join_df = breaks_mains_join_df.dropna()

            # make a dataframe from the mains and only keep facilityid, and drop rows with null values
            water_main_df = pd.DataFrame.spatial.from_featureclass(water_main)
            water_main_df = water_main_df[[UniqueID]]
            water_main_df = water_main_df.dropna()

            # Ensure UniqueID columns are of the same type
            breaks_mains_join_df[UniqueID] = breaks_mains_join_df[UniqueID].astype(str)
            water_main_df[UniqueID] = water_main_df[UniqueID].astype(str)
            WM_sl_Calc_df[UniqueID] = WM_sl_Calc_df[UniqueID].astype(str)

            # use the breaks dataframe to get the count of breaks for each pipe and add it to the water main dataframe in a Breaks column
            # Group the breaks_mains_join_df by UniqueID and count the number of breaks for each UniqueID
            breaks_count = breaks_mains_join_df.groupby(UniqueID).size().reset_index(name='Breaks')

            # Merge the water_main_df with the breaks_count dataframe on UniqueID
            breaks_df = pd.merge(water_main_df, breaks_count, on=UniqueID, how='left')

            # Fill NaN values in the 'Breaks' column with 0
            breaks_df['Breaks'] = breaks_df['Breaks'].fillna(0)
            # remove rows where Breaks is 0
            breaks_df = breaks_df[breaks_df['Breaks'] > 0]

            # score the breaks
            def score_breaks(breaks):
                if breaks == 1:
                    return 8
                elif breaks >= 2:
                    return 10

            # Apply the function to the 'Breaks' column to calculate the 'Breaks_score'
            breaks_df['Breaks_score'] = breaks_df['Breaks'].apply(score_breaks)
            # save to csv
            output_path = os.path.join(results_folder, "Breaks.csv")
            breaks_df.to_csv(output_path, index=False)

            # Merge the dataframes on UniqueID
            LOF_df = pd.merge(WM_sl_Calc_df, breaks_df, on=UniqueID, how='left')
            # Fill NaN values in the 'Breaks_score' column with 0
            LOF_df['Breaks_score'] = LOF_df['Breaks_score'].fillna(0)

        else:
            print("No features in the spatial join result")
            LOF_df = WM_sl_Calc_df.copy()
except NameError:
    print("Variable 'breaks' is not defined")
    LOF_df = WM_sl_Calc_df.copy()

# Break Density Scoring
# make feature classes for CAS and DIP mains
arcpy.conversion.ExportFeatures(in_features=water_main, out_features="mains_CAS", where_clause=f"{Material} = 'CAS'")
arcpy.conversion.ExportFeatures(in_features=water_main, out_features="mains_DIP", where_clause=f"{Material} = 'DIP'")
# make a list of the feature classes
break_mains_fcs = ["mains_CAS", "mains_DIP"]
# for each feature class, calculate the break density, and merge the results into one dataframe
dfs = []
for fc in break_mains_fcs:
    df = calculate_break_density(fc, breaks, results_folder, UniqueID)
    dfs.append(df)
# concatenate the dataframes
break_density_df = pd.concat(dfs)
# merge the break density dataframe with the LOF dataframe
LOF_df = pd.merge(LOF_df, break_density_df, on=UniqueID, how='left')
# fill in missing break density scores with 0
LOF_df['Break Density Score'] = LOF_df['Break Density Score'].fillna(0)

# drop missing values again
# LOF_df= LOF_df.dropna()
# calculate LOF based of availible fields and make a csv of scores and weights used
# calculate the LOF as (Service Life Score x 0.50) + (Break Density Score x 0.25) + (Breaks Score x 0.25) if Break Density Score column exists
if 'Break Density Score' in LOF_df.columns:
    LOF_df.loc[:, 'LOF'] = (LOF_df['Service Life Score'] * 0.5) + (LOF_df['Break Density Score'] * 0.25) + (LOF_df['Breaks_score'] * 0.25)
    scores = [0.5, 0.25, 0.25]
    weights = ['Service Life Score', 'Break Density Score', 'Breaks_score']
    score_weights = pd.DataFrame({'Score': scores, 'Weight': weights})
    output_path = os.path.join(results_folder, "LOF_Score_Weights.csv")
# calculate the LOF as (Service Life Score x 0.50) + (Breaks Score x 0.50) if Breaks_score column exists
elif 'Breaks_score' in LOF_df.columns:
    LOF_df.loc[:, 'LOF'] = (LOF_df['Service Life Score'] * 0.5) + (LOF_df['Breaks_score'] * 0.5)
    scores = [0.5, 0.5]
    weights = ['Service Life Score', 'Breaks_score']
    score_weights = pd.DataFrame({'Score': scores, 'Weight': weights})
    output_path = os.path.join(results_folder, "LOF_Score_Weights.csv")
else:
    LOF_df.loc[:, 'LOF'] = LOF_df['Service Life Score']
    scores = [1]
    weights = ['Service Life Score']
    score_weights = pd.DataFrame({'Score': scores, 'Weight': weights})
    output_path = os.path.join(results_folder, "LOF_Score_Weights.csv")
LOF_df.loc[:, 'LOF'] = np.ceil(LOF_df['LOF'])
# save the score weights to a csv
score_weights.to_csv(output_path, index=False)

# Save the final dataframe to a csv file
output_path = os.path.join(results_folder, "Final_LOF.csv")
LOF_df.to_csv(output_path, index=False)

# erase the memory workspace
arcpy.management.Delete(workspace)