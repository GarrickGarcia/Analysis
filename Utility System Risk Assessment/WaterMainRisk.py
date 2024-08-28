import pandas as pd
import math
import plotly.express as px

def normalize_column(df, column_name):
    """
    Normalizes a column in the DataFrame to a scale from 1 to 10.
    Adds the normalized values in a new column with a suffix '_normalized'.
    
    :param df: DataFrame
    :param column_name: Name of the column to normalize
    """
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    
    # Define the new column name for normalized scores
    normalized_column_name = column_name + '_normalized'
    
    # Avoid division by zero if all values are the same
    if min_value == max_value:
        df[normalized_column_name] = df[column_name]
    else:
        df[normalized_column_name] = ((df[column_name] - min_value) * (9 / (max_value - min_value)) + 1).apply(math.ceil)
    
    return df

def create_heatmap(df: pd.DataFrame, lof_column: str, cof_column: str, length_column: str):
    """
    Creates a plotly heatmap of COF vs LOF

    :param df: DataFrame containing the data
    :type df: pd.DataFrame
    :param lof_column: Name of the LOF column
    :type lof_column: str
    :param cof_column: Name of the COF column
    :type cof_column: str
    :param length_column: Name of the length column
    :type length_column: str
    """
    # Create the density heatmap
    fig = px.density_heatmap(
        df,
        x=lof_column,
        y=cof_column,
        z=length_column,
        histfunc='sum',
        text_auto=True,
        nbinsx=11,  # Ensure bins match with x-axis range
        nbinsy=11   # Ensure bins match with y-axis range
    )

    # Adjust the layout, including the plot_bgcolor to make 'empty' cells visible (as white)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # Set background to transparent
        width=900,  # Adjust as needed
        height=750,  # Adjust as needed
        margin=dict(t=50, l=50, r=50, b=50),  # Adjust margins if needed
        coloraxis_colorbar=dict(title='sum of Length (Ft)')  # Set colorbar title
    )

    # Update axes ranges to ensure cells are not cut off
    fig.update_xaxes(range=[-0.5, 10.5])
    fig.update_yaxes(range=[-0.5, 10.5])

    # Return the figure object
    return fig

# File paths for the COF and LOF CSV files
cof_file_path = r"C:\Users\ggarcia\OneDrive - Abonmarche\Documents\GitHub\Utility-System-Risk\DecaturResults\Final_COF.csv"
lof_file_path = r"C:\Users\ggarcia\OneDrive - Abonmarche\Documents\GitHub\Utility-System-Risk\DecaturResults\Final_LOF.csv"
risk_file_path = r"C:\Users\ggarcia\OneDrive - Abonmarche\Documents\GitHub\Utility-System-Risk\DecaturResults\Final_Risk.csv"
unique_id = 'FACILITYID'
length_column = 'LENGTH'
image_path = r"C:\Users\ggarcia\OneDrive - Abonmarche\Documents\GitHub\Utility-System-Risk\DecaturResults\heatmap.html"

# make dfs from csv files
cof_df = pd.read_csv(cof_file_path)
lof_df = pd.read_csv(lof_file_path)

# merge the two dfs
risk_df = pd.merge(cof_df, lof_df, on=unique_id, suffixes=('_cof', '_lof'))

# Normalize the COF and LOF columns
risk_df = normalize_column(risk_df, 'COF')
risk_df = normalize_column(risk_df, 'LOF')

# Calculate the risk score as COF * LOF
risk_df['RISK'] = risk_df['COF_normalized'] * risk_df['LOF_normalized']

# create the heatmap
plot = create_heatmap(risk_df, 'LOF_normalized', 'COF_normalized', length_column)

# Save the heatmap as an image file
plot.write_html(image_path)

# save the risk_df as a csv
risk_df.to_csv(risk_file_path, index=False)