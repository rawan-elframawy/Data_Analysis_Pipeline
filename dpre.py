import numpy as np   # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation and analysis
import seaborn as sns # Creating visually appealing statistical plots.
import matplotlib.pyplot as plt  # Import Matplotlib for data visualization
import subprocess # To invoke the next file
import sys # Get dataset path argument from invoker file


def preprocess_and_save():
    clean_data()
    reduce_data()
    encoded_df = transform_data()
    discretize_data()

    save(encoded_df)


def clean_data():
    # 1. Remove null values
    df.dropna(inplace=True)
    # 2. Drop non-correlated columns
    df.drop(columns=["Z_CostContact", "Z_Revenue"], axis=1, inplace=True)


def reduce_data():
    # The lines below group "Married" and "Together" into the category "Couple" and "Divorced", "Widow", "Alone", "YOLO", "Absurd" into the category "Alone".
    df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'], 'Couple')
    df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'], 'Alone')

    # Combining different dataframe into a single column to reduce the number of dimension
    df['num_of_children'] = df['Kidhome'] + df['Teenhome']
    df['total_money_spent'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
    df['total_num_of_accepted_cam'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']
    df['total_num_of_purchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']

    # We group "Basic" education into "1n Cycle", "2n Cycle", "Graduation", and "Master" into "2n Cycle", and "PhD" into "3n Cycle".
    df['Education'] = df['Education'].replace(['Basic'], '1n Cycle')
    df['Education'] = df['Education'].replace(['2n Cycle', 'Graduation', 'Master'], '2n Cycle')
    df['Education'] = df['Education'].replace(['PhD'], '3n Cycle')

    # Make an Age Column
    df['Age'] = 2024 - df["Year_Birth"]

    # Convert "Dt_Customer" to datetime with the correct format
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
    # Define the reference date
    reference_date = pd.to_datetime('03-11-2023')
    # Calculate the number of days
    df['days_since_enroll'] = (reference_date - df['Dt_Customer']).dt.days

    # Deleting some column to reduce dimension and complexity of model
    columns_to_delte = ["AcceptedCmp1" , "AcceptedCmp2", "AcceptedCmp3" , "AcceptedCmp4","AcceptedCmp5", "Response",
                        "NumWebVisitsMonth", "NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases" , 
                        "Kidhome", "Teenhome","MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", 
                        "MntGoldProds", "Year_Birth", "Dt_Customer","ID"]
    df.drop(columns=columns_to_delte, axis=1, inplace=True)


def transform_data():
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=["Education", "Marital_Status"])

    # note that you can still deal with the non-one hot encoded dataframe using df which i used globally throughout the whole code
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Apply Min-Max scaling to the selected columns
    columns_to_scale = df_encoded.columns.tolist()
    df_encoded[columns_to_scale] = scaler.fit_transform(df_encoded[columns_to_scale])
    return df_encoded


def discretize_data():
    # Do only two
    # binning / histogram analysis / clustering analysis / decision tree analysis
    pass


def save(encoded_df):
    encoded_df.to_csv('res_dpre.csv', index=False)
    # We save the non-encoded dataframe as well for future visualizations
    df.to_csv('non-encoded.csv', index=False)


# Import dataset & perform pre-processing
df = pd.read_csv(sys.argv[1], sep='\t')
preprocess_and_save()

# Invoke the next file
subprocess.run(['python', 'eda.py', sys.argv[1]])
