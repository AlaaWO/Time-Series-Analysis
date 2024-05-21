import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.impute import KNNImputer

def upload_file():
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()  # This hides the main window
    
    # Open a file dialog and get the file path
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("All files", "*.*")]  # You can specify file types here
    )
    
    if file_path:
        print("File uploaded:", file_path)
        
        # Import the file as a DataFrame based on file type
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            print(df)
        else:
            print("Unsupported file type.")
            return None

        return df
    else:
        print("No file selected.")
        return None

def is_df(var):
    if isinstance(var, pd.DataFrame):
        print("is dataframe")
    else:
        print("is not dataframe")

def percentage_Function(data, nominator):
    total_rows = data.shape[0]
    percentage = (nominator/ total_rows) * 100
    return percentage

def find_nan(data):
    # Count the rows where any column contains either 0 or NaN values
    rows_with_nan = data.isna().any(axis=1)
    rows_with_nan_count = rows_with_nan.sum()
    percentage = percentage_Function(data, rows_with_nan_count)
    # Display the count of rows with any 0 or NaN values
    print(f"NaN values remaining in the dataset: {rows_with_nan_count}")
    print(f"Percentage of NaN values remaining in the dataset: {percentage:.2f}%")


def remove_rows_with_nulls(data, maximum_null_allowed):
    missing_values_per_row = data.isnull().sum(axis=1)
    #rows_with_missing_values = missing_values_per_row[missing_values_per_row > 0]
    sorted_indices = missing_values_per_row.sort_values(ascending=False).index
    top_10_indices = sorted_indices[:10]
    missing_values_df = missing_values_per_row[top_10_indices].reset_index()
    missing_values_df.columns = ['index', 'missing_values']
    # Merge the original DataFrame with the missing values count
    #result_df = data.loc[top_10_indices].reset_index().merge(missing_values_df, on='index')

    rows_over_threshold = missing_values_per_row[missing_values_per_row > maximum_null_allowed]

    # Count the number of such rows
    num_rows_over_threshold = rows_over_threshold.shape[0]

    # Step 6: Calculate the percentage of these rows over the overall data
    percentage = percentage_Function(data, num_rows_over_threshold)
    # Output the results
    print(f"Number of rows with more than {maximum_null_allowed} missing values: {num_rows_over_threshold}")
    print(f"Percentage of rows with more than {maximum_null_allowed} missing values: {percentage:.2f}%")
    rows_with_nan = data.isna().any(axis=1)


    data = data.dropna(thresh=len(data.columns) - maximum_null_allowed)
    #count after drop
    missing_values_per_row = data.isnull().sum(axis=1)
    sorted_indices = missing_values_per_row.sort_values(ascending=False).index
    top_10_indices = sorted_indices[:10]
    missing_values_df = missing_values_per_row[top_10_indices].reset_index()
    missing_values_df.columns = ['index', 'missing_values']
    

    rows_over_threshold = missing_values_per_row[missing_values_per_row > maximum_null_allowed]

    num_rows_over_threshold = rows_over_threshold.shape[0]

    percentage = percentage_Function(data, num_rows_over_threshold)
    print(f"Number of rows with more than {maximum_null_allowed} missing values after processing: {num_rows_over_threshold}")
    print(f"Percentage of rows with more than {maximum_null_allowed} missing values after processing: {percentage:.2f}%")
    
    find_nan(data)
    
    return data

def impute_data(data, number_of_id_rows):
    numeric_data = data.select_dtypes(include=['number'])

    imputer = KNNImputer(n_neighbors=5, weights= 'distance')
    ## data preprocessing
    imputed_data = imputer.fit_transform(numeric_data)
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_data.columns)

    X_data=data.iloc[:,:number_of_id_rows]

    X_data_reset = X_data.reset_index(drop=True)
    imputed_df_reset = imputed_df.reset_index(drop=True)
    final_imputed_df = pd.concat([X_data_reset, imputed_df_reset], axis=1)

    return final_imputed_df

def column_question(data):
    column_names = data.columns.tolist()
    print("Column names:", column_names)
    column_name = input("Please specify the name of the column: ")  # Prompt user to input column name

    # Check if the specified column name exists
    if column_name not in column_names:
        raise ValueError(f"Column '{column_name}' not found in the data.")

    # Find the index of the column
    column_index = column_names.index(column_name)
    return column_index


def slice_data(data, column_index):
    # Slice the data to include the specified column and what is after it
    X_data = data.iloc[:, :column_index]
    Y_data = data.iloc[:, column_index:]

    # Print the sliced data
    print("\nSliced Data:")
    return X_data,Y_data

def split_test_columns(data):
    # Check if the necessary columns exist in the dataframe
    for year in ['2020', '2021', '2022']:
        if year not in data.columns:
            raise ValueError(f"Column '{year}' not found in the data.")
    
    column_names = data.columns.tolist()
    column_index = column_names.index('2020')

    data_train = data.iloc[:, :column_index]
    # Split the data into three separate dataframes based on test columns
    data_2020_test = data[['2020']]
    data_2021_test = data[['2021']]
    data_2022_test = data[['2022']]

    # Print the separated data
    print("\ntrain Data:")
    print(data_train)

    print("\n2020 Test Data:")
    print(data_2020_test)
    
    print("\n2021 Test Data:")
    print(data_2021_test)
    
    print("\n2022 Test Data:")
    print(data_2022_test)

    return data_train,data_2020_test, data_2021_test, data_2022_test
# Split the sliced data into test columns

file_path = upload_file()
print(type(file_path))
is_df(file_path) #to confirm that the data got parsed correctly
time_series_index = column_question(file_path)
file_path = file_path.replace(0, np.nan)
find_nan(file_path)
index =len(file_path.columns[time_series_index:])/2
print (index)
if (index%2 == 1):
    index = index + 1
maximum_null_allowed = index
remove_rows_with_nulls(file_path, maximum_null_allowed)
file_path = impute_data(file_path, time_series_index)
find_nan(file_path)
X_data,Y_data = slice_data(file_path,time_series_index)
data_train,data_2020_test, data_2021_test, data_2022_test = split_test_columns(Y_data)



