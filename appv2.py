import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.impute import KNNImputer
from datetime import datetime, date
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


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
    column_name = input("Please specify the name of the column where the time series starts: ")  # Prompt user to input column name

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
def concat_with_X(X_data, data):
    full_set = pd.concat([X_data, data], axis=1)
    return full_set

def long_f(data,number_of_id_rows):
    
    id_vars = list(data.columns)[:number_of_id_rows]
    value_vars = list(data.columns)[number_of_id_rows:]

    data = pd.melt(data, id_vars=id_vars, value_vars=value_vars, var_name='Year', value_name='Value')
    return (data)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# Function to calculate MAPE

def process_format(format,time_series_index):
    format['Year'] = pd.to_datetime(format.Year, format = '%Y')
    format['Value'] = pd.to_numeric(format.Value)
    format= format.iloc[:,time_series_index:]
    return format

def process_format_for_ARIMA(format,time_series_index):
    
    format = process_format(format,time_series_index)
    format.index = pd.DatetimeIndex(format.index).to_period('Y')
    format=format.set_index('Year')
    return format

def ARIMA__train(train,test2020,test2021,test2022,time_series_index):
    ##train
    train = process_format_for_ARIMA(train,time_series_index)
    mape_scores = []

    model = ARIMA(train['Value'], order=(5, 1, 0))  # Example order for ARIMA
    fitted_model = model.fit()
    ##test2020
    test2020=process_format_for_ARIMA(test2020,time_series_index)
    predictions2020 = fitted_model.forecast(steps=len(test2020))
    print(predictions2020)

    model = ARIMA(test2020['Value'], order=(5, 1, 0))  # Example order for ARIMA
    fitted_model = model.fit()

    mape_scores.append(mean_absolute_percentage_error(test2020, predictions2020))
    print(mape_scores)

    #test 2021
    test2021=process_format_for_ARIMA(test2021,time_series_index)
    predictions2021 = fitted_model.forecast(steps=len(test2021))
    print(predictions2021)

    model = ARIMA(test2021['Value'], order=(5, 1, 0))  # Example order for ARIMA
    fitted_model = model.fit()

    mape_scores.append( mean_absolute_percentage_error(test2021, predictions2021))
    print(mape_scores)

    model = ARIMA(test2022['Value'], order=(5, 1, 0))  # Example order for ARIMA
    fitted_model = model.fit()
    #test 2022
    test2022=process_format_for_ARIMA(test2022,time_series_index)
    predictions2022 = fitted_model.forecast(steps=len(test2022))
    print(predictions2022)

    mape_scores.append(mean_absolute_percentage_error(test2022, predictions2022))
    print(mape_scores)

    predictions2020 = predictions2020.reset_index(drop=True)
    predictions2021 = predictions2021.reset_index(drop=True)
    predictions2022 = predictions2022.reset_index(drop=True)



    Arima_predictions_df = pd.DataFrame({'ARIMA_Prediction_2020': predictions2020,'ARIMA_Prediction_2021': predictions2021, 
                                     'ARIMA_Prediction_2022': predictions2022})

    return Arima_predictions_df, mape_scores

def process_format_for_prophet(format,time_series_index):
    format = process_format(format,time_series_index)
    format = format.reset_index(drop= True).rename(columns={'Year': 'ds', 'Value': 'y'})
    return format

def prophet_time_series():
    with open('serialized_model.json', 'r') as fin:
    m = model_from_json(fin.read())  # Load model

def merge_columns(df1,df2):
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index (drop = True)
    output_db = pd.merge(df1, df2, left_index=True, right_index=True)  
    return output_db


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

## make train long format
data_train = concat_with_X(X_data, data_train)
data_train_long = long_f(data_train,time_series_index)
print(data_train_long)

## make test long format
data_2020_test = concat_with_X(X_data, data_2020_test)
data_2020_test_long = long_f(data_2020_test,time_series_index)
print(data_2020_test_long)

data_2021_test = concat_with_X(X_data, data_2021_test)
data_2021_test_long = long_f(data_2021_test,time_series_index)
print(data_2021_test_long)

data_2022_test = concat_with_X(X_data, data_2022_test)
data_2022_test_long = long_f(data_2022_test,time_series_index)


Arima_predictions_db, mape_scores = ARIMA__train(data_train_long,data_2020_test_long,data_2021_test_long,data_2022_test_long,time_series_index)
print(Arima_predictions_db)
output_db = merge_columns(file_path,Arima_predictions_db)
print(output_db)
# Calculate MAPE for each year and add them as new columns
output_db['ARIMA_MAPE_2020'] = abs((output_db['2020'] - output_db['ARIMA_Prediction_2020']) / output_db['2020']) * 100
output_db['ARIMA_MAPE_2021'] = abs((output_db['2021'] - output_db['ARIMA_Prediction_2021']) / output_db['2021']) * 100
output_db['ARIMA_MAPE_2022'] = abs((output_db['2022'] - output_db['ARIMA_Prediction_2022']) / output_db['2022']) * 100

