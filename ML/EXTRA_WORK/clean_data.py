import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Load the CSV file
file_path = 'D:/ML/python/POWER_Regional_Daily_20240101_20240630.csv'
df = pd.read_csv(file_path)

# Replace -999.0 with NaN to mark missing values
columns_to_check = ['ALLSKY_SFC_UVA', 'ALLSKY_SFC_UVB']
df[columns_to_check] = df[columns_to_check].replace(-999.0, np.nan)

# Separate features and target columns for imputation
features = df.drop(columns=columns_to_check)
X_imputed = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(features), columns=features.columns)

for col in columns_to_check:
    # Separate the complete cases
    complete_cases = df[df[col].notnull()]
    X_complete = X_imputed.loc[complete_cases.index]
    y_complete = complete_cases[col]

    # Train the RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(X_complete, y_complete)

    # Predict missing values
    missing_values = df[df[col].isnull()]
    X_missing = X_imputed.loc[missing_values.index]
    predicted_values = model.predict(X_missing)

    # Update the DataFrame with predicted values
    df.loc[missing_values.index, col] = predicted_values

# Save the imputed DataFrame to a new CSV file
filtered_df = df[(df['LAT'].between(28.6, 28.8)) & (df['LON'].between(77.2, 77.3))]
output_file_path = 'D:/ML/python/filtered_data_delhi_2024.xlsx'
filtered_df.to_excel(output_file_path, index=False)

print(f"Data imputed with RandomForestRegressor and saved to '{output_file_path}'")
