import pandas as pd

try:
    df = pd.read_csv('Bitcoin_Historical_Data.csv')
    print(df.head())
except FileNotFoundError:
    print("Error: 'Bitcoin_Historical_Data.csv' not found.")
    df = None

###Data Exploration
# Examine Data Shape and Types
print("Data Shape:", df.shape)
print("\nData Types:\n", df.dtypes)

# Identify Missing Values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
print("\nMissing Values:\n", missing_values)
print("\nMissing Value Percentage:\n", missing_percentage)

# Analyze Key Variable Distributions
key_variables = ['Price', 'Open', 'High', 'Low', 'Vol.']
for col in key_variables:
    try:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    except:
        pass
print("\nDescriptive Statistics for Key Variables:\n", df[key_variables].describe())

# Determine Date Range
print("\nDate Range:")
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
print("Earliest Date:", df['Date'].min())
print("Latest Date:", df['Date'].max())
print("\nDate Differences:\n", df['Date'].diff().value_counts())

# Summarize Initial Findings
print("\nSummary:")
print("The dataset has", df.shape[0], "rows and", df.shape[1], "columns.")
print("The 'Date' column has been converted to datetime objects.")
print("Key variables' descriptive statistics are printed.")
print("The analysis shows the following potential issues:", missing_percentage[missing_percentage > 0])

#### Data Cleaning
# Handle missing values (already handled in previous steps, but checking again)
if df.isnull().values.any():
    print("Warning: Missing values found.")
    # If missing values are present, choose an appropriate imputation strategy.
    # For example, using forward fill for time series data:
    df.fillna(method='ffill', inplace=True)
    print("Missing values filled using forward fill.")

# Convert 'Date' to datetime and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Remove duplicates
num_duplicates = df.duplicated().sum()
if num_duplicates > 0:
    print(f"Removed {num_duplicates} duplicate rows.")
    df.drop_duplicates(inplace=True)

# Check and convert data types of numerical columns
for col in ['Price', 'Open', 'High', 'Low', 'Vol.']:
    if not pd.api.types.is_numeric_dtype(df[col]):
        try:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',','').str.replace('K',''), errors='coerce')
        except:
            print(f"Error converting {col} to numeric type.")
            # Handle the error appropriately, e.g., remove the column or fill with a default value.
            # For now skip the problematic column, but in a real-world scenario, a more robust error handling mechanism should be in place.
            continue

print(df.head())
print(df.dtypes)


#### Data Preparation
# Create lagged features
for lag in [1, 2, 3, 7]:
    df[f'Price_Lag_{lag}'] = df['Price'].shift(lag)

# Handle missing values after creating lagged features
df.fillna(method='ffill', inplace=True)

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

# Separate features and target variable
features = [col for col in df.columns if 'Price_Lag' in col or col in ['Open', 'High', 'Low', 'Vol.']]
X_train = train_df[features]
y_train = train_df['Price']
X_test = test_df[features]
y_test = test_df['Price']

print(X_train.head())
print(y_train.head())

# Create lagged features
for lag in [1, 2, 3, 7]:
    df[f'Price_Lag_{lag}'] = df['Price'].shift(lag)

# Handle missing values after creating lagged features
df.ffill(inplace=True)

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

# Separate features and target variable
features = [col for col in df.columns if 'Price_Lag' in col or col in ['Open', 'High', 'Low', 'Vol.']]
X_train = train_df[features]
y_train = train_df['Price']
X_test = test_df[features]
y_test = test_df['Price']

print(X_train.head())
print(y_train.head())


##### Reasoning
# Create lagged features
for lag in [1, 2, 3, 7]:
    df[f'Price_Lag_{lag}'] = df['Price'].shift(lag)

# Handle missing values after creating lagged features
df.ffill(inplace=True)

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

# Separate features and target variable
features = [col for col in df.columns if 'Price_Lag' in col or col in ['Open', 'High', 'Low', 'Vol.']]
# Remove features with NaN values (introduced by lagging) after ffill
features = [f for f in features if f not in ['Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'Price_Lag_7']]
X_train = train_df[features]
y_train = train_df['Price']
X_test = test_df[features]
y_test = test_df['Price']

print(X_train.head())
print(y_train.head())


#### Feature engineering
from sklearn.preprocessing import MinMaxScaler

# Scale numerical features
scaler = MinMaxScaler()
numerical_features = ['Open', 'High', 'Low', 'Vol.']
X_train_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])

# Convert X_train_scaled and X_test_scaled back to dataframes
X_train_scaled = pd.DataFrame(X_train_scaled, columns=numerical_features, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=numerical_features, index=X_test.index)

# Handle 'Change %' column
try:
    df['Change %'] = df['Change %'].str.rstrip('%').astype('float') / 100
    X_train_scaled['Change %'] = df.loc[X_train.index]['Change %']
    X_test_scaled['Change %'] = df.loc[X_test.index]['Change %']
except:
    print("Error converting 'Change %' column. Dropping column.")

# Impute any remaining missing values
if X_train_scaled.isnull().values.any() or X_test_scaled.isnull().values.any():
    print("Warning: Missing values found after scaling. Imputing with mean.")
    for col in X_train_scaled.columns:
        if X_train_scaled[col].isnull().any():
            X_train_scaled[col].fillna(X_train_scaled[col].mean(), inplace=True)
            X_test_scaled[col].fillna(X_train_scaled[col].mean(), inplace=True)

print(X_train_scaled.head())
print(X_test_scaled.head())

#### Model training
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Train a Linear Regression model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train_scaled, y_train)

# Train a Random Forest Regressor model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators
random_forest_model.fit(X_train_scaled, y_train)

#### Model evaluation
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Make predictions
y_pred_lr = linear_regression_model.predict(X_test_scaled)
y_pred_rf = random_forest_model.predict(X_test_scaled)

# Calculate RMSE, MAE, and MAPE
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mape_lr = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100

# Store the metrics
model_performance = {
    'Linear Regression': {'RMSE': rmse_lr, 'MAE': mae_lr, 'MAPE': mape_lr},
    'Random Forest': {'RMSE': rmse_rf, 'MAE': mae_rf, 'MAPE': mape_rf}
}

print(model_performance)


###### Model optimization
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

# Linear Regression Hyperparameter Tuning
param_grid_lr = {
    'fit_intercept': [True, False],
    'positive': [True, False]
}

grid_search_lr = GridSearchCV(LinearRegression(), param_grid_lr, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=5)
grid_search_lr.fit(X_train_scaled, y_train)
best_lr_model = grid_search_lr.best_estimator_
y_pred_lr_tuned = best_lr_model.predict(X_test_scaled)
rmse_lr_tuned = np.sqrt(mean_squared_error(y_test, y_pred_lr_tuned))
mae_lr_tuned = mean_absolute_error(y_test, y_pred_lr_tuned)
mape_lr_tuned = np.mean(np.abs((y_test - y_pred_lr_tuned) / y_test)) * 100


# Random Forest Hyperparameter Tuning
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=5)
grid_search_rf.fit(X_train_scaled, y_train)
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf_tuned = best_rf_model.predict(X_test_scaled)
rmse_rf_tuned = np.sqrt(mean_squared_error(y_test, y_pred_rf_tuned))
mae_rf_tuned = mean_absolute_error(y_test, y_pred_rf_tuned)
mape_rf_tuned = np.mean(np.abs((y_test - y_pred_rf_tuned) / y_test)) * 100

print("Tuned Linear Regression Metrics:", rmse_lr_tuned, mae_lr_tuned, mape_lr_tuned)
print("Tuned Random Forest Metrics:", rmse_rf_tuned, mae_rf_tuned, mape_rf_tuned)

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data for the bar chart
model_names = ['Linear Regression', 'Random Forest']
rmse_values = [rmse_lr_tuned, rmse_rf_tuned]
mae_values = [mae_lr_tuned, mae_rf_tuned]
mape_values = [mape_lr_tuned, mape_rf_tuned]

# Create the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
index = range(len(model_names))

ax.bar(index, rmse_values, bar_width, label='RMSE', color='skyblue')
ax.bar([i + bar_width for i in index], mae_values, bar_width, label='MAE', color='lightcoral')
ax.bar([i + 2 * bar_width for i in index], mape_values, bar_width, label='MAPE', color='lightgreen')

ax.set_xlabel('Model')
ax.set_ylabel('Metric Value')
ax.set_title('Model Performance Comparison')
ax.set_xticks([i + bar_width for i in index])
ax.set_xticklabels(model_names)
ax.legend()
fig.tight_layout()

# Display the bar chart in Streamlit
st.pyplot(fig)

# Create line plots for predictions vs. actual values
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test.index, y_test, label='Actual Price', color='blue')
ax.plot(y_test.index, y_pred_lr_tuned, label='Linear Regression Predictions', color='red', linestyle='--')
ax.plot(y_test.index, y_pred_rf_tuned, label='Random Forest Predictions', color='green', linestyle='-.')

ax.set_xlabel('Date')
ax.set_ylabel('Bitcoin Price')
ax.set_title('Predicted vs. Actual Bitcoin Prices')
ax.legend()
fig.tight_layout()

# Display the line plot in Streamlit
st.pyplot(fig)

#### Model Evaluation
import pandas as pd

# Create a dictionary to store model performance
data = {
    'Model': ['Linear Regression (Initial)', 'Linear Regression (Tuned)', 'Random Forest (Initial)', 'Random Forest (Tuned)'],
    'RMSE': [rmse_lr, rmse_lr_tuned, rmse_rf, rmse_rf_tuned],
    'MAE': [mae_lr, mae_lr_tuned, mae_rf, mae_rf_tuned],
    'MAPE': [mape_lr, mape_lr_tuned, mape_rf, mape_rf_tuned],
}


# Create the DataFrame
model_performance_df = pd.DataFrame(data)
model_performance_df.set_index('Model', inplace=True)


# print the DataFrame
print(model_performance_df)
