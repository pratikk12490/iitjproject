import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split


try:
    df_bitcoin = pd.read_csv('Bitcoin_Historical_Data.csv')
    print(df_bitcoin.head())

    # Check data types and convert 'Date' column to datetime if needed
    print(df_bitcoin.dtypes)
    if df_bitcoin['Date'].dtype != 'datetime64[ns]':
        df_bitcoin['Date'] = pd.to_datetime(df_bitcoin['Date'], dayfirst=True, format='mixed')
    print(df_bitcoin.dtypes)

except FileNotFoundError:
    print("Error: 'Bitcoin_Historical_Data.csv' not found.")
    df_bitcoin = None  # Set df_bitcoin to None to indicate failure
except Exception as e:
    print(f"An error occurred: {e}")
    df_bitcoin = None

# Convert relevant columns to numeric, handling commas and other issues
for col in ['Price', 'Open', 'High', 'Low']:
    try:
        df_bitcoin[col] = df_bitcoin[col].astype(str).str.replace(',', '').astype(float)
    except Exception as e:
        print(f"Error converting column '{col}': {e}")

# Convert 'Vol.' column to numeric, handling 'K' suffix
try:
    df_bitcoin['Vol.'] = df_bitcoin['Vol.'].astype(str).str.replace('K', '').astype(float) * 1000
except Exception as e:
    print(f"Error converting column 'Vol.': {e}")

# Convert 'Change %' column to numeric, handling '%' suffix
try:
    df_bitcoin['Change %'] = df_bitcoin['Change %'].astype(str).str.replace('%', '').astype(float) / 100
except Exception as e:
    print(f"Error converting column 'Change %': {e}")

print(df_bitcoin.dtypes)

"""## Data exploration

Explore the loaded Bitcoin price dataset (df_bitcoin) to understand its characteristics.

"""
# 1. Data Shape and Types
print("Data Shape:", df_bitcoin.shape)
print("\nData Types:\n", df_bitcoin.dtypes)

# 2. Missing Values
missing_values = df_bitcoin.isnull().sum()
missing_percentage = (missing_values / len(df_bitcoin)) * 100
print("\nMissing Values:\n", missing_values)
print("\nMissing Value Percentage:\n", missing_percentage)

# 3. Descriptive Statistics
numerical_features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
print("\nDescriptive Statistics:\n", df_bitcoin[numerical_features].describe())

# 4. Data Distribution
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df_bitcoin[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
# plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=df_bitcoin[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
# plt.show()


# 5. Correlation Analysis
correlation_matrix = df_bitcoin[numerical_features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
# plt.show()

# 6. Time Series Analysis
plt.figure(figsize=(12, 6))
plt.plot(df_bitcoin['Date'], df_bitcoin['Price'])
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Price Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

# 7. Outlier Detection (Observations based on boxplots and descriptive stats)
# Observations documented in the finish_task command.

# 8. Duplicate Rows
duplicate_rows = df_bitcoin[df_bitcoin.duplicated()]
print("\nNumber of duplicate rows:", len(duplicate_rows))
print("\nDuplicate Rows:\n", duplicate_rows)

"""## Data cleaning

### Subtask:
Clean the Bitcoin price dataset by handling outliers and ensuring data consistency.

"""

import pandas as pd
import numpy as np

# Outlier handling using IQR and capping
numerical_cols = ['Price', 'Open', 'High', 'Low', 'Vol.']
for col in numerical_cols:
    Q1 = df_bitcoin[col].quantile(0.25)
    Q3 = df_bitcoin[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap outliers at the 99th percentile
    upper_limit = df_bitcoin[col].quantile(0.99)
    df_bitcoin[col] = np.where(df_bitcoin[col] > upper_limit, upper_limit, df_bitcoin[col])


# Data consistency checks
print("\nMissing values after outlier handling:\n", df_bitcoin.isnull().sum())

# Data type validation
print("\nData types after outlier handling and imputation:\n", df_bitcoin.dtypes)

# Feature engineering
# Rolling Statistics
df_bitcoin['Price_Rolling_Mean_7'] = df_bitcoin['Price'].rolling(window=7).mean()
df_bitcoin['Price_Rolling_Mean_30'] = df_bitcoin['Price'].rolling(window=30).mean()
df_bitcoin['Volume_Rolling_Mean_7'] = df_bitcoin['Vol.'].rolling(window=7).mean()
df_bitcoin['Volume_Rolling_Mean_30'] = df_bitcoin['Vol.'].rolling(window=30).mean()
df_bitcoin['Change_Rolling_Mean_7'] = df_bitcoin['Change %'].rolling(window=7).mean()
df_bitcoin['Change_Rolling_Mean_30'] = df_bitcoin['Change %'].rolling(window=30).mean()

# Price Changes and High-Low Difference
df_bitcoin['Price_Change_Percentage'] = df_bitcoin['Price'].pct_change() * 100
df_bitcoin['High_Low_Difference'] = df_bitcoin['High'] - df_bitcoin['Low']

# Volatility Indicators
df_bitcoin['Price_Rolling_Std_7'] = df_bitcoin['Price'].rolling(window=7).std()
df_bitcoin['Price_Rolling_Std_30'] = df_bitcoin['Price'].rolling(window=30).std()

# Time-Based Features
df_bitcoin['Day_of_Week'] = df_bitcoin['Date'].dt.dayofweek
df_bitcoin['Month'] = df_bitcoin['Date'].dt.month
df_bitcoin['Day_of_Week_sin'] = np.sin(2 * np.pi * df_bitcoin['Day_of_Week'] / 7)
df_bitcoin['Day_of_Week_cos'] = np.cos(2 * np.pi * df_bitcoin['Day_of_Week'] / 7)
df_bitcoin['Month_sin'] = np.sin(2 * np.pi * df_bitcoin['Month'] / 12)
df_bitcoin['Month_cos'] = np.cos(2 * np.pi * df_bitcoin['Month'] / 12)

# Lagged Features
df_bitcoin['Price_Lag_1'] = df_bitcoin['Price'].shift(1)
df_bitcoin['Volume_Lag_1'] = df_bitcoin['Vol.'].shift(1)

# Calculate MACD features
df_bitcoin['EMA_12'] = df_bitcoin['Price'].ewm(span=12, adjust=False).mean()
df_bitcoin['EMA_26'] = df_bitcoin['Price'].ewm(span=26, adjust=False).mean()
df_bitcoin['MACD'] = df_bitcoin['EMA_12'] - df_bitcoin['EMA_26']
df_bitcoin['Signal_Line'] = df_bitcoin['MACD'].ewm(span=9, adjust=False).mean()
df_bitcoin['MACD_Histogram'] = df_bitcoin['MACD'] - df_bitcoin['Signal_Line']

# Handle NaN values (forward fill for rolling stats and lagged features)
for col in ['Price_Rolling_Mean_7', 'Price_Rolling_Mean_30', 'Volume_Rolling_Mean_7', 'Volume_Rolling_Mean_30',
            'Change_Rolling_Mean_7', 'Change_Rolling_Mean_30', 'Price_Rolling_Std_7', 'Price_Rolling_Std_30',
            'Price_Lag_1', 'Volume_Lag_1','MACD', 'Signal_Line', 'MACD_Histogram']:
    df_bitcoin[col] = df_bitcoin[col].fillna(0)

# Create a list of numerical features to standardize
numerical_features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'Price_Rolling_Mean_7', 'Price_Rolling_Mean_30',
                     'Volume_Rolling_Mean_7', 'Volume_Rolling_Mean_30', 'Change_Rolling_Mean_7', 'Change_Rolling_Mean_30',
                     'Price_Change_Percentage', 'High_Low_Difference', 'Price_Rolling_Std_7', 'Price_Rolling_Std_30',
                     'Price_Lag_1', 'Volume_Lag_1','MACD', 'Signal_Line', 'MACD_Histogram']


