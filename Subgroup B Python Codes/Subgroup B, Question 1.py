# %%
import pandas as pd
import numpy as np
import os
from faker import Faker
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Load data from CSVs

# %%
weather_df = pd.read_csv('../data/weather_cleandata.csv')
waittime_df = pd.read_csv('../data/waittime_cleandata.csv')

# %% [markdown]1
# ## 2. Generate External Factors (Date Sensitive Factors)
# We will be considering the following events: 
# - Public Holidays and School Holidays
# - Special/ Seasonal events
#     - Halloween Horror Nights
#     - Minion Land Grand Opening
#     - A Universal Christmas: A *Wicked* Christmas
# 
# #### Step 1: Two dictionaries on holidays and seasonal events are created.

# %%
# Define dictionary for Public holidays and School holidays between 1 Jan 2024- 29 Feb 2025
holidays = {
    2024: {
    "New Year’s Day": '2024-01-01',
    "Chinese New Year": '2024-02-10',
    "Chinese New Year": '2024-02-11',
    "Chinese New Year": '2024-02-12',
    "March Holidays": ['2024-03-09','2024-03-10', '2024-03-11', '2024-03-12', '2024-03-13', '2024-03-14', '2024-03-15', '2024-03-16', '2024-03-17'],
    "Good Friday": '2024-03-29',
    "Hari Raya Puasa": '2024-04-10',
    "Labour Day": '2024-05-01',
    "Vesak Day": '2024-05-22',
    "June Holidays": ['2024-05-25', '2024-05-26', '2024-05-27', '2024-05-28', '2024-05-29', '2024-05-30', '2024-05-31',
                      '2024-06-01','2024-06-02', '2024-06-03', '2024-06-04', '2024-06-05', '2024-06-06', '2024-06-07', '2024-06-08', '2024-06-09', '2024-06-10',
                      '2024-06-11', '2024-06-12', '2024-06-13', '2024-06-14', '2024-06-15', '2024-06-16', '2024-06-17', '2024-06-18', '2024-06-19', '2024-06-20',
                      '2024-06-21', '2024-06-22', '2024-06-23'],
    "Hari Raya Haji": '2024-06-17',
    "National Day": '2024-08-09',
    "September Holidays": ['2024-09-31', '2024-09-01', '2024-09-02', '2024-09-03', '2024-09-04', '2024-09-05', '2024-09-06', '2024-09-07', '2024-09-08'],
    "Deepavali": '2024-10-31',
    "December Holidays": ['2024-11-16', '2024-11-17', '2024-11-18', '2024-11-19', '2024-11-20', '2024-11-21', '2024-11-22', '2024-11-23', '2024-11-24', '2024-11-25',
                          '2024-11-26', '2024-11-27', '2024-11-28', '2024-11-29', '2024-11-30', '2024-12-01', '2024-12-02', '2024-12-03', '2024-12-04', '2024-12-05',
                          '2024-12-06', '2024-12-07', '2024-12-08', '2024-12-09', '2024-12-10', '2024-12-11', '2024-12-12', '2024-12-13', '2024-12-14', '2024-12-15',
                          '2024-12-16', '2024-12-17', '2024-12-18', '2024-12-19', '2024-12-20', '2024-12-21', '2024-12-22', '2024-12-23', '2024-12-24', '2024-12-25',
                          '2024-12-26', '2024-12-27', '2024-12-28', '2024-12-29', '2024-12-30', '2024-12-31'],
    "Christmas Day": '2024-12-25'
    },

    2025: {
    "New Year’s Day": '2025-01-01',
    "Chinese New Year": '2025-01-29',
    "Chinese New Year": '2025-01-30',
    "March Holidays": ['2025-03-15', '2025-03-16', '2025-03-17', '2025-03-18', '2025-03-19', '2025-03-20', '2025-03-21', '2025-03-22', '2025-03-23'],
    "Hari Raya Puasa": '2025-03-31',
    "Good Friday": '2025-04-18',
    "Labour Day": '2025-05-01',
    "Vesak Day": '2025-05-12',
    "June Holidays": ['2025-05-31', '2025-06-01', '2025-06-02', '2025-06-03', '2025-06-04', '2025-06-05', '2025-06-06', '2025-06-07', '2025-06-08', '2025-06-09',
                      '2025-06-10', '2025-06-11', '2025-06-12', '2025-06-13', '2025-06-14', '2025-06-15', '2025-06-16', '2025-06-17', '2025-06-18', '2025-06-19',
                      '2025-06-20', '2025-06-21', '2025-06-22', '2025-06-23', '2025-06-24', '2025-06-25', '2025-06-26', '2025-06-27', '2025-06-28', '2025-06-29'],
    "Hari Raya Haji": '2025-06-07',
    "National Day": '2025-08-09',
    "September Holidays" :['2025-09-06', '2025-09-07', '2025-09-08', '2025-09-09', '2025-09-10', '2025-09-11', '2025-09-12', '2025-09-13', '2025-09-14'],
    "Deepavali": '2025-10-20',
    "Decemeber Holidays": ['2025-11-22', '2025-11-23', '2025-11-24', '2025-11-25', '2025-11-26', '2025-11-27', '2025-11-28', '2025-11-29', '2025-11-30', '2025-12-01',
                           '2025-12-02', '2025-12-03', '2025-12-04', '2025-12-05', '2025-12-06', '2025-12-07', '2025-12-08', '2025-12-09', '2025-12-10', '2025-12-11',
                           '2025-12-12', '2025-12-13', '2025-12-14', '2025-12-15', '2025-12-16', '2025-12-17', '2025-12-18', '2025-12-19', '2025-12-20', '2025-12-21',
                           '2025-12-22', '2025-12-23', '2025-12-24', '2025-12-25', '2025-12-26', '2025-12-27', '2025-12-28', '2025-12-29', '2025-12-30', '2025-12-31'],
    "Christmas Day": '2025-12-25'
    }
}

# Define dictionary for USS special/seaonal events between 1 Jan 2024- 29 Feb 2025
seasonal_events = {
    2024: {
        "HHN": ['2024-09-27', '2024-09-28', '2024-10-03', '2024-10-04', '2024-10-05', '2024-10-10', '2024-10-11', '2024-10-12', '2024-10-17', '2024-10-18','2024-10-19',
                '2024-10-17', '2024-10-18', '2024-10-19', '2024-10-24', '2024-10-25', '2024-10-26', '2024-10-31', '2024-10-31', '2024-11-01', '2024-11-02'],
        "A Universal Christmas": ['2024-11-29', '2024-11-30', '2024-12-01', '2024-12-02', '2024-12-03', '2024-12-04', '2024-12-05', '2024-12-06', '2024-12-07', '2024-12-08',
                                  '2024-12-09', '2024-12-10', '2024-12-11', '2024-12-12', '2024-12-13', '2024-12-14', '2024-12-15', '2024-12-16', '2024-12-17', '2024-12-18',
                                 '2024-12-19', '2024-12-20', '2024-12-21', '2024-12-22', '2024-12-23', '2024-12-24', '2024-12-25', '2024-12-26', '2024-12-27', '2024-12-28',
                                 '2024-12-29', '2024-12-30', '2024-12-31']
        
    },

    2025: {
        "A Universal Christmas": '2025-01-01',
        "Minion Land Opening": '2025-02-14'
    }
}

# %% [markdown]
# #### Step 2: Create functions to check whether date is a holiday or seasonal event day

# %%
# Define function to check if a date is a public holiday
def is_public_holiday(date):
    for year in holidays:
        for holiday in holidays[year]:
            if date in holidays[year][holiday]:
                return 1
    return 0

# Define function to check if a date is a seasonal event day
def is_seasonal_event(date):
    for year in seasonal_events:
        for event in seasonal_events[year]:
            if date in seasonal_events[year][event]:
                return 1
    return 0

# %% [markdown]
# ## 3. Generate Datafames and Synthetic Data
# The 3 dataframes we will be creating are:
# - attractions_df : dataframe that that contains the list of attractions
# - dine_df: dataframe that lists dining options
#     - calculate visitor count data every 2 hours using Survey Q14.6
# - retail_df: dataframe that lists retail options
#     - calculate visitor count data every 2 hours using Survey Q14.7

# %% [markdown]
# ### 3.1 attractions_df

# %%
attractions_info = {
    "Ride": [
        "Accelerator",
        "Battlestar Galactica: CYLON",
        "Battlestar Galactica: HUMAN",
        "Canopy Flyer",
        "Dino-Soarin",
        "Enchanted Airways",
        "Jurassic Park Rapids Adventure",
        "Lights, Camera, Action! Hosted by Steven Spielberg",
        "Magic Potion Spin",
        "Puss In Boots’ Giant Journey",
        "Revenge of the Mummy",
        "Sesame Street Spaghetti Space Chase",
        "Shrek 4-D Adventure",
        "Transformers The Ride: The Ultimate 3D Battle",
        "Treasure Hunters"
    ],
    "Zone": [
        "Sci-Fi City", "Sci-Fi City", "Sci-Fi City", "The Lost World", "The Lost World", "Far Far Away", 
        "The Lost World", "New York", "Far Far Away", "Far Far Away", "Ancient Egypt", "New York", "Far Far Away",
        "Sci-Fi City", "Ancient Egypt"
    ],
    "Thrill": [0,1,1,1,0,1,0,0,0,0,1,0,0,0,0],  #thrill ride =1, non-thrill ride =0
    "Indoor": [1,0,0,0,0,0,0,1,1,1,1,1,1,1,0]  #indoors =1, outdoors =0
}

attractions_df = pd.DataFrame(attractions_info)
attractions_df.to_csv("attractions_df.csv")

# %% [markdown]
# ### 3.2 dine_df
# - Assumes visitors will dine at zones they are currently at
# - Step 1: Generate the count of dining visitors using the number of Visitors for attractions (i.e., "Visitor Count" of waittime_df)
# - Step 2: Factor in weights for time slot and number of restaurants in zone

# %%
## Step 1: Generate count of dining visitors
# Group Watitime data by time slots
dine_df = waittime_df.copy()
dine_df["Time Slot"] = pd.cut(
    dine_df["Date/Time"],
    bins=[10, 12, 14, 16, 18, 20],
    labels=["10-12", "12-2", "2-4", "4-6", "6-8"],
    right=False
)
# Get total visitors per time slot
dine_df = dine_df.groupby(["Date", "Zone", "Time Slot"])["Visitor Count"].sum().reset_index()
# Generate percentage of visitors dining per time slot dictionary based on survey Q14.6
dining_pct = {
        "10-12" : 29.3 ,
        "12-2": 43.1,
        "2-4": 36.1,
        "4-6": 33.9,
        "6-8": 31.7,
}
# Normalize dining percentages
dining_pct_vals = {k: v / 100 for k, v in dining_pct.items()}

# Apply dining percetanges to adjust dining count
dine_df["Dining Visitor Count"] = dine_df.apply(lambda row: int(row["Visitor Count"] * dining_pct_vals[row["Time Slot"]]), axis=1)

# Scale number of dining visitors proportionate to attraction visitors
dine_df["Dining Visitor Count"] = dine_df["Dining Visitor Count"] * 0.5
dine_df.drop(columns=["Visitor Count"], inplace=True)
print(dine_df.head())

# %%
# Step 2: Adjust dining visitor count based on time slot and number of restaurants in the zone
# Create dictionary to mapo out diners and zone they belong to
dine_info = {
    "Dine": [
        "KT’s Grill", 
        "Loui’s NY Pizza Parlor", 
        "StarBot Café", 
        "Star Dots", 
        "Oasis Spice Café", 
        "Frozen Fuels", 
        "Planet Yen", 
        "Cairo Market", 
        "Discovery Food Court", 
        "Fossil Fuels", 
        "Jungle Bites", 
        "Goldilocks", 
        "Friar’s"
    ],
    "Zone":[
        "New York", "New York", "Sci-Fi City", "Sci-Fi City",
        "Ancient Egypt", "Sci-Fi City", "Sci-Fi City", "Ancient Egypt", 
        "The Lost World", "The Lost World", "The Lost World",
        "Far Far Away", "Far Far Away"
        ]
}
dine_info = pd.DataFrame(dine_info)

# Understand the number of restaurants in each zone
restaurant_count_per_zone = dine_info.groupby("Zone")["Dine"].count()
print(restaurant_count_per_zone)

def apply_timeslot_weight(row):
    if row["Time Slot"] in ["12-2", "6-8"]:
        # Increase weight for lunch and dinner time slots
        weight = 1.0
    else:
        weight = 0.8  
    adjusted_visitor_count = row["Dining Visitor Count"] * weight
    return adjusted_visitor_count

def apply_restaurant_weight(row):
    if row["Zone"] == "Sci-Fi City":
        # Increase weight for Sci-Fi City
        weight = 1.0
    if row["Zone"] == "The Lost World":
        # Increase weight for The Lost World
        weight = 0.9
    elif row["Zone"] in ["New York", "Ancient Egypt", "Far Far Away"]:
        # Increase weight for New York, Ancient Egypt, Far Far Away
        weight = 0.8
    # Apply the weight to the visitor count
    adjusted_visitor_count = row["Dining Visitor Count"] * weight
    return adjusted_visitor_count

dine_df["Adjusted Visitor Count"] = dine_df.apply(apply_timeslot_weight, axis=1)
dine_df["Adjusted Visitor Count"] = dine_df.apply(apply_restaurant_weight, axis=1).round(0).astype(int)
dine_df.drop(columns=["Dining Visitor Count"], inplace=True)
print(dine_df.head())
# dine_df.to_csv('../data/dine_cleandata.csv', index=False)

# %% [markdown]
# 

# %% [markdown]
# ### 3.3 retail_df

# %%
retail_info = {
    "Retail": [
        "Universal Studios Store",
        "That's a Wrap",
        "Hello Kitty Studio Store",
        "The Brown Derby",
        "Big Bird's Emporium",
        "Fairy Godmother's Potion Shop",
        "Jurassic Outfitters",
        "The Dino-Store",
        "Transformers Supply Vault" 
    ],
    "Zone": ["Hollywood", "Hollywood", "Hollywood", "Hollywood", "New York", 
            "Far Far Away", "The Lost World", "The Lost World","Sci-Fi City"
    ]
            }

retail_df = pd.DataFrame(retail_info)
print(retail_df.head())

# %% [markdown]
# ## 4. Feature Engineering

# %%
# Add features as columns to waittime_df
# for day of week, 0 = Monday,..,, 6 = Sunday
waittime_df["day_of_week"] = pd.to_datetime(waittime_df["Date"]).dt.dayofweek
waittime_df["is_holiday"] = waittime_df["Date"].apply(is_public_holiday)
waittime_df["is_seasonal_event"] = waittime_df["Date"].apply(is_seasonal_event)

# Merge waititme_df with weather_df on "Date"
waittime_df = waittime_df.merge(weather_df[["Date", 'Daily Rainfall Total (mm)', 'Mean Temperature (°C)']], on="Date", how="left")
waittime_df.rename(columns={'Date/Time': 'Time', 'Daily Rainfall Total (mm)': 'Rainfall (mm)', 'Mean Temperature (°C)': 'Temperature (°C)'}, inplace=True)

features = ["is_holday", "is_seasonal_event", "day_of_week", "Daily Rainfall Total (mm)", "Mean Temperature (°C)"]

waittime_df['Date'] = pd.to_datetime(waittime_df['Date'])
waittime_df['Timestamp'] = waittime_df['Date'] + pd.to_timedelta(waittime_df['Time'], unit='h')
waittime_df.to_csv("waittime_df.csv")

# %% [markdown]
# ### Modelling 
# #### 5.1 Using Multiple Linear Regression to determine key factors

# %%
# Linear regression
from sklearn import datasets, linear_model, metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Identify categorical columns for linear regression
categorical_columns = ["Ride", "Time", "Zone","day_of_week", "is_holiday", "is_seasonal_event"]

# Prepare OneHotEncoder for categorical columns
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(waittime_df[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded = pd.concat([waittime_df, one_hot_df], axis=1)
df_encoded = df_encoded.drop(categorical_columns, axis=1)
df_encoded = df_encoded.drop(['Timestamp', 'Date'], axis=1)
# print(f"Encoded Waititme data : \n{df_encoded}")

# Identify independent and dependent variables
X = df_encoded.drop(columns=['Wait Time'])
y = df_encoded['Wait Time']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Multiple Linear Regression Model
MLR_model = LinearRegression()
MLR_model.fit(X_train_scaled, y_train)

# Predict on test data
predicted_waittimes = MLR_model.predict(X_test_scaled)

# Evaluate Model Performance using RMSE
rmse = mean_squared_error(y_test, predicted_waittimes, squared=False)  #
r2 = r2_score(y_test, predicted_waittimes)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# %% [markdown]
# #### 5.1 (a) Plot Actual vs Predicted Waititmes between 2024-01 and 2025-02

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Extract the 'Timestamp' from the original dataset
waittime_df_test = waittime_df.iloc[X_test.index]  

# Create a DataFrame for plotting: Actual vs Predicted
plot_df = pd.DataFrame({
    'Timestamp': waittime_df_test['Timestamp'],
    'Actual Wait Time': y_test.values, 
    'Predicted Wait Time': predicted_waittimes 
})

# Sort by Timestamp to ensure proper time series plotting
plot_df = plot_df.sort_values(by='Timestamp')

# Plot the results
plt.figure(figsize=(12, 6))
sns.lineplot(x=plot_df['Timestamp'], y=plot_df['Actual Wait Time'], label='Actual', color='blue')
sns.lineplot(x=plot_df['Timestamp'], y=plot_df['Predicted Wait Time'], label='Predicted', color='red')

# Formatting the plot
plt.xlabel('Date')
plt.ylabel('Wait Time (minutes)')
plt.title('Actual vs Predicted Wait Times Over Time')
plt.xticks(rotation=45)  
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# %% [markdown]
# #### 5.1 (b) Test the MLR model for average waittiime per each ride in February 2025

# %% [markdown]
# - Hourly waititme predictions have low accuracy since multiple linear regression assumes a linear relationship between features and waittimes
# - In reality, hourly waititmes follow a non-linear pattern

# %%
# Identify categorical columns for linear regression
categorical_columns = ["Ride", "Time", "Zone","day_of_week", "is_holiday", "is_seasonal_event"]

# Prepare OneHotEncoder for categorical columns
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(waittime_df[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

df_encoded = pd.concat([waittime_df, one_hot_df], axis=1)
df_encoded = df_encoded.drop(categorical_columns, axis=1)
# print(f"Encoded Waititme data : \n{df_encoded}")

# Identify independent and dependent variables
X = df_encoded.drop(columns=['Wait Time'])
y = df_encoded['Wait Time']

# Split data into training (before Feb 2025) and future prediction (Feb 2025)
train_mask = X['Date'] < "2025-02-01"  
future_mask = X['Date'] >= "2025-02-01"

X_train, y_train = X[train_mask].drop(columns=['Date', 'Timestamp']), y[train_mask] 
X_future = X[future_mask].drop(columns=['Date', 'Timestamp'])

# Train Multiple Linear Regression Model
MLR_model = LinearRegression()
MLR_model.fit(X_train, y_train)

# Predict on test data
predicted_waittimes = MLR_model.predict(X_future)

# Output results
print(f"Predicted wait times for Feb 2025: \n{predicted_waittimes}")

# Extract actual wait times for February 2025 from df
df_feb_2025 = waittime_df[waittime_df['Date'].dt.strftime('%Y-%m') == '2025-02']

# Create a DataFrame for predictions with corresponding rides
df_predictions = X_future.copy()
df_predictions['Predicted Wait Time'] = predicted_waittimes
df_predictions['Date'] = "2025-02"  
df_predictions['Ride'] = waittime_df.loc[X_future.index, 'Ride'].values

# Group actual and predicted data by Ride
actual_wait_times = df_feb_2025.groupby('Ride')['Wait Time'].mean()
predicted_wait_times = df_predictions.groupby('Ride')['Predicted Wait Time'].mean()

# Merge actual and predicted values for comparison
wait_time_comparison = pd.DataFrame({
    'Actual Wait Time': actual_wait_times,
    'Predicted Wait Time': predicted_wait_times
}).dropna()  

# Plot comparison of actual vs predicted wait times
plt.figure(figsize=(12, 6))
wait_time_comparison.plot(kind='bar', figsize=(14, 6), colormap='viridis')

plt.title("Actual vs. Predicted Wait Times for February 2025")
plt.xlabel("Ride")
plt.ylabel("Wait Time (minutes)")
plt.xticks(rotation=45, ha="right")
plt.legend(["Actual Wait Time", "Predicted Wait Time"])
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()


