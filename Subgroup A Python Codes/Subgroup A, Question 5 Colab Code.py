import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print

print("\n[bold]Investigating the correlation between average crowd levels, weather and holidays[bold]")
df = pd.read_csv("themepark_weather_holiday.csv")
df

# Standardise avg_temp using z-score normalisation since different countries have different temperature ranges

def standardise_temperature(df, temp_column="avg_temp", country_column="country"):
    df["standardised_temp"] = df.groupby(country_column)[temp_column].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return df

# Apply function to dataset
df = standardise_temperature(df, temp_column="avg_temp", country_column="country")

df

# Remove first 3 columns
df_corr = df.copy()
df_corr = df_corr.iloc[:, 3:]
df_corr.head()

# Correlation plot
def plot_cor_matrix(df, title= "Correlation Matrix for Themepark-Weather-Holiday Data"):
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()

plot_cor_matrix(df_corr)

print("[bold]Analysis:[/bold]\n"
      "- school_holiday had the strongest positive correlation with avg_crowd_level among all variables.\n"
      "- standardised_temp also had a positive correlation with avg_crowd_level, but it was weaker than school_holiday's correlation.\n"
      "- avg_precipitation, avg_humidity, and public_holiday had weak negative correlations with avg_crowd_level.\n"
      "- The strong positive correlation between avg_temp and standardised_temp shows that the z-score normalisation process was able to preserve the original temperature trends.")

print("\n[bold]Construct Model using Multiple Linear Regression[bold]")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

features = ["standardised_temp", "avg_precipitation", "avg_humidity", "public_holiday", "school_holiday"]
target = "avg_crowd_level"

# Split dataset into test and training data
X_train, X_test, y_train, y_test = train_test_split(df_corr[features], df_corr[target], test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

print("\n[bold]Evaluate model performance[bold]")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-Squared Score:", r2_score(y_test, y_pred))

# Print coefficients
coefficients = model.coef_
intercept = model.intercept_
print("Model Coefficients:", coefficients)
print("Model Intercept:", intercept)

print("\n[bold]Construct Model using Random Forest (RF)[bold]")

from sklearn.ensemble import RandomForestRegressor


# Split dataset into test and training data
X_train, X_test, y_train, y_test = train_test_split(df_corr[features], df_corr[target], test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

print("\n[bold]Evaluate model performance[bold]")
print("Mean Absolute Error:",  mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-Squared Score:", r2_score(y_test, y_pred))
print("\n[bold]Print feature importance[bold]")
feature_importance = pd.DataFrame({"Feature": features, "Importance": rf_model.feature_importances_}).sort_values(by="Importance", ascending=False)
print("Feature Importance:")
print(feature_importance)

print("[bold]Analysis[/bold]\n"
      "- The mean squared error for the random forest model is lower than that of the multiple linear regression model, suggesting that the RF model is better at predicting average crowd level.\n"
      "- The negative r-squared score for the linear regression model indicates that it fails to capture meaningful relationships in the data. This suggests that the relationship between the predictors and crowd levels is highly non-linear, making linear regression an unsuitable choice.\n"
      "- While the random forest model achieves a positive r-squared score, its relatively low value implies that it only captures a limited portion of the variance in crowd levels.\n"
      "- Clustering guests into distinct segments may provide deeper insights into the different factors influencing their decision to visit theme parks. Identifying these visitor groups can enable a more targeted analysis and lead to better-informed recommendations.")

df_2 = pd.read_csv("dsa3101_clustered_data.csv")
df_2.head()
print("*********************************************")
print("[bold]EDA of clusters[bold]")

# extract relevant columns
data = df_2[['q1', 'q2_1','q3', 'q6', 'q7', 'q8', 'q10', 'q12', 'cluster']]
data.info()
data.isna().sum()

data = data.dropna(subset=['cluster'])
data.info()
data.isna().sum()
data.head()

cluster_counts = data['cluster'].value_counts().sort_index()
print(cluster_counts)

def plot_bar_per_cluster(df, question, cluster_col='cluster'):
    clusters = sorted(df[cluster_col].dropna().unique())
    all_labels = df[question].dropna().unique()
    plot_data = pd.DataFrame(index=all_labels)

    for c in clusters:
        responses = df[df[cluster_col] == c][question].value_counts()
        responses = responses.reindex(all_labels, fill_value=0)
        plot_data[f'Cluster {c}'] = responses
    return plot_data

plot_data = plot_bar_per_cluster(data, 'q1')

plot_data.plot(kind = 'bar', figsize = (12,6))
plt.title("Visitor Group Type Across Clusters")
plt.xlabel("")
plt.ylabel("Count")
plt.xticks(rotation = 45)

plot_data = plot_bar_per_cluster(data, 'q2_1')

plot_data.plot(kind = 'bar', figsize = (12,6))
plt.title("Distribution of Visitor Age Across Clusters")
plt.xlabel("")
plt.ylabel("Count")
plt.xticks(rotation = 45)

plot_data = plot_bar_per_cluster(data, 'q12')

plot_data.plot(kind = 'bar', figsize = (12,6))
plt.title("Preferred Park Exploration Method Across Clusters")
plt.xlabel("")
plt.ylabel("Count")
plt.xticks(rotation = 45)

print("[bold]Heatmaps: To assess the degree to which various factors are prevalent in each cluster, we will visualise their relative frequencies using heatmaps.[bold]")

def plot_multiselect_heatmap(df, question, cluster_col='cluster'):
    all_options = set()
    df[question].str.split(',').apply(lambda x: all_options.update([i.strip() for i in x if isinstance(i, str)]))
    all_options = sorted(all_options)


    clusters = sorted(df[cluster_col].dropna().unique())
    heatmap_data = pd.DataFrame(0.0, index=clusters, columns=all_options, dtype=float)

    for cluster in clusters:
        cluster_df = df[df[cluster_col] == cluster]
        total = len(cluster_df)

        exploded = cluster_df[question].str.split(',').explode().str.strip()
        option_counts = exploded.value_counts()

        for option in all_options:
            percent = (option_counts.get(option, 0) / total) * 100 if total > 0 else 0
            heatmap_data.loc[cluster, option] = round(percent, 2)

    return heatmap_data

# Some options in q6 have commas in them. Since we are using commas to split the selected options, this would lead to incorrect splitting of multi-word options. Hence, we should first clean the data before creating the heatmap

def clean_q6(response):
    options = [opt.strip().title() for opt in response.split(',')]
    cleaned = []

    for opt in options:
        if opt == "Other Rides (Teacup Ride":
            cleaned.append("Other rides")
        elif opt == "Carousel Rides)" or opt == "Suspended Coasters":
          continue
        else:
            cleaned.append(opt)

    return ', '.join(cleaned)

# Apply cleaning to q6 column
data['q6'] = data['q6'].apply(clean_q6)

q6_heatmap_data = plot_multiselect_heatmap(data, question='q6')

plt.figure(figsize=(12, 6))
sns.heatmap(q6_heatmap_data.astype(float), annot=True, cmap = "Greens", fmt= ".1f")
plt.title("Attraction Preference Across Clusters")
plt.xlabel(" ")
plt.xticks(rotation=45)
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()

# Similarly for q7, there are inconsistencies in the options selected. Therefore, we need to clean the data before creating the heatmap.

def clean_q7(response):
    options = [opt.strip().title() for opt in response.split(',')]
    cleaned = []

    for opt in options:
        if opt == "Thrill Factor (Not To Be Confused With Scare Factor)" or opt == "Aesthetics":
            continue
        elif opt == "Holiday Seasons" or opt == "Holiday seasons":
            cleaned.append("Holiday Seasons")
        elif opt == "Weather Conditions" or opt == "Weather conditions":
            cleaned.append("Weather Condition")
        else:
            cleaned.append(opt)

    return ', '.join(cleaned)

# Apply cleaning to q7 column
data['q7'] = data['q7'].apply(clean_q7)

q7_heatmap_data = plot_multiselect_heatmap(data, question='q7')

plt.figure(figsize=(12, 6))
sns.heatmap(q7_heatmap_data.astype(float), annot=True, cmap = "Blues", fmt= ".1f")
plt.title("Factors Influencing Visit Decisions by Cluster")
plt.xlabel(" ")
plt.xticks(rotation=90)
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()

q8_heatmap_data = plot_multiselect_heatmap(data, question='q8')

plt.figure(figsize=(12, 6))
sns.heatmap(q8_heatmap_data.astype(float), annot=True, cmap = "Greens", fmt= ".1f")
plt.title("USS Events Influencing Visit Decisions by Cluster")
plt.xlabel(" ")
plt.xticks(rotation=0)
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()

def clean_q10(response):
    options = [opt.strip().title() for opt in response.split(',')]
    cleaned = []

    for opt in options:
        if opt == "Special Events (Halloween":
            cleaned.append("Special events")
        elif opt in ["Special Events (Christmas Etc.)", "Christmas Etc.)", "Weather Conditions", "Summer Festival"]:
          continue
        else:
            cleaned.append(opt)

    return ', '.join(cleaned)

# Apply cleaning to q10 column
data['q10'] = data['q10'].apply(clean_q10)

q10_heatmap_data = plot_multiselect_heatmap(data, question='q10')

plt.figure(figsize=(12, 6))
sns.heatmap(q10_heatmap_data.astype(float), annot=True, cmap = "Purples", fmt= ".1f")
plt.title("Preferred Visit Periods by Cluster")
plt.xlabel(" ")
plt.xticks(rotation=0)
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()
