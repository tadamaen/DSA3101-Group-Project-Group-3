import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print

df = pd.read_csv("themepark_weather_holiday.csv")
df.head()

# Standardise avg_temp using z-score normalisation

def standardise_temperature(df, temp_column="avg_temp", country_column="country"):
    df["standardised_temp"] = df.groupby(country_column)[temp_column].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return df

# Apply function to dataset
df = standardise_temperature(df, temp_column="avg_temp", country_column="country")

df.head()

# Construct correlation matrix

df_corr = df.copy()
df_corr = df_corr.iloc[:, 3:]
df_corr.head()

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

# Evaluate model performance
print("[bold]Evaluate model performance[bold]")
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

# Evaluate model performance
print("[bold]Evaluate model performance[bold]")
print("Mean Absolute Error:",  mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-Squared Score:", r2_score(y_test, y_pred))

# Print feature importance
feature_importance = pd.DataFrame({"Feature": features, "Importance": rf_model.feature_importances_}).sort_values(by="Importance", ascending=False)
print("Feature Importance:")
print(feature_importance)

print("[bold]Analysis[/bold]\n"
      "- The mean squared error for the random forest model is lower than that of the multiple linear regression model, suggesting that the RF model is better at predicting average crowd level.\n"
      "- The negative r-squared score for the linear regression model indicates that it fails to capture meaningful relationships in the data. This suggests that the relationship between the predictors and crowd levels is highly non-linear, making linear regression an unsuitable choice.\n"
      "- While the random forest model achieves a positive r-squared score, its relatively low value implies that it only captures a limited portion of the variance in crowd levels.\n"
      "- Clustering guests into distinct segments may provide deeper insights into the different factors influencing their decision to visit theme parks. Identifying these visitor groups can enable a more targeted analysis and lead to better-informed recommendations.")

