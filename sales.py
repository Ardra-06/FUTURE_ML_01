import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


# 2 LOAD DATASET

df = pd.read_csv("superstore.csv", encoding="latin1")

print("Dataset Shape:", df.shape)
print(df.head())

# 3 DATA CLEANING

# convert order date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# remove missing rows
df = df.dropna()

# sort by date
df = df.sort_values("Order Date")

print("\nAfter Cleaning:", df.shape)

# 4 AGGREGATE DAILY SALES

sales_daily = df.groupby("Order Date")["Sales"].sum().reset_index()

print("\nDaily Sales Data:")
print(sales_daily.head())

# 5 FEATURE ENGINEERING (TIME FEATURES)

sales_daily["year"] = sales_daily["Order Date"].dt.year
sales_daily["month"] = sales_daily["Order Date"].dt.month
sales_daily["day"] = sales_daily["Order Date"].dt.day
sales_daily["dayofweek"] = sales_daily["Order Date"].dt.dayofweek
sales_daily["quarter"] = sales_daily["Order Date"].dt.quarter

# LAG FEATURES (important for time-series forecasting)

sales_daily["lag_1"] = sales_daily["Sales"].shift(1)
sales_daily["lag_7"] = sales_daily["Sales"].shift(7)

# rolling trend feature
sales_daily["rolling_mean_7"] = sales_daily["Sales"].rolling(7).mean()

# remove rows created by lagging
sales_daily = sales_daily.dropna()

# 6 TRAIN TEST SPLIT

# last 30 days for testing
train = sales_daily[:-30]
test = sales_daily[-30:]

features = [
    "year",
    "month",
    "day",
    "dayofweek",
    "quarter",
    "lag_1",
    "lag_7",
    "rolling_mean_7"
]

X_train = train[features]
y_train = train["Sales"]

X_test = test[features]
y_test = test["Sales"]

# 7 TRAIN MODEL

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Complete")


# 8 PREDICTIONS

predictions = model.predict(X_test)

# 9 MODEL EVALUATION

mae = mean_absolute_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)

print("\nModel Performance")
print("MAE:", mae)
print("MAPE:", mape)

# 10 FUTURE FORECAST (NEXT 30 DAYS)

last_date = sales_daily["Order Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=30
)

future_df = pd.DataFrame()
future_df["Order Date"] = future_dates

future_df["year"] = future_df["Order Date"].dt.year
future_df["month"] = future_df["Order Date"].dt.month
future_df["day"] = future_df["Order Date"].dt.day
future_df["dayofweek"] = future_df["Order Date"].dt.dayofweek
future_df["quarter"] = future_df["Order Date"].dt.quarter

# use last known values for lag features
last_lag1 = sales_daily["Sales"].iloc[-1]
last_lag7 = sales_daily["Sales"].iloc[-7]
last_roll = sales_daily["Sales"].rolling(7).mean().iloc[-1]

future_df["lag_1"] = last_lag1
future_df["lag_7"] = last_lag7
future_df["rolling_mean_7"] = last_roll

future_sales = model.predict(future_df[features])

future_df["Forecast Sales"] = future_sales

print("\nFuture Forecast:")
print(future_df.head())

# 11 VISUALIZATION

plt.figure(figsize=(14,6))

plt.plot(
    sales_daily["Order Date"],
    sales_daily["Sales"],
    label="Historical Sales"
)

plt.plot(
    future_df["Order Date"],
    future_df["Forecast Sales"],
    label="Forecast Sales",
    color="violet"
)

plt.title("Sales Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()

plt.show()

# 12 MONTHLY TREND ANALYSIS

monthly_sales = df.groupby(df["Order Date"].dt.month)["Sales"].sum()

plt.figure(figsize=(10,5))

monthly_sales.plot(kind="pie")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.show()

# 13 BUSINESS INSIGHTS

avg_sales = sales_daily["Sales"].mean()
max_sales = sales_daily["Sales"].max()

print("\nBusiness Insights")
print("-------------------")
print("Average Daily Sales:", avg_sales)
print("Maximum Daily Sales:", max_sales)

print("\nForecast Summary:")
print("Predicted sales for next 30 days:",
      round(future_sales.sum(),2))

print("\nPossible Business Actions:")
print("• Increase inventory if forecast trend rises")
print("• Plan promotions during low demand months")
print("• Allocate staff based on predicted demand")
