Sales Demand Forecasting:

A machine learning project to predict future sales demand using historical business data. This model helps businesses optimize inventory, manage cash flow, and improve staffing efficiency.

Project Objective:

The goal of this project is to build a reliable forecasting system that predicts the next 30 days of sales. By analyzing trends and seasonality, the model provides data-driven insights to support strategic business decisions.

Tools & Technologies

 Language: Python
 Libraries: Pandas, NumPy, Scikit-learn, Matplotlib
 Model: Random Forest Regressor
 Dataset: Superstore Sales Dataset

Key Features:

 Data Preprocessing: Handling missing values and date-time conversion.
 Feature Engineering:Time-based features (Year, Month, Day, Quarter).
 Lag Features : Capturing previous sales data to predict future trends.
 Rolling Averages:Smoothing data to identify underlying trends.
 Forecasting:Predicting sales for a 30-day future horizon.
 Evaluation: Using Mean Absolute Error (MAE) and MAPE to ensure accuracy.

Visualizations:

The script generates two key insights:

1. Sales Demand Forecast: A line graph comparing historical data against the 30-day predicted forecast.
2. Monthly Trend Analysis: A visualization of sales distribution across different months to identify seasonality.

Business Impact:

This forecast model allows a business to:
 Inventory Planning: Increase stock during predicted peaks and reduce it during dips.
 Staffing: Allocate more resources during high-demand periods.
 Financial Prep: Predict cash inflow more accurately for the upcoming month.

 How to Run:

1. Clone this repository.
2. Ensure you have the `superstore.csv` file in the root directory.
3. Install dependencies:
   pip install -r requirements.txt


<img width="1400" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/e0ee8823-e942-41ef-a708-5274f4a87e0e" />

<img width="1000" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/11c51b2d-7484-419d-b0c3-b30ed1ad2827" />



