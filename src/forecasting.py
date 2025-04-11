import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
from prophet import Prophet 
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value 
from loguru import logger
import warnings
warnings.simplefilter(action='ignore')
 
 
def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 
    df_long = df.melt(id_vars='Year', value_vars=month_cols, 
                  var_name='Month', value_name='Demand') 
    df_long['Date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Month'], format='%Y-%b') 
    df_long = df_long.sort_values('Date').reset_index(drop=True) 
    df_long.set_index('Date', inplace=True) 
    return df_long
 
#  2. Helper: Add Promotion Events as a Regressor  
def add_promotion_factors(df): 
    df['Promotion'] = 0  # Default value 
    for index, row in df.iterrows(): 
        if (row['ds'].month == 4 and row['ds'].year in [2023, 2024]) or (row['ds'].month == 5 and row['ds'].year in [2020, 2021, 2022]) or (row['ds'].month == 6 and row['ds'].year == 2019): 
            df.at[index, 'Promotion'] = 1  # Eid al-Fitr 
        elif row['ds'].month == 9: 
            df.at[index, 'Promotion'] = 1  # Saudi National Day 
        elif row['ds'].month == 2 and row['ds'].year >= 2022: 
            df.at[index, 'Promotion'] = 1  # Founding Day 
        elif row['ds'].month == 11: 
            df.at[index, 'Promotion'] = 1  # White Friday & Singles Day 
        elif row['ds'].month == 12: 
            df.at[index, 'Promotion'] = 1  # End-of-Year Sales 
    return df 
 
#  3. Cross-Validation to Find Optimal Weights  
def cross_validation(df:pd.DataFrame):
    initial_window = 36 
    n_splits = 12 
    actuals, sarima_preds, prophet_preds, hw_preds = [], [], [], [] 
 
    for i in range(n_splits): 
        train_end = initial_window + i 
        train = df.iloc[:train_end] 
        test = df.iloc[train_end:train_end + 1] 
        if len(test) == 0: 
            break 
        # SARIMA
        try: 
            sarima_model = SARIMAX(train['Demand'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False) 
            sarima_forecast = sarima_model.get_forecast(steps=1).predicted_mean.values[0] 
        except: 
            sarima_forecast = 0 

        # Prophet
        try:
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            prophet_df = train.reset_index()
            prophet_df = prophet_df.rename(columns={'index': 'ds', 'Demand': 'y'})
            
            # Initialize and fit the Prophet model
            prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            prophet_model.fit(prophet_df)
            
            # Create future dataframe for prediction
            future = prophet_model.make_future_dataframe(periods=1, freq='M')
            forecast = prophet_model.predict(future)
            
            # Get the last predicted value
            prophet_forecast = forecast['yhat'].iloc[-1]
        except:
            prophet_forecast = train['Demand'].mean()

        # Holt-Winters 
        try: 
            hw_model = ExponentialSmoothing(train['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit() 
            hw_forecast = hw_model.forecast(1).values[0] 
        except: 
            hw_forecast = train['Demand'].mean() 
 
        actuals.append(test['Demand'].values[0]) 
        sarima_preds.append(sarima_forecast) 
        prophet_preds.append(prophet_forecast) 
        hw_preds.append(hw_forecast) 
    return actuals, sarima_preds, prophet_preds, hw_preds
 
#  4. Optimize Blending Weights (MAE)  
def optimize_weights(actuals, sarima_preds, prophet_preds, hw_preds):   
    best_mae = float('inf') 
    best_weights = (1/3, 1/3, 1/3) 
    for w1 in np.linspace(0, 1, 21): 
        for w2 in np.linspace(0, 1 - w1, 21): 
            w3 = 1 - w1 - w2 
            blended = w1 * np.array(sarima_preds) + w2 * np.array(prophet_preds) + w3 * np.array(hw_preds) 
            mae = mean_absolute_error(actuals, blended) 
            if mae < best_mae: 
                best_mae = mae 
                best_weights = (w1, w2, w3) 
    return best_weights, best_mae
 
#  5. Forecast 2025 SARIMA Forecast 
def get_sarima_forecast(df: pd.DataFrame):
    sarima_model = SARIMAX(df['Demand'], order=(1,1,1), 
    seasonal_order=(1,1,1,12)).fit() 
    sarima_future = sarima_model.get_forecast(steps=12).predicted_mean 
    future_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS') 
    sarima_future.index = future_index 
    return sarima_future
 
# Prophet Forecast (Enhanced with factors) 
def get_prophet_forecast(df:pd.DataFrame):
    df_prophet = df.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'}) 
    df_prophet['cap'] = df_prophet['y'].max() * 3   
    df_prophet['floor'] = df_prophet['y'].min() * 0.5 
    df_prophet['company_growth'] = df_prophet['ds'].dt.year - 2017 
    df_prophet = add_promotion_factors(df_prophet) 
 
    model_prophet = Prophet(growth='logistic', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False) 
    model_prophet.add_regressor('company_growth') 
    model_prophet.add_regressor('Promotion') 
    model_prophet.fit(df_prophet[['ds', 'y', 'cap', 'floor', 'company_growth', 'Promotion']]) 
 
    future = model_prophet.make_future_dataframe(periods=12, freq='MS') 
    future['cap'] = df_prophet['cap'].iloc[0] 
    future['floor'] = df_prophet['floor'].iloc[0] 
    future['company_growth'] = future['ds'].dt.year - 2017 
    future = add_promotion_factors(future) 
 
    prophet_future = model_prophet.predict(future)['yhat'].values[-12:] 
    return prophet_future
 
# Holt-Winters Forecast 
def get_holt_winter_forecast(df:pd.DataFrame):
    hw_model_full = ExponentialSmoothing(df['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit() 
    hw_future = hw_model_full.forecast(12).values 
    return hw_future
 
#  6. Workforce Scheduling using PuLP  
def get_workforce(combined_forecast, df:pd.DataFrame, best_weights, best_mae):
    future_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    w1, w2, w3 = best_weights
    M = 12  # Months 
    S = 3   # Shift types 
    Productivity = 23 
    Cost = 8.5 
    Days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] 
    Hours = [6, 6, 6] 
 
    model = LpProblem("Workforce_Scheduling", LpMinimize) 
    X = {(i, j): LpVariable(f"X_{i}_{j}", lowBound=0, cat='Integer') for i in range(M) for j in range(S)} 
    model += lpSum(Cost * X[i, j] * Hours[j] * Days[i] for i in range(M) for j in range(S)) 
 
    for i in range(M): 
        model += lpSum(Productivity * X[i, j] * Hours[j] * Days[i] for j in range(S)) >= combined_forecast[i] 
 
    model.solve() 
  
    print(f"\nOptimal Weights: SARIMA={w1:.2f}, Prophet={w2:.2f}, HW={w3:.2f} | MAE (CV): {best_mae:.2f}\n") 
    print(f"{'Month':<10} {'Forecasted Demand':>20} {'Workers Required':>20}") 
    print("=" * 55) 
    workers_list = []
    for i in range(M): 
        month_name = future_index[i].strftime('%b') 
        demand = combined_forecast[i] 
        workers = sum(value(X[i, j]) for j in range(S)) 
        workers_list.append(int(workers))
        print(f"{month_name:<10} {demand:>20.2f} {workers:>20.0f}") 
    df_result = pd.DataFrame({"Month": future_index.strftime("%b-%Y"), "Forecasted Demand":combined_forecast, "Workers Required": workers_list})
    df_result["Forecasted Demand"] = df_result["Forecasted Demand"].round(2)
    # print("\n2025 Forecasted Demand Values (Optimized Weighted Forecast):") 
    # for i in range(M): 
    #     print(f"{future_index[i].strftime('%b %Y')}: {combined_forecast[i]:.2f}") 
    return df_result
 
#################################################################
 ##################################################### 
 
#  9. In-Sample Fitted Forecast  
def in_sample_fit_forecast(df:pd.DataFrame, best_weights):
    sarima_model = SARIMAX(df['Demand'], order=(1,1,1), 
    seasonal_order=(1,1,1,12)).fit() 
    sarima_fitted = sarima_model.fittedvalues 
 
    hw_model_full = ExponentialSmoothing(df['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit() 
    hw_fitted = hw_model_full.fittedvalues 
 
    df_prophet_fit = df.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'}) 
    df_prophet_fit['cap'] = df_prophet_fit['y'].max() * 3 
    df_prophet_fit['floor'] = df_prophet_fit['y'].min() * 0.5 
    df_prophet_fit['company_growth'] = df_prophet_fit['ds'].dt.year - 2017 
    df_prophet_fit = add_promotion_factors(df_prophet_fit) 
 
    model_prophet_fit = Prophet(growth='logistic', 
    yearly_seasonality=True, weekly_seasonality=False, 
    daily_seasonality=False) 
    model_prophet_fit.add_regressor('company_growth') 
    model_prophet_fit.add_regressor('Promotion') 
    model_prophet_fit.fit(df_prophet_fit[['ds', 'y', 'cap', 'floor', 'company_growth', 'Promotion']]) 
 
    future_fit = df_prophet_fit[['ds', 'cap', 'floor', 'company_growth', 'Promotion']] 
    prophet_fitted = model_prophet_fit.predict(future_fit)['yhat'].values 
 
    w1, w2, w3 = best_weights 
    combined_fitted = w1 * sarima_fitted.values + w2 * prophet_fitted + w3 * hw_fitted.values 
 
    common_index = df.index.intersection(df.index[:len(combined_fitted)]) 
    combined_fitted_series = pd.Series(combined_fitted, index=common_index) 
 
    #  10. Evaluation  
    print(f"\nOptimal Weights: SARIMA={w1:.2f}, Prophet={w2:.2f}, HW={w3:.2f} | In-Sample MAE: {mean_absolute_error(df['Demand'], combined_fitted_series):.2f}") 
    print(f"In-Sample MAPE: {mean_absolute_percentage_error(df['Demand'], combined_fitted_series) * 100:.2f}%") 

    insample_df = pd.DataFrame({"Month":df.index, 
                                "Actual": df['Demand'],
                                "SARIMA": sarima_fitted.round(2),
                                "Prophet": prophet_fitted.round(2),
                                "Holt-Winters": hw_fitted.round(2),
                                "Combined": combined_fitted_series.round(2)})
    insample_df["Month"] = insample_df["Month"].dt.strftime("%Y-%m")
    insample_df.reset_index(drop=True, inplace=True)
    return insample_df

def get_result(df:pd.DataFrame):
    df_proc = preprocess_data(df)
    logger.info("loaded data")
    actuals, sarima_preds, prophet_preds, hw_preds = cross_validation(df_proc)
    logger.info("actuals and predictions")
    best_weights, best_mae = optimize_weights(actuals, sarima_preds, prophet_preds, hw_preds)
    sarima_forecast = get_sarima_forecast(df_proc)
    prophet_forecast = get_prophet_forecast(df_proc)
    hw_forecast = get_holt_winter_forecast(df_proc)
    w1, w2, w3 = best_weights 
    combined_forecast = w1 * sarima_forecast.values + w2 * prophet_forecast + w3 * hw_forecast
    future_df = pd.DataFrame({
        "Month": pd.date_range(start=df_proc.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS'),
        "SARIMA": sarima_forecast.values.round(2),
        "Prophet": prophet_forecast.round(2),
        "Holt-Winters": hw_forecast.round(2),
        "Combined": combined_forecast.round(2)
    })
    future_df["Month"] = future_df["Month"].dt.strftime("%Y-%m")
    print(future_df)
    result_df = get_workforce(combined_forecast, df_proc, best_weights, best_mae)
    insample_df = in_sample_fit_forecast(df_proc, best_weights)
    return result_df, insample_df, future_df

if __name__ == "__main__":
    df = pd.read_excel(r"C:\Users\yeswa\Downloads\Demand.xlsx")
    result_df, insample_df, future_df = get_result(df)
    logger.info("Forecasting Result")   
    print(result_df)
    logger.info("Insample Result")   
    print(insample_df)
    logger.info("Future Result")   
    print(future_df)

 
