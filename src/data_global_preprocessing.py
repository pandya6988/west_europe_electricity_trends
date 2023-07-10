import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import zscore

df_global = pd.read_csv("west_europe_electricity_data.csv")
df_global["date"] = pd.to_datetime(df_global["date"])

df_global = df_global.groupby('date').agg({'price': 'sum', 'demand': 'sum', "supply":"sum"}).reset_index()

def get_fig_for_moving_window(df_global=df_global, window=365):
    df_global[f"demand_moving_avg"] = df_global["demand"].rolling(window=window).mean()
    df_global[f"supply_moving_avg"] = df_global["supply"].rolling(window=window).mean()
    df_global[f"price_moving_avg"] = df_global["price"].rolling(window=window).mean()

    #fig = px.line(df_global, x="date", y=["supply_moving_avg", "demand_moving_avg", "price"],
    #              hover_name="date", title="Global trends").update_xaxes(dtick="M12", tickformat="%Y")
    return df_global


def remove_outliers_zscore(df=df_global, series="price", threshold=4):
    z_scores = zscore(df[series].to_numpy())

    # df = df[np.abs(z_scores) < threshold]
    # df[series].where(np.abs(z_scores) > threshold, df[series].shift())

    last_valid = None
    for idx, value in enumerate(df[series]):
        if np.abs(z_scores[idx]) > threshold:
            if last_valid is not None:
                df.loc[idx, series] = last_valid
        else:
            last_valid = value
    return df

def moving_avg_plt(df:pd.DataFrame):

    # Plot time series for demand, supply, and price
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(df['demand_moving_avg'], label='demand_moving_avg', color='blue')
    plt.title('Electricity Demand')
    plt.ylabel('demand_moving_avg')

    plt.subplot(3, 1, 2)
    plt.plot(df['supply_moving_avg'], label='Supply', color='green')
    plt.title('Electricity Supply')
    plt.ylabel('supply_moving_avg')

    plt.subplot(3, 1, 3)
    plt.plot(df['price_moving_avg'], label='price_moving_avg', color='red')
    plt.title('Electricity Price')
    plt.ylabel('price_moving_avg')

    # Automatically adjust subplot parameters to give specified padding
    plt.tight_layout()

    plt.show()
