import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from src import data_global_preprocessing as dgp

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.set_option('deprecation.showPyplotGlobalUse', False)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.header("West europe dashboard")
st.sidebar.header("control panel")
with st.sidebar:
    vis_type = st.selectbox(label="Visualization type", options=["Global", "By country"])

@st.cache_data
def load_data():
    return pd.read_csv("west_europe_electricity_data.csv")

df_global = load_data()

@st.cache_data
def get_df_new(df, window, threshold):
    df = dgp.remove_outliers_zscore(df, threshold=threshold)
    df = dgp.get_fig_for_moving_window(df_global=df ,window=window)
    return df

if vis_type == "Global":
    df_global["date"] = pd.to_datetime(df_global["date"])

    df_global = df_global.groupby('date').agg({'price': 'sum', 'demand': 'sum', "supply": "sum"}).reset_index()

    window = st.sidebar.number_input("Moving window", 2, 1000, step=1, value=365)
    threshold = st.sidebar.slider(label="zscore threshold to detect outliers",
                          min_value=1, max_value=5, value=4)
    moving_avg_year = st.sidebar.number_input(label="Metrics year", min_value=2015, max_value=2023, value=2021)

    df_new = get_df_new(df_global, window, threshold)

    df_new_m = df_new.copy()
    df_new_m.set_index("date", inplace=True)
    df_new_m["hour"] = df_new_m.index.hour
    df_new_m["dayofweek"] = df_new_m.index.dayofweek
    df_new_m["quarter"] = df_new_m.index.quarter
    df_new_m["month"] = df_new_m.index.month
    df_new_m["dayofyear"] = df_new_m.index.dayofyear

    st.markdown('### Metrics')
    col1, col2, col3 = st.columns(3)

    price_mean_diff = df_new_m[df_new_m.index>f'31-12-{moving_avg_year}']['price'].mean() - df_new_m[df_new_m.index<f'31-12-{moving_avg_year}']['price'].mean()
    col1.metric(f"Price moving avg from {moving_avg_year}", f"{round(df_new_m[df_new_m.index>f'31-12-{moving_avg_year}']['price'].mean(), 2)}€", f"{round(price_mean_diff,2)}€")

    demand_mean_diff = df_new_m[df_new_m.index > f'31-12-{moving_avg_year}']['demand'].mean() - df_new_m[df_new_m.index <f'31-12-{moving_avg_year}'][
        'demand'].mean()
    col2.metric(f"Demand moving avg from {moving_avg_year}", f"{round(df_new_m[df_new_m.index >f'31-12-{moving_avg_year}']['demand'].mean(), 2)}MW",
                f"{round(demand_mean_diff, 2)}MW")
    supply_mean_diff = df_new_m[df_new_m.index >f'31-12-{moving_avg_year}']['supply'].mean() - df_new_m[df_new_m.index <f'31-12-{moving_avg_year}'][
        'supply'].mean()
    col3.metric(f"Supply moving avg from {moving_avg_year}", f"{round(df_new_m[df_new_m.index >f'31-12-{moving_avg_year}']['supply'].mean(), 2)}MW",
                f"{round(supply_mean_diff, 2)}MW")
    show_plotly = st.checkbox("Show plotly", value=False)
    #st.plotly_chart(fig2, use_container_width=True)
    if show_plotly:
        fig2 = px.line(df_new, x="date",
                       y=["supply_moving_avg", "demand_moving_avg", "price_moving_avg"],
                       hover_name="date", title="Data Moving Average Value").update_xaxes(dtick="M12", tickformat="%Y")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.pyplot( dgp.moving_avg_plt(df_new_m) )

    st.markdown("### Trends")
    col_plot, col_heat_map = st.columns(2)
    with col_plot:
        col1, col2 = st.columns(2)
        x = col1.selectbox("X-axis", options=["supply_moving_avg", "demand_moving_avg", "price_moving_avg"])
        x_name = x
        x = df_new[df_new[x] >0 ][x] .to_numpy()
        y = col2.selectbox("Y-axis", options=["price_moving_avg", "supply_moving_avg", "demand_moving_avg"])
        y_name = y
        y = df_new[df_new[y] >0 ][y] .to_numpy()
        #slope, intercept, r_value, p_value, std_err = linregress(df_new[x].to_numpy(), df_new[y].to_numpy())
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        fig4, ax = plt.subplots(figsize=(20, 8))
        ax.plot(x, p(x), 'r', label='fitted line', c="b")
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title("Tred plot")
        st.pyplot(fig4, use_container_width=True)
    with col_heat_map:
        df_new_m["demand_moving_avg_minus_supply_moving_avg"] = df_new_m["demand_moving_avg"] - df_new_m["supply_moving_avg"]
        important_features = ["demand_moving_avg", "supply_moving_avg", "price_moving_avg", "hour", "dayofweek",
                              "month", "dayofyear", "quarter", "demand_moving_avg_minus_supply_moving_avg"]
        correlation_matrix = np.corrcoef(df_new_m[important_features].values.T)
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                    xticklabels=df_new_m[important_features].columns, yticklabels=df_new_m[important_features].columns)
        st.pyplot(fig)

    st.markdown("### Detailed graph")

    col1, col2 = st.columns(2)
    time = col1.selectbox(label="Time parameter", options=["hour", "dayofweek", "quarter", "month", "dayofyear"])
    feature = col2.selectbox("select the feature", ["demand", "supply", "price"] )

    fig, ax = plt.subplots(figsize=(20,8))
    sns.boxplot(df_new_m, x=time, y=feature, palette="Blues")
    ax.set_title(f"{time} X {feature}")

    st.pyplot(fig, use_container_width=True)

if vis_type == "By country":
    df_global["date"] = pd.to_datetime(df_global["date"])
    window = st.sidebar.number_input("Moving window", 2, 1000, step=1, value=365)
    threshold = st.sidebar.slider(label="zscore threshold to detect outliers",
                                  min_value=1, max_value=5, value=3)
    df_new = get_df_new(df_global, window, threshold)

    #fig = px.line(df_new, x="date",
    #               y=["supply_moving_avg", "demand_moving_avg", "price"], color="country",
    #               hover_name="date", title="Data Moving Average Value").update_xaxes(dtick="M12", tickformat="%Y")
    #st.plotly_chart(fig, use_container_width=True)
    countries = df_global['country'].unique()
    df_global["date"] = pd.to_datetime(df_global["date"])
    df_global.set_index('date', inplace=True)
    df_new1 = df_global.groupby('country').resample('D').mean()

    fig, ax = plt.subplots(len(countries), 3, figsize=(18, 2.5 * len(countries)))

    for i, country in enumerate(countries):
        ax[i, 0].plot(df_new1.loc[country, 'demand'], color='blue')
        ax[i, 0].set_title(f'Daily Electricity Demand in {country}')
        ax[i, 0].set_ylabel('Demand')

        ax[i, 1].plot(df_new1.loc[country, 'supply'], color='green')
        ax[i, 1].set_title(f'Daily Electricity Supply in {country}')
        ax[i, 1].set_ylabel('Supply')

        ax[i, 2].plot(df_new1.loc[country, 'price'], color='red')
        ax[i, 2].set_title(f'Daily Electricity Price in {country}')
        ax[i, 2].set_ylabel('Price')

    plt.tight_layout()
    st.pyplot(fig)