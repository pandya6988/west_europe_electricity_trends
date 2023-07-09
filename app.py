import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src import data_global_preprocessing as dgp

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

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
    df = dgp.get_fig_for_moving_window(df_global=df ,window=window)

    df = dgp.remove_outliers_zscore(df, threshold=threshold)
    return df

if vis_type == "Global":
    df_global["date"] = pd.to_datetime(df_global["date"])

    df_global = df_global.groupby('date').agg({'price': 'sum', 'demand': 'sum', "supply": "sum"}).reset_index()

    window = st.sidebar.number_input("Moving window", 2, 1000, step=1, value=365)
    threshold = st.sidebar.slider(label="zscore threshold to detect outliers",
                          min_value=1, max_value=5, value=4)
    moving_avg_year = st.sidebar.number_input(label="Metrics year", min_value=2015, max_value=2023, value=2021)

    df_new = get_df_new(df_global, window, threshold)
    fig2 = px.line(df_new, x="date",
                   y=["supply_moving_avg", "demand_moving_avg", "price"],
                   hover_name="date", title="Trends").update_xaxes(dtick="M12", tickformat="%Y")

    df_new.set_index("date", inplace=True)
    df_new["hour"] = df_new.index.hour
    df_new["dayofweek"] = df_new.index.dayofweek
    df_new["quarter"] = df_new.index.quarter
    df_new["month"] = df_new.index.month
    df_new["dayofyear"] = df_new.index.dayofyear

    st.markdown('### Metrics')
    col1, col2, col3 = st.columns(3)

    price_mean_diff = df_new[df_new.index>f'31-12-{moving_avg_year}']['price'].mean() - df_new[df_new.index<f'31-12-{moving_avg_year}']['price'].mean()
    col1.metric(f"Price avg from {moving_avg_year}", f"{round(df_new[df_new.index>f'31-12-{moving_avg_year}']['price'].mean(), 2)}€", f"{round(price_mean_diff,2)}€")

    demand_mean_diff = df_new[df_new.index > f'31-12-{moving_avg_year}']['demand'].mean() - df_new[df_new.index <f'31-12-{moving_avg_year}'][
        'demand'].mean()
    col2.metric(f"Demand avg from {moving_avg_year}", f"{round(df_new[df_new.index >f'31-12-{moving_avg_year}']['demand'].mean(), 2)}MW",
                f"{round(demand_mean_diff, 2)}€")
    supply_mean_diff = df_new[df_new.index >f'31-12-{moving_avg_year}']['supply'].mean() - df_new[df_new.index <f'31-12-{moving_avg_year}'][
        'supply'].mean()
    col3.metric(f"Supply avg from {moving_avg_year}", f"{round(df_new[df_new.index >f'31-12-{moving_avg_year}']['supply'].mean(), 2)}MW",
                f"{round(supply_mean_diff, 2)}€")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Detailed graph")

    col1, col2 = st.columns(2)
    time = col1.selectbox(label="Time parameter", options=["hour", "dayofweek", "quarter", "month", "dayofyear"])
    feature = col2.selectbox("select the feature", ["demand", "supply", "price"] )

    fig, ax = plt.subplots(figsize=(20,8))
    sns.boxplot(df_new, x=time, y=feature, palette="Blues")
    ax.set_title(f"{time} X {feature}")

    st.pyplot(fig, use_container_width=True)