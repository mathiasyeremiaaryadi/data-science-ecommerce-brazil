import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import altair as alt
import plotly.graph_objects as go
from babel.numbers import format_currency
import numpy as np

def create_rfm_df(main_df):
    recency_df = main_df[["customer_id", "order_purchase_timestamp"]]
    recency_df["order_purchase_timestamp"] = pd.to_datetime(recency_df["order_purchase_timestamp"]).dt.date
    latest_order_date = main_df["order_purchase_timestamp"].dt.date.max()
    recency_df.loc[:, "recency"] = recency_df["order_purchase_timestamp"].apply(lambda x: (latest_order_date - x).days)
    recency_df.drop_duplicates(inplace=True)

    frequency_df = main_df.groupby("customer_id").order_purchase_timestamp.count().sort_values(ascending=False).reset_index()
    frequency_df.columns = ["customer_id", "frequency"]

    monetary_df = main_df[["customer_id", "payment_value"]]
    monetary_df = monetary_df.groupby("customer_id").payment_value.sum().sort_values(ascending=False).reset_index()
    monetary_df.columns = ["customer_id", "monetary"]

    rfm_df = recency_df.merge(monetary_df, on="customer_id")
    rfm_df = rfm_df.merge(frequency_df, on="customer_id")
    rfm_df.drop(columns=["order_purchase_timestamp"], inplace=True)

    rfm_df["r_rank"] = rfm_df["recency"].rank(ascending=False)
    rfm_df["f_rank"] = rfm_df["frequency"].rank(ascending=True)
    rfm_df["m_rank"] = rfm_df["monetary"].rank(ascending=True)

    rfm_df["r_rank_norm"] = (rfm_df["r_rank"] / rfm_df["r_rank"].max()) * 100
    rfm_df["f_rank_norm"] = (rfm_df["f_rank"] / rfm_df["f_rank"].max()) * 100
    rfm_df["m_rank_norm"] = (rfm_df["m_rank"] / rfm_df["m_rank"].max()) * 100

    rfm_df["r_score"] = rfm_df["r_rank_norm"] * 0.05
    rfm_df["f_score"] = rfm_df["f_rank_norm"] * 0.05
    rfm_df["m_score"] = rfm_df["m_rank_norm"] * 0.05

    rfm_df["rfm_score"] = rfm_df["r_score"] * 0.2 + rfm_df["f_score"] * 0.3 + rfm_df["m_score"] * 0.5

    rfm_df["category"] = np.where(rfm_df["rfm_score"] > 4.0, "Top Customer",
                        np.where(rfm_df["rfm_score"] > 3.0, "High Value Customer",
                        np.where(rfm_df["rfm_score"] > 2.0, "Medium Value Customer",
                        np.where(rfm_df["rfm_score"] > 1.0, "Low Value Customer", "Bottom"))))
    
    return rfm_df

def visualize_geospatial(main_df):
    customer_geospatial_df = main_df[["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state", "customer_geolocation_lat", "customer_geolocation_lng"]]

    fig = px.scatter_geo(customer_geospatial_df,
                    lat=customer_geospatial_df.customer_geolocation_lat,
                    lon=customer_geospatial_df.customer_geolocation_lng,
                    hover_name="customer_city")

    fig.update_layout(
        title_text="Geospatial Distribution of Customers",
        geo_scope='south america',
    )

    return fig

def visualize_most_customer_city(main_df):
    customer_top10_city = main_df.groupby("customer_city").customer_id.count().sort_values(ascending=False).head(10).reset_index()
    customer_top10_city.sort_values(by="customer_id", inplace=True)
    colors = ['lightslategray',] * len(customer_top10_city)
    colors[-1] = 'crimson'

    fig = go.Figure(data=[go.Bar(
        x=customer_top10_city.customer_id,
        y=customer_top10_city.customer_city,
        marker_color=colors,
        orientation='h',
    )])

    fig.update_layout(title_text='Top 10 Most Customer Cities')
    return fig

def visualize_most_customer_state(main_df):
    customer_top10_state = main_df.groupby("customer_state").customer_id.count().sort_values(ascending=False).head(10).reset_index()
    customer_top10_state.sort_values(by="customer_id", inplace=True)
    colors = ['lightslategray',] * len(customer_top10_state)
    colors[-1] = 'crimson'

    fig = go.Figure(data=[go.Bar(
        x=customer_top10_state.customer_id,
        y=customer_top10_state.customer_state,
        marker_color=colors,
        orientation='h',
    )])

    fig.update_layout(title_text='Top 10 Most Customer States')
    return fig


def visualize_payment_method_by_usage(main_df):
    payment_type_df = main_df[["customer_id", "order_id", "payment_type", "payment_installments", "payment_sequential", "payment_value", "order_year", "order_month", "order_day"]]
    most_payment_type_df = payment_type_df.groupby("payment_type").customer_id.nunique().sort_values(ascending=False).reset_index()
    fig = px.pie(most_payment_type_df, values="customer_id", names="payment_type", title="Payment Types By Usage")
    return fig

def visualize_payment_method_by_sequential(main_df):
    most_payment_sequential_df = main_df.groupby("payment_type").payment_sequential.nunique().sort_values(ascending=False).reset_index()
    fig = px.pie(most_payment_sequential_df, values="payment_sequential", names="payment_type", title="Payment Method By Sequential (N Times)")
    return fig

def visualize_payment_method_by_installments(main_df):
    most_payment_type_df = main_df.groupby("payment_type").payment_installments.nunique().sort_values(ascending=False).reset_index()  
    fig = px.pie(most_payment_type_df, values="payment_installments", names="payment_type", title="Payment Method By Installments (N Times)")
    return fig

def visualize_payment_method_growth(main_df):
    payment_type_df = main_df[["customer_id", "order_id", "payment_type", "payment_installments", "payment_sequential", "payment_value", "order_year", "order_month", "order_day"]]
    payment_type_time_df = payment_type_df.groupby(["order_year", "payment_type"]).customer_id.nunique().reset_index()

    fig = px.line(payment_type_time_df, x='order_year', y='customer_id', color='payment_type', markers=True)

    fig.update_layout(
        title_text="Payment Method Usage Growth",
        xaxis=dict(
            tickmode='linear',
            tick0=2016,
            dtick=1
        ),
        xaxis_title="Year",
        yaxis_title="Total Customer"
    )
   
    return fig

def visualize_best_selling_product(main_df):
    product_top10_df = main_df.groupby("product_category_name").customer_id.count().sort_values(ascending=False).head(10).reset_index()
    product_top10_df.sort_values(by="customer_id", inplace=True)
    colors = ['lightslategray',] * len(product_top10_df)
    colors[-1] = 'crimson'

    fig = go.Figure(data=[go.Bar(
        x=product_top10_df.customer_id,
        y=product_top10_df.product_category_name,
        marker_color=colors,
        orientation='h',
    )])

    fig.update_layout(title_text='Top 10 Best Selling Product')
    return fig

def visualize_worst_selling_product(main_df):
    product_down10_df = main_df.groupby("product_category_name").customer_id.count().sort_values().head(10).reset_index()
    colors = ['lightslategray',] * len(product_down10_df)
    colors[0] = 'crimson'

    fig = go.Figure(data=[go.Bar(
        x=product_down10_df.customer_id,
        y=product_down10_df.product_category_name,
        marker_color=colors,
        orientation='h',
    )])

    fig.update_layout(title_text='Top 10 Worst Selling Product')
    return fig

def visualize_customer_review_score(main_df):
    all_reviews_df = main_df.groupby("review_score").customer_id.count().reset_index()
    fig = px.pie(all_reviews_df, values="customer_id", names="review_score", title="Based On Review Score")
    return fig

def visualize_customer_review_order_status(main_df):
    review_status_df = main_df.groupby(["review_score", "order_status"]).order_id.count().reset_index()
    
    fig = px.histogram(
        review_status_df, 
        x="review_score", 
        y="order_id",
        color='order_status',
        barmode='group',
        height=500, 
        title="Based on Order Status"
    )

    fig.update_layout(
        xaxis_title="Review Score",
        yaxis_title="Total Review"
    )

    fig.update_yaxes(type="log")
    return fig

def visualize_customer_satisification_growth(main_df):
    review_time_df = main_df.groupby(["order_year", "review_score"]).customer_id.count().reset_index()
    fig = px.line(review_time_df, x='order_year', y='customer_id', color='review_score', markers=True)

    fig.update_layout(
        title_text="Customer Satisaction Growth",
        xaxis=dict(
            tickmode='linear',
            tick0=2016,
            dtick=1
        ),
        xaxis_title="Year",
        yaxis_title="Total Score Review"
    )

    return fig

def visualize_customer_review_category(main_df):
    review_category_df = main_df.groupby("review_category").customer_id.count().reset_index()
    fig = px.pie(review_category_df, values="customer_id", names="review_category", title="Based On Review Categories")
    return fig

def visualize_customer_review_score_category(main_df):
    score_category_df = main_df.groupby(["review_score", "review_category"]).customer_id.count().reset_index()

    fig = px.histogram(score_category_df, x="review_score", y="customer_id",
                color='review_category', barmode='group',
                height=500, title="Total Review Based on Review Score")

    fig.update_layout(
        xaxis_title="Review Score",
        yaxis_title="Total Review"
    )

    return fig

def visualize_order_status(main_df):
    order_status_percent_df = main_df.groupby("order_status").order_id.count().reset_index()
    order_status_percent_df.sort_values(by="order_id", ascending=False, inplace=True)
    colors = ['lightslategray',] * len(order_status_percent_df)
    colors[0] = 'crimson'

    fig = go.Figure(data=[go.Bar(
        x=order_status_percent_df.order_status,
        y=order_status_percent_df.order_id,
        marker_color=colors,
    )])

    fig.update_yaxes(type="log")
    fig.update_layout(title_text='Basend On Order Status')
    return fig

def visualize_order_status_by_year(main_df):
    order_status_year_df = main_df.groupby(["order_year", "order_status"]).order_id.count().reset_index()

    fig = px.histogram(order_status_year_df, x="order_year", y="order_id",
                color='order_status', barmode='group',
                height=500, title="Based On Year")

    fig.update_layout(yaxis_type="log", xaxis_title="Year", yaxis_title="Total Order")
    return fig

def visualize_order_status_by_month(main_df):
    order_status_month_df = main_df.groupby(["order_month", "order_status"]).order_id.count().reset_index()
    fig = px.histogram(order_status_month_df, x="order_month", y="order_id",
                color='order_status', barmode='group',
                height=500, title="Based On Month")

    fig.update_layout(yaxis_type="log", xaxis_title="Month", yaxis_title="Total Order")
    return fig

def visualize_order_status_by_day(main_df):
    order_status_month_df = main_df.groupby(["order_day", "order_status"]).order_id.count().reset_index()

    fig = px.histogram(order_status_month_df, x="order_day", y="order_id",
                color='order_status', barmode='group',
                height=500, title="Based On Day")

    fig.update_layout(yaxis_type="log", xaxis_title="Day", yaxis_title="Total Order")
    return fig

def visualize_recency(main_df):
    recency_df = main_df[["customer_id", "order_purchase_timestamp"]]
    recency_df["order_purchase_timestamp"] = recency_df["order_purchase_timestamp"].dt.date
    latest_order_date = main_df["order_purchase_timestamp"].dt.date.max()
    recency_df.loc[:, "recency"] = recency_df["order_purchase_timestamp"].apply(lambda x: (latest_order_date - x).days)
    recency_df.drop_duplicates(inplace=True)
    fig = px.bar(recency_df.sort_values(by="recency").head(10), x="customer_id", y="recency", title="Top 10 Customer Recency", height=550)
    fig.update_layout(xaxis_title="Customer ID", yaxis_title="Recency (Day)")
    return fig

def visualize_frequency(main_df):
    frequency_df = main_df.groupby("customer_id").order_purchase_timestamp.count().sort_values(ascending=False).reset_index()
    frequency_df.columns = ["customer_id", "frequency"]
    fig = px.bar(frequency_df.head(10), x="customer_id", y="frequency", title="Top 10 Customer Frequency", height=550)
    fig.update_layout(xaxis_title="Customer ID", yaxis_title="Frequency (N Times)")
    return fig

def visualize_monetary(main_df):
    monetary_df = main_df[["customer_id", "payment_value"]]
    monetary_df = monetary_df.groupby("customer_id").payment_value.sum().sort_values(ascending=False).reset_index()
    monetary_df.columns = ["customer_id", "monetary"]
    fig = px.bar(monetary_df.head(10), x="customer_id", y="monetary", title="Top 10 Customer Monetary", height=550)
    fig.update_layout(xaxis_title="Customer ID", yaxis_title="Monetary (BRL)")
    return fig

def visualize_customer_segmentation(main_df):
    rfm_df_count = rfm_df.groupby("category").customer_id.count().reset_index()
    fig = px.pie(rfm_df_count, values="customer_id", names="category", title="Customer Segmentation")
    return fig

#########################################################################################################################
list_month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
list_day = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

main_df = pd.read_csv("dashboard/main_data.csv")
main_df["review_score"].astype("int")
main_df["order_purchase_timestamp"] = pd.to_datetime(main_df["order_purchase_timestamp"])
main_df["order_month"] = pd.Categorical(main_df["order_month"], categories=list_month, ordered=True)
main_df["order_day"] = pd.Categorical(main_df["order_day"], categories=list_day, ordered=True)

min_order_date = main_df["order_purchase_timestamp"].dt.date.min()
max_order_date = main_df["order_purchase_timestamp"].dt.date.max()

rfm_df = create_rfm_df(main_df)

st.set_page_config(
    page_title="Brazil E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

with st.sidebar:
    st.title('🛒 Brazil E-Commerce Dashboard')
    st.write("By Mathias Yeremia Aryadi")
    
    selected_start_order_date, selected_end_order_date = st.date_input(
        label="Date Filter",
        min_value=min_order_date,
        max_value=max_order_date,
        value=[min_order_date, max_order_date]
    )

    st.caption('Copyright (C) Mathias Yeremia Aryadi 2024')

filtered_df = main_df[(main_df["order_purchase_timestamp"] >= str(selected_start_order_date)) & 
                (main_df["order_purchase_timestamp"] <= str(selected_end_order_date))]


##################### OVERVIEW METRICS
total_customer = filtered_df["customer_unique_id"].nunique()
total_product = filtered_df["product_category_name"].nunique()
total_order = filtered_df["order_id"].count()
total_seller = filtered_df["seller_id"].nunique()
total_city = filtered_df["customer_city"].unique()

st.header("Overview Metric", divider=True, anchor=False)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Customers", value=f"{total_customer:,}".replace(",", "."))
with col2:
    st.metric(label="Total Products", value=f"{total_product:,}".replace(",", "."))
with col3:
    st.metric(label="Total Orders", value=f"{total_order:,}".replace(",", "."))
with col4:
    st.metric(label="Total Sellers", value=f"{total_seller:,}".replace(",", "."))


total_payment_method = filtered_df["payment_type"].nunique()
total_income = filtered_df["payment_value"].sum()
total_good_revies = filtered_df[(filtered_df["review_score"] == 4) | (filtered_df["review_score"] == 5)]["review_score"].count()
total_bad_revies = filtered_df[(filtered_df["review_score"] == 1) | (filtered_df["review_score"] == 2)]["review_score"].count()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Payment Method", value=f"{total_payment_method:,}".replace(",", "."))
with col2:
    total_income = format_currency(total_income, "BRL", locale='pt_BR') 
    st.metric(label="Total Income (BRL)", value=total_income)
with col3:
    st.metric(label="Total Good Reviews (4-5)", value=f"{total_good_revies:,}".replace(",", "."))
with col4:
    st.metric(label="Total Bad Reviews (1-2)", value=f"{total_bad_revies:,}".replace(",", "."))

average_income = filtered_df["payment_value"].mean()
max_income = filtered_df["payment_value"].max()
min_income = filtered_df["payment_value"].min()
total_city = filtered_df["customer_city"].nunique()

col1, col2, col3, col4 = st.columns(4)
with col1:
    average_income = format_currency(average_income, "BRL", locale='pt_BR') 
    st.metric(label="Average Income (BRL)", value=average_income)
with col2:
    max_income = format_currency(max_income, "BRL", locale='pt_BR') 
    st.metric(label="Maximum Income (BRL)", value=max_income)
with col3:
    min_income = format_currency(min_income, "BRL", locale='pt_BR') 
    st.metric(label="Minium Income (BRL)", value=min_income)
with col4:
    st.metric(label="Total Customer Cities", value=f"{total_city:,}".replace(",", "."))
################################################################################


##################### RFM METRICS
st.text("")
st.text("")
st.text("")
st.header("RFM (Recency, Frequency, Monetary) Metrics", divider=True, anchor=False)

average_recency = rfm_df["recency"].mean()
average_frequency = rfm_df["frequency"].mean()
average_monetary = rfm_df["monetary"].mean()

col1, col2, col3 = st.columns(3)
with col1:
    average_recency = round(average_recency)
    st.metric(label="Average Recency (Days)", value=average_recency)
with col2:
    average_frequency = round(average_frequency, 2)
    st.metric(label="Average Frequency (Times)", value=average_frequency)
with col3:
    average_monetary = format_currency(average_monetary, "BRL", locale='pt_BR') 
    st.metric(label="Average Monetary (BRL)", value=average_monetary)
################################################################################


##################### Geospatial
st.text("")
st.text("")
st.text("")
st.header("Customer Distribution Geographically", divider=True, anchor=False)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(visualize_geospatial(main_df), use_container_width=True)
with col2:
    st.plotly_chart(visualize_most_customer_city(main_df), use_container_width=True)

_, col2, _ = st.columns(3)
with col2:
    st.plotly_chart(visualize_most_customer_state(main_df), use_container_width=True)
################################################################################


##################### Payment Method
st.text("")
st.text("")
st.text("")
st.header("Payment Method", divider=True, anchor=False)

col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(visualize_payment_method_by_usage(main_df), use_container_width=True)
with col2:
    st.plotly_chart(visualize_payment_method_by_sequential(main_df), use_container_width=True)
with col3:
    st.plotly_chart(visualize_payment_method_by_installments(main_df), use_container_width=True)

st.plotly_chart(visualize_payment_method_growth(main_df), use_container_width=True)
################################################################################


##################### Products
st.text("")
st.text("")
st.text("")
st.header("Product Sales", divider=True, anchor=False)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(visualize_best_selling_product(main_df), use_container_width=True)
with col2:
    st.plotly_chart(visualize_worst_selling_product(main_df), use_container_width=True)
################################################################################


##################### Customer Review Score
st.text("")
st.text("")
st.text("")
st.header("Customer Satisfication", divider=True, anchor=False)

review_df = main_df[["customer_id", "review_id", "order_id", "review_score", "review_category", "order_status", "order_year"]]
st.plotly_chart(visualize_customer_review_score(review_df), use_container_width=True)
st.plotly_chart(visualize_customer_review_order_status(review_df), use_container_width=True)
st.plotly_chart(visualize_customer_satisification_growth(review_df), use_container_width=True)
################################################################################


##################### Customer Review Engagement
st.text("")
st.text("")
st.text("")
st.header("Customer Review Engagement", divider=True, anchor=False)

review_df = main_df[["customer_id", "review_id", "review_score", "review_category", "order_year", "order_month", "order_day"]]
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(visualize_customer_review_category(review_df), use_container_width=True)
with col2:
    st.plotly_chart(visualize_customer_review_score_category(review_df), use_container_width=True)
################################################################################


##################### Order Performance
st.text("")
st.text("")
st.text("")
st.header("Order Performance", divider=True, anchor=False)

order_status_df = main_df[["order_id", "order_status", "order_year", "order_month", "order_day"]]
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(visualize_order_status(order_status_df), use_container_width=True)
with col2:
    st.plotly_chart(visualize_order_status_by_year(order_status_df), use_container_width=True)

st.plotly_chart(visualize_order_status_by_month(order_status_df), use_container_width=True)
st.plotly_chart(visualize_order_status_by_day(order_status_df), use_container_width=True)
################################################################################


##################### Customer Segmentation
st.text("")
st.text("")
st.text("")
st.header("Customer Segmentation", divider=True, anchor=False)


st.plotly_chart(visualize_recency(main_df), use_container_width=True)
st.plotly_chart(visualize_frequency(main_df), use_container_width=True)
st.plotly_chart(visualize_monetary(main_df), use_container_width=True)
st.plotly_chart(visualize_customer_segmentation(create_rfm_df(main_df)), use_container_width=True)
################################################################################
