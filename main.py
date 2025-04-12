import streamlit as st
import pandas as pd
import datetime
from io import BytesIO
from src import forecast_demand_workers
import plotly.graph_objects as go

st.set_page_config(layout="centered", page_title="Demand - Workers Forecasting")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"], .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al,
        h1, h2, h3, h4, h5, h6, .stTabs [data-baseweb="tab-list"], .stTabs [data-baseweb="tab"],
        button, .streamlit-expanderHeader, p, div, span {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Specific styling for title */
        h1 {
            font-weight: 700 !important;
            color: #111827 !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 4px 4px 0 0;
            padding: 10px 16px;
            font-weight: 600;
        }
        
        /* Background color */
        .stApp {
            background-color: #fffff;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Demand Forecast & Staffing</h1>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

def create_forecast_chart(df, x_col, y_cols, title):
    """
    Create a beautiful Plotly chart for forecast visualization
    
    Parameters:
    df (DataFrame): The dataframe containing the data
    x_col (str): The column name for x-axis (dates)
    y_cols (list): List of column names to plot
    title (str): Chart title
    
    Returns:
    plotly figure object
    """
    # Custom color palette for the lines
    colors = {
        "Actual": "#3366CC",       # Blue for actual values
        "SARIMA": "#DC3912",       # Red for SARIMA
        "Prophet": "#FF9900",      # Orange for Prophet
        "Holt_Winters": "#109618", # Green for Holt-Winters
        "Combined": "#990099"      # Purple for Combined
    }
    fig = go.Figure()
    
    for col in y_cols:
        line_width = 2 if col == "Actual" else 1
        dash_style = None if col == "Actual" else None
        
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[col],
                name=col.replace("_", "-"),
                line=dict(color=colors.get(col, "#000000"), width=line_width, dash=dash_style),
                mode='lines',
                marker=dict(size=6),
                hovertemplate=f"{col}: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>"
            )
        )
    
    fig.update_layout(
        title={
            'text': title,
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, family='Inter, sans-serif', color="#111827")
        },
        xaxis_title=None,
        yaxis_title="Demand",
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.2,
            'xanchor': 'center',
            'x': 0.5
        },
        plot_bgcolor='rgba(250,250,250,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=60, b=60),
        hovermode="x unified",
        font=dict(family='Inter, sans-serif'),
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211,211,211,0.5)',
        tickfont=dict(family='Inter, sans-serif')
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211,211,211,0.5)',
        tickfont=dict(family='Inter, sans-serif')
    )
    
    return fig

if uploaded_file:
    try:
        with st.spinner('Generating demand forecasts and staffing requirements...'):
            df = pd.read_excel(uploaded_file)
            result_df, insample_df, future_df = forecast_demand_workers(df)
        st.success('Forecast completed successfully!')
        dates = result_df["Month"].tolist()
        workers = result_df["Workers Required"].tolist()
        demand = result_df["Forecasted Demand"].tolist()
        insample = {
            "Month": insample_df["Month"].tolist(),
            "Actual": insample_df["Actual"].tolist(),
            "SARIMA": insample_df["SARIMA"].tolist(),
            "Prophet": insample_df["Prophet"].tolist(),
            "Holt_Winters": insample_df["Holt-Winters"].tolist(),
            "Combined": insample_df["Combined"].tolist()
        }
        future = {
            "Month": future_df["Month"].tolist(),
            "SARIMA": future_df["SARIMA"].tolist(),
            "Prophet": future_df["Prophet"].tolist(),
            "Holt_Winters": future_df["Holt-Winters"].tolist(),
            "Combined": future_df["Combined"].tolist()
        }

        tab1, tab2, tab3 = st.tabs(["Results", "In-Sample Analysis", "Future Forecast"])

        with tab1:
            result_df = pd.DataFrame({
                "Month": dates, "Forecasted Demand": demand, "Workers Required": workers
            })
            st.dataframe(result_df, use_container_width=True, hide_index=True)
        with tab2:
            insample_df = pd.DataFrame(insample)
            x_cols = "Month"
            y_cols = ["Actual", "SARIMA", "Prophet", "Holt_Winters", "Combined"]
            insample_fig = create_forecast_chart(insample_df, x_cols, y_cols, "In-Sample Fit Comparison")
            st.plotly_chart(insample_fig, use_container_width=True)
        with tab3:
            future_df = pd.DataFrame(future)
            x_cols = "Month"
            y_cols = ["SARIMA", "Prophet", "Holt_Winters", "Combined"]
            forecast_fig = create_forecast_chart(future_df, x_cols, y_cols, "Future Forecast Comparison")
            st.plotly_chart(forecast_fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error processing the file: {e}")
        st.stop()