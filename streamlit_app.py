# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from inflation import adjust_for_inflation

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Financial ML App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'show_raw_data' not in st.session_state:
    st.session_state.show_raw_data = False
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False
if 'inflation_adjusted' not in st.session_state:
    st.session_state.inflation_adjusted = False
if 'features_engineered' not in st.session_state:
    st.session_state.features_engineered = False

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stSidebar {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stTitle {
        color: #1f77b4;
        font-size: 2.5em;
        text-align: center;
        padding: 20px;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
    }
    .stInfo {
        background-color: #cce5ff;
        color: #004085;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with enhanced styling
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <h2>📊 Data Input</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload Kragle dataset
    st.markdown("### 📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV from Kaggle", type=["csv"])
    
    st.markdown("---")
    
    # Fetch Yahoo Finance data
    st.markdown("### 📈 Fetch Stock Data")
    ticker = st.text_input("Enter Stock Ticker (Yahoo Finance)", "AAPL")
    date_range = st.date_input("Date Range", [])
    fetch_data = st.button("🔍 Fetch Stock Data", use_container_width=True)

# Main content
if not st.session_state.data_loaded:
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.data_loaded = True
        st.success("✅ Kragle dataset uploaded successfully!")
    elif fetch_data and len(date_range) == 2:
        with st.spinner('Fetching data...'):
            st.session_state.df = yf.download(ticker, start=date_range[0], end=date_range[1])
            st.session_state.df.reset_index(inplace=True)
            st.session_state.data_loaded = True
            st.success(f"✅ Data for {ticker} fetched successfully!")

if not st.session_state.data_loaded:
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1>📈 Financial Machine Learning with Streamlit</h1>
            <p style='font-size: 1.2em;'>Welcome to your interactive financial analysis platform!</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center;'>
            <img src='https://media.giphy.com/media/3o7buirYof8fILTkU0/giphy.gif' width='400'>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h2>🚀 Get Started</h2>
            <p>Upload your dataset or fetch real-time stock data to begin your analysis!</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("### Data Analysis Steps")
    
    # Show Raw Data
    if st.button("🔍 Show Raw Data", use_container_width=True):
        st.session_state.show_raw_data = True
    
    if st.session_state.show_raw_data:
        st.markdown("### Raw Data Preview")
        st.dataframe(st.session_state.df.head().style.set_properties(**{
            'background-color': '#f0f2f6',
            'color': '#1f77b4',
            'border': '1px solid #ddd'
        }))
        st.markdown(f"**Total rows:** {len(st.session_state.df)}")
        st.markdown("**Column Information:**")
        for col in st.session_state.df.columns:
            st.markdown(f"- {col} (Type: {type(col)})")

    # Preprocess Data
    if st.button("⚙️ Preprocess Data", use_container_width=True):
        st.session_state.preprocessed = True
    
    if st.session_state.preprocessed:
        st.markdown("### Preprocessing Results")
        original_rows = len(st.session_state.df)
        st.session_state.df.dropna(inplace=True)
        st.info(f"🧹 Missing values removed. Remaining rows: {len(st.session_state.df)}")
        st.markdown(f"**Rows removed:** {original_rows - len(st.session_state.df)}")
        st.dataframe(st.session_state.df.head())

    # Apply Inflation Adjustment
    if st.button("📐 Apply Inflation Adjustment", use_container_width=True):
        st.session_state.inflation_adjusted = True
    
    if st.session_state.inflation_adjusted:
        st.markdown("### Inflation Adjustment Results")
        try:
            # Create a copy of the dataframe to avoid modifying the original
            df_adjusted = st.session_state.df.copy()
            
            # Flatten the MultiIndex columns if they exist
            if isinstance(df_adjusted.columns, pd.MultiIndex):
                df_adjusted.columns = [col[0] for col in df_adjusted.columns]
            
            # Calculate inflation factor based on dates
            if 'Date' in df_adjusted.columns:
                # Convert Date column to datetime if it's not already
                df_adjusted['Date'] = pd.to_datetime(df_adjusted['Date'])
                
                # Calculate years from the start date
                start_date = df_adjusted['Date'].min()
                df_adjusted['Years'] = (df_adjusted['Date'] - start_date).dt.days / 365.25
                
                # Calculate inflation factor (assuming 2% annual inflation)
                df_adjusted['Inflation Factor'] = (1.02) ** df_adjusted['Years']
            else:
                # If no Date column, use index as years
                df_adjusted['Years'] = df_adjusted.index / 365.25
                df_adjusted['Inflation Factor'] = (1.02) ** df_adjusted['Years']
            
            # Apply inflation adjustment to price columns
            price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
            for col in price_columns:
                if col in df_adjusted.columns:
                    # Create new column for adjusted price
                    new_col = f'{col}_Adj'
                    # Calculate adjusted price
                    df_adjusted[new_col] = df_adjusted[col].astype(float) * df_adjusted['Inflation Factor']
            
            # Update the session state with adjusted data
            st.session_state.df = df_adjusted
            
            # Show the results
            st.success("💰 Inflation factor calculated and applied to prices.")
            st.markdown("#### Original vs Adjusted Prices")
            
            # Prepare columns for display
            display_cols = ['Date'] if 'Date' in df_adjusted.columns else []
            for col in price_columns:
                if col in df_adjusted.columns:
                    display_cols.append(col)
                    display_cols.append(f'{col}_Adj')
            
            # Display the results
            st.dataframe(df_adjusted[display_cols].head())
            
            # Show inflation factor over time
            if 'Date' in df_adjusted.columns:
                fig = px.line(df_adjusted, x='Date', y='Inflation Factor',
                            title='Inflation Factor Over Time')
                st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"❌ Error applying inflation adjustment: {e}")
            st.error("Please make sure your data has the required price columns.")
            # Add debug information
            st.markdown("**Available columns:**")
            st.write(df_adjusted.columns.tolist())
            st.markdown("**DataFrame Info:**")
            st.write(df_adjusted.info())

    # Feature Engineering
    if st.button("🏗️ Feature Engineering", use_container_width=True):
        st.session_state.features_engineered = True
    
    if st.session_state.features_engineered:
        st.markdown("### Feature Engineering Results")
        st.session_state.df['Day'] = pd.to_datetime(st.session_state.df['Date']).dt.dayofyear if 'Date' in st.session_state.df.columns else np.arange(len(st.session_state.df))
        st.session_state.df['Price'] = st.session_state.df['Close'] if 'Close' in st.session_state.df.columns else st.session_state.df.iloc[:, 1]
        st.success("✨ Features engineered successfully!")
        st.dataframe(st.session_state.df[['Day', 'Price']].head())
        
        # Show feature distribution
        fig = px.histogram(st.session_state.df, x='Price', title='Price Distribution')
        st.plotly_chart(fig)

    # Train/Test Split
    if st.button("🧪 Train/Test Split", use_container_width=True):
        if not st.session_state.features_engineered:
            st.error("❌ Please perform feature engineering first!")
        else:
            X = st.session_state.df[['Day']]
            y = st.session_state.df['Price']
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Show split visualization
            fig = px.pie(
                names=['Train', 'Test'],
                values=[len(st.session_state.X_train), len(st.session_state.X_test)],
                color_discrete_sequence=px.colors.qualitative.Set3,
                title='Train/Test Split Distribution'
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig)
            
            st.info(f"📊 Train set size: {len(st.session_state.X_train)}")
            st.info(f"📊 Test set size: {len(st.session_state.X_test)}")

    # Model Training
    if st.button("🤖 Train Linear Regression Model", use_container_width=True):
        if st.session_state.X_train is None or st.session_state.y_train is None:
            st.error("❌ Please perform train/test split first!")
        else:
            with st.spinner('Training model...'):
                st.session_state.model = LinearRegression()
                st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
                st.success("🎯 Model trained successfully!")
                
                # Show model coefficients
                st.markdown("#### Model Coefficients")
                st.markdown(f"**Slope:** {st.session_state.model.coef_[0]:.4f}")
                st.markdown(f"**Intercept:** {st.session_state.model.intercept_:.4f}")

    # Model Evaluation
    if st.button("📊 Evaluate Model", use_container_width=True):
        if st.session_state.model is None:
            st.error("❌ Please train the model first!")
        else:
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            mse = mean_squared_error(st.session_state.y_test, y_pred)
            r2 = r2_score(st.session_state.y_test, y_pred)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Squared Error", f"{mse:.2f}")
            with col2:
                st.metric("R-squared", f"{r2:.2f}")
            
            # Show predictions plot
            fig = px.scatter(
                x=st.session_state.X_test.Day,
                y=st.session_state.y_test,
                labels={'x': 'Day', 'y': 'Actual'},
                color_discrete_sequence=['#1f77b4'],
                title='Model Predictions vs Actual Values'
            )
            fig.add_scatter(
                x=st.session_state.X_test.Day,
                y=y_pred,
                mode='lines',
                name='Predicted',
                line=dict(color='#ff7f0e')
            )
            fig.update_layout(
                xaxis_title="Day",
                yaxis_title="Price",
                template="plotly_white"
            )
            st.plotly_chart(fig)

    # Download Results
    if st.button("📥 Download Results", use_container_width=True):
        if st.session_state.model is None:
            st.error("❌ Please train the model first!")
        else:
            st.session_state.df['Predicted'] = st.session_state.model.predict(st.session_state.df[['Day']])
            st.download_button(
                "📊 Download CSV",
                st.session_state.df.to_csv(index=False),
                file_name="predicted_results.csv",
                mime="text/csv"
            )
            st.success("✅ Results ready for download!")
