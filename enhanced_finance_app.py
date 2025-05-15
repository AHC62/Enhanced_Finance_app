# Core Libraries
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# ML Essentials
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    HistGradientBoostingRegressor  # Recommended addition
)
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Conditional Imports (uncomment if needed)
import yfinance as yf  # For stock data
import joblib  # For model saving
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
# ======================
# 🎮 DUAL THEME ENGINE
# ======================

CYBERPUNK_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle, #000428 0%, #004e92 100%);
    color: #00f3ff;
    font-family: 'Courier New', monospace;
}

[data-testid="stSidebar"] {
    background: rgba(0, 4, 40, 0.9) !important;
    border-right: 2px solid #00f3ff !important;
}

h1, h2, h3 {
    color: #ff00ff !important;
    text-shadow: 0 0 10px #ff00ff;
}

.stButton>button {
    background: linear-gradient(45deg, #00f3ff, #ff00ff);
    border: none !important;
    color: #000 !important;
    border-radius: 25px;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.1);
    box-shadow: 0 0 20px #00f3ff;
}

.stTextInput>div>div>input {
    background: rgba(0, 4, 40, 0.8) !important;
    color: #00f3ff !important;
    border: 1px solid #00f3ff !important;
}

.stSelectbox>div>div>select {
    background: rgba(0, 4, 40, 0.8) !important;
    color: #00f3ff !important;
}

.stSlider>div>div>div>div {
    background: #00f3ff !important;
}
</style>
"""

MATRIX_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background: #000000;
    color: #00ff00;
    font-family: 'Courier New', monospace;
}

[data-testid="stSidebar"] {
    background: rgba(0, 30, 0, 0.9) !important;
    border-right: 2px solid #00ff00 !important;
}

h1, h2, h3 {
    color: #00ff00 !important;
    text-shadow: 0 0 10px #00ff00;
}

.stButton>button {
    background: #000 !important;
    border: 2px solid #00ff00 !important;
    color: #00ff00 !important;
    border-radius: 0;
}

.stButton>button:hover {
    box-shadow: 0 0 20px #00ff00;
}

.stTextInput>div>div>input {
    background: rgba(0, 30, 0, 0.8) !important;
    color: #00ff00 !important;
    border: 1px solid #00ff00 !important;
}

.stSelectbox>div>div>select {
    background: rgba(0, 30, 0, 0.8) !important;
    color: #00ff00 !important;
}

.stSlider>div>div>div>div {
    background: #00ff00 !important;
}
</style>
"""

def apply_theme(theme):
    if theme == "Cyberpunk 2077":
        st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)
        st.image("https://media.giphy.com/media/3o7aCTPPm4OHfRLSH6/giphy.gif", 
                use_column_width=True)
        st.markdown("""
        <div style="text-align: center; border: 2px solid #00f3ff; padding: 20px; margin: 20px 0;">
            <h1>🛸 CYBER FINANCE AI 3000 🛸</h1>
            <h3>«« NEURAL NETWORK ACTIVATED »»</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(MATRIX_CSS, unsafe_allow_html=True)
        st.image("https://media.giphy.com/media/l0MYEqEzwMWFCg8rm/giphy.gif", 
                use_column_width=True)
        st.markdown("""
        <div style="text-align: center; border: 2px solid #00ff00; padding: 20px; margin: 20px 0;">
            <h1>MATRIX FINANCIAL SYSTEM</h1>
            <h3>«« FOLLOW THE WHITE RABBIT »»</h3>
        </div>
        """, unsafe_allow_html=True)

# ======================
# 🚀 ORIGINAL FUNCTIONALITY
# ======================

def init_session_state():
    session_defaults = {
        'df': None,
        'original_df': None,
        'models': {},
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'y_pred': None,
        'preprocessor': None,
        'feature_selector': None,
        'current_model': None,
        'metrics': {},
        'feature_importance': None,
        'preprocessing_done': False,
        'preprocessing_active': False,
        'missing_values': "Drop rows",
        'outlier_threshold': 1.5,
        'scaling_method': "None"
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_data():
    try:
        if st.session_state.data_source == "Upload File":
            if st.session_state.file_uploader is not None:
                if st.session_state.file_uploader.name.endswith('.csv'):
                    df = pd.read_csv(st.session_state.file_uploader)
                else:
                    df = pd.read_excel(st.session_state.file_uploader)
            else:
                st.warning("⚠️ Please upload a file")
                return None
                
        elif st.session_state.data_source == "Kaggle Dataset":
            if not all([st.session_state.kaggle_user, 
                       st.session_state.kaggle_key,
                       st.session_state.kaggle_dataset]):
                st.error("❌ Missing Kaggle credentials or dataset URL")
                return None
                
            try:
                import os
                os.environ['KAGGLE_USERNAME'] = st.session_state.kaggle_user
                os.environ['KAGGLE_KEY'] = st.session_state.kaggle_key
                from kaggle.api.kaggle_api_extended import KaggleApi
                
                api = KaggleApi()
                api.authenticate()
                with st.spinner(f"Downloading {st.session_state.kaggle_dataset}..."):
                    api.dataset_download_files(st.session_state.kaggle_dataset, unzip=True)
                    dataset_name = st.session_state.kaggle_dataset.split('/')[-1]
                    df = pd.read_csv(f"{dataset_name}.csv")
                    
            except Exception as e:
                st.error(f"❌ Kaggle Error: {str(e)}")
                return None
                
        elif st.session_state.data_source == "Yahoo Finance":
            try:
                import yfinance as yf
                with st.spinner(f"Fetching {st.session_state.yahoo_ticker} data..."):
                    df = yf.download(
                        tickers=st.session_state.yahoo_ticker,
                        start=st.session_state.yahoo_start,
                        end=st.session_state.yahoo_end,
                        interval=st.session_state.yahoo_interval
                    ).reset_index()
                    
            except Exception as e:
                st.error(f"❌ Yahoo Finance Error: {str(e)}")
                return None
                
        st.session_state.original_df = df.copy()
        st.session_state.df = df.copy()
        st.session_state.preprocessing_done = False
        
        with st.expander("🔍 Data Preview (First 5 Rows)"):
            st.dataframe(df.head())
        
        with st.expander("📊 Data Summary (Statistics)"):
            st.write(df.describe().style.format("{:.2f}"))
        
        with st.expander("🧐 Data Information (dtypes/memory)"):
            buffer = BytesIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue().decode('utf-8'))
        
        if len(df.select_dtypes(include=np.number).columns) > 1:
            sample_size = min(100, len(df))
            sample_df = df.sample(sample_size) if len(df) > 100 else df
            fig = px.scatter_matrix(
                sample_df,
                dimensions=df.select_dtypes(include=np.number).columns[:4],
                title="Scatter Matrix of First 4 Numeric Columns"
            )
            st.plotly_chart(fig)
        else:
            st.warning("⚠️ Not enough numeric columns for visualization")
        
        return df
        
    except Exception as e:
        st.error(f"💥 Error loading data: {str(e)}")
        return None

def update_preprocessing():
    if st.session_state.df is None:
        return
    
    df = st.session_state.original_df.copy()
    
    # Date conversion
    date_cols = []
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='ignore')
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
        except:
            pass
    
    if date_cols:
        df = df.drop(columns=date_cols)
    
    # Numeric conversion
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.replace("%", ""), errors="ignore")
        except:
            pass
    
    # Get numeric columns after conversion
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Skip missing value handling if no numeric columns exist
    if not numeric_cols:
        st.warning("⚠️ No numeric columns found after conversion")
        st.session_state.df = df
        return
    
    # Handle missing values - with safety checks
    missing_option = st.session_state.get('missing_values', "Drop rows")
    
    try:
        if missing_option == "Drop rows":
            df = df.dropna(subset=numeric_cols)
        elif missing_option == "Drop columns":
            # Only drop columns that actually exist in the DataFrame
            cols_to_drop = [col for col in numeric_cols if col in df.columns]
            df = df.dropna(axis=1, subset=cols_to_drop)
        elif missing_option == "Fill with mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif missing_option == "Fill with median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    except Exception as e:
        st.error(f"❌ Error during missing value handling: {str(e)}")
        return
    
    # Outlier handling - only on numeric columns
    outlier_threshold = st.session_state.get('outlier_threshold', 1.5)
    for col in numeric_cols:
        if col in df.columns:  # Additional safety check
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (outlier_threshold * iqr)
            upper_bound = q3 + (outlier_threshold * iqr)
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    # Feature scaling - only on numeric columns
    scaling_method = st.session_state.get('scaling_method', "None")
    if scaling_method == "Standard":
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.session_state.preprocessor = scaler
    elif scaling_method == "MinMax":
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.session_state.preprocessor = scaler
    
    st.session_state.df = df
    st.session_state.preprocessing_done = True
def show_preprocessing_ui():
    if st.session_state.df is None:
        st.error("❌ Please load data first")
        return
    
    with st.expander("⚙️ Preprocessing Options", expanded=True):
        st.radio(
            "Handle Missing Values",
            ["Drop rows", "Drop columns", "Fill with mean", "Fill with median"],
            key='missing_values',
            on_change=update_preprocessing
        )
        
        st.slider(
            "Outlier Threshold (IQR multiplier)",
            0.0, 5.0, 1.5, 0.1,
            key='outlier_threshold',
            on_change=update_preprocessing
        )
        
        st.selectbox(
            "Feature Scaling",
            ["None", "Standard", "MinMax"],
            key='scaling_method',
            on_change=update_preprocessing
        )
    
    if st.session_state.get('preprocessing_done', False):
        st.success(f"✅ Preprocessing complete. Final shape: {st.session_state.df.shape}")
        
        if len(st.session_state.df.columns) > 0:
            col_to_plot = st.selectbox(
                "Select column to visualize distribution",
                st.session_state.df.columns,
                key='viz_column'
            )
            fig = px.histogram(st.session_state.df, x=col_to_plot, nbins=50, 
                             title=f"Distribution of {col_to_plot}")
            st.plotly_chart(fig)

def feature_engineering():
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        with st.expander("🔧 Feature Engineering Options"):
            # Convert financial abbreviations - with proper type checking
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        # Handle string conversion safely
                        df[col] = df[col].astype(str).str.replace(',', '').str.upper()
                        
                        # Process financial abbreviations
                        mask_k = df[col].str.contains('K', na=False)
                        mask_m = df[col].str.contains('M', na=False)
                        mask_b = df[col].str.contains('B', na=False)
                        mask_pct = df[col].str.contains('%', na=False)
                        
                        df.loc[mask_k, col] = df.loc[mask_k, col].str.replace('K', '').astype(float) * 1_000
                        df.loc[mask_m, col] = df.loc[mask_m, col].str.replace('M', '').astype(float) * 1_000_000
                        df.loc[mask_b, col] = df.loc[mask_b, col].str.replace('B', '').astype(float) * 1_000_000_000
                        df.loc[mask_pct, col] = df.loc[mask_pct, col].str.replace('%', '').astype(float) / 100
                        
                        # Convert to numeric where possible
                        try:
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        except:
                            pass
                    except Exception as e:
                        st.warning(f"Could not process column {col}: {str(e)}")
                        continue
            
            # Date features
            date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
            if date_cols:
                st.write("Date columns detected. Creating temporal features...")
                for col in date_cols:
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df = df.drop(columns=date_cols)
            
            # Interaction features
            if len(df.columns) > 1:
                num_cols = df.select_dtypes(include=[np.number]).columns
                if st.checkbox("Create interaction features"):
                    for i in range(len(num_cols)):
                        for j in range(i+1, len(num_cols)):
                            col1 = num_cols[i]
                            col2 = num_cols[j]
                            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2].replace(0, np.nan))
            
            # Polynomial features
            if st.checkbox("Create polynomial features (degree 2)"):
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                poly_features = poly.fit_transform(df[numeric_cols])
                poly_col_names = poly.get_feature_names_out(numeric_cols)
                df_poly = pd.DataFrame(poly_features, columns=poly_col_names)
                df = pd.concat([df, df_poly], axis=1)
        
        # Feature selection
        if st.session_state.feature_selection_method != "None":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if st.session_state.target_col and st.session_state.target_col in df.columns:
                y = df[st.session_state.target_col]
                X = df.drop(columns=[st.session_state.target_col])
            else:
                y = df.iloc[:, -1]
                X = df.iloc[:, :-1]
            
            X = X.select_dtypes(include=[np.number])
            
            if st.session_state.feature_selection_method == "SelectKBest":
                selector = SelectKBest(f_regression, k=st.session_state.n_features)
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()]
                df = pd.concat([pd.DataFrame(X_selected, columns=selected_features), y], axis=1)
                st.session_state.feature_selector = selector
                st.info(f"Selected features: {', '.join(selected_features)}")
            
            elif st.session_state.feature_selection_method == "PCA":
                pca = PCA(n_components=st.session_state.n_features)
                X_pca = pca.fit_transform(X)
                df = pd.concat([pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(st.session_state.n_features)]), y], axis=1)
                st.session_state.feature_selector = pca
                
                fig = px.bar(x=[f"PC{i+1}" for i in range(st.session_state.n_features)], 
                            y=pca.explained_variance_ratio_,
                            labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'},
                            title="PCA Explained Variance")
                st.plotly_chart(fig)
        
        st.session_state.df = df
        st.success(f"✅ Feature engineering complete. Final shape: {df.shape}")
        
        # Correlation matrix
        if len(df.columns) > 1:
            corr = df.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='Viridis'
            ))
            fig.update_layout(title="Feature Correlation Matrix")
            st.plotly_chart(fig)
    else:
        st.error("❌ Please load and preprocess data first")

def train_test_split_data():
    if st.session_state.df is not None:
        df = st.session_state.df
        
        if len(df.columns) < 2:
            st.error("❌ Not enough columns for train/test split")
            return
        
        if st.session_state.target_col and st.session_state.target_col in df.columns:
            y = df[st.session_state.target_col]
            X = df.drop(columns=[st.session_state.target_col])
        else:
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=st.session_state.test_size, random_state=st.session_state.random_state
        )
        
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        
        st.success(f"✅ Data split complete (Train: {len(X_train)}, Test: {len(X_test)})")
        
        fig = px.pie(values=[len(X_train), len(X_test)], 
                     names=["Train", "Test"], 
                     title="Train-Test Split",
                     hole=0.4)
        st.plotly_chart(fig)
        
        with st.expander("👀 View Training Data Sample"):
            st.dataframe(pd.concat([X_train, y_train], axis=1).head())
    else:
        st.error("❌ Please complete previous steps first")

def update_model_params():
    """Update model parameters when selections change"""
    if 'X_train' not in st.session_state:
        return
    
    model_type = st.session_state.get('selected_model', "Linear Regression")
    
    # Initialize model with current parameters
    if model_type == "Ridge Regression":
        st.session_state.model = Ridge(alpha=st.session_state.get('ridge_alpha', 1.0))
    elif model_type == "Lasso Regression":
        st.session_state.model = Lasso(alpha=st.session_state.get('lasso_alpha', 1.0))
    elif model_type == "Random Forest":
        st.session_state.model = RandomForestRegressor(
            n_estimators=st.session_state.get('rf_n_estimators', 100),
            max_depth=st.session_state.get('rf_max_depth', 10),
            random_state=st.session_state.random_state
        )
    elif model_type == "Gradient Boosting":
        st.session_state.model = GradientBoostingRegressor(
            n_estimators=st.session_state.get('gb_n_estimators', 100),
            learning_rate=st.session_state.get('gb_learning_rate', 0.1),
            random_state=st.session_state.random_state
        )
    elif model_type == "Support Vector Machine":
        st.session_state.model = SVR(
            kernel=st.session_state.get('svm_kernel', "rbf"),
            C=st.session_state.get('svm_c', 1.0)
        )
    else:  # Linear Regression
        st.session_state.model = LinearRegression()

def train_model_ui():
    """Display model training UI with persistent state"""
    if 'X_train' not in st.session_state:
        st.error("❌ Please split the data first")
        return
    
    model_options = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Support Vector Machine": SVR()
    }
    
    # Enhanced NaN handling with multiple safety checks
    with st.expander("🛠️ Missing Value Handling", expanded=True):
        nan_handling = st.selectbox(
            "Handle Missing Values Before Training",
            ["Drop rows with NaN", "Simple Imputer (mean)", "Simple Imputer (median)", "Iterative Imputer"],
            key='nan_handling'
        )
    
    st.selectbox(
        "Select Model",
        list(model_options.keys()),
        key='selected_model',
        on_change=update_model_params
    )
    
    with st.expander("⚙️ Hyperparameters", expanded=True):
        if st.session_state.selected_model == "Ridge Regression":
            st.slider(
                "Alpha (Regularization)", 
                0.01, 10.0, 1.0, 0.1,
                key='ridge_alpha',
                on_change=update_model_params
            )
        elif st.session_state.selected_model == "Lasso Regression":
            st.slider(
                "Alpha (Regularization)", 
                0.01, 10.0, 1.0, 0.1,
                key='lasso_alpha',
                on_change=update_model_params
            )
        elif st.session_state.selected_model == "Random Forest":
            st.slider(
                "Number of Trees", 
                10, 500, 100, 10,
                key='rf_n_estimators',
                on_change=update_model_params
            )
            st.slider(
                "Max Depth", 
                1, 50, 10, 1,
                key='rf_max_depth',
                on_change=update_model_params
            )
        elif st.session_state.selected_model == "Gradient Boosting":
            st.slider(
                "Number of Trees", 
                10, 500, 100, 10,
                key='gb_n_estimators',
                on_change=update_model_params
            )
            st.slider(
                "Learning Rate", 
                0.01, 1.0, 0.1, 0.01,
                key='gb_learning_rate',
                on_change=update_model_params
            )
        elif st.session_state.selected_model == "Support Vector Machine":
            st.selectbox(
                "Kernel", 
                ["linear", "poly", "rbf", "sigmoid"],
                key='svm_kernel',
                on_change=update_model_params
            )
            st.slider(
                "C (Regularization)", 
                0.1, 10.0, 1.0, 0.1,
                key='svm_c',
                on_change=update_model_params
            )
    
    if st.button("🚀 Train Model"):
        if hasattr(st.session_state, 'model'):
            with st.spinner(f"Training {st.session_state.selected_model}..."):
                try:
                    X_train = st.session_state.X_train.copy()
                    y_train = st.session_state.y_train.copy()
                    
                    # 1. Show raw NaN count before processing
                    initial_nans = X_train.isna().sum().sum()
                    if initial_nans > 0:
                        st.warning(f"⚠️ Found {initial_nans} NaN values in features before processing")
                    
                    # 2. Handle missing values
                    if nan_handling == "Drop rows with NaN":
                        mask = ~X_train.isna().any(axis=1)
                        rows_before = len(X_train)
                        X_train = X_train[mask]
                        y_train = y_train[mask]
                        rows_after = len(X_train)
                        st.info(f"📉 Dropped {rows_before - rows_after} rows ({rows_after} remaining)")
                        
                    elif "Simple Imputer" in nan_handling:
                        strategy = 'mean' if "mean" in nan_handling else 'median'
                        imputer = SimpleImputer(strategy=strategy)
                        X_train = pd.DataFrame(imputer.fit_transform(X_train), 
                                            columns=X_train.columns)
                        st.info(f"🔄 Imputed {initial_nans} NaN values using {strategy}")
                    
                    elif nan_handling == "Iterative Imputer":
                        imputer = IterativeImputer(max_iter=10, random_state=st.session_state.random_state)
                        X_train = pd.DataFrame(imputer.fit_transform(X_train), 
                                            columns=X_train.columns)
                        st.info(f"🔄 Iteratively imputed {initial_nans} NaN values")
                    
                    # 3. Final validation
                    remaining_nans = X_train.isna().sum().sum()
                    if remaining_nans > 0:
                        st.error(f"❌ Critical: {remaining_nans} NaN values remain after processing!")
                        st.session_state.model_trained = False
                        return
                    
                    if len(X_train) == 0:
                        st.error("❌ No training data remains after processing!")
                        st.session_state.model_trained = False
                        return
                        
                    # 4. ACTUAL MODEL TRAINING
                    model = st.session_state.model
                    with st.spinner("Finalizing training..."):
                        model.fit(X_train, y_train)
                    
                    # 5. Only mark as trained if we get here
                    st.session_state.models[st.session_state.selected_model] = model
                    st.session_state.current_model = st.session_state.selected_model
                    st.session_state.model_trained = True  # Critical flag for evaluation
                    
                    st.success(f"✅ {st.session_state.selected_model} trained successfully on {len(X_train)} samples!")
                    
                    # Feature importance/coefficients visualization
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        features = X_train.columns
                        st.session_state.feature_importance = pd.DataFrame({
                            'Feature': features,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False)
                        
                        st.subheader("📊 Feature Importance")
                        fig = px.bar(st.session_state.feature_importance.head(20),
                                    x='Feature',
                                    y='Importance',
                                    title="Top 20 Important Features")
                        st.plotly_chart(fig)

                    elif hasattr(model, 'coef_'):
                        coef = model.coef_
                        features = X_train.columns
                        st.session_state.feature_importance = pd.DataFrame({
                            'Feature': features,
                            'Coefficient': coef
                        }).sort_values('Coefficient', ascending=False)
                        
                        st.subheader("📊 Model Coefficients")
                        fig = px.bar(st.session_state.feature_importance,
                                    x='Feature',
                                    y='Coefficient',
                                    title="Feature Coefficients")
                        st.plotly_chart(fig)

                    # Store basic training metrics
                    train_score = model.score(X_train, y_train)
                    st.session_state.metrics[st.session_state.selected_model] = {
                        'Training Score': train_score,
                        'Features Used': len(X_train.columns),
                        'Samples Used': len(X_train)
                    }

                    # Show quick performance summary
                    st.subheader("🚀 Training Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Training R² Score", f"{train_score:.4f}")
                    col2.metric("Features Used", len(X_train.columns))
                    col3.metric("Training Samples", len(X_train))

                    # Enable evaluation and prediction tabs
                    st.session_state.model_trained = True
                    st.session_state.prediction_active = True
                    st.session_state.evaluation_active = True

                    # Show quick actions
                    st.markdown("### Next Steps")
                    if st.button("📉 Evaluate Model Now"):
                        evaluate_model()
                    if st.button("🔮 Make Predictions Now"):
                        prediction_interface()
                    
                except Exception as e:
                    st.session_state.model_trained = False
                    st.error(f"""
                    ❌ Training failed: {str(e)}
                    
                    Common fixes:
                    1. Try different NaN handling method
                    2. Check for infinite values with st.session_state.df.isin([np.inf, -np.inf]).sum()
                    3. Reduce model complexity
                    4. Check data types with st.session_state.df.info()
                    """)
        else:
            st.error("❌ No model initialized. Please select a model type first.")
def evaluate_model():
    if not st.session_state.get('current_model') or st.session_state.current_model not in st.session_state.models:
        st.error("❌ Please train a model first")
        return
    
    model = st.session_state.models[st.session_state.current_model]
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    try:
        y_pred = model.predict(X_test)
        st.session_state.y_pred = y_pred
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_name = st.session_state.current_model
        st.session_state.metrics[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        st.success(f"✅ {model_name} Evaluation Complete")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Squared Error", f"{mse:.4f}")
        col2.metric("Root Mean Squared Error", f"{rmse:.4f}")
        col3.metric("Mean Absolute Error", f"{mae:.4f}")
        col4.metric("R² Score", f"{r2:.4f}")
        
        fig1 = px.scatter(x=y_test, y=y_pred,
                         labels={'x': 'Actual', 'y': 'Predicted'},
                         title=f"{model_name} - Actual vs Predicted Values",
                         trendline="ols")
        fig1.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                      x1=y_test.max(), y1=y_test.max(),
                      line=dict(color="Red", width=2, dash="dot"))
        st.plotly_chart(fig1)
        
        residuals = y_test - y_pred
        fig2 = px.scatter(x=y_pred, y=residuals,
                         labels={'x': 'Predicted', 'y': 'Residuals'},
                         title=f"{model_name} - Residual Plot")
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig2)
        
        fig3 = px.histogram(residuals, nbins=50,
                           title=f"{model_name} - Distribution of Residuals",
                           labels={'value': 'Residuals'})
        st.plotly_chart(fig3)
        
        if len(st.session_state.models) > 1:
            st.markdown("### 📊 Model Comparison")
            comparison_df = pd.DataFrame.from_dict(st.session_state.metrics, orient='index')
            st.dataframe(comparison_df.style.format("{:.4f}").highlight_min(axis=0, color='#4CAF50'))
            
    except Exception as e:
        st.error(f"❌ Error evaluating model: {str(e)}")

def prediction_interface():
    if not st.session_state.get('current_model') or st.session_state.current_model not in st.session_state.models:
        st.error("❌ Please train a model first")
        return
    
    model = st.session_state.models[st.session_state.current_model]
    feature_names = st.session_state.X_train.columns
    
    st.header(f"🔮 {st.session_state.current_model} Predictions")
    
    with st.expander("✍️ Manual Input", expanded=True):
        input_data = {}
        cols = st.columns(3)
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                input_key = f"pred_input_{feature}"
                if input_key not in st.session_state:
                    st.session_state[input_key] = 0.0
                
                input_data[feature] = st.number_input(
                    feature, 
                    value=st.session_state[input_key],
                    key=input_key
                )
        
        if st.button("Predict", key="manual_predict_btn"):
            try:
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)
                st.success(f"Predicted Value: {prediction[0]:.4f}")
                st.session_state.last_prediction = {
                    'model': st.session_state.current_model,
                    'input': input_data,
                    'output': prediction[0]
                }
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
    
    with st.expander("📁 Batch Prediction (CSV)", expanded=True):
        pred_file = st.file_uploader("Upload CSV for prediction", 
                                    type=["csv"],
                                    key="batch_pred_uploader")
        
        if pred_file is not None:
            try:
                pred_df = pd.read_csv(pred_file)
                missing_features = set(feature_names) - set(pred_df.columns)
                
                if not missing_features:
                    predictions = model.predict(pred_df[feature_names])
                    pred_df['Prediction'] = predictions
                    st.success(f"✅ {len(predictions)} predictions generated using {st.session_state.current_model}")
                    
                    st.session_state.last_batch_prediction = {
                        'model': st.session_state.current_model,
                        'data': pred_df,
                        'timestamp': datetime.now()
                    }
                    
                    st.dataframe(pred_df)
                    
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name=f"predictions_{st.session_state.current_model.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="pred_download_btn"
                    )
                else:
                    st.error(f"❌ Missing required features: {', '.join(missing_features)}")
            except Exception as e:
                st.error(f"❌ Batch prediction failed: {str(e)}")
    
    if 'last_prediction' in st.session_state and st.session_state.last_prediction['model'] == st.session_state.current_model:
        with st.expander("⏱️ Last Prediction", expanded=False):
            st.json(st.session_state.last_prediction)

def model_management():
    st.header("💾 Model Management")
    
    if st.session_state.models:
        selected_model = st.selectbox("Select Model to Manage", list(st.session_state.models.keys()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_filename = st.text_input("Model filename", f"{selected_model.lower().replace(' ', '_')}.pkl")
            if st.button("💾 Save Model"):
                model = st.session_state.models[selected_model]
                joblib.dump(model, model_filename)
                st.success(f"Model saved as {model_filename}")
        
        with col2:
            uploaded_model = st.file_uploader("Upload Model (.pkl)", type=["pkl"])
            if uploaded_model is not None:
                try:
                    model = joblib.load(uploaded_model)
                    model_name = uploaded_model.name.replace(".pkl", "").replace("_", " ").title()
                    st.session_state.models[model_name] = model
                    st.success(f"Model '{model_name}' loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    
    if st.session_state.df is not None:
        st.markdown("---")
        st.header("📥 Download Processed Data")
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )

# ======================
# 🎮 MAIN APP
# ======================

def main():
    # Theme selection
    st.sidebar.title("🌐 SYSTEM THEMES")
    theme = st.sidebar.radio("Select Interface Skin:", 
                           ["Cyberpunk 2077", "The Matrix"],
                           index=0)
    apply_theme(theme)
    
    # Initialize session state
    init_session_state()
    
    # Sidebar configuration
    with st.sidebar.expander("⚙️ Global Settings", expanded=False):
        st.session_state.test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        st.session_state.random_state = st.number_input("Random State", 0, 100, 42)
        st.session_state.target_col = st.text_input("Target Column (leave blank for last column)", "")
        
        st.session_state.feature_selection_method = st.selectbox(
            "Feature Selection Method",
            ["None", "SelectKBest", "PCA"],
            index=0
        )
        
        if st.session_state.feature_selection_method != "None":
            st.session_state.n_features = st.number_input("Number of Features", 1, 20, 5)
    
    # Data source selection
    with st.sidebar.expander("📂 Data Configuration", expanded=False):
        st.session_state.data_source = st.radio("Data Source:", 
                                              ["Upload File", "Kaggle Dataset", "Yahoo Finance"],
                                              index=0)
        
        if st.session_state.data_source == "Upload File":
            st.session_state.file_uploader = st.file_uploader("Upload CSV/Excel", 
                                                            type=["csv", "xlsx"])
        elif st.session_state.data_source == "Kaggle Dataset":
            st.session_state.kaggle_user = st.text_input("Kaggle Username")
            st.session_state.kaggle_key = st.text_input("Kaggle API Key", type="password")
            st.session_state.kaggle_dataset = st.text_input("Dataset URL (format: owner/dataset)", 
                                                          placeholder="username/dataset-name")
        elif st.session_state.data_source == "Yahoo Finance":
            st.session_state.yahoo_ticker = st.text_input("Ticker Symbol", value="AAPL")
            st.session_state.yahoo_start = st.date_input("Start Date", 
                                                       value=pd.to_datetime('2020-01-01'))
            st.session_state.yahoo_end = st.date_input("End Date", 
                                                     value=pd.to_datetime('today'))
            st.session_state.yahoo_interval = st.selectbox("Interval", 
                                                         ["1d", "1wk", "1mo"], 
                                                         index=0)
    
  # ======================
# 🎮 MAIN WORKFLOW
# ======================

def main_workflow():
    """Main workflow steps with enhanced UI and functionality"""
    
    # Step 1: Load Data
    if st.sidebar.button("1️⃣ Load & Explore Data"):
        with st.spinner("🚀 Launching data acquisition..."):
            load_data()
    
    # Step 2: Preprocess Data
    if st.sidebar.button("2️⃣ Preprocess Data"):
        st.session_state.preprocessing_active = True
    
    if st.session_state.get('preprocessing_active', False):
        show_preprocessing_ui()
    
    # Step 3: Feature Engineering
    if st.sidebar.button("3️⃣ Feature Engineering"):
        with st.spinner("⚙️ Engineering features..."):
            feature_engineering()
    
    # Step 4: Train/Test Split
    if st.sidebar.button("4️⃣ Train/Test Split"):
        with st.spinner("✂️ Splitting data..."):
            train_test_split_data()
    
    # Step 5: Model Training
    if st.sidebar.button("5️⃣ Train Model"):
        st.session_state.model_training_active = True
    
    if st.session_state.get('model_training_active', False):
        train_model_ui()
    
    # Step 6: Model Evaluation
    if st.sidebar.button("6️⃣ Evaluate Model"):
        with st.spinner("📊 Evaluating model performance..."):
            evaluate_model()
    
    # Step 7: Predictions
    if st.sidebar.button("7️⃣ Make Predictions"):
        st.session_state.prediction_active = True
    
    if st.session_state.get('prediction_active', False):
        prediction_interface()
    
    # Step 8: Model Management
    if st.sidebar.button("8️⃣ Model Management"):
        with st.spinner("💾 Managing model artifacts..."):
            model_management()

# ======================
# 🎮 MAIN APP EXECUTION
# ======================

def main():
    # Theme selection
    st.sidebar.title("🌐 SYSTEM THEMES")
    theme = st.sidebar.radio("Select Interface Skin:", 
                           ["Cyberpunk 2077", "The Matrix"],
                           index=0)
    apply_theme(theme)
    
    # Initialize session state
    init_session_state()
    
    # Sidebar configuration
    with st.sidebar.expander("⚙️ Global Settings", expanded=False):
        st.session_state.test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        st.session_state.random_state = st.number_input("Random State", 0, 100, 42)
        st.session_state.target_col = st.text_input("Target Column (leave blank for last column)", "")
        
        st.session_state.feature_selection_method = st.selectbox(
            "Feature Selection Method",
            ["None", "SelectKBest", "PCA"],
            index=0
        )
        
        if st.session_state.feature_selection_method != "None":
            st.session_state.n_features = st.number_input("Number of Features", 1, 20, 5)
    
    # Data source selection
    with st.sidebar.expander("📂 Data Configuration", expanded=False):
        st.session_state.data_source = st.radio("Data Source:", 
                                              ["Upload File", "Kaggle Dataset", "Yahoo Finance"],
                                              index=0)
        
        if st.session_state.data_source == "Upload File":
            st.session_state.file_uploader = st.file_uploader("Upload CSV/Excel", 
                                                            type=["csv", "xlsx"])
        elif st.session_state.data_source == "Kaggle Dataset":
            st.session_state.kaggle_user = st.text_input("Kaggle Username")
            st.session_state.kaggle_key = st.text_input("Kaggle API Key", type="password")
            st.session_state.kaggle_dataset = st.text_input("Dataset URL (format: owner/dataset)", 
                                                          placeholder="username/dataset-name")
        elif st.session_state.data_source == "Yahoo Finance":
            st.session_state.yahoo_ticker = st.text_input("Ticker Symbol", value="AAPL")
            st.session_state.yahoo_start = st.date_input("Start Date", 
                                                       value=pd.to_datetime('2020-01-01'))
            st.session_state.yahoo_end = st.date_input("End Date", 
                                                     value=pd.to_datetime('today'))
            st.session_state.yahoo_interval = st.selectbox("Interval", 
                                                         ["1d", "1wk", "1mo"], 
                                                         index=0)
    
    # Execute main workflow
    main_workflow()
    
    # Help/About section
    with st.sidebar.expander("ℹ️ Help & About", expanded=False):
        st.markdown("""
        **Advanced Financial ML Dashboard**
        
        This application provides a comprehensive platform for financial data analysis and predictive modeling.
        
        **Workflow:**
        1. Upload your financial dataset
        2. Preprocess and clean the data
        3. Engineer new features
        4. Split into train/test sets
        5. Train machine learning models
        6. Evaluate model performance
        7. Make predictions
        8. Manage and save models
        
        **Features:**
        - Multiple regression algorithms
        - Hyperparameter tuning
        - Feature selection
        - Comprehensive visualizations
        - Model persistence
        
        **Developers:**
        """, unsafe_allow_html=True)
        
        # Developer 1
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Abdul Hadi Cheema**  
            [![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?logo=github)](https://github.com/AHC62)  
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/abdul-hadi-cheema-238562220/)
            """, unsafe_allow_html=True)
        
        # Developer 2
        with col2:
            st.markdown("""
            **Hafiz Muhammad Danish**  
            [![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?logo=github)](https://github.com/muhdanish119)  
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/muhammad-danish-b47478293/)
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        

if __name__ == "__main__":
    main()