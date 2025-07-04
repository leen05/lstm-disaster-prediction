import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage
from fpdf import FPDF
from io import StringIO
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, confusion_matrix
import logging
import os # <--- ADDED THIS LINE

warnings.filterwarnings('ignore')

# ==============================================
# LOGGING CONFIGURATION
# ==============================================
log_stream = StringIO()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=log_stream)

# ==============================================
# PAGE CONFIGURATION
# ==============================================
st.set_page_config(
    page_title="⚡️ Critical Event Forecasting",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================
# SESSION STATE INITIALIZATION
# ==============================================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'alert_sent' not in st.session_state:
    st.session_state.alert_sent = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'raw_predictions' not in st.session_state:
    st.session_state.raw_predictions = None
if 'applied_threshold' not in st.session_state:
    st.session_state.applied_threshold = 100.0
if 'data_stats' not in st.session_state:
    st.session_state.data_stats = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'train_test_info' not in st.session_state:
    st.session_state.train_test_info = None
if 'last_retrain_date' not in st.session_state:
    st.session_state.last_retrain_date = None
if 'model_version' not in st.session_state:
    st.session_state.model_version = "1.0.0"
if 'alert_history' not in st.session_state: # NEW: To store alert history
    st.session_state.alert_history = []
if 'default_email_subject' not in st.session_state: # NEW: Customizable email subject
    st.session_state.default_email_subject = "🚨 Critical Event Alert: Threshold Exceeded!"
if 'default_email_body' not in st.session_state: # NEW: Customizable email body
    st.session_state.default_email_body = """
Dear User,

This is an automated alert from your AI-Powered Critical Event Forecasting System.
We detected that {alert_count} predictions for the next {future_steps} days are forecasted to exceed the set threshold of {threshold:.2f}.

--- Alert Details ---
{alert_summary_table}

This is a critical notification requiring attention.
Please log in to the system for more detailed analysis.

Best regards,
Your AI Prediction Team
"""
if 'alert_levels' not in st.session_state: # NEW: Multiple alert levels
    st.session_state.alert_levels = [
        {"level": "Warning", "threshold": 100.0, "color": "orange", "icon": "⚠️"},
        {"level": "Critical", "threshold": 120.0, "color": "red", "icon": "🚨"},
        {"level": "Emergency", "threshold": 150.0, "color": "darkred", "icon": "🆘"},
    ]

# ==============================================
# ENHANCED HELPER FUNCTIONS
# ==============================================
def preprocess_data(df, date_col_name, value_col_name, missing_strategy, outlier_strategy):
    """Clean and prepare the dataset with robust error handling and options"""
    df = df.copy()
    initial_missing_count = df.isnull().sum().sum()
    
    # Data Validation Feedback
    if date_col_name not in df.columns:
        st.error(f"Error: Date column '{date_col_name}' not found in the uploaded file.")
        return None
    if value_col_name not in df.columns:
        st.error(f"Error: Value column '{value_col_name}' not found in the uploaded file.")
        return None

    try:
        # Date column processing
        df['Date'] = pd.to_datetime(df[date_col_name], errors='coerce')
        df = df.dropna(subset=['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index() # Ensure chronological order
        logging.info("Date column processed and set as index.")
        
        # Rename value column
        df.rename(columns={value_col_name: 'Value'}, inplace=True)
        logging.info(f"Value column identified as '{value_col_name}'.")
        
        # Missing Value Handling
        if df['Value'].isnull().sum() > 0:
            st.warning(f"Detected {df['Value'].isnull().sum()} missing values in 'Value' column.")
            if missing_strategy == "Interpolate":
                df['Value'] = df['Value'].interpolate(method='linear')
                logging.info(f"Interpolated {initial_missing_count} missing values.")
            elif missing_strategy == "Drop Rows":
                df.dropna(subset=['Value'], inplace=True)
                logging.info(f"Dropped rows with {initial_missing_count} missing values.")
        else:
            logging.info("No missing values detected in 'Value' column.")

        # Outlier Treatment (Basic IQR method)
        if outlier_strategy != "None":
            Q1 = df['Value'].quantile(0.25)
            Q3 = df['Value'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_count = df[(df['Value'] < lower_bound) | (df['Value'] > upper_bound)].shape[0]
            if outliers_count > 0:
                st.info(f"Detected {outliers_count} potential outliers based on IQR method.")
                if outlier_strategy == "Cap (IQR)":
                    df['Value'] = np.where(df['Value'] < lower_bound, lower_bound, df['Value'])
                    df['Value'] = np.where(df['Value'] > upper_bound, upper_bound, df['Value'])
                    logging.info(f"Capped {outliers_count} outliers.")
                elif outlier_strategy == "Remove (IQR)":
                    df = df[(df['Value'] >= lower_bound) & (df['Value'] <= upper_bound)]
                    logging.info(f"Removed {outliers_count} outliers.")
            else:
                logging.info("No significant outliers detected.")

        return df
    except Exception as e:
        st.error(f"Data processing error: {str(e)}. Please check your data format and selected columns.")
        logging.error(f"Data preprocessing failed: {e}")
        return None

def create_sequences(data, seq_length):
    """Create time-series sequences for LSTM with validation"""
    try:
        X, y = [], []
        for i in range(len(data)-seq_length-1):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        logging.info(f"Created {len(X)} sequences with length {seq_length}.")
        return np.array(X), np.array(y)
    except Exception as e:
        st.error(f"Sequence creation error: {str(e)}")
        logging.error(f"Sequence creation failed: {e}")
        return None, None

# ==============================================
# Model Evaluation
# ==============================================
def evaluate_model(X_test, y_test, model, scaler, threshold):
    try:
        predictions_scaled = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions_scaled)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / (actual + 1e-8))) * 100
        
        st.subheader("📊 Model Evaluation (Regression Metrics)")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
        logging.info(f"Model evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%.")

        eval_fig = go.Figure()
        eval_fig.add_trace(go.Scatter(y=actual.flatten(), mode='lines', name='Actual'))
        eval_fig.add_trace(go.Scatter(y=predictions.flatten(), mode='lines', name='Predicted'))
        eval_fig.update_layout(title="Actual vs Predicted (Evaluation Set)", xaxis_title="Time Step", yaxis_title="Value")
        st.plotly_chart(eval_fig, use_container_width=True)

        st.subheader("🚨 Alerting System Evaluation (Classification Metrics)")
        y_true_alerts = (actual > threshold).astype(int)
        y_pred_alerts = (predictions > threshold).astype(int)

        precision = precision_score(y_true_alerts, y_pred_alerts)
        recall = recall_score(y_true_alerts, y_pred_alerts)
        f1 = f1_score(y_true_alerts, y_pred_alerts)
        
        st.write(f"**Threshold for Alerting:** {threshold:.2f}")
        st.write(f"**Precision (True Alerts / All Predicted Alerts):** {precision:.2f}")
        st.write(f"**Recall (True Alerts / All Actual Alerts):** {recall:.2f}")
        st.write(f"**F1-Score:** {f1:.2f}")
        logging.info(f"Alert system evaluation: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}.")

    except Exception as e:
        st.error(f"Evaluation failed: {str(e)}")
        logging.error(f"Model evaluation failed: {e}")

def threshold_analysis_dashboard(df, threshold):
    """Enhanced threshold analysis with visualizations"""
    try:
        with st.expander("📊 Threshold Analysis Dashboard", expanded=False):
            total_points = len(df)
            exceedances = (df['Value'] > threshold).sum()
            safe_points = total_points - exceedances
            mean_val = df['Value'].mean()
            std_val = df['Value'].std()
            suggested_threshold = round(mean_val + 1.5 * std_val, 2)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Data Points", total_points)
                st.metric("Current Threshold", f"{threshold:.2f}")
                
            with col2:
                st.metric("Exceedances", 
                         f"{exceedances} ({exceedances/total_points*100:.1f}%)",
                         delta_color="inverse")
                
            with col3:
                st.metric("Safe Values", 
                         f"{safe_points} ({safe_points/total_points*100:.1f}%)")
                st.metric("Suggested Threshold", f"{suggested_threshold:.2f}")
            
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                fig_pie = px.pie(
                    names=['Exceedances', 'Safe Values'],
                    values=[exceedances, safe_points],
                    color_discrete_sequence=['red', 'green'],
                    title="Threshold Distribution",
                    hole=0.3
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_viz2:
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Value'],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='royalblue')
                ))
                fig_line.add_hline(
                    y=threshold,
                    line=dict(color='red', dash='dash'),
                    annotation_text=f"Disaster Risk Threshold ({threshold})",
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color="red"
                )
                fig_line.add_hline(
                    y=suggested_threshold,
                    line=dict(color='orange', dash='dot'),
                    annotation_text=f"Suggested Threshold ({suggested_threshold})"
                )
                fig_line.update_layout(
                    title="Data with Thresholds",
                    height=300
                )
                st.plotly_chart(fig_line, use_container_width=True)
            
            auto_adjust = st.checkbox(
                "🔄 Auto-adjust to suggested threshold",
                value=False,
                help=f"Will use {suggested_threshold} instead of {threshold}"
            )
            if auto_adjust:
                st.session_state['applied_threshold'] = suggested_threshold
                st.success(f"Threshold updated to {suggested_threshold}!")
                logging.info(f"Threshold auto-adjusted to {suggested_threshold}.")
            else:
                st.session_state['applied_threshold'] = threshold
                logging.info(f"Threshold set to {threshold}.")
                
    except Exception as e:
        st.error(f"Threshold analysis failed: {str(e)}")
        logging.error(f"Threshold analysis failed: {e}")

def create_download_button(data, filename, button_text=None):
    """Create a styled download button"""
    csv = data.to_csv(index=False)
    st.download_button(
        label=button_text or f"⬇️ Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv",
        help=f"Export {filename} for further analysis"
    )

def send_email_alert(recipient_email, subject, body):
    """Sends an email alert using SMTP server configured via Streamlit Secrets."""
    sender_email = st.secrets.get("email_sender") 
    sender_password = st.secrets.get("email_password")
    smtp_server = st.secrets.get("smtp_server", "smtp.gmail.com")
    smtp_port = st.secrets.get("smtp_port", 465)

    if not recipient_email or recipient_email == "your-email@example.com":
        st.warning("Please configure a valid recipient email in the sidebar to send alerts.")
        logging.warning("Email alert skipped: No valid recipient email configured.")
        return False
    
    if not sender_email or not sender_password:
        st.error("Email alert not configured. Please set `email_sender` and `email_password` in your `.streamlit/secrets.toml` file.")
        st.info("For Gmail, you might need to generate an 'App password'.")
        logging.error("Email alert skipped: Sender credentials missing in Streamlit Secrets.")
        return False

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        st.success(f"Email alert sent to {recipient_email}!")
        logging.info(f"Email alert successfully sent to {recipient_email}.")
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}. Check email settings and app password in `.streamlit/secrets.toml`.")
        logging.error(f"Failed to send email to {recipient_email}: {e}")
        return False

# ==============================================
# SIDEBAR CONFIGURATION
# ==============================================
with st.sidebar:
    st.title("⚙️ System Configuration")
    
    # Model parameters
    st.subheader("Model Parameters")
    seq_length = st.slider("Sequence Length", 5, 60, 30, 
                          help="Number of historical time steps to consider")
    epochs = st.slider("Training Epochs", 10, 500, 100)
    lstm_units = st.slider("LSTM Units", 16, 256, 64)
    dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.05)
    
    # Alert system with dynamic threshold
    st.subheader("Alert Thresholds")
    
    # NEW: Multiple Alert Levels
    st.markdown("Define your alert levels (e.g., Warning, Critical):")
    edited_alert_levels = st.data_editor(
        st.session_state.alert_levels,
        column_config={
            "level": st.column_config.TextColumn("Level", help="e.g., Warning, Critical", required=True),
            "threshold": st.column_config.NumberColumn("Threshold", help="Value at which this alert level triggers", required=True, format="%.2f"),
            "color": st.column_config.TextColumn("Color", help="Visual color for this alert level (e.g., 'red', '#FF0000'). Requires Streamlit >= 1.27.0 for ColorColumn."), # Changed to TextColumn
            "icon": st.column_config.TextColumn("Icon", help="Emoji icon for this level (e.g., ⚠️, 🚨, 🆘)"),
        },
        num_rows="dynamic",
        hide_index=True,
        key="alert_levels_editor"
    )
    # Sort alert levels by threshold in ascending order for proper logic
    st.session_state.alert_levels = sorted(edited_alert_levels, key=lambda x: x["threshold"])

    # Use the lowest threshold as the primary one for general display/evaluation if needed
    primary_disaster_threshold = st.session_state.alert_levels[0]["threshold"] if st.session_state.alert_levels else 100.0
    st.info(f"Primary alert threshold set to: **{primary_disaster_threshold:.2f}** (from '{st.session_state.alert_levels[0]['level']}' level)")
    st.session_state.applied_threshold = primary_disaster_threshold # Ensure applied_threshold reflects the lowest level

    notification_email = st.text_input("Notification Email", 
                                     "your-email@example.com",
                                     help="Email address to send alerts to.")
    enable_email_alerts = st.checkbox("Enable Email Alerts", False, help="Check to receive email notifications for alerts.")
    # NEW: Customizable Email Content
    with st.expander("✉️ Customize Email Alerts", expanded=False):
        st.session_state.default_email_subject = st.text_input(
            "Email Subject Template", 
            st.session_state.default_email_subject,
            help="Use {alert_count}, {future_steps}, {threshold} placeholders."
        )
        st.session_state.default_email_body = st.text_area(
            "Email Body Template", 
            st.session_state.default_email_body, 
            height=300,
            help="Use {alert_count}, {future_steps}, {threshold}, {alert_summary_table} placeholders."
        )
    
    # SMS Integration Placeholder
    st.subheader("SMS Alerts (Advanced)")
    st.info("SMS integration (e.g., via Twilio) would require an external API setup and credentials.")
    # enable_sms_alerts = st.checkbox("Enable SMS Alerts", False) # Uncomment if you implement SMS
    
    enable_sound = st.checkbox("Enable Alert Sounds", True, help="Visual indicator only. Actual sound requires custom HTML/JS.")
    
    # Model management
    st.subheader("Model Operations")
    if st.button("💾 Save Model", help="Save current model to disk"):
        if st.session_state.model:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"disaster_model_{timestamp}.h5"
                st.session_state.model.save(filename)
                st.success(f"Model saved as {filename}!")
                logging.info(f"Model saved as {filename}.")
            except Exception as e:
                st.error(f"Model save failed: {str(e)}")
                logging.error(f"Model save failed: {e}")
        else:
            st.warning("No trained model available")
            logging.warning("Save Model: No model available to save.")
    
    uploaded_model = st.file_uploader("Load Model", type=['h5'])
    if uploaded_model:
        try:
            st.session_state.model = load_model(uploaded_model)
            st.success("Model loaded successfully!")
            logging.info(f"Model loaded from {uploaded_model.name}.")
            st.session_state.last_retrain_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            logging.error(f"Model loading failed for {uploaded_model.name}: {e}")

# ==============================================
# MAIN APPLICATION INTERFACE
# ==============================================
st.title(" ⚡️ AI-Powered Critical Event Forecasting System")
st.markdown("""
    ### Predict future values and get early warnings for critical thresholds.
    Upload your time-series data to generate predictions and analyze threshold behavior.
""")

# File upload section
uploaded_file = st.file_uploader("📤 Upload Sensor Data (CSV/Excel)", 
                                type=['csv', 'xlsx'],
                                help="Should contain at least one date/timestamp column and one numeric value column.")

if uploaded_file:
    logging.info(f"File '{uploaded_file.name}' uploaded.")
    
    # Read the uploaded file to infer columns
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        
        st.write("Preview of uploaded data:")
        st.dataframe(df_raw.head())

        # NEW: Column Selection Flexibility
        all_columns = df_raw.columns.tolist()
        col1, col2 = st.columns(2)
        with col1:
            selected_date_col = st.selectbox("Select Date/Timestamp Column", all_columns, help="Choose the column containing dates or timestamps.")
        with col2:
            numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.warning("No numeric columns detected in your data for 'Value'. Please ensure your data contains numeric values.")
                st.stop() # Stop execution if no numeric columns
            selected_value_col = st.selectbox("Select Value Column", numeric_cols, help="Choose the column containing the numeric values to predict.")

        # NEW: Missing Value and Outlier Handling Options
        st.subheader("Data Preprocessing Options")
        col3, col4 = st.columns(2)
        with col3:
            missing_strategy = st.radio(
                "Missing Value Strategy",
                ["Interpolate", "Drop Rows"],
                help="How to handle missing values in the selected Value column."
            )
        with col4:
            outlier_strategy = st.radio(
                "Outlier Treatment Strategy",
                ["None", "Cap (IQR)", "Remove (IQR)"],
                help="How to handle outliers (values far from the norm) using Interquartile Range (IQR) method."
            )

        if st.button("✨ Process Data & Visualize"):
            with st.spinner("Processing data..."):
                df_processed = preprocess_data(df_raw, selected_date_col, selected_value_col, missing_strategy, outlier_strategy)
                st.session_state.df_processed = df_processed
                
                if df_processed is not None:
                    st.success("Data processed successfully!")
                    st.session_state.data_stats = {
                        'min': df_processed['Value'].min(),
                        'max': df_processed['Value'].max(),
                        'mean': df_processed['Value'].mean(),
                        'std': df_processed['Value'].std(),
                        'initial_missing_values': df_raw.isnull().sum().sum(),
                        'processed_missing_values': df_processed.isnull().sum().sum()
                    }
                    logging.info("Data processed and statistics generated.")
                    
                    # Data Insights Section
                    st.subheader("📊 Data Insights")
                    with st.expander("Explore your dataset", expanded=False):
                        st.write("**Dataset Head (Processed):**")
                        st.dataframe(df_processed.head())
                        
                        st.write("**Descriptive Statistics (Processed Value Column):**")
                        st.dataframe(df_processed['Value'].describe())

                        col_data_viz1, col_data_viz2 = st.columns(2)
                        with col_data_viz1:
                            fig_box = px.box(df_processed, y="Value", title="Value Distribution (Outliers)")
                            st.plotly_chart(fig_box, use_container_width=True)

                        with col_data_viz2:
                            st.write("#### Missing Data Handling Summary")
                            if st.session_state.data_stats['initial_missing_values'] > 0:
                                st.info(f"Initial missing values detected: {st.session_state.data_stats['initial_missing_values']}")
                                st.success(f"Missing values after '{missing_strategy}': {st.session_state.data_stats['processed_missing_values']}")
                            else:
                                st.info("No missing values detected in the uploaded data.")
                            
                            st.write("#### Outlier Treatment Summary")
                            if outlier_strategy != "None":
                                st.info(f"Outliers treated using '{outlier_strategy}' strategy.")
                            else:
                                st.info("No outlier treatment applied.")


                    # Threshold analysis dashboard
                    threshold_analysis_dashboard(df_processed, st.session_state.applied_threshold)
                    
                    # Data visualization (Historical Data Trend)
                    st.subheader("📈 Data Overview")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_processed.index,
                        y=df_processed['Value'],
                        mode='lines+markers',
                        name='Historical Data',
                        line=dict(color='royalblue', width=2)
                    ))
                    fig.add_hline(
                        y=st.session_state['applied_threshold'],
                        line=dict(color='red', dash='dash', width=2),
                        annotation_text=f"Applied Threshold ({st.session_state['applied_threshold']})"
                    )
                    fig.update_layout(
                        title=f"Historical Data Trend of {selected_value_col}",
                        xaxis_title="Date",
                        yaxis_title=f"{selected_value_col} (e.g., Sensor Reading, Disaster Index)", # Dynamic Y-axis label
                        height=400,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Data processing failed. Please check your file and selections.")
        else:
            st.info("Select your columns and preprocessing options, then click 'Process Data & Visualize'.")

    except Exception as e:
        st.error(f"Failed to load the uploaded file: {str(e)}. Please ensure it's a valid CSV/Excel.")
        logging.error(f"File upload failed for {uploaded_file.name}: {e}")

# Only show model training and prediction if data is processed
if st.session_state.get('df_processed') is not None:
    # Model training section
    st.subheader("🤖 Model Training")
    if st.button("🚀 Train Prediction Model", 
                help="Train a new LSTM model with current parameters"):
        with st.spinner("Training AI model. This may take several minutes..."):
            try:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(st.session_state.df_processed[['Value']])
                st.session_state.scaler = scaler
                
                X, y = create_sequences(scaled_data.flatten(), seq_length)
                if X is not None and y is not None:
                    X = X.reshape((X.shape[0], X.shape[1], 1))
                    
                    split = int(0.8 * len(X))
                    X_train, X_test = X[:split], X[split:]
                    y_train, y_test = y[:split], y[split:]
                    
                    st.session_state.train_test_info = {
                        'X_test': X_test,
                        'y_test': y_test
                    }
                    
                    model = Sequential([
                        LSTM(lstm_units, 
                            input_shape=(seq_length, 1),
                            return_sequences=True),
                        Dropout(dropout_rate),
                        LSTM(lstm_units//2),
                        Dropout(dropout_rate),
                        Dense(1)
                    ])
                    
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss='mse',
                        metrics=['mae']
                    )
                    
                    early_stop = EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                    
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop],
                        verbose=0
                    )
                    
                    st.session_state.model = model
                    st.session_state.alert_sent = False
                    st.session_state.last_retrain_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("✅ Model trained successfully!")
                    logging.info("Model training completed.")

                    st.subheader("🧠 Model Architecture")
                    model_summary = StringIO()
                    model.summary(print_fn=lambda x: model_summary.write(x + '\n'))
                    st.text(model_summary.getvalue())
                    st.info(f"Input Shape: (None, {seq_length}, 1) | Output Layer: Dense (1)")
                    
                    st.subheader("📈 Training Performance")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history.history['loss'],
                        name="Training Loss",
                        line=dict(color='royalblue')
                    ))
                    fig.add_trace(go.Scatter(
                        y=history.history['val_loss'],
                        name="Validation Loss",
                        line=dict(color='firebrick')
                    ))
                    fig.update_layout(
                        title="Model Training Progress (Loss Curve)",
                        xaxis_title="Epochs",
                        yaxis_title="Loss",
                        height=300,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    evaluate_model(X_test, y_test, model, scaler, st.session_state.applied_threshold)
                    
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                logging.error(f"Model training failed: {e}")

    # Prediction section
    if st.session_state.model and st.session_state.scaler:
        st.subheader("🔮 Future Predictions")
        
        with st.expander("ℹ️ Prediction Details & Assumptions", expanded=False):
            st.write(f"**Model Version:** {st.session_state.model_version}")
            st.write(f"**Data Window Used (Sequence Length):** {seq_length} historical data points")
            st.write(f"**Last Retrain Date:** {st.session_state.last_retrain_date if st.session_state.last_retrain_date else 'N/A (Model not yet trained/loaded)'}")
            st.markdown("""
            **Assumptions Made:**
            * **Stationarity:** The underlying statistical properties of the time series (mean, variance, autocorrelation) remain constant over time.
            * **Independence:** Future values depend only on past values within the defined sequence length.
            * **No External Factors:** The model primarily relies on the 'Value' itself, assuming no significant unmodeled external influences.
            * **Data Quality:** Input data is clean and representative of the phenomenon.
            * **Future Resembles Past:** The patterns and relationships learned from historical data will continue into the future.
            """)

        future_steps = st.slider("Prediction Horizon (days)", 
                               1, 90, 14,
                               help="Number of future days to predict")
        
        if st.button("Generate Forecast", 
                   help="Generate predictions using the trained model"):
            with st.spinner("Calculating future risks..."):
                try:
                    scaled_data = st.session_state.scaler.transform(
                        st.session_state.df_processed[['Value']])
                    last_sequence = scaled_data[-seq_length:].reshape(
                        1, seq_length, 1)
                    
                    predictions_scaled = []
                    error_margin = 0
                    if st.session_state.train_test_info:
                        X_test_for_err = st.session_state.train_test_info['X_test']
                        y_test_for_err = st.session_state.train_test_info['y_test']
                        if len(X_test_for_err) > 0:
                            test_predictions_scaled = st.session_state.model.predict(X_test_for_err)
                            test_predictions_actual_scale = st.session_state.scaler.inverse_transform(test_predictions_scaled)
                            test_actual_values = st.session_state.scaler.inverse_transform(y_test_for_err.reshape(-1, 1))
                            error_margin = mean_absolute_error(test_actual_values, test_predictions_actual_scale) * 1.5
                            logging.info(f"Estimated error margin for CI: {error_margin:.2f}")

                    current_sequence = last_sequence.copy()
                    
                    for step in range(future_steps):
                        pred = st.session_state.model.predict(
                            current_sequence, verbose=0)
                        predictions_scaled.append(pred[0,0])
                        current_sequence = np.roll(current_sequence, -1, axis=1)
                        current_sequence[0, -1, 0] = pred[0,0]
                    
                    predictions = st.session_state.scaler.inverse_transform(
                        np.array(predictions_scaled).reshape(-1, 1))
                    
                    predictions_upper = predictions.flatten() + error_margin
                    predictions_lower = predictions.flatten() - error_margin

                    future_dates = [st.session_state.df_processed.index[-1] + timedelta(days=i) 
                                  for i in range(1, future_steps+1)]
                    pred_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted_Value': predictions.flatten(),
                        'Upper_Bound': predictions_upper,
                        'Lower_Bound': predictions_lower,
                        # NEW: Assign alert level based on multiple thresholds
                        'Alert_Level': [
                            next((level['level'] for level in st.session_state.alert_levels if val >= level['threshold']), "Safe")
                            for val in predictions.flatten()
                        ],
                        'Threshold_Exceeded': predictions.flatten() >= st.session_state.applied_threshold # Using primary threshold for this boolean
                    })
                    st.session_state.predictions = pred_df
                    
                    st.success("Forecast generated successfully!")
                    logging.info(f"Generated {future_steps}-day forecast.")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=st.session_state.df_processed.index[-60:],
                        y=st.session_state.df_processed['Value'].values[-60:],
                        name='Historical',
                        line=dict(color='royalblue', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=pred_df['Predicted_Value'],
                        name='Forecast',
                        line=dict(color='darkorange', width=3, dash='dot')
                    ))
                    fig.add_trace(go.Scatter(
                        x=future_dates + future_dates[::-1],
                        y=pred_df['Upper_Bound'].tolist() + pred_df['Lower_Bound'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255,165,0,0.3)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval (Est.)',
                        showlegend=True
                    ))
                    # Add all defined alert level thresholds to the plot
                    for level_info in st.session_state.alert_levels:
                        fig.add_hline(
                            y=level_info['threshold'],
                            line=dict(color=level_info['color'], dash='dash', width=1),
                            annotation_text=f"{level_info['level']} Threshold ({level_info['threshold']})",
                            annotation_position="top right",
                            annotation_font_size=10,
                            annotation_font_color=level_info['color']
                        )

                    fig.update_layout(
                        title=f"{future_steps}-Day Forecast of {selected_value_col} with Estimated Confidence",
                        height=400,
                        showlegend=True,
                        xaxis_title="Date",
                        yaxis_title=f"{selected_value_col} (e.g., Sensor Reading, Index)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # NEW: In-App Notification & Alert History Logic
                    alert_summary_table_str = ""
                    triggered_alerts = pred_df[pred_df['Alert_Level'] != "Safe"].copy()
                    
                    if not triggered_alerts.empty:
                        # Sort by highest severity level first for display
                        # Map levels to an arbitrary numerical order for sorting
                        level_order = {level['level']: i for i, level in enumerate(st.session_state.alert_levels)}
                        triggered_alerts['Sort_Order'] = triggered_alerts['Alert_Level'].map(level_order)
                        triggered_alerts = triggered_alerts.sort_values(by=['Sort_Order', 'Date']).drop(columns=['Sort_Order'])

                        st.error(f"🚨 {len(triggered_alerts)} predictions indicate critical events!")
                        
                        # In-App Notification (Visual)
                        most_severe_alert = triggered_alerts['Alert_Level'].iloc[-1] # Get the most severe from sorted list
                        most_severe_icon = next((level['icon'] for level in st.session_state.alert_levels if level['level'] == most_severe_alert), "⚠️")
                        st.warning(f"{most_severe_icon} **Attention!** Predicted values are entering **{most_severe_alert}** levels. Review details below.")
                        
                        # Alert History Log
                        current_alert_log = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "num_alerts": len(triggered_alerts),
                            "highest_level": most_severe_alert,
                            "threshold": st.session_state.applied_threshold,
                            "details": triggered_alerts[['Date', 'Predicted_Value', 'Alert_Level']].to_dict(orient='records')
                        }
                        st.session_state.alert_history.append(current_alert_log)
                        
                        # Format dates in table more cleanly
                        triggered_alerts_display = triggered_alerts.copy()
                        triggered_alerts_display['Date'] = triggered_alerts_display['Date'].dt.strftime('%b %d, %Y')
                        
                        st.dataframe(
                            triggered_alerts_display,
                            column_config={
                                "Date": "Date",
                                "Predicted_Value": st.column_config.NumberColumn(
                                    "Predicted Value",
                                    format="%.2f"
                                ),
                                "Upper_Bound": st.column_config.NumberColumn(
                                    "Upper CI",
                                    format="%.2f"
                                ),
                                "Lower_Bound": st.column_config.NumberColumn(
                                    "Lower CI",
                                    format="%.2f"
                                ),
                                "Alert_Level": st.column_config.TextColumn(
                                    "Alert Level"
                                ),
                                "Threshold_Exceeded": st.column_config.CheckboxColumn(
                                    "Primary Alert",
                                    disabled=True,
                                    default=True
                                )
                            },
                            hide_index=True
                        )
                        
                        # Alert Logic Explanation
                        with st.expander("🛈 How Alerts Are Triggered", expanded=False):
                            st.markdown(f"""
                            Alerts are triggered based on the **predicted value** exceeding predefined thresholds.
                            
                            You have configured the following alert levels:
                            """)
                            for level_info in st.session_state.alert_levels:
                                st.markdown(f"- {level_info['icon']} **{level_info['level']}**: Predicted value $\\ge$ **{level_info['threshold']:.2f}**")
                            st.markdown(f"""
                            The primary alert (used for email notification and initial warning) is based on the lowest defined threshold: **{st.session_state.applied_threshold:.2f}**.
                            This system provides an early warning to allow for proactive measures.
                            """)

                        # Email Alert
                        if enable_email_alerts and notification_email and notification_email != "your-email@example.com":
                            # Prepare table for email body
                            email_table_df = triggered_alerts[['Date', 'Predicted_Value', 'Alert_Level']].copy()
                            email_table_df['Date'] = email_table_df['Date'].dt.strftime('%Y-%m-%d')
                            email_table_df['Predicted_Value'] = email_table_df['Predicted_Value'].apply(lambda x: f"{x:.2f}")
                            
                            alert_summary_table_str = email_table_df.to_markdown(index=False)

                            subject = st.session_state.default_email_subject.format(
                                alert_count=len(triggered_alerts),
                                future_steps=future_steps,
                                threshold=st.session_state.applied_threshold
                            )
                            body = st.session_state.default_email_body.format(
                                alert_count=len(triggered_alerts),
                                future_steps=future_steps,
                                threshold=st.session_state.applied_threshold,
                                alert_summary_table=alert_summary_table_str
                            )
                            
                            if send_email_alert(notification_email, subject, body):
                                st.session_state.alert_sent = True
                        else:
                            if enable_email_alerts:
                                st.warning("Email alerts are enabled but not sent. Please ensure a valid 'Notification Email' is provided in the sidebar.")
                            else:
                                st.info("Email alerts are disabled. Check 'Enable Email Alerts' in sidebar if you wish to receive them.")
                    else:
                        st.info("✅ No immediate disaster predicted. All future values are within the safe threshold.")
                        st.session_state.alert_sent = False # Reset if no alert needed
                    
                    # Display all predictions in a collapsible section
                    with st.expander("Show All Future Predictions"):
                        st.write("All predicted values for the next 14 days:")
                        all_pred_df_display = pred_df.copy()
                        all_pred_df_display['Date'] = all_pred_df_display['Date'].dt.strftime('%b %d, %Y')
                        st.dataframe(
                            all_pred_df_display,
                            column_config={
                                "Date": "Date",
                                "Predicted_Value": st.column_config.NumberColumn(
                                    "Predicted Value",
                                    format="%.2f"
                                ),
                                "Upper_Bound": st.column_config.NumberColumn(
                                    "Upper CI",
                                    format="%.2f"
                                ),
                                "Lower_Bound": st.column_config.NumberColumn(
                                    "Lower CI",
                                    format="%.2f"
                                ),
                                "Alert_Level": st.column_config.TextColumn(
                                    "Alert Level"
                                ),
                                "Threshold_Exceeded": st.column_config.CheckboxColumn(
                                    "Primary Alert",
                                    disabled=True
                                )
                            },
                            hide_index=True
                        )
                        create_download_button(pred_df, "future_predictions.csv")

                except Exception as e:
                    st.error(f"Prediction generation failed: {str(e)}")
                    logging.error(f"Prediction generation failed: {e}")
    else:
        st.info("Please process your data and train a model to generate future predictions.")
else:
    st.info("Upload your sensor data to get started, or generate sample data to test the system.")
    if st.button("🎲 Generate Sample Data"):
        dates = pd.date_range(start='2023-01-01', periods=100)
        values = np.random.normal(loc=50, scale=20, size=100).cumsum() + np.sin(np.arange(100)/10)*50
        sample_df = pd.DataFrame({'Date': dates, 'Value': values})
        create_download_button(sample_df, "sample_data.csv")
        logging.info("Sample data generated and download button created.")

# ==============================================
# DEBUG PANEL
# ==============================================
if st.checkbox("🐛 Show Debug Panel", False):
    st.subheader("System Diagnostics")
    
    tab1, tab2, tab3 = st.tabs(["Session State", "Data Stats", "Logs"])
    with tab1:
        st.json({
            "model_loaded": st.session_state.model is not None,
            "scaler_loaded": st.session_state.scaler is not None,
            "applied_threshold": st.session_state.get('applied_threshold', 100.0),
            "predictions_available": st.session_state.predictions is not None,
            "notification_email_set": notification_email != "your-email@example.com",
            "alert_sent_status": st.session_state.alert_sent,
            "model_version": st.session_state.model_version,
            "last_retrain_date": st.session_state.last_retrain_date,
            "alert_levels_configured": st.session_state.alert_levels,
            "alert_history_count": len(st.session_state.alert_history)
        })
    
    with tab2:
        if st.session_state.data_stats:
            st.write("Data Statistics:")
            st.json(st.session_state.data_stats)
            st.write("Value Distribution (Sampled Normal):")
            st.line_chart(pd.DataFrame({'Value': np.random.normal(
                st.session_state.data_stats['mean'],
                st.session_state.data_stats['std'],
                1000
            )}))
        else:
            st.info("Upload data to see statistics.")

    with tab3: # Logs tab
        st.subheader("Application Logs (In-Memory Stream):")
        log_stream.seek(0) # Go to the beginning of the stream
        st.code(log_stream.read())
        st.info("Note: For persistent logs in deployment, refer to your Streamlit Cloud dashboard or server logs.")

# ==============================================
# FOOTER
# ==============================================
st.markdown("---")
st.markdown("Developed by Your Name/Team Name")
st.markdown("For educational and demonstration purposes only.")
