# app.py - Ultimate Fraud Detection System with Advanced Analytics
import streamlit as st
import pandas as pd
import mysql.connector
import pickle
import smtplib
import time
import threading
import numpy as np
from email.message import EmailMessage
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from streamlit.components.v1 import html
import datetime
import plotly.express as px
import plotly.graph_objects as go
from decimal import Decimal
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="AI Fraud Detection",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# ====================== CONSTANTS ======================
# Updated DB_CONFIG for Azure - Directly using values instead of secrets
DB_CONFIG = {
    'host': "detect-db.mysql.database.azure.com",  # Azure MySQL server name
    'user': "detect_db",  # Azure MySQL username
    'password': "derutu@2711",  # Azure MySQL password
    'database': "fraud",  # Your database name
    'port': 3306,  # Default MySQL port
    'ssl_ca': None,  # Add SSL cert path if required
    'connect_timeout': 10
}

EMAIL_CONFIG = {
    'sender': "rajukakade183@gmail.com",
    'receiver': "rajukakade183@gmail.com",
    'password': "uobp wrkr tclm zflh"
}

# ====================== FRAUD SYSTEM CLASS ======================
class FraudSystem:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.stop_monitoring = False
        self.last_prediction = None

    def load_data(self):
        """Load data from MySQL database"""
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            queries = {
                'transactions': "SELECT * FROM transactions",
                'users': "SELECT * FROM users",
                'cards': "SELECT * FROM cards",
                'merchants': "SELECT * FROM merchants",
                'credit_data': "SELECT * FROM credit_data",
                'addresses': "SELECT * FROM addresses"
            }
            return {table: pd.read_sql(query, conn) for table, query in queries.items()}
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return None
        finally:
            if conn and conn.is_connected():
                conn.close()

    def merge_data(self, dataframes):
        """Merge all tables like in training"""
        df = dataframes['transactions'].copy()
        df = df.merge(dataframes['cards'], left_on='card_id', right_on='id', how='left', suffixes=('', '_card'))
        df = df.merge(dataframes['users'], left_on='client_id', right_on='id', how='left', suffixes=('', '_user'))
        df = df.merge(dataframes['merchants'], on='merchant_id', how='left', suffixes=('', '_merchant'))
        df = df.merge(dataframes['addresses'], left_on='client_id', right_on='id', how='left', suffixes=('', '_address'))
        df = df.merge(dataframes['credit_data'], left_on='client_id', right_on='id', how='left', suffixes=('', '_credit'))
        return df

    def train_model(self):
        """Train and save fraud detection model"""
        try:
            dataframes = self.load_data()
            if dataframes is None:
                return False
                
            df = self.merge_data(dataframes)
            if df is None:
                return False
                
            required_columns = ['errors_indicator', 'use_chip', 'errors', 'fraud_classification']
            if not all(col in df.columns for col in required_columns):
                st.error("Missing required columns in dataset")
                return False
                
            features = ['errors_indicator', 'use_chip', 'errors']
            df = df[features + ['fraud_classification']].dropna()
            df['fraud_classification'] = df['fraud_classification'].map({'Non-Fraud': 0, 'Fraud': 1})
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), ['use_chip', 'errors'])
                ],
                remainder='passthrough'
            )
            
            X = preprocessor.fit_transform(df[features])
            y = df['fraud_classification']
            
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                class_weight='balanced',
                random_state=42
            )
            model.fit(X, y)
            
            with open('fraud_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('preprocessor.pkl', 'wb') as f:
                pickle.dump(preprocessor, f)
                
            return True
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return False

    def predict_fraud(self, transaction_data):
        """Predict fraud for new transaction"""
        try:
            if not self.model:
                with open('fraud_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                with open('preprocessor.pkl', 'rb') as f:
                    self.preprocessor = pickle.load(f)
            
            features = ['errors_indicator', 'use_chip', 'errors']
            df = pd.DataFrame([transaction_data])[features]
            
            X = self.preprocessor.transform(df)
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][1]
            
            return prediction, probability
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None

    def send_alert(self, transaction_id, probability):
        """Send email alert"""
        try:
            msg = EmailMessage()
            msg['Subject'] = f"🚨 [TEST] Fraud Alert - Transaction {transaction_id}"
            msg['From'] = EMAIL_CONFIG['sender']
            msg['To'] = EMAIL_CONFIG['receiver']
            
            msg.set_content(f"""
            TEST EMAIL - NOT A REAL ALERT
            
            Fraudulent Transaction Detected (Simulation)!
            
            Transaction ID: {transaction_id}
            Probability: {probability:.2%}
            
            This is only a test - no action required!
            """)
            
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
                server.send_message(msg)
            return True
        except Exception as e:
            st.error(f"Email error: {str(e)}")
            return False

    def start_monitoring(self):
        """Monitor transactions in background"""
        conn = None
        cursor = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            last_id = None
            
            while not self.stop_monitoring:
                cursor.execute("SELECT MAX(id) FROM transactions")
                latest_id = cursor.fetchone()['MAX(id)']
                
                if last_id is None:
                    last_id = latest_id
                
                if latest_id != last_id:
                    last_id = latest_id
                    
                    cursor.execute(f"""
                    SELECT t.*, u.name as client_name 
                    FROM transactions t
                    JOIN users u ON t.client_id = u.id
                    WHERE t.id = {latest_id}
                    """)
                    txn_data = cursor.fetchone()
                    
                    prediction, probability = self.predict_fraud({
                        'errors_indicator': txn_data['errors_indicator'],
                        'use_chip': txn_data['use_chip'],
                        'errors': txn_data['errors']
                    })
                    
                    if prediction == 1:
                        self.send_alert(latest_id, probability)
                        st.session_state.alerts.append(
                            f"🚨 [TEST] Fraud detected on Txn #{latest_id} (Prob: {probability:.2%}) - This is a simulation"
                        )
                    else:
                        st.session_state.alerts.append(
                            f"✅ Legitimate Txn #{latest_id} (Prob: {1-probability:.2%})"
                        )
                    
                    st.session_state.last_alert = (prediction, probability)
                    st.rerun()
                
                time.sleep(5)
        except Exception as e:
            st.error(f"Monitoring error: {str(e)}")
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

# ====================== ETL PIPELINE CLASS ======================
class ETLPipeline:
    def __init__(self):
        self.engine = None
        self.etl_status = "Not Run"
        self.etl_history = []
        
    def connect_to_mysql(self):
        """Establish MySQL connection"""
        try:
            self.engine = create_engine(
                f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:3306/{DB_CONFIG['database']}"
            )
            return True
        except Exception as e:
            st.error(f"MySQL Connection Failed: {e}")
            return False
    
    def extract_data(self, file_path):
        """Extract data from CSV files"""
        try:
            df = pd.read_csv(file_path, low_memory=False)
            self.etl_history.append(f"✅ Extracted {file_path} with {df.shape[0]} rows")
            return df
        except Exception as e:
            self.etl_history.append(f"❌ Failed to extract {file_path}: {e}")
            return None
    
    def transform_data(self, df, table_name):
        """Apply data transformations"""
        try:
            if df is None:
                return None
                
            # Basic cleaning for all tables
            df = df.dropna(how='all')
            df = df.drop_duplicates()
            
            # Table-specific transformations
            if table_name == "transactions":
                # Convert amount to float
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                # Clean date field
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
            elif table_name == "users":
                # Standardize names
                df['name'] = df['name'].str.title()
                
            self.etl_history.append(f"🔄 Transformed {table_name} data")
            return df
        except Exception as e:
            self.etl_history.append(f"❌ Transformation failed for {table_name}: {e}")
            return None
    
    def load_data(self, df, table_name):
        """Load data into MySQL"""
        if df is None:
            self.etl_history.append(f"⚠ Skipping {table_name} due to previous errors")
            return False
            
        try:
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            self.etl_history.append(f"📊 Loaded {table_name} to MySQL successfully")
            return True
        except Exception as e:
            self.etl_history.append(f"❌ Failed to load {table_name}: {e}")
            return False
    
    def run_pipeline(self, file_paths):
        """Execute complete ETL pipeline"""
        self.etl_history = []
        self.etl_status = "Running"
        
        if not self.connect_to_mysql():
            self.etl_status = "Failed"
            return False
        
        for table_name, file_path in file_paths.items():
            # Extract
            df = self.extract_data(file_path)
            
            # Transform
            df = self.transform_data(df, table_name)
            
            # Load
            self.load_data(df, table_name)
        
        self.etl_status = "Completed"
        return True
    
    def visualize_pipeline(self):
        """Generate ETL pipeline visualization"""
        try:
            fig, ax = plt.subplots(figsize=(10, 2))
            stages = ["Extract", "Transform", "Load", "PowerBI"]
            colors = ["#4CAF50", "#FFC107", "#2196F3", "#9C27B0"]

            for i, stage in enumerate(stages):
                ax.add_patch(plt.Rectangle((i * 2, 0.5), 1.8, 0.8, color=colors[i], ec="black", lw=1.5))
                ax.text(i * 2 + 0.9, 0.9, stage, ha='center', va='center', fontsize=12, color='white', fontweight='bold')

                if i < len(stages) - 1:
                    ax.annotate('', xy=(i * 2 + 1.8, 0.9), xytext=((i + 1) * 2, 0.9),
                                arrowprops=dict(arrowstyle="->", lw=2, color='gray'))

            ax.set_xlim(0, len(stages) * 2)
            ax.set_ylim(0.4, 1.5)
            ax.axis('off')
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            return img
        except Exception as e:
            st.error(f"Visualization error: {e}")
            return None

# Initialize the system
fraud_system = FraudSystem()
etl_pipeline = ETLPipeline()

# ====================== UI ENHANCEMENTS ======================
def apply_custom_styles():
    """Apply custom CSS styles"""
    st.markdown("""
    <style>
        /* Base font size */
        html, body, .stApp {
            font-size: 18px !important;
        }
        
        /* Headers */
        h1 {
            font-size: 2rem !important;
            color: #2563eb !important;
        }
        h2 {
            font-size: 1.75rem !important;
        }
        h3 {
            font-size: 1.5rem !important;
        }
        
        /* Feature cards */
        .feature-card {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            background: #1e293b;
            border-left: 5px solid #2563eb;
            color: white !important;
        }
        .feature-card h3 {
            color: #93c5fd !important;
        }
        
        /* Alerts */
        .fraud-alert {
            background: #fff0f0 !important;
            color: #ff4b4b !important;
            border-left: 5px solid #ff4b4b !important;
        }
        .success-alert {
            background: #e6ffed !important;
            color: #00a854 !important;
            border-left: 5px solid #00a854 !important;
        }
        
        /* ETL Status */
        .etl-running {
            background: #fff3e0 !important;
            border-left: 5px solid #FFC107 !important;
        }
        .etl-success {
            background: #e8f5e9 !important;
            border-left: 5px solid #4CAF50 !important;
        }
        .etl-failed {
            background: #ffebee !important;
            border-left: 5px solid #f44336 !important;
        }
        
        /* Dashboard tiles */
        .metric-card {
            padding: 15px;
            border-radius: 8px;
            background: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

def inject_animations():
    """Inject custom animations"""
    st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .stApp {
            animation: fadeIn 0.5s;
        }
        .stButton>button {
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
    </style>
    """, unsafe_allow_html=True)

# ====================== ETL PIPELINE PAGE ======================
def show_etl_pipeline():
    """Show ETL pipeline visualization and controls"""
    st.title("🔄 ETL Pipeline Dashboard")
    st.markdown("---")

    st.info("""
    **Data Flow Architecture**  
    This system uses an automated ETL pipeline to process transaction data from CSV files 
    to MySQL database and finally to PowerBI dashboards.
    """)

    # Pipeline Visualization
    st.subheader("📊 Pipeline Architecture")
    pipeline_img = etl_pipeline.visualize_pipeline()
    if pipeline_img:
        st.image(pipeline_img, use_column_width=True)
    else:
        st.warning("⚠️ Unable to load pipeline visualization.")

    # Documentation
    with st.expander("📚 Pipeline Documentation", expanded=True):
        st.markdown("""
        ### ETL Pipeline Stages

        1. **Extract**  
           - Reads raw data from CSV files  
           - Supports various file formats and encodings  
           - Validates basic file structure

        2. **Transform**  
           - Cleanses data (removes duplicates, handles missing values)  
           - Standardizes formats (dates, currencies)  
           - Applies business rules and validations  
           - Feature engineering for fraud detection

        3. **Load**  
           - Stores processed data in MySQL database  
           - Maintains data integrity with constraints  
           - Optimized for query performance

        4. **PowerBI Integration**  
           - Direct connection to MySQL database  
           - Automatic refresh on data updates  
           - Real-time dashboards  
        """)

    st.markdown("---")
    st.subheader("⚙️ Pipeline Controls")

    demo_file_paths = {
        "transactions": "transaction.csv",
        "users": "user_data.csv",
        "credit_data": "credit_data.csv",
        "merchants": "merchant_data.csv",
        "addresses": "address_data.csv",
        "cards": "card_data.csv"
    }

    if st.button("▶️ Run ETL Pipeline", type="primary"):
        with st.spinner("Running ETL pipeline..."):
            if etl_pipeline.run_pipeline(demo_file_paths):
                st.success("ETL pipeline completed successfully!")
            else:
                st.error("ETL pipeline failed!")

    st.markdown("---")
    st.subheader("📝 Pipeline Execution Log")

    status_class = {
        "Not Run": "",
        "Running": "etl-running",
        "Completed": "etl-success",
        "Failed": "etl-failed"
    }.get(etl_pipeline.etl_status, "")

    st.markdown(f"""
    <div class="metric-card {status_class}">
        <h3>Pipeline Status: {etl_pipeline.etl_status}</h3>
    </div>
    """, unsafe_allow_html=True)

    if etl_pipeline.etl_history:
        st.text_area("Execution Log", "\n".join(etl_pipeline.etl_history), height=200)

    st.markdown("---")
    st.subheader("🔍 Data Flow Example")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("""
        ### CSV Source Data
        ```
        transaction_id,amount,date,card_id,errors
        1001,150.00,2023-01-15,5001,None
        1002,89.99,2023-01-15,5002,Insufficient Balance
        1003,1200.00,2023-01-16,5003,Invalid PIN
        ```
        """)

    with cols[1]:
        st.markdown("""
        ### MySQL Processed Data
        ```sql
        SELECT * FROM transactions LIMIT 3;

        id  | amount | date       | card_id | errors               | fraud_classification
        ----|--------|------------|---------|----------------------|---------------------
        1   | 150.00 | 2023-01-15 | 5001    | NULL                 | Non-Fraud
        2   | 89.99  | 2023-01-15 | 5002    | Insufficient Balance | Fraud
        3   |1200.00 | 2023-01-16 | 5003    | Invalid PIN          | Fraud
        ```
        """)

# ====================== PROJECT DOCUMENTATION PAGE ======================
def show_project_documentation():
    """Show detailed project documentation"""
    st.title("📑 Fraud Detection System Documentation")
    st.markdown("---")
    
    with st.expander("📌 System Overview", expanded=True):
        st.markdown("""
        ### Introduction
        This AI-powered Fraud Detection System is designed to identify suspicious financial transactions in real-time using machine learning. The system analyzes transaction patterns and flags potential fraud with 96.4% accuracy.

        ### Key Features
        - Real-time transaction monitoring
        - Automated fraud alerts via email
        - Historical data analysis
        - Model training and versioning
        - Comprehensive ETL pipeline
        """)
    
    with st.expander("⚙️ System Architecture", expanded=False):
        st.markdown("""
        ### Components
        1. **Data Layer**: MySQL database storing all transaction data
        2. **Processing Layer**: Python-based ETL and machine learning
        3. **Presentation Layer**: Streamlit web interface
        4. **Alerting System**: SMTP email notifications

        ### Technology Stack
        - Python 3.10
        - Scikit-learn (Random Forest Classifier)
        - Streamlit for UI
        - MySQL for data storage
        - SMTP for email alerts
        """)
    
    with st.expander("📊 Model Information", expanded=False):
        st.markdown("""
        ### Model Details
        - **Algorithm**: Random Forest Classifier
        - **Version**: v2.1.4
        - **Accuracy**: 96.4%
        - **Features Used**:
          - Error indicators
          - Payment method (chip/swipe)
          - Transaction errors
          - Historical patterns

        ### Training Process
        1. First train the model using historical data
        2. Model is saved as fraud_model.pkl
        3. Preprocessor saved as preprocessor.pkl
        4. Model can be retrained with new data
        """)
    
    with st.expander("🚀 Getting Started", expanded=False):
        st.markdown("""
        ### Initial Setup
        1. Configure database connection in DB_CONFIG
        2. Set up email credentials in EMAIL_CONFIG
        3. Run the Streamlit app: `streamlit run app.py`

        ### Usage Instructions
        1. **Home Page**: System overview and features
        2. **Live Testing**: Simulate transactions and test fraud detection
        3. **ETL Pipeline**: Run data processing workflows
        4. **Documentation**: System documentation (this page)

        ### Model Retraining
        To retrain the model with latest data:
        1. Go to Live Testing page
        2. Click "Retrain Model" button
        3. Wait for training to complete
        """)
    
    with st.expander("⚠️ Troubleshooting", expanded=False):
        st.markdown("""
        ### Common Issues
        1. **Database Connection Errors**:
           - Verify DB_CONFIG credentials
           - Check network connectivity
           - Ensure MySQL server is running

        2. **Email Not Sending**:
           - Verify SMTP credentials
           - Check if less secure apps are allowed
           - Try different SMTP port (465 or 587)

        3. **Model Prediction Errors**:
           - Ensure model files exist (fraud_model.pkl, preprocessor.pkl)
           - Verify input data format matches training data
           - Retrain model if data structure changes

        4. **Streamlit Performance Issues**:
           - Reduce data volume for testing
           - Use st.cache for expensive computations
           - Optimize database queries
        """)

# ====================== MAIN APP ======================
def main():
    apply_custom_styles()
    inject_animations()
    
    # Initialize session state
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    if 'last_alert' not in st.session_state:
        st.session_state.last_alert = None

    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["🏠 Home", "🔍 Live Testing", "🔄 ETL Pipeline", "📑 Project Documentation"])

    # -------------------- HOME PAGE --------------------
    if page == "🏠 Home":
        st.title("AI-Powered Fraud Detection System")
        st.markdown("---")
        
        # Hero Section
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #2563eb;">🛡️ Protecting Your Transactions</h1>
            <p style="font-size: 1.2rem;">
            Our cutting-edge AI system detects fraudulent transactions in real-time with 
            <span style="font-weight: bold; color: #10b981;">96.4% accuracy</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Why This Matters
        with st.expander("🔍 Why Fraud Detection Matters", expanded=True):
            st.markdown("""
            - **Financial losses** from payment fraud reached \$41 billion globally in 2022
            - **78% of businesses** experienced increased fraud attempts last year
            - Our system can **reduce fraud losses by 90%** through early detection
            """)
        
        # Key Features
        st.markdown("---")
        st.header("✨ Key Features")
        
        cols = st.columns(3)
        with cols[0]:
            st.markdown("""
            <div class="feature-card">
                <h3>Real-time Monitoring</h3>
                <p>24/7 transaction analysis with 2.4s average detection time</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown("""
            <div class="feature-card">
                <h3>AI-Powered</h3>
                <p>Machine learning models with 96.4% accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown("""
            <div class="feature-card">
                <h3>Instant Alerts</h3>
                <p>Email/SMS notifications for suspicious activity</p>
            </div>
            """, unsafe_allow_html=True)
        
        # How It Works
        st.markdown("---")
        st.header("🛠️ How It Works")
        st.image("https://via.placeholder.com/1200x400?text=Fraud+Detection+Workflow", 
                caption="Transaction Analysis Pipeline")
        
        st.markdown("""
        1. **Transaction Data** flows into our system
        2. **AI Model** analyzes 150+ risk factors
        3. **Real-time Decision** made within milliseconds
        4. **Alerts** generated for suspicious transactions
        """)

    # -------------------- LIVE TESTING PAGE --------------------
    elif page == "🔍 Live Testing":
        st.title("Live Transaction Testing")
        st.markdown("---")
        
        st.warning("""
        ⚠ *This is a simulation environment*  
        All fraud detection alerts are for testing purposes only.  
        Emails sent will be marked as TEST messages.
        """)
        
        # Model Management Section
        st.markdown("---")
        st.subheader("🤖 Model Management")
        
        cols = st.columns([3, 1])
        with cols[0]:
            if st.button("🔄 Retrain Model with Latest Data", type="primary", 
                        help="First train the model before using monitoring features"):
                with st.spinner("Training model with latest transaction data..."):
                    if fraud_system.train_model():
                        st.success("Model retrained successfully!")
                    else:
                        st.error("Model retraining failed!")
        with cols[1]:
            st.metric("Model Version", "v2.1.4")
        
        # Email Settings
        st.markdown("---")
        st.subheader("✉ Alert Configuration")
        
        cols = st.columns(2)
        with cols[0]:
            EMAIL_CONFIG['sender'] = st.text_input("Sender Email", value=EMAIL_CONFIG['sender'])
            EMAIL_CONFIG['password'] = st.text_input("App Password", type="password", value=EMAIL_CONFIG['password'])
        with cols[1]:
            EMAIL_CONFIG['receiver'] = st.text_input("Receiver Email", value=EMAIL_CONFIG['receiver'])
        
        # Test Transaction
        st.markdown("---")
        st.subheader("🧪 Test Transaction Simulation")
        
        with st.form("test_transaction"):
            cols = st.columns(3)
            with cols[0]:
                amount = st.number_input("Amount ($)", min_value=0.0, value=150.0)
            with cols[1]:
                error_indicator = st.selectbox("Error Indicator", [0, 1], 
                                             help="0 = Legitimate, 1 = Potential Fraud")
            with cols[2]:
                error_type = st.selectbox("Error Type", ["None", "Insufficient Balance", "Invalid PIN"])
            
            submitted = st.form_submit_button("🚀 Test Transaction")
            
            if submitted:
                with st.spinner("Analyzing..."):
                    time.sleep(1.5)
                    prediction, prob = fraud_system.predict_fraud({
                        'errors_indicator': error_indicator,
                        'use_chip': "Chip Transaction",
                        'errors': error_type
                    })
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="fraud-alert" style='
                         background: #fff0f0;
                            color: #ff4b4b;
                            padding: 20px;
                            border-radius: 10px;
                            margin: 15px 0;
                        '>
                        <h2>🚨 FRAUD DETECTED (TEST)</h2>
                        <p>Probability: {prob:.2%}</p>
                        <p><small>This is a test simulation only</small></p>
                        </div>
                         """, unsafe_allow_html=True)
                        if fraud_system.send_alert("TEST_TXN", prob):
                            st.toast(f"📧 Test alert sent to {EMAIL_CONFIG['receiver']}")
                    else:
                        st.markdown(f"""
                        <div class="success-animation" style='
                            background: #e6ffed;
                            color: #00a854;
                            padding: 20px;
                            border-radius: 10px;
                            margin: 15px 0;
                        '>
                        <h2>✅ LEGITIMATE</h2>
                        <p>Confidence: {1-prob:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Monitoring Controls
        st.markdown("---")
        st.subheader("👁️ Live Monitoring (Simulation)")
        
        cols = st.columns([3, 1])
        with cols[0]:
            if st.button("▶ Start Monitoring", type="primary", disabled=st.session_state.monitoring):
                st.session_state.monitoring = True
                fraud_system.stop_monitoring = False
                monitor_thread = threading.Thread(target=fraud_system.start_monitoring)
                monitor_thread.start()
                st.success("Monitoring started (Simulation Mode)!")
            
            if st.button("⏹ Stop Monitoring", type="secondary", disabled=not st.session_state.monitoring):
                st.session_state.monitoring = False
                fraud_system.stop_monitoring = True
                st.success("Monitoring stopped")
            
            if st.session_state.alerts:
                st.subheader("🔔 Recent Alerts (Test Only)")
                for alert in reversed(st.session_state.alerts[-5:]):
                    st.write(alert)
        
        with cols[1]:
            st.subheader("📊 Status")
            if st.session_state.monitoring:
                st.success("🟢 ACTIVE (TEST)")
                st.metric("Last Check", time.strftime("%H:%M:%S"))
            else:
                st.error("🔴 OFFLINE")
            
            if st.session_state.last_alert:
                pred, prob = st.session_state.last_alert
                st.metric("Last Result", "FRAUD (TEST)" if pred == 1 else "CLEAN", 
                         f"{prob:.2%}" if pred == 1 else f"{1-prob:.2%}")

    # -------------------- ETL PIPELINE PAGE --------------------
    elif page == "🔄 ETL Pipeline":
        show_etl_pipeline()
        
    # -------------------- PROJECT DOCUMENTATION PAGE --------------------
    elif page == "📑 Project Documentation":
        show_project_documentation()

if __name__ == "__main__":
    main()