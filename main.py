import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bank Marketing Analysis Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset function
@st.cache_data
def load_data():
    """Load the Bank Marketing dataset"""
    # For demo purposes, we'll create a sample dataset structure
    # In real implementation, you would load from: 
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
    # df = pd.read_csv(url, sep=';')
    
    np.random.seed(42)
    n_samples = 5000
    
    # Create sample data mimicking the Bank Marketing dataset structure
    jobs = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
            'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']
    marital = ['divorced', 'married', 'single']
    education = ['primary', 'secondary', 'tertiary', 'unknown']
    default = ['no', 'yes']
    housing = ['no', 'yes']
    loan = ['no', 'yes']
    contact = ['cellular', 'telephone', 'unknown']
    month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    poutcome = ['failure', 'other', 'success', 'unknown']
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 95, n_samples),
        'job': np.random.choice(jobs, n_samples),
        'marital': np.random.choice(marital, n_samples),
        'education': np.random.choice(education, n_samples),
        'default': np.random.choice(default, n_samples, p=[0.98, 0.02]),
        'balance': np.random.normal(1362, 3044, n_samples).astype(int),
        'housing': np.random.choice(housing, n_samples, p=[0.44, 0.56]),
        'loan': np.random.choice(loan, n_samples, p=[0.84, 0.16]),
        'contact': np.random.choice(contact, n_samples, p=[0.65, 0.30, 0.05]),
        'day': np.random.randint(1, 32, n_samples),
        'month': np.random.choice(month, n_samples),
        'duration': np.random.exponential(258, n_samples).astype(int),
        'campaign': np.random.poisson(2.76, n_samples) + 1,
        'pdays': np.random.choice([-1] + list(range(1, 1000)), n_samples, p=[0.82] + [0.18/999]*999),
        'previous': np.random.poisson(0.58, n_samples),
        'poutcome': np.random.choice(poutcome, n_samples, p=[0.49, 0.20, 0.12, 0.19]),
        'y': np.random.choice(['no', 'yes'], n_samples, p=[0.89, 0.11])
    })
    
    return df

# Main app
def main():
    st.markdown('<h1 class="main-header">🏦 Bank Marketing Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">Navigation</div>', unsafe_allow_html=True)
    
    # Feature selection
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["📊 Data Overview", "🔍 Exploratory Data Analysis", "📈 Statistical Analysis", 
         "🎯 Predictive Modeling", "🔮 Customer Segmentation", "📋 Campaign Performance", 
         "💼 Business Intelligence", "🛠️ Data Quality Assessment"]
    )
    
    # Data filters
    st.sidebar.markdown("### Data Filters")
    age_range = st.sidebar.slider("Age Range", int(df['age'].min()), int(df['age'].max()), 
                                 (int(df['age'].min()), int(df['age'].max())))
    
    job_filter = st.sidebar.multiselect("Job Type", df['job'].unique(), default=df['job'].unique())
    marital_filter = st.sidebar.multiselect("Marital Status", df['marital'].unique(), default=df['marital'].unique())
    
    # Apply filters
    filtered_df = df[
        (df['age'] >= age_range[0]) & 
        (df['age'] <= age_range[1]) & 
        (df['job'].isin(job_filter)) & 
        (df['marital'].isin(marital_filter))
    ]
    
    # Feature 1: Data Overview
    if page == "📊 Data Overview":
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(filtered_df))
        with col2:
            st.metric("Success Rate", f"{(filtered_df['y'] == 'yes').mean():.1%}")
        with col3:
            st.metric("Avg Age", f"{filtered_df['age'].mean():.1f}")
        with col4:
            st.metric("Avg Balance", f"€{filtered_df['balance'].mean():.0f}")
        
        # Dataset info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", filtered_df.shape)
            st.write("**Data Types:**")
            st.write(filtered_df.dtypes)
        
        with col2:
            st.write("**Missing Values:**")
            st.write(filtered_df.isnull().sum())
            
        # Sample data
        st.subheader("Sample Data")
        st.dataframe(filtered_df.head(10))
        
        # Basic statistics
        st.subheader("Descriptive Statistics")
        st.write(filtered_df.describe())
    
    # Feature 2: Exploratory Data Analysis
    elif page == "🔍 Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        
        # Target variable distribution
        st.subheader("Target Variable Distribution")
        fig = px.pie(filtered_df, names='y', title="Term Deposit Subscription Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Age distribution
        st.subheader("Age Distribution by Subscription")
        fig = px.histogram(filtered_df, x='age', color='y', nbins=30, 
                          title="Age Distribution by Term Deposit Subscription")
        st.plotly_chart(fig, use_container_width=True)
        
        # Job distribution
        st.subheader("Job Distribution")
        job_counts = filtered_df['job'].value_counts()
        fig = px.bar(x=job_counts.index, y=job_counts.values, 
                    title="Distribution of Job Types")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Matrix")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       title="Correlation Matrix of Numeric Variables",
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature 3: Statistical Analysis
    elif page == "📈 Statistical Analysis":
        st.header("Statistical Analysis")
        
        # Campaign success by different factors
        st.subheader("Success Rate Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Success rate by job
            job_success = filtered_df.groupby('job')['y'].apply(lambda x: (x == 'yes').mean()).sort_values(ascending=False)
            fig = px.bar(x=job_success.index, y=job_success.values, 
                        title="Success Rate by Job Type")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Success rate by marital status
            marital_success = filtered_df.groupby('marital')['y'].apply(lambda x: (x == 'yes').mean())
            fig = px.bar(x=marital_success.index, y=marital_success.values, 
                        title="Success Rate by Marital Status")
            st.plotly_chart(fig, use_container_width=True)
        
        # Duration vs Success
        st.subheader("Call Duration Impact")
        fig = px.box(filtered_df, x='y', y='duration', 
                    title="Call Duration Distribution by Subscription")
        st.plotly_chart(fig, use_container_width=True)
        
        # Campaign frequency analysis
        st.subheader("Campaign Frequency Analysis")
        campaign_success = filtered_df.groupby('campaign')['y'].apply(lambda x: (x == 'yes').mean())
        fig = px.line(x=campaign_success.index, y=campaign_success.values, 
                     title="Success Rate by Number of Campaigns")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature 4: Predictive Modeling
    elif page == "🎯 Predictive Modeling":
        st.header("Predictive Modeling")
        
        # Prepare data for modeling
        df_model = filtered_df.copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
        
        for col in categorical_cols:
            df_model[col] = le.fit_transform(df_model[col])
        
        df_model['y'] = le.fit_transform(df_model['y'])
        
        X = df_model.drop('y', axis=1)
        y = df_model['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model selection
        model_choice = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression"])
        
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Model performance
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        with col2:
            st.metric("ROC AUC", f"{roc_auc_score(y_test, y_pred):.3f}")
        with col3:
            st.metric("Test Size", len(y_test))
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (for Random Forest)
        if model_choice == "Random Forest":
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(feature_importance.head(10), x='importance', y='feature', 
                        orientation='h', title="Top 10 Most Important Features")
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature 5: Customer Segmentation
    elif page == "🔮 Customer Segmentation":
        st.header("Customer Segmentation Analysis")
        
        # Age groups
        filtered_df['age_group'] = pd.cut(filtered_df['age'], bins=[0, 30, 40, 50, 60, 100], 
                                         labels=['18-30', '31-40', '41-50', '51-60', '60+'])
        
        # Balance groups
        filtered_df['balance_group'] = pd.cut(filtered_df['balance'], bins=[-np.inf, 0, 1000, 5000, np.inf], 
                                             labels=['Negative', 'Low', 'Medium', 'High'])
        
        # Segment analysis
        st.subheader("Customer Segments")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age group success rate
            age_success = filtered_df.groupby('age_group')['y'].apply(lambda x: (x == 'yes').mean())
            fig = px.bar(x=age_success.index, y=age_success.values, 
                        title="Success Rate by Age Group")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Balance group success rate
            balance_success = filtered_df.groupby('balance_group')['y'].apply(lambda x: (x == 'yes').mean())
            fig = px.bar(x=balance_success.index, y=balance_success.values, 
                        title="Success Rate by Balance Group")
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment overview
        st.subheader("Segment Overview")
        segment_table = filtered_df.groupby(['age_group', 'balance_group']).agg({
            'y': lambda x: (x == 'yes').mean(),
            'age': 'count'
        }).round(3)
        segment_table.columns = ['Success_Rate', 'Count']
        st.dataframe(segment_table)
    
    # Feature 6: Campaign Performance
    elif page == "📋 Campaign Performance":
        st.header("Campaign Performance Analysis")
        
        # Monthly performance
        st.subheader("Monthly Campaign Performance")
        monthly_stats = filtered_df.groupby('month').agg({
            'y': lambda x: (x == 'yes').mean(),
            'duration': 'mean',
            'campaign': 'mean'
        }).round(3)
        monthly_stats.columns = ['Success_Rate', 'Avg_Duration', 'Avg_Campaigns']
        
        fig = px.line(x=monthly_stats.index, y=monthly_stats['Success_Rate'], 
                     title="Monthly Success Rate Trend")
        st.plotly_chart(fig, use_container_width=True)
        
        # Contact method effectiveness
        st.subheader("Contact Method Effectiveness")
        contact_stats = filtered_df.groupby('contact').agg({
            'y': lambda x: (x == 'yes').mean(),
            'duration': 'mean'
        }).round(3)
        
        fig = px.bar(x=contact_stats.index, y=contact_stats['y'], 
                    title="Success Rate by Contact Method")
        st.plotly_chart(fig, use_container_width=True)
        
        # Previous outcome impact
        st.subheader("Previous Campaign Outcome Impact")
        poutcome_stats = filtered_df.groupby('poutcome')['y'].apply(lambda x: (x == 'yes').mean()).sort_values(ascending=False)
        fig = px.bar(x=poutcome_stats.index, y=poutcome_stats.values, 
                    title="Success Rate by Previous Outcome")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature 7: Business Intelligence
    elif page == "💼 Business Intelligence":
        st.header("Business Intelligence Dashboard")
        
        # Key metrics
        st.subheader("Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_contacts = len(filtered_df)
            st.metric("Total Contacts", total_contacts)
        
        with col2:
            successful_contacts = len(filtered_df[filtered_df['y'] == 'yes'])
            st.metric("Successful Contacts", successful_contacts)
        
        with col3:
            avg_duration = filtered_df['duration'].mean()
            st.metric("Avg Call Duration", f"{avg_duration:.1f} sec")
        
        with col4:
            avg_campaigns = filtered_df['campaign'].mean()
            st.metric("Avg Campaigns per Client", f"{avg_campaigns:.1f}")
        
        # ROI Analysis
        st.subheader("Return on Investment Analysis")
        
        # Assume cost per call and revenue per subscription
        cost_per_call = 5  # euros
        revenue_per_subscription = 1000  # euros
        
        total_cost = total_contacts * cost_per_call
        total_revenue = successful_contacts * revenue_per_subscription
        roi = ((total_revenue - total_cost) / total_cost) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Cost", f"€{total_cost:,.0f}")
        with col2:
            st.metric("Total Revenue", f"€{total_revenue:,.0f}")
        with col3:
            st.metric("ROI", f"{roi:.1f}%")
        
        # Recommendations
        st.subheader("Strategic Recommendations")
        
        # Find best performing segments
        best_job = filtered_df.groupby('job')['y'].apply(lambda x: (x == 'yes').mean()).idxmax()
        best_marital = filtered_df.groupby('marital')['y'].apply(lambda x: (x == 'yes').mean()).idxmax()
        
        st.write(f"**Target Demographics:**")
        st.write(f"- Focus on {best_job} professionals")
        st.write(f"- Prioritize {best_marital} customers")
        st.write(f"- Optimal call duration: {filtered_df[filtered_df['y'] == 'yes']['duration'].mean():.0f} seconds")
        
        # Risk analysis
        st.subheader("Risk Analysis")
        high_campaign_clients = filtered_df[filtered_df['campaign'] > 5]
        st.write(f"**High-touch clients (>5 campaigns):** {len(high_campaign_clients)} ({len(high_campaign_clients)/len(filtered_df)*100:.1f}%)")
        st.write(f"**Success rate for high-touch clients:** {(high_campaign_clients['y'] == 'yes').mean():.1%}")
    
    # Feature 8: Data Quality Assessment
    elif page == "🛠️ Data Quality Assessment":
        st.header("Data Quality Assessment")
        
        # Missing data analysis
        st.subheader("Missing Data Analysis")
        missing_data = filtered_df.isnull().sum()
        missing_percent = (missing_data / len(filtered_df)) * 100
        
        quality_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percent.values
        })
        
        st.dataframe(quality_df)
        
        # Data distribution analysis
        st.subheader("Data Distribution Quality")
        
        # Check for outliers in numeric columns
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in filtered_df.columns:
                Q1 = filtered_df[col].quantile(0.25)
                Q3 = filtered_df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = filtered_df[(filtered_df[col] < Q1 - 1.5 * IQR) | (filtered_df[col] > Q3 + 1.5 * IQR)]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{col}:**")
                    st.write(f"- Outliers: {len(outliers)} ({len(outliers)/len(filtered_df)*100:.1f}%)")
                    st.write(f"- Range: {filtered_df[col].min()} to {filtered_df[col].max()}")
                
                with col2:
                    fig = px.box(filtered_df, y=col, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Data consistency checks
        st.subheader("Data Consistency Checks")
        
        # Check for negative balances
        negative_balance = len(filtered_df[filtered_df['balance'] < 0])
        st.write(f"**Negative balances:** {negative_balance} ({negative_balance/len(filtered_df)*100:.1f}%)")
        
        # Check for unrealistic ages
        unrealistic_age = len(filtered_df[(filtered_df['age'] < 18) | (filtered_df['age'] > 100)])
        st.write(f"**Unrealistic ages:** {unrealistic_age}")
        
        # Check for duration = 0 (unsuccessful calls)
        zero_duration = len(filtered_df[filtered_df['duration'] == 0])
        st.write(f"**Zero duration calls:** {zero_duration} ({zero_duration/len(filtered_df)*100:.1f}%)")
    
    # Download filtered data
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Download Data")
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_bank_marketing_data.csv',
        mime='text/csv',
    )

if __name__ == "__main__":
    main()
