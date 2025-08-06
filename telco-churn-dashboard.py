"""
Telco Customer Churn Analytics Dashboard
A comprehensive Streamlit dashboard with visualizations and ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    .stAlert {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📊 Telco Customer Churn Analytics Dashboard")
st.markdown("### Comprehensive Analysis of Customer Churn Patterns and Predictive Modeling")

# Sidebar
st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.markdown("---")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('TelcoCustomerChurn.csv')
    
    # Clean TotalCharges column
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
    
    # Create additional features
    df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                                labels=['0-1 Year', '1-2 Years', '2-4 Years', '4+ Years'])
    df['MonthlyChargesGroup'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 65, 95, 120], 
                                        labels=['Low', 'Medium', 'High', 'Very High'])
    
    return df

# Load data
df = load_data()

# Sidebar filters
st.sidebar.subheader("🔍 Data Filters")
selected_gender = st.sidebar.multiselect("Gender", options=df['gender'].unique(), default=df['gender'].unique())
selected_contract = st.sidebar.multiselect("Contract Type", options=df['Contract'].unique(), default=df['Contract'].unique())
selected_internet = st.sidebar.multiselect("Internet Service", options=df['InternetService'].unique(), default=df['InternetService'].unique())
tenure_range = st.sidebar.slider("Tenure (months)", 0, int(df['tenure'].max()), (0, int(df['tenure'].max())))

# Filter data
filtered_df = df[
    (df['gender'].isin(selected_gender)) &
    (df['Contract'].isin(selected_contract)) &
    (df['InternetService'].isin(selected_internet)) &
    (df['tenure'].between(tenure_range[0], tenure_range[1]))
]

# Key Metrics
st.markdown("## 📈 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_customers = len(filtered_df)
    st.metric("Total Customers", f"{total_customers:,}")

with col2:
    churn_rate = (filtered_df['Churn'] == 'Yes').mean() * 100
    st.metric("Churn Rate", f"{churn_rate:.1f}%", delta=f"{churn_rate - 26.5:.1f}%")

with col3:
    avg_tenure = filtered_df['tenure'].mean()
    st.metric("Avg Tenure", f"{avg_tenure:.1f} months")

with col4:
    avg_monthly = filtered_df['MonthlyCharges'].mean()
    st.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")

with col5:
    total_revenue = filtered_df['TotalCharges'].sum()
    st.metric("Total Revenue", f"${total_revenue/1e6:.2f}M")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "🤖 ML Models", "🎯 Interactive Analysis", "📋 Data Overview"])

with tab1:
    st.markdown("## 📊 Data Visualizations")
    
    # Row 1: 2 visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Visualization 1: Churn Distribution
        st.subheader("1. Churn Distribution")
        churn_counts = filtered_df['Churn'].value_counts()
        fig1 = px.pie(values=churn_counts.values, names=churn_counts.index, 
                      title="Customer Churn Distribution",
                      color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'},
                      hole=0.4)
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Visualization 2: Churn by Contract Type
        st.subheader("2. Churn Rate by Contract Type")
        churn_by_contract = filtered_df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean() * 100).reset_index()
        churn_by_contract.columns = ['Contract', 'Churn Rate (%)']
        fig2 = px.bar(churn_by_contract, x='Contract', y='Churn Rate (%)',
                      title="Churn Rate by Contract Type",
                      color='Churn Rate (%)',
                      color_continuous_scale='RdYlGn_r')
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Row 2: 2 visualizations
    col3, col4 = st.columns(2)
    
    with col3:
        # Visualization 3: Monthly Charges Distribution by Churn
        st.subheader("3. Monthly Charges Distribution")
        fig3 = px.box(filtered_df, x='Churn', y='MonthlyCharges',
                      title="Monthly Charges by Churn Status",
                      color='Churn',
                      color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'})
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        # Visualization 4: Tenure Distribution by Churn
        st.subheader("4. Customer Tenure Analysis")
        fig4 = px.histogram(filtered_df, x='tenure', color='Churn',
                           title="Tenure Distribution by Churn Status",
                           nbins=30,
                           color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'},
                           barmode='overlay',
                           opacity=0.7)
        fig4.update_layout(xaxis_title="Tenure (months)", yaxis_title="Count")
        st.plotly_chart(fig4, use_container_width=True)
    
    # Row 3: 2 visualizations
    col5, col6 = st.columns(2)
    
    with col5:
        # Visualization 5: Payment Method Analysis
        st.subheader("5. Churn by Payment Method")
        payment_churn = filtered_df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x=='Yes').mean() * 100).reset_index()
        payment_churn.columns = ['Payment Method', 'Churn Rate (%)']
        fig5 = px.bar(payment_churn, y='Payment Method', x='Churn Rate (%)',
                      title="Churn Rate by Payment Method",
                      orientation='h',
                      color='Churn Rate (%)',
                      color_continuous_scale='Viridis')
        fig5.update_layout(showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)
    
    with col6:
        # Visualization 6: Internet Service Impact
        st.subheader("6. Internet Service Type Analysis")
        internet_churn = filtered_df.groupby(['InternetService', 'Churn']).size().reset_index(name='Count')
        fig6 = px.sunburst(internet_churn, path=['InternetService', 'Churn'], values='Count',
                          title="Customer Distribution by Internet Service",
                          color='Count',
                          color_continuous_scale='Blues')
        st.plotly_chart(fig6, use_container_width=True)
    
    # Row 4: 2 visualizations
    col7, col8 = st.columns(2)
    
    with col7:
        # Visualization 7: Senior Citizen Analysis
        st.subheader("7. Age Demographics Impact")
        senior_analysis = filtered_df.groupby(['SeniorCitizen', 'Churn']).size().unstack(fill_value=0)
        senior_analysis.index = ['Non-Senior', 'Senior']
        fig7 = go.Figure()
        fig7.add_trace(go.Bar(name='No Churn', x=senior_analysis.index, y=senior_analysis['No'], marker_color='#4ECDC4'))
        fig7.add_trace(go.Bar(name='Churn', x=senior_analysis.index, y=senior_analysis['Yes'], marker_color='#FF6B6B'))
        fig7.update_layout(title="Churn by Senior Citizen Status", barmode='group')
        st.plotly_chart(fig7, use_container_width=True)
    
    with col8:
        # Visualization 8: Service Adoption Heatmap
        st.subheader("8. Service Adoption Patterns")
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        service_churn = []
        for service in services:
            churn_rate = filtered_df[filtered_df[service] == 'Yes']['Churn'].apply(lambda x: x == 'Yes').mean() * 100
            service_churn.append(churn_rate)
        
        fig8 = go.Figure(data=go.Heatmap(
            z=[service_churn],
            x=services,
            y=['Churn Rate (%)'],
            colorscale='RdYlGn_r',
            text=[[f'{rate:.1f}%' for rate in service_churn]],
            texttemplate='%{text}',
            textfont={"size": 12},
            showscale=True
        ))
        fig8.update_layout(title="Churn Rate by Service Adoption", height=250)
        st.plotly_chart(fig8, use_container_width=True)

with tab2:
    st.markdown("## 🤖 Machine Learning Models")
    
    # Prepare data for ML
    @st.cache_data
    def prepare_ml_data(df):
        # Create a copy for ML
        ml_df = df.copy()
        
        # Drop unnecessary columns
        ml_df = ml_df.drop(['customerID', 'TenureGroup', 'MonthlyChargesGroup'], axis=1, errors='ignore')
        
        # Encode categorical variables
        label_encoders = {}
        categorical_columns = ml_df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != 'Churn':
                le = LabelEncoder()
                ml_df[col] = le.fit_transform(ml_df[col])
                label_encoders[col] = le
        
        # Prepare features and target
        X = ml_df.drop('Churn', axis=1)
        y = (ml_df['Churn'] == 'Yes').astype(int)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, X.columns
    
    X, y, feature_names = prepare_ml_data(filtered_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Model training section
    st.subheader("🎯 Model Training & Evaluation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Model Selection")
        model_type = st.selectbox("Choose Model", 
                                  ["Random Forest", "Gradient Boosting", "Logistic Regression"])
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                # Train selected model
                if model_type == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
                elif model_type == "Gradient Boosting":
                    model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
                else:
                    model = LogisticRegression(random_state=42, max_iter=1000)
                
                model.fit(X_train, y_train)
                
                # Store model in session state
                st.session_state['trained_model'] = model
                st.session_state['model_type'] = model_type
                st.success(f"✅ {model_type} trained successfully!")
    
    with col2:
        if 'trained_model' in st.session_state:
            model = st.session_state['trained_model']
            model_type = st.session_state['model_type']
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            st.markdown(f"### {model_type} Performance Metrics")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.3f}")
            
            with metric_col2:
                precision = precision_score(y_test, y_pred)
                st.metric("Precision", f"{precision:.3f}")
            
            with metric_col3:
                recall = recall_score(y_test, y_pred)
                st.metric("Recall", f"{recall:.3f}")
            
            with metric_col4:
                f1 = f1_score(y_test, y_pred)
                st.metric("F1 Score", f"{f1:.3f}")
    
    # Model Comparison
    st.markdown("### 📊 Model Comparison")
    
    if st.button("🔄 Compare All Models"):
        with st.spinner("Training all models..."):
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            results = []
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                results.append({
                    'Model': name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1 Score': f1_score(y_test, y_pred),
                    'ROC AUC': roc_auc_score(y_test, y_pred_proba)
                })
            
            results_df = pd.DataFrame(results)
            
            # Display comparison table
            st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']),
                        use_container_width=True)
            
            # Visualization of model comparison
            fig_comparison = go.Figure()
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
            for _, row in results_df.iterrows():
                fig_comparison.add_trace(go.Scatterpolar(
                    r=[row[metric] for metric in metrics],
                    theta=metrics,
                    fill='toself',
                    name=row['Model']
                ))
            
            fig_comparison.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Model Performance Comparison"
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Feature Importance (if Random Forest or Gradient Boosting is selected)
    if 'trained_model' in st.session_state:
        model = st.session_state['trained_model']
        model_type = st.session_state['model_type']
        
        if model_type in ['Random Forest', 'Gradient Boosting']:
            st.markdown("### 🎯 Feature Importance Analysis")
            
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig_importance = px.bar(feature_importance, x='Importance', y='Feature',
                                   orientation='h',
                                   title=f"Top 10 Most Important Features ({model_type})",
                                   color='Importance',
                                   color_continuous_scale='Viridis')
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)
    
    # Confusion Matrix
    if 'trained_model' in st.session_state:
        st.markdown("### 🎯 Confusion Matrix")
        
        model = st.session_state['trained_model']
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No Churn', 'Predicted Churn'],
            y=['Actual No Churn', 'Actual Churn'],
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='Blues'
        ))
        
        fig_cm.update_layout(
            title=f'Confusion Matrix - {st.session_state["model_type"]}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)

with tab3:
    st.markdown("## 🎯 Interactive Customer Analysis")
    
    # Interactive Scatter Plot
    st.subheader("Interactive Churn Analysis Explorer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_axis = st.selectbox("X-axis", 
                             ['MonthlyCharges', 'TotalCharges', 'tenure', 'AvgChargesPerMonth'])
    
    with col2:
        y_axis = st.selectbox("Y-axis", 
                             ['TotalCharges', 'MonthlyCharges', 'tenure', 'AvgChargesPerMonth'])
    
    with col3:
        color_by = st.selectbox("Color by", 
                               ['Churn', 'Contract', 'InternetService', 'PaymentMethod', 'gender'])
    
    # Create interactive scatter plot
    fig_interactive = px.scatter(filtered_df, 
                                 x=x_axis, 
                                 y=y_axis,
                                 color=color_by,
                                 size='MonthlyCharges',
                                 hover_data=['customerID', 'tenure', 'Contract', 'Churn'],
                                 title=f"Interactive Analysis: {x_axis} vs {y_axis}",
                                 color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'} if color_by == 'Churn' else None,
                                 opacity=0.6)
    
    fig_interactive.update_layout(height=500)
    st.plotly_chart(fig_interactive, use_container_width=True)
    
    # Customer Segmentation Analysis
    st.subheader("🎨 Customer Segmentation Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Segment by multiple factors
        st.markdown("#### Segment Definition")
        segment_contract = st.multiselect("Contract Type", 
                                         options=df['Contract'].unique(),
                                         default=['Month-to-month'])
        segment_tenure = st.slider("Tenure Range", 0, 72, (0, 12))
        segment_charges = st.slider("Monthly Charges Range", 
                                   float(df['MonthlyCharges'].min()),
                                   float(df['MonthlyCharges'].max()),
                                   (20.0, 50.0))
    
    with col2:
        # Apply segmentation
        segment_df = df[
            (df['Contract'].isin(segment_contract)) &
            (df['tenure'].between(segment_tenure[0], segment_tenure[1])) &
            (df['MonthlyCharges'].between(segment_charges[0], segment_charges[1]))
        ]
        
        if len(segment_df) > 0:
            st.markdown("#### Segment Statistics")
            st.metric("Segment Size", f"{len(segment_df):,} customers")
            st.metric("Segment Churn Rate", f"{(segment_df['Churn'] == 'Yes').mean() * 100:.1f}%")
            st.metric("Avg Monthly Revenue", f"${segment_df['MonthlyCharges'].mean():.2f}")
        else:
            st.warning("No customers match the selected criteria")
    
    # Dynamic Risk Score Calculator
    st.subheader("🔮 Customer Risk Score Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        calc_tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        calc_monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=70.0)
        calc_contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    
    with col2:
        calc_internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        calc_payment = st.selectbox("Payment Method", 
                                   ['Electronic check', 'Mailed check', 
                                    'Bank transfer (automatic)', 'Credit card (automatic)'])
        calc_paperless = st.selectbox("Paperless Billing", ['Yes', 'No'])
    
    with col3:
        # Calculate risk score based on patterns
        risk_score = 50  # Base score
        
        # Adjust based on tenure
        if calc_tenure < 6:
            risk_score += 20
        elif calc_tenure < 12:
            risk_score += 10
        elif calc_tenure > 24:
            risk_score -= 20
        
        # Adjust based on contract
        if calc_contract == 'Month-to-month':
            risk_score += 25
        elif calc_contract == 'Two year':
            risk_score -= 25
        
        # Adjust based on payment method
        if calc_payment == 'Electronic check':
            risk_score += 15
        elif 'automatic' in calc_payment:
            risk_score -= 10
        
        # Adjust based on charges
        if calc_monthly > 80:
            risk_score += 10
        
        # Ensure score is between 0 and 100
        risk_score = max(0, min(100, risk_score))
        
        # Display risk score with color coding
        if risk_score < 30:
            color = "🟢"
            risk_level = "Low Risk"
        elif risk_score < 60:
            color = "🟡"
            risk_level = "Medium Risk"
        else:
            color = "🔴"
            risk_level = "High Risk"
        
        st.markdown("#### Risk Assessment")
        st.metric("Risk Score", f"{risk_score}/100")
        st.markdown(f"### {color} {risk_level}")
        
        # Recommendations
        st.markdown("#### Retention Recommendations")
        if risk_score > 60:
            st.info("• Offer contract upgrade incentive\n• Provide loyalty discount\n• Assign account manager")
        elif risk_score > 30:
            st.info("• Send satisfaction survey\n• Offer service bundle\n• Regular check-ins")
        else:
            st.success("• Maintain current relationship\n• Consider for upselling\n• Reward loyalty")

with tab4:
    st.markdown("## 📋 Data Overview & Insights")
    
    # Data Quality Report
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Statistics")
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Total Features:** {len(df.columns)}")
        st.write(f"**Numerical Features:** {len(df.select_dtypes(include=[np.number]).columns)}")
        st.write(f"**Categorical Features:** {len(df.select_dtypes(include=['object']).columns)}")
        st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values check
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.warning(f"⚠️ Found {missing_values.sum()} missing values")
        else:
            st.success("✅ No missing values detected")
    
    with col2:
        st.subheader("💡 Key Insights")
        
        # Calculate key insights
        high_risk_segment = df[(df['Contract'] == 'Month-to-month') & 
                               (df['tenure'] < 12) & 
                               (df['PaymentMethod'] == 'Electronic check')]
        
        loyal_customers = df[df['tenure'] > 48]
        
        st.info(f"""
        **🎯 High Risk Segment:**
        - Size: {len(high_risk_segment):,} customers ({len(high_risk_segment)/len(df)*100:.1f}%)
        - Churn Rate: {(high_risk_segment['Churn'] == 'Yes').mean() * 100:.1f}%
        
        **💎 Loyal Customers (4+ years):**
        - Size: {len(loyal_customers):,} customers
        - Churn Rate: {(loyal_customers['Churn'] == 'Yes').mean() * 100:.1f}%
        
        **📈 Revenue Impact:**
        - Lost Revenue from Churn: ${df[df['Churn'] == 'Yes']['MonthlyCharges'].sum():,.2f}/month
        - Avg Customer Lifetime Value: ${df['TotalCharges'].mean():,.2f}
        """)
    
    # Detailed Data View
    st.subheader("🔍 Detailed Data Explorer")
    
    # Filter options for data view
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_churn = st.selectbox("Show Churn Status", ['All', 'Yes', 'No'])
    
    with col2:
        sort_by = st.selectbox("Sort by", df.columns.tolist())
    
    with col3:
        sort_order = st.selectbox("Order", ['Ascending', 'Descending'])
    
    with col4:
        num_rows = st.number_input("Rows to display", min_value=5, max_value=100, value=10)
    
    # Apply filters to display
    display_df = filtered_df.copy()
    
    if show_churn != 'All':
        display_df = display_df[display_df['Churn'] == show_churn]
    
    display_df = display_df.sort_values(by=sort_by, ascending=(sort_order == 'Ascending'))
    
    # Display the data
    st.dataframe(display_df.head(num_rows), use_container_width=True)
    
    # Export functionality
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Filtered Data (CSV)",
            data=csv,
            file_name='filtered_telco_data.csv',
            mime='text/csv'
        )
    
    with col2:
        # Summary statistics
        summary_stats = filtered_df.describe().to_csv()
        st.download_button(
            label="📊 Download Summary Statistics",
            data=summary_stats,
            file_name='telco_summary_stats.csv',
            mime='text/csv'
        )
    
    with col3:
        # Churn analysis report
        churn_report = f"""Telco Customer Churn Analysis Report
        Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        OVERALL METRICS:
        - Total Customers: {len(filtered_df):,}
        - Overall Churn Rate: {(filtered_df['Churn'] == 'Yes').mean() * 100:.2f}%
        - Average Tenure: {filtered_df['tenure'].mean():.1f} months
        - Average Monthly Charges: ${filtered_df['MonthlyCharges'].mean():.2f}
        
        HIGH RISK SEGMENTS:
        - Month-to-month contracts: {(filtered_df[filtered_df['Contract'] == 'Month-to-month']['Churn'] == 'Yes').mean() * 100:.1f}% churn
        - Electronic check payments: {(filtered_df[filtered_df['PaymentMethod'] == 'Electronic check']['Churn'] == 'Yes').mean() * 100:.1f}% churn
        - New customers (<6 months): {(filtered_df[filtered_df['tenure'] < 6]['Churn'] == 'Yes').mean() * 100:.1f}% churn
        
        RECOMMENDATIONS:
        1. Focus retention efforts on month-to-month contract customers
        2. Incentivize automatic payment methods
        3. Implement early engagement programs for new customers
        4. Develop loyalty programs for long-tenure customers
        """
        
        st.download_button(
            label="📑 Download Analysis Report",
            data=churn_report,
            file_name='telco_churn_report.txt',
            mime='text/plain'
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 Telco Customer Churn Analytics Dashboard | Built with Streamlit & Plotly | 
    Data contains {len(df):,} customer records</p>
</div>
""".format(len(df)), unsafe_allow_html=True)

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About This Dashboard")
st.sidebar.info("""
This comprehensive dashboard provides:

**📊 8 Key Visualizations:**
- Churn distribution analysis
- Contract type impact
- Payment method analysis
- Service adoption patterns
- And more...

**🤖 3 ML Models:**
- Random Forest Classifier
- Gradient Boosting
- Logistic Regression

**🎯 Interactive Features:**
- Dynamic filtering
- Risk score calculator
- Customer segmentation
- Real-time analysis

**📈 Business Insights:**
- Identify high-risk customers
- Understand churn drivers
- Revenue impact analysis
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Quick Start Guide")
st.sidebar.markdown("""
1. **Explore Visualizations**: View 8 comprehensive charts in the Visualizations tab
2. **Train ML Models**: Compare 3 different algorithms for churn prediction
3. **Interactive Analysis**: Use the dynamic explorer to find patterns
4. **Export Results**: Download filtered data and reports
""")

# Display current filter status
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Current Filters")
st.sidebar.write(f"**Records displayed:** {len(filtered_df):,} / {len(df):,}")
st.sidebar.write(f"**Filtered Churn Rate:** {(filtered_df['Churn'] == 'Yes').mean() * 100:.1f}%")