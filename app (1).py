import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the trained model and scaler
try:
    xgb_model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    st.success("Model and scaler loaded successfully.")
except FileNotFoundError:
    st.error("Error: Model or scaler file not found. Please make sure 'xgb_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop() # Stop execution if files are not found

# Define the prediction function
def predict_churn(user_input):
    """
    Predicts churn based on user input.

    Args:
        user_input (dict): A dictionary containing user input features.

    Returns:
        float: The predicted churn probability.
    """
    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_input])

    # Ensure the input DataFrame has the same columns as the training data after preprocessing
    # This is a simplified approach. In a real app, you would need to
    # apply the same preprocessing steps (like one-hot encoding) to the input_df
    # as were applied to the training data.

    # For demonstration purposes, let's assume the input_df columns match the scaled training columns
    # after manual mapping of user inputs to the expected format.
    # A more robust solution would involve saving the list of columns after preprocessing
    # and ensuring the input_df matches that structure before scaling.

    # Example of handling categorical features - this needs to match your preprocessing
    # For simplicity here, we assume user input directly maps to features after one-hot encoding
    # This part needs to be adapted based on how you handled categorical features during training.
    # A common way is to create a template DataFrame with all possible columns after one-hot encoding.

    # --- Placeholder for robust preprocessing of user input ---
    # This section needs to be filled in based on the exact preprocessing
    # steps applied to the original data.
    # Example (assuming one-hot encoding was applied to categorical features):
    # user_input_processed = {}
    # for col in X.columns: # Assuming X is the DataFrame before splitting
    #     if col in user_input:
    #         user_input_processed[col] = user_input[col]
    #     else:
    #         user_input_processed[col] = 0 # Or appropriate default for dummy variables

    # input_df_processed = pd.DataFrame([user_input_processed])
    # input_scaled = scaler.transform(input_df_processed)
    # -------------------------------------------------------

    # Using the provided user_input_features structure,
    # we need to reconstruct the processed dataframe structure.
    # This requires access to the original column names and dtypes after one-hot encoding.
    # A more robust approach would save the preprocessor object or the list of columns.

    # For now, let's assume the input_df columns match the structure expected by the scaler
    # after implicit handling of categorical features in the user_input_features function.
    # **IMPORTANT:** This is a simplification for demonstration.
    # You need to ensure the user input is preprocessed identically to the training data.

    # Let's attempt to preprocess the input_df based on the original notebook's steps
    # This requires re-creating the preprocessing logic within the app or saving the preprocessor.

    # Re-applying a simplified preprocessing for the single input row:
    # Convert 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling' which were binary and LabelEncoded
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
         if col in input_df.columns:
             # Assuming 'Yes' maps to 1 and 'No' maps to 0
             input_df[col] = input_df[col].apply(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else x))

    # Apply one-hot encoding to other categorical features
    categorical_cols_to_onehot = ['gender', 'MultipleLines', 'InternetService',
                                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                  'TechSupport', 'StreamingTV', 'StreamingMovies',
                                  'Contract', 'PaymentMethod']

    # Need to handle missing columns after one-hot encoding
    input_df_processed = pd.get_dummies(input_df, columns=categorical_cols_to_onehot)

    # Align columns with training data - fill missing columns with 0
    # This requires knowing the columns of the training data after preprocessing (X_train.columns)
    # In a real app, you would save X_train.columns and load it here.
    # For now, we'll assume X_train.columns is available from the notebook context.
    try:
        for col in X_train.columns:
            if col not in input_df_processed.columns:
                input_df_processed[col] = 0
        # Ensure the order of columns is the same
        input_df_processed = input_df_processed[X_train.columns]

        input_scaled = scaler.transform(input_df_processed)
        churn_probability = xgb_model.predict_proba(input_scaled)[:, 1]
        return churn_probability[0]
    except NameError:
         st.error("Error: Training columns (X_train.columns) not available. Cannot preprocess user input correctly.")
         st.error("Please ensure X_train is defined in your training script and its columns are accessible or saved.")
         return None # Indicate failure


# Streamlit UI
st.title("Telco Customer Churn Prediction Dashboard")
st.write("This dashboard predicts customer churn based on their service usage and demographics.")

st.sidebar.header("User Input Features")

def user_input_features():
    gender = st.sidebar.selectbox('Gender',('Female','Male'))
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen',(0,1))
    Partner = st.sidebar.selectbox('Partner',('Yes','No'))
    Dependents = st.sidebar.selectbox('Dependents',('Yes','No'))
    tenure = st.sidebar.slider('Tenure', 0, 72, 1)
    PhoneService = st.sidebar.selectbox('Phone Service',('Yes','No'))
    MultipleLines = st.sidebar.selectbox('Multiple Lines',('No phone service','No','Yes'))
    InternetService = st.sidebar.selectbox('Internet Service',('DSL','Fiber optic','No'))
    OnlineSecurity = st.sidebar.selectbox('Online Security',('No','Yes','No internet service'))
    OnlineBackup = st.sidebar.selectbox('Online Backup',('No','Yes','No internet service'))
    DeviceProtection = st.sidebar.selectbox('Device Protection',('No','Yes','No internet service'))
    TechSupport = st.sidebar.selectbox('Tech Support',('No','Yes','No internet service'))
    StreamingTV = st.sidebar.selectbox('Streaming TV',('No','Yes','No internet service'))
    StreamingMovies = st.sidebar.selectbox('Streaming Movies',('No','Yes','No internet service'))
    Contract = st.sidebar.selectbox('Contract',('Month-to-month','One year','Two year'))
    PaperlessBilling = st.sidebar.selectbox('Paperless Billing',('Yes','No'))
    PaymentMethod = st.sidebar.selectbox('Payment Method',('Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'))
    MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0, 120.0, 50.0)
    TotalCharges = st.sidebar.slider('Total Charges', 0.0, 9000.0, 1000.0)

    data = {'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# Create a button to trigger the prediction
if st.button('Predict Churn'):
    churn_probability = predict_churn(input_df.iloc[0].to_dict())

    if churn_probability is not None: # Check if prediction was successful
        st.subheader('Prediction Result')
        st.write(f"Churn Probability: {churn_probability:.2f}")

        threshold = 0.5
        if churn_probability > threshold:
            st.write("Prediction: Customer is likely to churn.")
        else:
            st.write("Prediction: Customer is not likely to churn.")

        # SHAP explanation
        st.subheader('Explanation of the Prediction')
        explainer = shap.Explainer(xgb_model)
        # Need to use the scaled input for SHAP
        # Re-preprocess the input_df to get the scaled version for SHAP
        # This is a repeat of preprocessing in predict_churn - consider refactoring
        # For now, let's re-apply the preprocessing steps
        input_df_shap = input_df.copy()
        for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            if col in input_df_shap.columns:
                input_df_shap[col] = input_df_shap[col].apply(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else x))

        categorical_cols_to_onehot = ['gender', 'MultipleLines', 'InternetService',
                                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                      'TechSupport', 'StreamingTV', 'StreamingMovies',
                                      'Contract', 'PaymentMethod']
        input_df_shap_processed = pd.get_dummies(input_df_shap, columns=categorical_cols_to_onehot)

        try:
            # Align columns with training data
            for col in X_train.columns:
                if col not in input_df_shap_processed.columns:
                    input_df_shap_processed[col] = 0
            input_df_shap_processed = input_df_shap_processed[X_train.columns] # Ensure order

            input_scaled_shap = scaler.transform(input_df_shap_processed)
            shap_values = explainer(input_scaled_shap)

            shap.initjs()
            # Use st.pyplot for SHAP plots
            st_shap = shap.plots.force(shap_values[0], matplotlib=True)
            st.pyplot(st_shap.fig)
            plt.close(st_shap.fig) # Close the figure

        except NameError:
             st.warning("Cannot generate SHAP plot: Training columns (X_train.columns) not available.")
             st.warning("Please ensure X_train is defined in your training script and its columns are accessible or saved.")


# Model Performance Evaluation (XGBoost)
st.subheader("Model Performance Evaluation (XGBoost)")

try:
    # Assuming X_test_scaled and y_test are available from the training notebook context
    # In a real standalone app, you would load these from saved files.
    # For this interactive environment, we'll use the variables if they exist.

    # 1. Classification Report
    st.text("Classification Report:")
    # Need to ensure xgb_model.predict works with the available test data
    if 'X_test_scaled' in locals() and 'y_test' in locals():
        classification_rep = classification_report(y_test, xgb_model.predict(X_test_scaled))
        st.text(classification_rep)

        # 2. Confusion Matrix
        st.text("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, xgb_model.predict(X_test_scaled))
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("XGBoost Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(plt)
        plt.close() # Close the figure to free memory

        # 3. ROC Curve
        st.text("\nROC Curve:")
        xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr_xgb, tpr_xgb, label='XGBoost ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('XGBoost ROC Curve')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        plt.close() # Close the figure to free memory
    else:
        st.warning("Test data (X_test_scaled, y_test) not available in this context. Cannot display model performance metrics.")
        st.warning("In a real Streamlit app, you would need to load or generate this data.")

except NameError:
    st.warning("Error: Could not access variables for model performance evaluation.")


# Data Visualizations
# Load data for visualizations
try:
    df_viz = pd.read_csv("Telco-Customer-Churn.csv")
    df_viz.drop('customerID', axis=1, inplace=True)
    df_viz['TotalCharges'] = pd.to_numeric(df_viz['TotalCharges'], errors='coerce')
    df_viz['TotalCharges'].fillna(df_viz['TotalCharges'].median(), inplace=True)

    # Convert binary columns for visualization purposes if needed (depends on viz type)
    # For visualizations like countplot, scatterplot with hue, raw data is often fine.
    # For correlation heatmap, numerical/encoded data is needed.
    # Let's re-apply encoding/dummies for visualizations that require it.

    # Re-apply encoding for visualizations if needed
    binary_cols_viz = [col for col in df_viz.columns if df_viz[col].nunique() == 2]
    for col in binary_cols_viz:
        if df_viz[col].dtype == 'object': # Only encode if not already numerical
             df_viz[col] = LabelEncoder().fit_transform(df_viz[col])

    # Apply one-hot encoding for visualizations like heatmap
    df_viz_encoded = pd.get_dummies(df_viz)


    st.subheader("Data Visualizations")

    # 1. Monthly Charges Distribution
    st.write("Monthly Charges Distribution:")
    plt.figure(figsize=(8, 5))
    sns.histplot(df_viz['MonthlyCharges'], bins=30, kde=True)
    plt.title("Monthly Charges Distribution")
    st.pyplot(plt)
    plt.close()

    # 2. Total Charges Boxplot
    st.write("Total Charges Boxplot:")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df_viz['TotalCharges'])
    plt.title("Total Charges Boxplot")
    st.pyplot(plt)
    plt.close()

    # 3. Churn Count
    st.write("Churn Count:")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Churn', data=df_viz) # Use original Churn column
    plt.title("Churn Count")
    st.pyplot(plt)
    plt.close()

    # 4. Feature Correlation Heatmap (Optional - can be too large)
    # Using the encoded dataframe for heatmap
    # st.write("Feature Correlation Heatmap:")
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(df_viz_encoded.corr(), cmap='coolwarm', annot=False)
    # plt.title("Feature Correlation Heatmap")
    # st.pyplot(plt)
    # plt.close()

    # 5. Monthly vs Total Charges by Churn
    st.write("Monthly vs Total Charges by Churn:")
    plt.figure(figsize=(8, 5))
    # Use original Churn column for hue in scatterplot
    sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=df_viz)
    plt.title("Monthly vs Total Charges by Churn")
    st.pyplot(plt)
    plt.close()

    # 6. Churn Distribution (Pie Chart)
    st.write("Churn Distribution:")
    churn_labels = ['No Churn', 'Churn']
    churn_values = df_viz['Churn'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(churn_values, labels=churn_labels, autopct='%1.1f%%')
    plt.title("Churn Distribution")
    st.pyplot(plt)
    plt.close()

    # 7. Monthly Charges Distribution by Churn (Violin Plot)
    st.write("Monthly Charges Distribution by Churn:")
    plt.figure(figsize=(8, 5))
    # Use original Churn column for x in violinplot
    sns.violinplot(x='Churn', y='MonthlyCharges', data=df_viz)
    plt.title("Monthly Charges Distribution by Churn")
    st.pyplot(plt)
    plt.close()

except FileNotFoundError:
    st.error("Telco-Customer-Churn.csv not found. Please make sure the data file is in the correct directory.")
except Exception as e:
    st.error(f"An error occurred during visualization: {e}")


# Deployment Instructions
st.subheader("Deployment Instructions")

st.write("""
To deploy this Streamlit application, follow these steps:

1.  **Save the application file:**
    Make sure you have saved the Python code as `app.py`.

2.  **Save the model and scaler:**
    Ensure that the trained XGBoost model (`xgb_model.pkl`) and the StandardScaler (`scaler.pkl`) are saved in the same directory as `app.py`. If you trained your model in a separate notebook, make sure to save them using `joblib.dump()` and transfer them to the app directory.
