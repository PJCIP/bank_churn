import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_curve,precision_recall_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.preprocessing import LabelEncoder, StandardScaler # Added StandardScaler
import warnings
import numpy as np
warnings.filterwarnings("ignore") #suppress warnings

def log_transform(x):
    # Handle non-positive values by adding a small constant
    x_adjusted = np.where(x <= 0, 1e-6, x)
    return np.log(x_adjusted)


# Load the pre-trained model and data
try:
    with open('bank_churn_mod_v2.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
     st.warning("Model pickle file not found. Using a dummy model for demonstration.")


try:
    X_test = pd.read_csv('x_test.csv')
    data = pd.read_csv('x_test.csv')
except FileNotFoundError:
    st.warning("x_test.csv file not found. Using dummy data for demonstration.")

# Page setup
st.set_page_config(page_title="Bank Churn Model Portfolio", layout="wide")
st.title("Bank Churn Model Portfolio")

# Sidebar navigation
st.sidebar.title("Table of Contents")
page = st.sidebar.radio("Go to", ["About Dataset","Model Performance", "SHAP Analysis", "Individual Prediction"])

#0. About Dataset
if page == "About Dataset":
    st.header("About Dataset")
    # Model Evaluation
    st.markdown('''
    ### 
    **Description:** Account information for 10,000 customers at a European bank, including details on their credit score, balance, products, and whether they have churned.
    
    **Data Source:** Bank_Churn.xlsx with two sheets (Customer_info and Accounts_info)
    
    **No. of records :** 10,000 records
    
    **No. of Fields :** 13

    With this data, had created a visualization to come up with some basic insights.

    **Visualization:** https://mavenanalytics.io/project/32515            
    
    **Github:** https://github.com/PJCIP/bank_churn
                
    To interact feel free to download the powerbi desktop version from my github Mentioned above 

    **Key insights:**
                
    - Out of 10,000 customers, 2,037 had exited hence dataset is an imbalanced dataset 
    - Female contribute 60%  of exit rate.
    - Senior Age Group tend to exit a lot out of contributing 50% of the exit
    - People who purchase 2+ products tend to exit.
                
        
    ''')

# 1. Model Performance Page
if page == "Model Performance":
    st.header("Model Performance")

    # Model Evaluation
    if 'Exited' in X_test.columns:  # check if target is in the provided dataset.
        y_test = X_test['Exited']
        X_test = X_test.drop('Exited', axis=1)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1

        # Key Metrics
        st.subheader("Performance by Class")
        report = classification_report(y_test, y_pred, output_dict=True)
        precision_class_0 = report['0']['precision']
        recall_class_0 = report['0']['recall']
        f1_score_class_0 = report['0']['f1-score']
        precision_class_1 = report['1']['precision']
        recall_class_1 = report['1']['recall']
        f1_score_class_1 = report['1']['f1-score']
        accuracy = report['accuracy']
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)

        col1, col2, col3 = st.columns(3)
        
        col1.metric("F1-Score (Class 0)", f"{f1_score_class_0:.2f}")
        col2.metric("Precision (Class 0)", f"{precision_class_0:.2f}")
        col3.metric("Recall (Class 0)", f"{recall_class_0:.2f}")
        col1, col2, col3 = st.columns(3)
        col1.metric("F1-Score (Class 1)", f"{f1_score_class_1:.2f}")
        col2.metric("Precision (Class 1)", f"{precision_class_1:.2f}")
        col3.metric("Recall (Class 1)", f"{recall_class_1:.2f}")
        
        st.subheader("Overall Performance ")
        col1, col2, col3,col4,col5 = st.columns(5)
        col1.metric("Accuracy", f"{accuracy:.2f}")
        col2.metric("F1_score", f"{f1_score(y_test, y_pred):.4f}")
        col3.metric("Precisison", f"{precision_score(y_test, y_pred):.4f}")
        col4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        col5.metric("Specificity", f"{specificity:.4f}")

        st.markdown("""**Inference**:

        Here, is the catch you might have noticed that the recall mentioning we are 70% of the churn pattern is recognized 
    but there exist only 50% chance that he/she will actually exit.
        
    This model is designed with objective that the cost of losing a customer (false negative) is significantly higher 
    than the cost of a wasted retention effort (false positive), hence satisfied with  model's recall (70%). 
    
    Meaning, Bank is ready to  risk spending resources on some who weren't going to leave to ensure you catch more of those who were.
        
    Just in case the cost of a wasted retention effort is very high (e.g., very expensive campaigns, risk of alienating customers). 
                    
    In this case, just be setting threshold as .69. Would set the precision to 70% and recall to be 50%.
     """)
        
        # Plots
        st.subheader("Precision-Recall and ROC Curves")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ax1.plot(recall, precision, label='Precision-Recall')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curve')
        ax1.legend()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random classifier
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()

        st.pyplot(fig)

        st.markdown("""**Inference**:

                     An AUC of 0.8543 suggests that the model is effective at identifying customers at risk of churn. 
    This is a positive result, indicating that the model has the potential to be valuable for your bank in implementing 
    churn prevention strategies.
                    
                     """)
    else:
        st.write(
            "Target variable 'Exited' not found in x_test.csv.  Model performance metrics and plots are not available.")

# 2. SHAP Plots Page
elif page == "SHAP Analysis":
    st.header("SHAP Analysis")
    
    if 'Exited' in X_test.columns:  # check if target is in the provided dataset.
        log_transform_cols  = ['EstimatedSalary', 'Balance']
        one_hot_labels = ['Geography', 'Age_Group_Custom']
        labels_endcode = ['Gender']
        feature_names_after_onehot = model.named_steps['preprocessor'].named_transformers_['onehot_encode']['onehot_encoder'].get_feature_names_out(input_features=one_hot_labels)
        onehot_cols = list(feature_names_after_onehot)
        # Get column names after transformation
        original_columns = list(X_test.columns)
        feature_names = (
            labels_endcode
            +onehot_cols
            + log_transform_cols
            + [col for col in original_columns if col not in labels_endcode + onehot_cols + log_transform_cols + one_hot_labels + ['Exited']]
        )
        
        # print(feature_names)
        # feature_names = log_transform_cols +[col for col in input_data.columns if col not in log_transform_cols+['Exited']]
        print(X_test.columns,print(X_test.shape))

        print(model.named_steps['preprocessor'].transform(X_test).shape)
        print(len(feature_names))
        print(feature_names)
        X_test_processed = pd.DataFrame(model.named_steps['preprocessor'].transform(X_test),columns=feature_names)

        # Explain the model's predictions using SHAP
        explainer = shap.TreeExplainer(model.named_steps["model"]) 
        shap_values = explainer.shap_values(X_test_processed)

          # SHAP Bar Plot
        st.subheader("SHAP Bar Plot")
        col1, col2 = st.columns([0.7, 0.3])  # Adjust column widths as needed
        with col1:
            fig_bar = plt.figure()
            shap.summary_plot(shap_values, X_test_processed, plot_type="bar", show=False)
            st.pyplot(fig_bar)
        with col2:
            st.markdown("""
            ### Key Insights:
                        
            From the MEan Absolute Shap Plot, the top 5 performing features are:
                        
            i) Number of the Products \
                        
            ii) isActiveMember \
                        
            iii) Under Age_Group - Senior \
                        
            iv) Gender \
                        
            v) Under Geography - Germany"""
             ) 

        # SHAP Beeswarm Plot
        st.subheader("SHAP Beeswarm Plot")
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            fig_beeswarm = plt.figure()
            shap.summary_plot(shap_values, X_test_processed, show=False)
            st.pyplot(fig_beeswarm)
        with col2:
            st.markdown("""
            ### Key Insights:
                        
            Let's go through some of the features in your plot:

            #### NumOfProducts:
              - High values (red dots, likely meaning more products) tend to have a negative SHAP value, pushing the model's output lower.
              - Low values (blue dots, likely meaning fewer products) tend to have a positive SHAP value, pushing the model's output higher.
              -  This suggests that having **more products** is associated with a **lower model output**.
            
            #### IsActiveMember:

            - High values (red dots, likely meaning the customer is an active member) tend to have a positive SHAP value.
            - Low values (blue dots, likely meaning the customer is not an active member) tend to have a negative SHAP value.
            - This indicates that being **an active member increases** the model's output.
            
            #### Age_Group_Custom_Senior:

            - High values (red dots, indicating the customer belongs to the "Senior" age group) tend to have a positive SHAP value.
            - Low values (blue dots, indicating the customer does not belong to this group) tend to have a negative SHAP value.
            - This suggests that being in the **"Senior" age group increases** the model's output.
            
           #### Tenure:
            - The impact of Tenure (how long the customer has been with the bank) seems to be mostly centered around zero, indicating a relatively **weaker overall impact** compared to some other features.
            """
             )



    else:
        st.write("Target variable 'Exited' not found in x_test.csv. SHAP plots are not available.")

# 3. Prediction Page
elif page == "Individual Prediction":
    st.header("Individual Prediction")

    # Input tabs
    with st.form("prediction_form"):
        Geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
        Gender = st.selectbox("Gender", ['Male', 'Female'])
        Age_Group_Custom = st.selectbox("Age_Group_Custom",
                                         ['Middle-Aged', 'Adult', 'Senior', 'Young Adult', 'Very Senior'])
        EstimatedSalary = st.number_input("EstimatedSalary", value=100000)
        Balance = st.number_input("Balance", value=50000)
        IsActiveMember = st.selectbox("IsActiveMember", [1, 0])
        Tenure = st.number_input("Tenure", value=5)
        NumOfProducts = st.selectbox("NumOfProducts", [1, 2, 3, 4])

        submit_button = st.form_submit_button("Make Prediction")

    # Prediction and Waterfall Plot
    if submit_button:
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame({
            'Geography': [Geography],
            'Gender': [Gender],
            'Age_Group_Custom': [Age_Group_Custom],
            'EstimatedSalary': [EstimatedSalary],
            'Balance': [Balance],
            'IsActiveMember': [IsActiveMember],
            'Tenure': [Tenure],
            'NumOfProducts': [NumOfProducts],
        })

        # # Preprocess the input data to match the training data format
        # input_data = pd.get_dummies(input_data, drop_first=True)
        # # Ensure the input data has the same columns as the training data
        # expected_cols = X_test.columns
        # for col in expected_cols:
        #     if col not in input_data.columns:
        #         input_data[col] = 0  # Add missing columns with 0 values
        # input_data = input_data[expected_cols]  # reorder

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Probability of class 1

        st.subheader("Prediction")
        if prediction == 1:
            st.write(f"The model predicts that this customer will churn. (Probability: {probability:.2f})")
        else:
            st.write(f"The model predicts that this customer will not churn. (Probability: {probability:.2f})")

        log_transform_cols  = ['EstimatedSalary', 'Balance']
        one_hot_labels = ['Geography', 'Age_Group_Custom']
        labels_endcode = ['Gender']
        feature_names_after_onehot = model.named_steps['preprocessor'].named_transformers_['onehot_encode']['onehot_encoder'].get_feature_names_out(input_features=one_hot_labels)
        onehot_cols = list(feature_names_after_onehot)
        # Get column names after transformation
        original_columns = list(input_data.columns)
        feature_names = (
            labels_endcode
            +onehot_cols
            + log_transform_cols
            + [col for col in original_columns if col not in labels_endcode + onehot_cols + log_transform_cols + one_hot_labels]
        )
        # print(feature_names)
        # # feature_names = log_transform_cols +[col for col in input_data.columns if col not in log_transform_cols+['Exited']]

        # print(input_data)
        # print(model.named_steps['preprocessor'].transform(input_data))
        # print(model.named_steps['preprocessor'].transform(input_data).shape)
        input_processed = pd.DataFrame(model.named_steps['preprocessor'].transform(input_data),columns=feature_names)

        # Waterfall Plot
        st.subheader("Feature Importance (Waterfall Plot)")
        explainer = shap.Explainer(model.named_steps["model"])  # Use training data for explainer
        # shap_values = explainer.shap_values(input_processed)
        shap_values = explainer(input_processed)
        fig_waterfall = plt.figure()
        shap.initjs()  # Initialize JavaScript for the waterfall plot
        shap.plots.waterfall(shap_values[0])
        # shap.force_plot(explainer.expected_value[1], shap_values[1], input_data, matplotlib=True, show=False)
        st.pyplot(fig_waterfall)
