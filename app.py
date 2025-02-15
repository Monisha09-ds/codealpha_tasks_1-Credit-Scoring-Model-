import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier

st.title("Credit Scorecard Analysis")


uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    # Filter necessary columns
    if 'loan_status' in df.columns:
        loan_filter = df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])
        df = df[loan_filter]
        df['Late_Loan'] = df['loan_status'].apply(lambda x: 0 if x == 'Fully Paid' else 1)
        
        
        drop_cols = ['id', 'member_id', 'loan_status', 'url', 'zip_code', 'policy_code', 'application_type']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
        
        
        df = df.fillna(df.mean(numeric_only=True))
        
        
        features = df.columns.difference(['Late_Loan'])
        X = df[features]
        y = df['Late_Loan']
        
       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
        
        logreg = LogisticRegression(max_iter=500)
        logreg.fit(X_train, y_train)
        logreg_acc = logreg.score(X_test, y_test)
        y_pred_log = logreg.predict(X_test)
        
       
        dtree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=100)
        dtree.fit(X_train, y_train)
        dtree_acc = dtree.score(X_test, y_test)
        y_pred_tree = dtree.predict(X_test)
        
        
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        xgb_acc = xgb.score(X_test, y_test)
        y_pred_xgb = xgb.predict(X_test)
        
    
        st.write("### Model Performance")
        st.write(f"**Logistic Regression Accuracy:** {logreg_acc:.2f}")
        st.write(f"**Decision Tree Accuracy:** {dtree_acc:.2f}")
        st.write(f"**XGBoost Accuracy:** {xgb_acc:.2f}")
        
        st.write("#### Confusion Matrix - Logistic Regression")
        st.write(pd.DataFrame(confusion_matrix(y_test, y_pred_log), columns=['Pred 0', 'Pred 1'], index=['Actual 0', 'Actual 1']))
        
        st.write("#### Confusion Matrix - Decision Tree")
        st.write(pd.DataFrame(confusion_matrix(y_test, y_pred_tree), columns=['Pred 0', 'Pred 1'], index=['Actual 0', 'Actual 1']))
        
        st.write("#### Confusion Matrix - XGBoost")
        st.write(pd.DataFrame(confusion_matrix(y_test, y_pred_xgb), columns=['Pred 0', 'Pred 1'], index=['Actual 0', 'Actual 1']))
        
        
        st.write("#### Classification Report - Logistic Regression")
        st.text(classification_report(y_test, y_pred_log))
        
        st.write("#### Classification Report - Decision Tree")
        st.text(classification_report(y_test, y_pred_tree))
        
        st.write("#### Classification Report - XGBoost")
        st.text(classification_report(y_test, y_pred_xgb))
        
        
        logit_roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
        tree_roc_auc = roc_auc_score(y_test, dtree.predict_proba(X_test)[:, 1])
        xgb_roc_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
        
        fpr_log, tpr_log, _ = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
        fpr_tree, tpr_tree, _ = roc_curve(y_test, dtree.predict_proba(X_test)[:, 1])
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb.predict_proba(X_test)[:, 1])
        
        fig, ax = plt.subplots()
        ax.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {logit_roc_auc:.2f})')
        ax.plot(fpr_tree, tpr_tree, label=f'Decision Tree (AUC = {tree_roc_auc:.2f})')
        ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("The dataset does not contain the 'loan_status' column.")

