import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("❤️ Heart Disease Prediction System")
st.markdown("An interactive machine learning application for predicting heart disease risk")

# Load sample data (creating a synthetic dataset similar to Cleveland Heart Disease dataset)
@st.cache_data
def load_heart_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic heart disease dataset
    data = {
        'age': np.random.randint(25, 80, n_samples),
        'sex': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
        'chest_pain_type': np.random.choice([0, 1, 2, 3], n_samples),  # 0-3: Different types
        'resting_bp': np.random.normal(130, 20, n_samples),
        'cholesterol': np.random.normal(240, 50, n_samples),
        'fasting_bs': np.random.choice([0, 1], n_samples),  # 0: <=120, 1: >120
        'resting_ecg': np.random.choice([0, 1, 2], n_samples),
        'max_heart_rate': np.random.normal(150, 25, n_samples),
        'exercise_angina': np.random.choice([0, 1], n_samples),
        'oldpeak': np.random.exponential(1, n_samples),
        'st_slope': np.random.choice([0, 1, 2], n_samples)
    }
    
    # Create target variable with some logic
    heart_disease_prob = (
        (data['age'] - 25) / 55 * 0.3 +
        data['sex'] * 0.2 +
        data['chest_pain_type'] / 3 * 0.2 +
        (data['resting_bp'] - 90) / 100 * 0.1 +
        (data['cholesterol'] - 150) / 150 * 0.1 +
        data['exercise_angina'] * 0.1
    )
    
    data['target'] = (heart_disease_prob + np.random.normal(0, 0.2, n_samples) > 0.5).astype(int)
    
    df = pd.DataFrame(data)
    
    # Ensure realistic ranges
    df['resting_bp'] = df['resting_bp'].clip(80, 200)
    df['cholesterol'] = df['cholesterol'].clip(100, 400)
    df['max_heart_rate'] = df['max_heart_rate'].clip(60, 220)
    df['oldpeak'] = df['oldpeak'].clip(0, 6)
    
    return df

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Overview", "📊 Data Analysis", "🤖 Model Training", "🔮 Prediction", "📈 Model Comparison"]
)

# Load data
df = load_heart_data()

if page == "🏠 Overview":
    st.header("Heart Disease Prediction Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About This Application
        This application uses machine learning to predict the likelihood of heart disease based on various medical indicators.
        
        **Features:**
        - Interactive data exploration
        - Multiple ML model training and comparison
        - Real-time prediction interface
        - Comprehensive model evaluation
        
        **Dataset Information:**
        The dataset contains medical information including:
        - Age, sex, and chest pain type
        - Blood pressure and cholesterol levels
        - Heart rate and exercise-induced symptoms
        - ECG results and other cardiac indicators
        """)
    
    with col2:
        st.subheader("Dataset Quick Stats")
        st.metric("Total Samples", len(df))
        st.metric("Features", len(df.columns) - 1)
        
        heart_disease_rate = df['target'].mean() * 100
        st.metric("Heart Disease Rate", f"{heart_disease_rate:.1f}%")
        
        # Age distribution
        fig = px.histogram(df, x='age', color='target', 
                          title="Age Distribution by Heart Disease",
                          labels={'target': 'Heart Disease', 'count': 'Count'})
        st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Data Analysis":
    st.header("Data Analysis and Visualization")
    
    # Feature descriptions
    feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Sex (0: Female, 1: Male)',
        'chest_pain_type': 'Chest pain type (0-3)',
        'resting_bp': 'Resting blood pressure (mm Hg)',
        'cholesterol': 'Serum cholesterol (mg/dl)',
        'fasting_bs': 'Fasting blood sugar > 120 mg/dl (0: No, 1: Yes)',
        'resting_ecg': 'Resting ECG results (0-2)',
        'max_heart_rate': 'Maximum heart rate achieved',
        'exercise_angina': 'Exercise induced angina (0: No, 1: Yes)',
        'oldpeak': 'ST depression induced by exercise',
        'st_slope': 'Slope of peak exercise ST segment (0-2)',
        'target': 'Heart disease (0: No, 1: Yes)'
    }
    
    # Data overview
    tab1, tab2, tab3 = st.tabs(["📋 Data Overview", "📈 Distributions", "🔗 Correlations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Sample")
            st.dataframe(df.head(10))
        
        with col2:
            st.subheader("Feature Descriptions")
            for feature, description in feature_descriptions.items():
                st.write(f"**{feature}:** {description}")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        # Feature selection for visualization
        feature_to_plot = st.selectbox("Select feature to analyze:", 
                                     [col for col in df.columns if col != 'target'])
        
        with col1:
            # Distribution by target
            fig = px.histogram(df, x=feature_to_plot, color='target',
                             title=f"{feature_to_plot} Distribution by Heart Disease",
                             labels={'target': 'Heart Disease'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(df, x='target', y=feature_to_plot,
                        title=f"{feature_to_plot} Box Plot by Heart Disease",
                        labels={'target': 'Heart Disease'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics by Heart Disease Status")
        summary_stats = df.groupby('target')[feature_to_plot].describe()
        st.dataframe(summary_stats)
    
    with tab3:
        st.subheader("Feature Correlation Matrix")
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        
        # Feature importance with target
        st.subheader("Correlation with Target Variable")
        target_corr = df.corr()['target'].sort_values(key=abs, ascending=False)[1:]
        
        fig = px.bar(x=target_corr.values, y=target_corr.index, orientation='h',
                    title="Feature Correlation with Heart Disease",
                    labels={'x': 'Correlation', 'y': 'Features'})
        st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Training":
    st.header("Model Training and Evaluation")
    
    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Model selection
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox("Select Model:", 
                                  ["Random Forest", "Logistic Regression", "Support Vector Machine"])
        test_size = st.slider("Test Set Size:", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        random_state = st.number_input("Random State:", value=42, min_value=0, max_value=1000)
        
        if model_choice == "Random Forest":
            n_estimators = st.slider("Number of Trees:", 10, 200, 100, 10)
            max_depth = st.slider("Max Depth:", 3, 20, 10)
    
    # Train model
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize model
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=n_estimators, 
                                             max_depth=max_depth, 
                                             random_state=random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(random_state=random_state, max_iter=1000)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
            elif model_choice == "Support Vector Machine":
                model = SVC(random_state=random_state, probability=True)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Store in session state for use in prediction
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.model_type = model_choice
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"Model trained successfully!")
                st.metric("Accuracy", f"{accuracy:.3f}")
                
                # Classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            
            with col2:
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
                
                # Feature importance (for Random Forest)
                if model_choice == "Random Forest":
                    st.subheader("Feature Importance")
                    importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig = px.bar(importance, x='importance', y='feature', orientation='h',
                               title="Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Prediction":
    st.header("Heart Disease Risk Prediction")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first in the 'Model Training' section.")
    else:
        st.success(f"Using {st.session_state.model_type} model for predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Information")
            
            # Input fields
            age = st.slider("Age", 25, 80, 50)
            sex = st.selectbox("Sex", ["Female", "Male"])
            chest_pain = st.selectbox("Chest Pain Type", 
                                    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
            
        with col2:
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
            max_hr = st.number_input("Maximum Heart Rate", 60, 220, 150)
            exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression", 0.0, 6.0, 0.0, 0.1)
            st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
        
        # Convert inputs to model format
        sex_encoded = 1 if sex == "Male" else 0
        chest_pain_encoded = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain)
        fasting_bs_encoded = 1 if fasting_bs == "Yes" else 0
        resting_ecg_encoded = ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(resting_ecg)
        exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0
        st_slope_encoded = ["Upsloping", "Flat", "Downsloping"].index(st_slope)
        
        # Create input array
        input_data = np.array([[age, sex_encoded, chest_pain_encoded, resting_bp, cholesterol,
                               fasting_bs_encoded, resting_ecg_encoded, max_hr, 
                               exercise_angina_encoded, oldpeak, st_slope_encoded]])
        
        # Make prediction
        if st.button("Predict Heart Disease Risk", type="primary"):
            if st.session_state.model_type in ["Logistic Regression", "Support Vector Machine"]:
                input_scaled = st.session_state.scaler.transform(input_data)
                prediction = st.session_state.model.predict(input_scaled)[0]
                probability = st.session_state.model.predict_proba(input_scaled)[0]
            else:
                prediction = st.session_state.model.predict(input_data)[0]
                probability = st.session_state.model.predict_proba(input_data)[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ High Risk of Heart Disease")
                else:
                    st.success("✅ Low Risk of Heart Disease")
                
                st.metric("Risk Probability", f"{probability[1]:.1%}")
            
            with col2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability[1] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Heart Disease Risk (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Model Comparison":
    st.header("Model Comparison")
    
    if st.button("Compare All Models"):
        with st.spinner("Training and comparing models..."):
            # Prepare data
            X = df.drop('target', axis=1)
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Scale data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Models to compare
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Support Vector Machine': SVC(random_state=42, probability=True)
            }
            
            results = {}
            
            for name, model in models.items():
                if name == 'Random Forest':
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {'accuracy': accuracy, 'predictions': y_pred, 'probabilities': y_prob}
            
            # Display comparison
            comparison_df = pd.DataFrame({name: [data['accuracy']] for name, data in results.items()}, 
                                       index=['Accuracy']).T
            
            st.subheader("Model Accuracy Comparison")
            st.dataframe(comparison_df.style.highlight_max(axis=0))
            
            # Bar chart
            fig = px.bar(x=comparison_df.index, y=comparison_df['Accuracy'], 
                        title="Model Accuracy Comparison")
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model recommendation
            best_model = comparison_df['Accuracy'].idxmax()
            st.success(f"🏆 Best performing model: **{best_model}** with {comparison_df.loc[best_model, 'Accuracy']:.3f} accuracy")

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This application is for educational purposes only and should not be used for actual medical diagnosis. Always consult with healthcare professionals for medical advice.")
