import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import r2_score, mean_squared_error

def show_ml_pipeline(df):
    # Provide target selection (including an unsupervised option
    target_options = ["(Unsupervised)"] + list(df.columns)
    target_choice = st.selectbox("Select target column for ML (or choose '(Unsupervised)' for clustering)", target_options)
    target_col = None if target_choice == "(Unsupervised)" else target_choice
    
    # Only proceed when user clicks the Run button
    if st.button("Run ML Pipeline"):
        result = {}
        if target_col is None:
            # Unsupervised clustering scenario
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            data = df.select_dtypes(include='number').dropna()
            if data.shape[0] < 2:
                st.error("Not enough numeric data for clustering.")
                return
            scaled = StandardScaler().fit_transform(data)
            # Try different cluster counts and pick the best via silhouette score
            best_score = -1
            best_k = None
            best_labels = None
            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(scaled)
                score = silhouette_score(scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
            result['task'] = 'clustering'
            result['best_k'] = best_k
            result['silhouette'] = best_score
            # Cluster size distribution
            clusters, counts = np.unique(best_labels, return_counts=True)
            cluster_sizes = {int(c): int(n) for c, n in zip(clusters, counts)}
            result['cluster_sizes'] = cluster_sizes
            # Save results to session and display them
            st.session_state['ml_results'] = result
            st.write(f"**Optimal number of clusters:** {best_k} (Silhouette Score = {best_score:.4f})")
            st.write("Cluster sizes:", cluster_sizes)
        else:
            # Supervised ML scenario
            y = df[target_col]
            X = df.drop(columns=[target_col])
            # Drop non-numeric features for simplicity (could be extended to handle encoding)
            X = X.select_dtypes(exclude=['object'])
            # Drop any rows with missing values in X or y to avoid errors
            data_clean = pd.concat([X, y], axis=1).dropna()
            X = data_clean.drop(columns=[target_col])
            y = data_clean[target_col]
            # Determine classification vs regression
            if y.dtype == object or y.dtype == bool or y.dtype.name == 'category':
                task_type = 'classification'
            elif pd.api.types.is_numeric_dtype(y):
                # Consider integer with small unique values as classification (e.g., 0/1)
                if pd.api.types.is_integer_dtype(y) and y.nunique() < 10:
                    task_type = 'classification'
                else:
                    task_type = 'regression'
            else:
                task_type = 'classification'
            result['task'] = task_type
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # Define candidate models and hyperparameter grids
            if task_type == 'classification':
                models = [
                    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
                    ('Random Forest', RandomForestClassifier(random_state=42)),
                    ('KNN Classifier', KNeighborsClassifier())
                ]
                param_grids = {
                    'Logistic Regression': {'C': [0.1, 1.0, 10.0]},
                    'Random Forest': {'n_estimators': [50, 100], 'max_depth': [None, 5]},
                    'KNN Classifier': {'n_neighbors': [3, 5, 7]}
                }
                scoring_metric = 'accuracy'
            else:  # regression
                models = [
                    ('Linear Regression', LinearRegression()),
                    ('Random Forest', RandomForestRegressor(random_state=42)),
                    ('KNN Regressor', KNeighborsRegressor())
                ]
                param_grids = {
                    'Linear Regression': {},  # no hyperparameters to tune
                    'Random Forest': {'n_estimators': [50, 100], 'max_depth': [None, 5]},
                    'KNN Regressor': {'n_neighbors': [3, 5, 7]}
                }
                scoring_metric = 'r2'
            
            best_model_name = None
            best_score = -np.inf
            best_model = None
            model_performance = []  # to store performance of each model
            for name, model in models:
                params = param_grids.get(name, {})
                if params:
                    # Perform GridSearchCV for hyperparameter tuning
                    grid = GridSearchCV(model, params, cv=3, scoring=scoring_metric)
                    grid.fit(X_train, y_train)
                    best_estimator = grid.best_estimator_
                    cv_score = grid.best_score_
                else:
                    # If no parameters to tune, fit directly
                    best_estimator = model.fit(X_train, y_train)
                    cv_score = None
                # Evaluate on test set
                if task_type == 'classification':
                    y_pred = best_estimator.predict(X_test)
                    test_score = accuracy_score(y_test, y_pred)
                    model_performance.append((name, cv_score, test_score))
                    score_to_compare = cv_score if cv_score is not None else test_score
                    if score_to_compare > best_score:
                        best_score = score_to_compare
                        best_model_name = name
                        best_model = best_estimator
                        # Save detailed metrics for the best classifier
                        result['accuracy'] = test_score
                        result['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
                        result['classification_report'] = classification_report(y_test, y_pred, output_dict=False)
                else:
                    y_pred = best_estimator.predict(X_test)
                    test_score = r2_score(y_test, y_pred)
                    model_performance.append((name, cv_score, test_score))
                    score_to_compare = cv_score if cv_score is not None else test_score
                    if score_to_compare > best_score:
                        best_score = score_to_compare
                        best_model_name = name
                        best_model = best_estimator
                        # Save detailed metrics for the best regressor
                        result['r2_score'] = test_score
                        result['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                        result['y_test_sample'] = list(y_test[:50])  # sample actual values
                        result['y_pred_sample'] = list(y_pred[:50])  # sample predicted values
            result['best_model'] = best_model_name
            # Store simplified model comparison results
            result['models_compared'] = [
                (name, None if cv is None else float(cv), float(test)) 
                for name, cv, test in model_performance
            ]
            # Capture top features or feature importances for the best model if available
            if best_model is not None:
                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                    feature_names = X.columns
                    sorted_idx = np.argsort(importances)[::-1]
                    top_features = [(feature_names[i], float(importances[i])) for i in sorted_idx[:5]]
                    result['top_features'] = top_features
                elif hasattr(best_model, 'coef_'):
                    coeff = best_model.coef_
                    feature_names = X.columns
                    if coeff.ndim == 1:
                        importance = np.abs(coeff)
                    else:
                        # For multi-class models, average coefficient magnitudes
                        importance = np.mean(np.abs(coeff), axis=0)
                    sorted_idx = np.argsort(importance)[::-1]
                    top_features = [(feature_names[i], float(importance[i])) for i in sorted_idx[:5]]
                    result['top_features'] = top_features
            
            # Save results to session and display key outcomes
            st.session_state['ml_results'] = result
            st.write(f"**Best Model:** {best_model_name}")
            if task_type == 'classification':
                st.write(f"Test Accuracy: {result['accuracy']:.4f}")
                # Display confusion matrix as a heatmap
                st.write("Confusion Matrix:")
                cm = pd.DataFrame(result['confusion_matrix'])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            else:
                # ----------- Regression results and plots -----------

            # Display key scores
                st.write(f"Test R² Score: {result['r2_score']:.4f}")

                # Compute RMSE (works on any sklearn version)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                st.write(f"Test RMSE: {rmse:.4f}")

                # Plot Actual vs Predicted for a sample of test points (50 if available)
                sample_size = min(len(y_test), 50)
                y_test_sample = y_test[:sample_size]
                y_pred_sample = y_pred[:sample_size]

                fig, ax = plt.subplots()
                ax.scatter(y_test_sample, y_pred_sample, alpha=0.7)
                # Reference line for perfect predictions
                min_val = min(y_test_sample.min(), y_pred_sample.min())
                max_val = max(y_test_sample.max(), y_pred_sample.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                ax.set_xlabel("Actual") 
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted (Sample)")
                ax.legend()
                st.pyplot(fig)

                # Display top features if available
                if result.get('top_features'):
                    st.write("**Top Features (importance or coefficient):**")
                    for feat, val in result['top_features']:
                        st.write(f"- {feat}: {val:.4f}")
