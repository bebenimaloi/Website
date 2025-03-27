import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt
import base64
import io
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors

app = Flask(__name__)

def perform_decision_tree_analysis(X_train, X_test, y_train, y_test):
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                             param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get best model and make predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'best_params': grid_search.best_params_
    }

    # Calculate AUC-ROC
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)
    y_pred_binarized = lb.transform(y_pred)
    metrics['auc_roc'] = roc_auc_score(y_test_binarized, y_pred_binarized, 
                                      average='weighted', multi_class='ovr')

    # Create visualizations
    images = {}
    try:
        # Feature Importance Plot with improved styling
        feature_importances = best_model.feature_importances_
        feature_importance_dict = dict(zip(X_train.columns, feature_importances))
        
        # Custom Skyblue Gradient
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            "skyblue_gradient", ["#B3E5FC", "#03A9F4"], N=256)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Normalize values for gradient
        norm = plt.Normalize(vmin=min(feature_importances), 
                           vmax=max(feature_importances))
        colors = custom_cmap(norm(feature_importances))
        
        # Plot bars
        bars = ax.barh(X_train.columns, feature_importances, 
                      color=colors, edgecolor=None, alpha=0.9)
        
        # Style adjustments
        ax.set_xlabel("Importance", fontsize=14, fontweight='bold', color="#444444")
        ax.set_ylabel("Features", fontsize=14, fontweight='bold', color="#444444")
        ax.set_title("Decision Tree Feature Importance", 
                    fontsize=16, fontweight='bold', color="#222222")
        
        # Add grid and remove spines
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Keep most important feature on top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        images['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
        
    finally:
        plt.close('all')

    # Create feature importance dictionary
    feature_importances = list(zip(X_train.columns, best_model.feature_importances_))
    
    return metrics, images, feature_importances

def perform_random_forest_analysis(X_train, X_test, y_train, y_test):
    # Simplified parameter grid with fewer combinations
    param_grid = {
        'n_estimators': [100],  # Reduced from [50, 100, 200]
        'max_depth': [10, 20],  # Reduced from [None, 10, 20]
        'max_features': ['sqrt'],  # Reduced from ['sqrt', 'log2', None]
        'min_samples_split': [5]  # Reduced from [2, 5, 10]
    }

    # Initialize and run GridSearchCV with optimized parameters
    grid_search = GridSearchCV(
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1,  # Parallel processing within each tree
            warm_start=True,  # Enable warm start for faster fitting
            class_weight='balanced'  # Better handling of imbalanced datasets
        ),
        param_grid,
        cv=3,  # Reduced from 5 to 3 folds
        n_jobs=-1,  # Parallel processing for GridSearchCV
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Get best model and make predictions
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test)

    # Calculate metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, average='weighted'),
        'recall': recall_score(y_test, y_test_pred, average='weighted'),
        'f1': f1_score(y_test, y_test_pred, average='weighted')
    }

    # Calculate AUC-ROC
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)
    y_test_pred_binarized = lb.transform(y_test_pred)
    auc_roc = roc_auc_score(y_test_binarized, y_test_pred_binarized, 
                           average='weighted', multi_class='ovr')

    # Feature importance
    importances = best_model.feature_importances_
    feature_importance = dict(zip(X_train.columns, importances))
    sorted_importance = dict(sorted(feature_importance.items(), 
                                  key=lambda x: x[1], reverse=True))

    # Create visualization with improved styling
    images = {}
    try:
        # Create custom color map
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            "skyblue_gradient", ["#B3E5FC", "#03A9F4"], N=256)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Normalize values for gradient
        norm = plt.Normalize(vmin=min(sorted_importance.values()), 
                           vmax=max(sorted_importance.values()))
        colors = custom_cmap(norm(list(sorted_importance.values())))
        
        # Plot bars
        bars = ax.barh(list(sorted_importance.keys()), 
                      list(sorted_importance.values()),
                      color=colors, edgecolor=None, alpha=0.9)
        
        # Style adjustments
        ax.set_xlabel("Importance", fontsize=14, fontweight='bold', color="#444444")
        ax.set_ylabel("Features", fontsize=14, fontweight='bold', color="#444444")
        ax.set_title("Random Forest Feature Importance", fontsize=16, fontweight='bold', color="#222222")
        
        # Add grid and remove spines
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Keep most important feature on top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        images['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
        
    finally:
        plt.close('all')

    return (grid_search.best_params_, test_metrics, auc_roc, 
            sorted_importance, images)

def perform_svm_analysis(X_train, X_test, y_train, y_test):
    # Store column names before scaling
    feature_names = X_train.columns if hasattr(X_train, 'columns') else None
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }

    # Initialize and train GridSearchCV
    grid_search = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    # Get best model and make predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    # Feature importance for linear kernel
    feature_importance = None
    images = {}
    
    if best_model.kernel == 'linear' and feature_names is not None:
        coef = np.abs(best_model.coef_).flatten()
        feature_importance = dict(zip(feature_names, coef))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        try:
            # Custom Skyblue Gradient
            custom_cmap = mcolors.LinearSegmentedColormap.from_list(
                "skyblue_gradient", ["#B3E5FC", "#03A9F4"], N=256)
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Normalize values for gradient
            norm = plt.Normalize(vmin=min(feature_importance.values()), 
                               vmax=max(feature_importance.values()))
            colors = custom_cmap(norm(list(feature_importance.values())))
            
            # Plot bars
            bars = ax.barh(list(feature_importance.keys()), 
                          list(feature_importance.values()),
                          color=colors, edgecolor=None, alpha=0.9)
            
            # Style adjustments
            ax.set_xlabel("Importance", fontsize=14, fontweight='bold', color="#444444")
            ax.set_ylabel("Features", fontsize=14, fontweight='bold', color="#444444")
            ax.set_title("SVM Feature Importance", fontsize=16, fontweight='bold', color="#222222")
            
            # Add grid and remove spines
            ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            # Keep most important feature on top
            ax.invert_yaxis()
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            images['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
            
        finally:
            plt.close('all')

    return grid_search.best_params_, metrics, feature_importance, images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'file' not in request.files:
            return "No file part in the request", 400

        file = request.files['file']
        model_choice = request.form.get('model')

        if file.filename == '':
            return "No selected file", 400

        if file:
            # Save and read the file
            upload_folder = "uploads"
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Read the CSV file
            data = pd.read_csv(file_path)

            # Prepare feature columns and target
            X = data[['Cognitive Abilities', 'Engagement Levels', 'Pre-Test Scores',
                     'Post-Test Scores', 'Time Spent on AR', 'Frequency of AR Use']]
            y = data['Performance']

            # Encode the target column
            le = LabelEncoder()
            y = le.fit_transform(y)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            if model_choice == 'decision_tree':
                metrics, images, feature_importances = perform_decision_tree_analysis(
                    X_train, X_test, y_train, y_test
                )
                return render_template(
                    'decision_tree_results.html',
                    metrics=metrics,
                    images=images,
                    feature_importances=feature_importances
                )
            
            elif model_choice == 'random_forest':
                (best_params, test_metrics, auc_roc, 
                 feature_importance, images) = perform_random_forest_analysis(
                    X_train, X_test, y_train, y_test
                )
                return render_template(
                    'random_forest_results.html',
                    best_params=best_params,
                    metrics=test_metrics,
                    auc_roc=round(auc_roc, 3),
                    feature_importance=feature_importance,
                    images=images
                )

            elif model_choice == 'svm':
                best_params, metrics, feature_importance, images = perform_svm_analysis(
                    X_train, X_test, y_train, y_test
                )
                return render_template(
                    'svm_results.html',
                    best_params=best_params,
                    metrics=metrics,
                    feature_importance=feature_importance,
                    images=images
                )

            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        print(f"Error: {str(e)}")
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
