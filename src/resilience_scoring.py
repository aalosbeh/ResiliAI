"""
Resilience Scoring Engine with Explainable AI for ResiliAI Framework

This module implements a resilience scoring system that uses explainable AI techniques
to identify the most influential variables affecting economic stability.

Author: Anas ALsobeh, Raneem Alkurdi
Date: July 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class ResilienceScoringEngine:
    """
    A scoring engine that uses explainable AI to evaluate economic resilience
    and identify key factors affecting stability.
    """
    
    def __init__(self):
        """
        Initialize the resilience scoring engine.
        """
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.explainer = None
        self.shap_values = None
        self.performance_metrics = {}
        
    def train(self, data, target_column='resilience_score', test_size=0.2, random_state=42):
        """
        Train the resilience scoring model on economic data.
        
        Args:
            data (pd.DataFrame): Economic data with features and target
            target_column (str): Name of the target column (resilience score)
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Model performance metrics
        """
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train a Gradient Boosting model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=random_state
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model performance
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Store performance metrics
        self.performance_metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X_test_scaled)
        
        return self.performance_metrics
    
    def predict_resilience(self, data):
        """
        Predict resilience scores for new economic data.
        
        Args:
            data (pd.DataFrame): Economic data with features
            
        Returns:
            np.ndarray: Predicted resilience scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Ensure data has the correct features
        if not all(feature in data.columns for feature in self.feature_names):
            missing_features = [f for f in self.feature_names if f not in data.columns]
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        # Select and scale features
        X = data[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def explain_predictions(self, data, output_dir=None):
        """
        Generate explanations for resilience predictions using SHAP.
        
        Args:
            data (pd.DataFrame): Economic data to explain
            output_dir (str, optional): Directory to save visualizations
            
        Returns:
            dict: Explanation data and visualization paths
        """
        if self.model is None or self.explainer is None:
            raise ValueError("Model has not been trained yet")
        
        # Ensure data has the correct features
        if not all(feature in data.columns for feature in self.feature_names):
            missing_features = [f for f in self.feature_names if f not in data.columns]
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        # Select and scale features
        X = data[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_scaled)
        
        # Prepare explanation data
        explanation = {
            'shap_values': shap_values,
            'feature_names': self.feature_names,
            'base_value': self.explainer.expected_value,
            'visualizations': []
        }
        
        # Generate visualizations if output directory is provided
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            summary_path = os.path.join(output_dir, 'shap_summary.png')
            plt.savefig(summary_path)
            plt.close()
            explanation['visualizations'].append(summary_path)
            
            # 2. Bar plot of feature importance
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, plot_type='bar', show=False)
            plt.tight_layout()
            bar_path = os.path.join(output_dir, 'shap_importance.png')
            plt.savefig(bar_path)
            plt.close()
            explanation['visualizations'].append(bar_path)
            
            # 3. Dependence plots for top features
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(-feature_importance)[:3]  # Top 3 features
            
            for idx in top_indices:
                feature_name = self.feature_names[idx]
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(idx, shap_values, X, feature_names=self.feature_names, show=False)
                plt.tight_layout()
                dep_path = os.path.join(output_dir, f'shap_dependence_{feature_name}.png')
                plt.savefig(dep_path)
                plt.close()
                explanation['visualizations'].append(dep_path)
        
        return explanation
    
    def identify_key_factors(self, data, n_factors=5):
        """
        Identify the most influential factors affecting economic resilience.
        
        Args:
            data (pd.DataFrame): Economic data to analyze
            n_factors (int): Number of top factors to identify
            
        Returns:
            dict: Top factors and their importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Ensure data has the correct features
        if not all(feature in data.columns for feature in self.feature_names):
            missing_features = [f for f in self.feature_names if f not in data.columns]
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        # Select and scale features
        X = data[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Calculate SHAP values if not already done
        if self.shap_values is None:
            self.shap_values = self.explainer.shap_values(X_scaled)
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        # Get top factors
        top_indices = np.argsort(-mean_abs_shap)[:n_factors]
        top_factors = {self.feature_names[i]: float(mean_abs_shap[i]) for i in top_indices}
        
        return {
            'top_factors': top_factors,
            'factor_ranking': dict(sorted(top_factors.items(), key=lambda x: x[1], reverse=True))
        }
    
    def analyze_policy_impact(self, base_data, policy_data, policy_name):
        """
        Analyze the impact of a policy intervention on resilience scores.
        
        Args:
            base_data (pd.DataFrame): Economic data before policy intervention
            policy_data (pd.DataFrame): Economic data after policy intervention
            policy_name (str): Name of the policy for reporting
            
        Returns:
            dict: Policy impact analysis
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Predict resilience scores before and after policy
        base_scores = self.predict_resilience(base_data)
        policy_scores = self.predict_resilience(policy_data)
        
        # Calculate changes
        score_changes = policy_scores - base_scores
        mean_change = np.mean(score_changes)
        median_change = np.median(score_changes)
        
        # Identify entities with most significant improvements
        entities = base_data.index if base_data.index.equals(policy_data.index) else np.arange(len(base_scores))
        improvements = sorted(zip(entities, score_changes), key=lambda x: x[1], reverse=True)
        
        # Calculate feature changes
        feature_changes = {}
        for feature in self.feature_names:
            if feature in base_data.columns and feature in policy_data.columns:
                base_values = base_data[feature].values
                policy_values = policy_data[feature].values
                changes = policy_values - base_values
                feature_changes[feature] = {
                    'mean_change': float(np.mean(changes)),
                    'median_change': float(np.median(changes)),
                    'max_improvement': float(np.max(changes)),
                    'correlation_with_score_change': float(np.corrcoef(changes, score_changes)[0, 1])
                }
        
        return {
            'policy_name': policy_name,
            'mean_resilience_change': float(mean_change),
            'median_resilience_change': float(median_change),
            'positive_impact_percentage': float(np.mean(score_changes > 0) * 100),
            'top_improvements': improvements[:5],
            'feature_changes': feature_changes,
            'most_influential_changes': sorted(
                [(f, c['correlation_with_score_change']) for f, c in feature_changes.items()],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
        }
    
    def visualize_model_performance(self, data, target_column='resilience_score', output_dir=None):
        """
        Visualize model performance and feature importance.
        
        Args:
            data (pd.DataFrame): Economic data with features and target
            target_column (str): Name of the target column
            output_dir (str, optional): Directory to save visualizations
            
        Returns:
            list: Paths to saved visualization files
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if output_dir is None:
            return []
            
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        visualization_paths = []
        
        # Prepare data
        X = data[self.feature_names]
        y = data[target_column]
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        # 1. Actual vs Predicted plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Actual Resilience Score')
        plt.ylabel('Predicted Resilience Score')
        plt.title('Actual vs Predicted Resilience Scores')
        plt.grid(True, alpha=0.3)
        
        # Add performance metrics to plot
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        plt.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
        
        # Save the figure
        fig_path = os.path.join(output_dir, 'actual_vs_predicted.png')
        plt.savefig(fig_path)
        visualization_paths.append(fig_path)
        plt.close()
        
        # 2. Feature importance plot
        plt.figure(figsize=(12, 8))
        feature_importance = self.model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        plt.barh(np.array(self.feature_names)[sorted_idx], feature_importance[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance (MDI)')
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(fig_path)
        visualization_paths.append(fig_path)
        plt.close()
        
        # 3. Residuals plot
        plt.figure(figsize=(10, 8))
        residuals = y - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='k', linestyles='--')
        plt.xlabel('Predicted Resilience Score')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        fig_path = os.path.join(output_dir, 'residuals.png')
        plt.savefig(fig_path)
        visualization_paths.append(fig_path)
        plt.close()
        
        return visualization_paths
    
    def save_model(self, model_path):
        """
        Save the trained model to a file.
        
        Args:
            model_path (str): Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        try:
            import joblib
            
            # Create a dictionary with all necessary components
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'scaler': self.scaler,
                'performance_metrics': self.performance_metrics
            }
            
            # Save to file
            joblib.dump(model_data, model_path)
            print(f"Model successfully saved to {model_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path):
        """
        Load a trained model from a file.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import joblib
            
            # Load from file
            model_data = joblib.load(model_path)
            
            # Extract components
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.scaler = model_data['scaler']
            self.performance_metrics = model_data['performance_metrics']
            
            # Initialize explainer
            self.explainer = shap.TreeExplainer(self.model)
            
            print(f"Model successfully loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False


def generate_synthetic_data(n_samples=100, n_features=10, random_state=42):
    """
    Generate synthetic economic data for testing the resilience scoring engine.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features to generate
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Synthetic economic data
    """
    np.random.seed(random_state)
    
    # Generate feature names
    feature_names = [
        'gdp_growth', 'unemployment_rate', 'inflation_rate', 'debt_to_gdp',
        'fiscal_balance', 'current_account', 'foreign_reserves', 'financial_depth',
        'economic_complexity', 'infrastructure_quality', 'governance_quality',
        'education_index', 'healthcare_access', 'digital_adoption', 'trade_openness'
    ]
    
    # Select a subset of features if n_features is less than available names
    selected_features = feature_names[:min(n_features, len(feature_names))]
    
    # Generate random data
    X = np.random.randn(n_samples, len(selected_features))
    
    # Scale features to realistic ranges
    feature_ranges = {
        'gdp_growth': (0, 10),
        'unemployment_rate': (2, 20),
        'inflation_rate': (0, 15),
        'debt_to_gdp': (20, 150),
        'fiscal_balance': (-10, 5),
        'current_account': (-15, 15),
        'foreign_reserves': (0, 100),
        'financial_depth': (20, 200),
        'economic_complexity': (0, 2),
        'infrastructure_quality': (1, 7),
        'governance_quality': (0, 100),
        'education_index': (0.3, 0.9),
        'healthcare_access': (40, 100),
        'digital_adoption': (0.2, 0.9),
        'trade_openness': (20, 200)
    }
    
    # Transform data to realistic ranges
    for i, feature in enumerate(selected_features):
        if feature in feature_ranges:
            min_val, max_val = feature_ranges[feature]
            X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min()) * (max_val - min_val) + min_val
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=selected_features)
    
    # Generate resilience score as a function of the features
    # Higher GDP growth, foreign reserves, education -> higher resilience
    # Higher unemployment, inflation, debt -> lower resilience
    weights = {}
    for feature in selected_features:
        if feature in ['gdp_growth', 'foreign_reserves', 'education_index', 
                      'healthcare_access', 'governance_quality', 'infrastructure_quality']:
            weights[feature] = np.random.uniform(0.5, 1.5)  # Positive impact
        elif feature in ['unemployment_rate', 'inflation_rate', 'debt_to_gdp']:
            weights[feature] = np.random.uniform(-1.5, -0.5)  # Negative impact
        else:
            weights[feature] = np.random.uniform(-0.5, 0.5)  # Mixed impact
    
    # Calculate base resilience score
    resilience = np.zeros(n_samples)
    for i, feature in enumerate(selected_features):
        resilience += df[feature].values * weights[feature]
    
    # Normalize to 0-100 scale and add some noise
    resilience = (resilience - resilience.min()) / (resilience.max() - resilience.min()) * 100
    resilience += np.random.normal(0, 5, n_samples)
    resilience = np.clip(resilience, 0, 100)
    
    # Add to DataFrame
    df['resilience_score'] = resilience
    
    # Add entity names (countries)
    countries = [
        'USA', 'China', 'Japan', 'Germany', 'UK', 'France', 'India', 'Italy', 'Brazil', 'Canada',
        'Russia', 'South Korea', 'Australia', 'Spain', 'Mexico', 'Indonesia', 'Netherlands', 'Saudi Arabia',
        'Turkey', 'Switzerland', 'Poland', 'Sweden', 'Belgium', 'Thailand', 'Ireland', 'Argentina',
        'Norway', 'Israel', 'Singapore', 'Malaysia', 'South Africa', 'Egypt', 'Philippines', 'Pakistan',
        'Colombia', 'Chile', 'Peru', 'Vietnam', 'Bangladesh', 'Nigeria', 'Kenya', 'Morocco', 'Ghana'
    ]
    
    # Assign countries to samples (with repetition if n_samples > len(countries))
    entity_names = [countries[i % len(countries)] for i in range(n_samples)]
    df['entity'] = entity_names
    
    return df


def main():
    """
    Main function to demonstrate the resilience scoring engine.
    """
    # Generate synthetic data
    data = generate_synthetic_data(n_samples=150, n_features=12)
    print(f"Generated synthetic data with {data.shape[0]} samples and {data.shape[1]} columns")
    
    # Initialize and train the resilience scoring engine
    engine = ResilienceScoringEngine()
    performance = engine.train(data.drop(columns=['entity']))
    print(f"Model trained with R² = {performance['r2']:.3f}, RMSE = {performance['rmse']:.3f}")
    
    # Identify key factors
    key_factors = engine.identify_key_factors(data.drop(columns=['entity', 'resilience_score']))
    print("Top factors affecting resilience:")
    for factor, importance in key_factors['factor_ranking'].items():
        print(f"  - {factor}: {importance:.4f}")
    
    # Generate explanations
    vis_dir = "/home/ubuntu/ESI2025_ResiliAI/data/visualizations/resilience"
    explanations = engine.explain_predictions(data.drop(columns=['entity', 'resilience_score']), vis_dir)
    print(f"Generated {len(explanations['visualizations'])} explanation visualizations")
    
    # Visualize model performance
    vis_paths = engine.visualize_model_performance(data.drop(columns=['entity']), output_dir=vis_dir)
    print(f"Generated {len(vis_paths)} model performance visualizations")
    
    # Save model
    model_path = "/home/ubuntu/ESI2025_ResiliAI/data/resilience_model.joblib"
    engine.save_model(model_path)


if __name__ == "__main__":
    main()
