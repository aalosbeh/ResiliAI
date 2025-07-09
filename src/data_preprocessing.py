"""
Data Preprocessing Module for ResiliAI Framework

This module handles the extraction, cleaning, and preprocessing of economic datasets
for use in the Multi-Agent Reinforcement Learning (MARL) framework and Digital Twin
economy simulation environment.

Author: Anas ALsobeh, Raneem Alkurdi
Date: July 2025
"""

import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    A class for preprocessing economic datasets for the ResiliAI framework.
    
    This class handles data loading, cleaning, normalization, and feature engineering
    for economic datasets used in the MARL framework and Digital Twin simulation.
    """
    
    def __init__(self, data_dir):
        """
        Initialize the DataPreprocessor with the directory containing the datasets.
        
        Args:
            data_dir (str): Path to the directory containing the datasets
        """
        self.data_dir = data_dir
        self.datasets = {}
        self.processed_data = {}
        self.scalers = {}
        
    def load_text_data(self, filename, dataset_name):
        """
        Load data from a text file extracted from PDF.
        
        Args:
            filename (str): Name of the text file
            dataset_name (str): Name to assign to the dataset
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = os.path.join(self.data_dir, filename)
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Store the raw content
            self.datasets[dataset_name] = {'raw_text': content}
            print(f"Successfully loaded {filename} as {dataset_name}")
            return True
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            return False
    
    def extract_country_data(self, dataset_name, country_pattern=r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\n'):
        """
        Extract country-specific data from text content.
        
        Args:
            dataset_name (str): Name of the dataset to process
            country_pattern (str): Regex pattern to identify country names
            
        Returns:
            dict: Dictionary of country data
        """
        if dataset_name not in self.datasets or 'raw_text' not in self.datasets[dataset_name]:
            print(f"Dataset {dataset_name} not found or raw text not available")
            return {}
        
        raw_text = self.datasets[dataset_name]['raw_text']
        countries = re.findall(country_pattern, raw_text)
        
        country_data = {}
        for country in countries:
            # Find the section for this country
            country_pattern = f"{country}\\s*\\n(.*?)(?=\\n\\n[A-Z][a-z]+\\s*\\n|$)"
            match = re.search(country_pattern, raw_text, re.DOTALL)
            if match:
                country_data[country] = match.group(1).strip()
        
        self.datasets[dataset_name]['country_data'] = country_data
        return country_data
    
    def extract_fiscal_measures(self, dataset_name):
        """
        Extract fiscal measures data from country text.
        
        Args:
            dataset_name (str): Name of the dataset to process
            
        Returns:
            pd.DataFrame: DataFrame of fiscal measures by country
        """
        if dataset_name not in self.datasets or 'country_data' not in self.datasets[dataset_name]:
            print(f"Country data not found for {dataset_name}")
            return pd.DataFrame()
        
        country_data = self.datasets[dataset_name]['country_data']
        
        # Define patterns for different fiscal measures
        patterns = {
            'above_the_line': r'Above-the-line measures\s*:?\s*([0-9.]+)',
            'below_the_line': r'Below-the-line measures\s*:?\s*([0-9.]+)',
            'contingent_liabilities': r'Contingent liabilities\s*:?\s*([0-9.]+)',
            'healthcare_spending': r'[Hh]ealth(?:care)?\s+spending\s*:?\s*([0-9.]+)',
            'household_support': r'[Hh]ousehold\s+support\s*:?\s*([0-9.]+)',
            'business_support': r'[Bb]usiness\s+support\s*:?\s*([0-9.]+)'
        }
        
        data = []
        for country, text in country_data.items():
            row = {'country': country}
            for measure, pattern in patterns.items():
                match = re.search(pattern, text)
                if match:
                    try:
                        row[measure] = float(match.group(1))
                    except ValueError:
                        row[measure] = np.nan
                else:
                    row[measure] = np.nan
            data.append(row)
        
        df = pd.DataFrame(data)
        self.datasets[dataset_name]['fiscal_measures'] = df
        return df
    
    def load_csv_data(self, filename, dataset_name):
        """
        Load data from a CSV file.
        
        Args:
            filename (str): Name of the CSV file
            dataset_name (str): Name to assign to the dataset
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        try:
            file_path = os.path.join(self.data_dir, filename)
            df = pd.read_csv(file_path)
            self.datasets[dataset_name] = {'dataframe': df}
            print(f"Successfully loaded {filename} as {dataset_name}")
            return df
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            return pd.DataFrame()
    
    def clean_data(self, dataset_name):
        """
        Clean the dataset by handling missing values and outliers.
        
        Args:
            dataset_name (str): Name of the dataset to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if dataset_name not in self.datasets or 'dataframe' not in self.datasets[dataset_name]:
            if dataset_name in self.datasets and 'fiscal_measures' in self.datasets[dataset_name]:
                df = self.datasets[dataset_name]['fiscal_measures']
            else:
                print(f"No dataframe found for {dataset_name}")
                return pd.DataFrame()
        else:
            df = self.datasets[dataset_name]['dataframe']
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # For numeric columns, impute with median (only for columns with at least one non-missing value)
        if len(numeric_cols) > 0:
            # Check which columns have at least one non-missing value
            valid_numeric_cols = [col for col in numeric_cols if df[col].notna().any()]
            
            if valid_numeric_cols:
                imputer = SimpleImputer(strategy='median')
                df[valid_numeric_cols] = imputer.fit_transform(df[valid_numeric_cols])
            
            # For columns with all missing values, fill with 0
            all_missing_cols = [col for col in numeric_cols if col not in valid_numeric_cols]
            for col in all_missing_cols:
                df[col] = 0
                print(f"Column {col} has all missing values, filling with 0")
        
        # For categorical columns, impute with most frequent value (only for columns with at least one non-missing value)
        if len(categorical_cols) > 0:
            # Check which columns have at least one non-missing value
            valid_categorical_cols = [col for col in categorical_cols if df[col].notna().any()]
            
            if valid_categorical_cols:
                imputer = SimpleImputer(strategy='most_frequent')
                df[valid_categorical_cols] = imputer.fit_transform(df[valid_categorical_cols])
            
            # For columns with all missing values, fill with "Unknown"
            all_missing_cols = [col for col in categorical_cols if col not in valid_categorical_cols]
            for col in all_missing_cols:
                df[col] = "Unknown"
                print(f"Column {col} has all missing values, filling with 'Unknown'")
        
        # Handle outliers using IQR method for numeric columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), 
                               df[col].median(), df[col])
        
        self.processed_data[dataset_name] = df
        return df
    
    def normalize_data(self, dataset_name):
        """
        Normalize the numeric features in the dataset.
        
        Args:
            dataset_name (str): Name of the dataset to normalize
            
        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        if dataset_name not in self.processed_data:
            print(f"No processed data found for {dataset_name}")
            return pd.DataFrame()
        
        df = self.processed_data[dataset_name].copy()
        
        # Normalize only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers[dataset_name] = scaler
        
        self.processed_data[dataset_name] = df
        return df
    
    def engineer_features(self, dataset_name):
        """
        Engineer additional features for the dataset.
        
        Args:
            dataset_name (str): Name of the dataset for feature engineering
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        if dataset_name not in self.processed_data:
            print(f"No processed data found for {dataset_name}")
            return pd.DataFrame()
        
        df = self.processed_data[dataset_name].copy()
        
        # Example feature engineering for fiscal measures data
        if 'above_the_line' in df.columns and 'below_the_line' in df.columns:
            # Total fiscal response
            df['total_fiscal_response'] = df['above_the_line'] + df['below_the_line']
            
            # Ratio of direct to indirect measures
            df['direct_to_indirect_ratio'] = df['above_the_line'] / df['below_the_line'].replace(0, 0.001)
        
        # Add resilience score if we have enough measures
        resilience_columns = ['above_the_line', 'below_the_line', 'contingent_liabilities']
        if all(col in df.columns for col in resilience_columns):
            # Simple weighted sum for resilience score
            weights = [0.5, 0.3, 0.2]  # Example weights
            df['resilience_score'] = sum(df[col] * weight for col, weight in zip(resilience_columns, weights))
        
        self.processed_data[dataset_name] = df
        return df
    
    def visualize_data(self, dataset_name, output_dir):
        """
        Create visualizations of the processed data.
        
        Args:
            dataset_name (str): Name of the dataset to visualize
            output_dir (str): Directory to save visualizations
            
        Returns:
            list: Paths to saved visualization files
        """
        if dataset_name not in self.processed_data:
            print(f"No processed data found for {dataset_name}")
            return []
        
        df = self.processed_data[dataset_name]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        visualization_paths = []
        
        # Visualize fiscal measures by country if applicable
        if 'country' in df.columns and any(col in df.columns for col in ['above_the_line', 'below_the_line']):
            plt.figure(figsize=(12, 8))
            
            # Select top 10 countries by total fiscal response if we have many countries
            if 'total_fiscal_response' in df.columns and len(df) > 10:
                plot_df = df.nlargest(10, 'total_fiscal_response')
            else:
                plot_df = df
            
            # Plot fiscal measures
            measures = [col for col in ['above_the_line', 'below_the_line', 'contingent_liabilities'] 
                       if col in df.columns]
            
            if measures:
                plot_df.set_index('country')[measures].plot(kind='bar', figsize=(12, 8))
                plt.title('Fiscal Measures by Country')
                plt.ylabel('Measure Value')
                plt.tight_layout()
                
                # Save the figure
                fig_path = os.path.join(output_dir, f'{dataset_name}_fiscal_measures.png')
                plt.savefig(fig_path)
                visualization_paths.append(fig_path)
                plt.close()
        
        # Create correlation heatmap for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            plt.imshow(corr, cmap='coolwarm')
            plt.colorbar()
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
            plt.yticks(range(len(numeric_cols)), numeric_cols)
            plt.title('Correlation Matrix')
            
            # Add correlation values
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    plt.text(j, i, f'{corr.iloc[i, j]:.2f}', 
                             ha='center', va='center', 
                             color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
            
            plt.tight_layout()
            
            # Save the figure
            fig_path = os.path.join(output_dir, f'{dataset_name}_correlation.png')
            plt.savefig(fig_path)
            visualization_paths.append(fig_path)
            plt.close()
        
        return visualization_paths
    
    def prepare_for_simulation(self, dataset_name):
        """
        Prepare the processed data for use in the simulation environment.
        
        Args:
            dataset_name (str): Name of the dataset to prepare
            
        Returns:
            dict: Data formatted for simulation
        """
        if dataset_name not in self.processed_data:
            print(f"No processed data found for {dataset_name}")
            return {}
        
        df = self.processed_data[dataset_name]
        
        # Convert DataFrame to dictionary format suitable for simulation
        simulation_data = {}
        
        # If we have country data, organize by country
        if 'country' in df.columns:
            for _, row in df.iterrows():
                country = row['country']
                country_data = row.drop('country').to_dict()
                simulation_data[country] = country_data
        else:
            # Otherwise, just convert to dict
            simulation_data = df.to_dict(orient='records')
        
        return simulation_data
    
    def save_processed_data(self, dataset_name, output_file):
        """
        Save the processed data to a CSV file.
        
        Args:
            dataset_name (str): Name of the dataset to save
            output_file (str): Path to save the processed data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if dataset_name not in self.processed_data:
            print(f"No processed data found for {dataset_name}")
            return False
        
        try:
            df = self.processed_data[dataset_name]
            df.to_csv(output_file, index=False)
            print(f"Successfully saved processed data to {output_file}")
            return True
        except Exception as e:
            print(f"Error saving processed data: {str(e)}")
            return False


def main():
    """
    Main function to demonstrate the data preprocessing workflow.
    """
    # Initialize the preprocessor
    data_dir = "/home/ubuntu/ESI2025_ResiliAI/data"
    preprocessor = DataPreprocessor(data_dir)
    
    # Load IMF fiscal policy data
    preprocessor.load_text_data("imf_fiscal_policy_covid19.txt", "imf_fiscal")
    
    # Extract country data
    country_data = preprocessor.extract_country_data("imf_fiscal")
    print(f"Extracted data for {len(country_data)} countries")
    
    # Extract fiscal measures
    fiscal_df = preprocessor.extract_fiscal_measures("imf_fiscal")
    print(f"Extracted fiscal measures: {fiscal_df.shape}")
    
    # Clean the data
    cleaned_df = preprocessor.clean_data("imf_fiscal")
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    # Normalize the data
    normalized_df = preprocessor.normalize_data("imf_fiscal")
    print(f"Normalized data shape: {normalized_df.shape}")
    
    # Engineer features
    engineered_df = preprocessor.engineer_features("imf_fiscal")
    print(f"Engineered data shape: {engineered_df.shape}")
    
    # Create visualizations
    vis_dir = os.path.join(data_dir, "visualizations")
    vis_paths = preprocessor.visualize_data("imf_fiscal", vis_dir)
    print(f"Created {len(vis_paths)} visualizations")
    
    # Prepare data for simulation
    sim_data = preprocessor.prepare_for_simulation("imf_fiscal")
    print(f"Prepared simulation data for {len(sim_data)} entities")
    
    # Save processed data
    output_file = os.path.join(data_dir, "processed_fiscal_data.csv")
    preprocessor.save_processed_data("imf_fiscal", output_file)


if __name__ == "__main__":
    main()
