"""
Data Analysis Module
Provides core functionality for CSV data loading, analysis, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from typing import List, Dict, Tuple, Optional


class DataAnalyzer:
    """Main class for data analysis operations."""
    
    def __init__(self):
        self.datasets = {}
        self.dataset_names = []
    
    def load_csv(self, file, name: str) -> pd.DataFrame:
        """
        Load a CSV file and store it in the datasets dictionary.
        
        Args:
            file: File object or path to CSV file
            name: Name to identify the dataset
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            df = pd.read_csv(file)
            self.datasets[name] = df
            if name not in self.dataset_names:
                self.dataset_names.append(name)
            return df
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get a dataset by name."""
        return self.datasets.get(name)
    
    def get_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with summary statistics
        """
        return df.describe()
    
    def get_correlation_matrix(self, df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for numerical columns.
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix as DataFrame
        """
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr(method=method)
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, method: str = 'pearson', 
                                 figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create a correlation heatmap.
        
        Args:
            df: Input DataFrame
            method: Correlation method
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        corr_matrix = self.get_correlation_matrix(df, method)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f'Correlation Matrix ({method.capitalize()})', fontsize=16, pad=20)
        plt.tight_layout()
        return fig
    
    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, 
                    color_col: Optional[str] = None, size_col: Optional[str] = None,
                    title: Optional[str] = None) -> go.Figure:
        """
        Create an interactive scatter plot.
        
        Args:
            df: Input DataFrame
            x_col: Column for x-axis
            y_col: Column for y-axis
            color_col: Optional column for color encoding
            size_col: Optional column for size encoding
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if title is None:
            title = f'{y_col} vs {x_col}'
        
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                        title=title, hover_data=df.columns)
        fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
        return fig
    
    def plot_histogram(self, df: pd.DataFrame, column: str, bins: int = 30,
                      color: str = 'steelblue') -> plt.Figure:
        """
        Create a histogram for a numerical column.
        
        Args:
            df: Input DataFrame
            column: Column name
            bins: Number of bins
            color: Bar color
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[column].dropna(), bins=bins, color=color, edgecolor='black', alpha=0.7)
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of {column}', fontsize=14, pad=20)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_box(self, df: pd.DataFrame, columns: List[str]) -> plt.Figure:
        """
        Create box plots for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of column names
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        data_to_plot = [df[col].dropna() for col in columns]
        ax.boxplot(data_to_plot, labels=columns)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title('Box Plot Comparison', fontsize=14, pad=20)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def plot_line(self, df: pd.DataFrame, x_col: str, y_cols: List[str],
                 title: Optional[str] = None) -> go.Figure:
        """
        Create a line plot with multiple y-axis variables.
        
        Args:
            df: Input DataFrame
            x_col: Column for x-axis
            y_cols: List of columns for y-axis
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if title is None:
            title = 'Line Plot'
        
        fig = go.Figure()
        for col in y_cols:
            fig.add_trace(go.Scatter(x=df[x_col], y=df[col], mode='lines+markers', name=col))
        
        fig.update_layout(title=title, xaxis_title=x_col, yaxis_title='Values',
                         hovermode='x unified')
        return fig
    
    def plot_pairplot(self, df: pd.DataFrame, columns: List[str], 
                     hue: Optional[str] = None) -> plt.Figure:
        """
        Create a pair plot for multiple columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to include
            hue: Optional column for color encoding
            
        Returns:
            Seaborn PairGrid figure
        """
        subset_df = df[columns + ([hue] if hue and hue not in columns else [])]
        pairplot = sns.pairplot(subset_df, hue=hue, diag_kind='hist', 
                               plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.7})
        pairplot.fig.suptitle('Pair Plot Analysis', y=1.02, fontsize=16)
        return pairplot.fig
    
    def get_column_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get information about DataFrame columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with column information
        """
        info_data = {
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        }
        return pd.DataFrame(info_data)
    
    def analyze_correlation_with_target(self, df: pd.DataFrame, target_col: str,
                                       method: str = 'pearson') -> pd.DataFrame:
        """
        Analyze correlation of all numerical columns with a target column.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            method: Correlation method
            
        Returns:
            DataFrame with correlations sorted by absolute value
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col not in numeric_df.columns:
            raise ValueError(f"Target column '{target_col}' must be numerical")
        
        correlations = numeric_df.corr(method=method)[target_col].sort_values(
            ascending=False, key=abs
        )
        
        result_df = pd.DataFrame({
            'Column': correlations.index,
            'Correlation': correlations.values,
            'Abs_Correlation': correlations.abs().values
        })
        
        return result_df[result_df['Column'] != target_col]
    
    def compute_bias_detection_performance(
        self,
        runs_df: pd.DataFrame,
        participants_df: pd.DataFrame,
        biased_only: bool = True
    ) -> pd.DataFrame:
        """
        Compute per-participant bias detection performance and merge with demographics.
        
        This helper assumes a schema similar to the provided study:
        - runs_df: has columns 'participant_id', 'is_correct' (bool/int), and optionally 'is_biased'
        - participants_df: has columns 'id' (participant id), 'age' and other attributes
        
        Args:
            runs_df: Scenario runs data
            participants_df: Participants metadata
            biased_only: If True, use only rows where is_biased == True
        
        Returns:
            DataFrame with one row per participant containing:
            - participant_id
            - n_trials, n_correct, accuracy
            - merged columns from participants_df (e.g. age, gender, ai_knowledge, ...)
        """
        df_runs = runs_df.copy()
        
        # Optional filtering to only biased scenarios if column exists and flag is set
        if biased_only and 'is_biased' in df_runs.columns:
            df_runs = df_runs[df_runs['is_biased'] == True]
        
        # Ensure required columns exist
        required_cols = {'participant_id', 'is_correct'}
        missing = required_cols - set(df_runs.columns)
        if missing:
            raise ValueError(f"runs_df is missing required columns: {', '.join(missing)}")
        
        # Group by participant and compute accuracy
        grouped = (
            df_runs
            .groupby('participant_id')['is_correct']
            .agg(n_trials='count', n_correct='sum')
            .reset_index()
        )
        
        # Avoid division by zero
        grouped['accuracy'] = grouped.apply(
            lambda row: row['n_correct'] / row['n_trials'] if row['n_trials'] > 0 else np.nan,
            axis=1
        )
        
        # Merge with participant metadata (expects participants_df.id to match participant_id)
        if 'id' not in participants_df.columns:
            raise ValueError("participants_df must contain an 'id' column to join on")
        
        merged = grouped.merge(
            participants_df,
            left_on='participant_id',
            right_on='id',
            how='left'
        )
        
        return merged
    
    def correlation_age_vs_bias_performance(
        self,
        performance_df: pd.DataFrame,
        min_trials: int = 1,
        method: str = 'spearman'
    ) -> Dict[str, float]:
        """
        Compute correlation between age and bias-detection accuracy.
        
        Args:
            performance_df: Output of compute_bias_detection_performance
            min_trials: Minimum number of trials per participant to be included
            method: Correlation method ('spearman' or 'pearson')
        
        Returns:
            Dictionary with keys: 'correlation', 'p_value', 'n'
        """
        if 'age' not in performance_df.columns or 'accuracy' not in performance_df.columns:
            raise ValueError("performance_df must contain 'age' and 'accuracy' columns")
        
        df = performance_df.copy()
        if 'n_trials' in df.columns:
            df = df[df['n_trials'] >= min_trials]
        
        df = df[['age', 'accuracy']].dropna()
        if len(df) < 2:
            return {'correlation': np.nan, 'p_value': np.nan, 'n': len(df)}
        
        if method == 'spearman':
            corr, p = stats.spearmanr(df['age'], df['accuracy'])
        elif method == 'pearson':
            corr, p = stats.pearsonr(df['age'], df['accuracy'])
        else:
            raise ValueError("method must be 'spearman' or 'pearson'")
        
        return {'correlation': float(corr), 'p_value': float(p), 'n': int(len(df))}
    
    def merge_datasets(self, name1: str, name2: str, on: str, how: str = 'inner') -> pd.DataFrame:
        """
        Merge two datasets.
        
        Args:
            name1: Name of first dataset
            name2: Name of second dataset
            on: Column to merge on
            how: Type of merge ('inner', 'outer', 'left', 'right')
            
        Returns:
            Merged DataFrame
        """
        df1 = self.get_dataset(name1)
        df2 = self.get_dataset(name2)
        
        if df1 is None or df2 is None:
            raise ValueError("One or both datasets not found")
        
        return pd.merge(df1, df2, on=on, how=how)
    
    def get_missing_data_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of missing data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing data statistics
        """
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum().values,
            'Missing_Percentage': (df.isnull().sum() / len(df) * 100).values
        })
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )
        return missing_data
