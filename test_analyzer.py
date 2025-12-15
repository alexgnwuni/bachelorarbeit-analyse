"""
Test script for the Data Analyzer module
"""

import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from data_analyzer import DataAnalyzer

def test_data_analyzer():
    """Test the DataAnalyzer class with sample data"""
    
    print("=" * 60)
    print("Testing Data Analyzer Module")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DataAnalyzer()
    print("\n✓ DataAnalyzer initialized")
    
    # Load sample CSV files
    try:
        df1 = analyzer.load_csv('sample_data_employees.csv', 'employees')
        print(f"✓ Loaded employees dataset: {len(df1)} rows, {len(df1.columns)} columns")
        
        df2 = analyzer.load_csv('sample_data_sales.csv', 'sales')
        print(f"✓ Loaded sales dataset: {len(df2)} rows, {len(df2.columns)} columns")
        
        df3 = analyzer.load_csv('sample_data_departments.csv', 'departments')
        print(f"✓ Loaded departments dataset: {len(df3)} rows, {len(df3.columns)} columns")
    except Exception as e:
        print(f"✗ Error loading CSV files: {e}")
        return False
    
    # Test summary statistics
    print("\n" + "-" * 60)
    print("Summary Statistics for Employees:")
    print("-" * 60)
    summary = analyzer.get_summary_statistics(df1)
    print(summary)
    
    # Test correlation matrix
    print("\n" + "-" * 60)
    print("Correlation Matrix for Employees:")
    print("-" * 60)
    corr_matrix = analyzer.get_correlation_matrix(df1)
    print(corr_matrix)
    
    # Test column info
    print("\n" + "-" * 60)
    print("Column Information for Employees:")
    print("-" * 60)
    col_info = analyzer.get_column_info(df1)
    print(col_info)
    
    # Test correlation with target
    print("\n" + "-" * 60)
    print("Correlation with Salary:")
    print("-" * 60)
    target_corr = analyzer.analyze_correlation_with_target(df1, 'Salary')
    print(target_corr)
    
    # Test missing data summary
    print("\n" + "-" * 60)
    print("Missing Data Summary for Employees:")
    print("-" * 60)
    missing = analyzer.get_missing_data_summary(df1)
    if len(missing) == 0:
        print("No missing data found ✓")
    else:
        print(missing)
    
    # Test merge functionality
    print("\n" + "-" * 60)
    print("Testing Data Merge:")
    print("-" * 60)
    try:
        merged = analyzer.merge_datasets('employees', 'departments', on='Department', how='left')
        print(f"✓ Merged datasets successfully: {len(merged)} rows, {len(merged.columns)} columns")
        print("\nFirst 5 rows of merged data:")
        print(merged.head())
    except Exception as e:
        print(f"✗ Error merging datasets: {e}")
    
    # Test visualizations
    print("\n" + "-" * 60)
    print("Testing Visualizations:")
    print("-" * 60)
    
    try:
        # Create temporary directory for output files
        temp_dir = tempfile.gettempdir()
        
        # Test histogram
        fig = analyzer.plot_histogram(df1, 'Age')
        plt.savefig(os.path.join(temp_dir, 'test_histogram.png'))
        plt.close()
        print("✓ Histogram created successfully")
        
        # Test correlation heatmap
        fig = analyzer.plot_correlation_heatmap(df1)
        plt.savefig(os.path.join(temp_dir, 'test_heatmap.png'))
        plt.close()
        print("✓ Correlation heatmap created successfully")
        
        # Test box plot
        fig = analyzer.plot_box(df1, ['Age', 'Salary', 'Experience'])
        plt.savefig(os.path.join(temp_dir, 'test_boxplot.png'))
        plt.close()
        print("✓ Box plot created successfully")
        
        # Test scatter plot
        fig = analyzer.plot_scatter(df1, 'Experience', 'Salary', color_col='Department')
        fig.write_html(os.path.join(temp_dir, 'test_scatter.html'))
        print("✓ Scatter plot created successfully")
        
    except Exception as e:
        print(f"✗ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_data_analyzer()
