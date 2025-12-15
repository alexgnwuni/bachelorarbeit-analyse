# Quick Start Guide

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alexgnwuni/bachelorarbeit-analyse.git
   cd bachelorarbeit-analyse
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Using the Application

### Step 1: Upload CSV Files
1. Use the sidebar on the left to upload up to 3 CSV files
2. Click "Browse files" for each dataset slot
3. Select your CSV files
4. Click "Load Datasets" button
5. Wait for success messages confirming all datasets are loaded

### Step 2: Explore Your Data
Navigate through the tabs:

#### 📋 Data Overview
- View your data in a table format
- See summary statistics (mean, median, std, min, max, etc.)
- Check column information (data types, null counts, unique values)
- Identify missing data

#### 📈 Correlations
- Select a correlation method (Pearson, Spearman, or Kendall)
- View the correlation heatmap
- Select a target variable to see its correlations with all other variables
- Analyze top correlations with bar charts

#### 📊 Visualizations
Choose from multiple visualization types:
- **Scatter Plot**: Compare two numerical variables
  - Select X and Y axes
  - Optionally color points by a category
  - Optionally size points by a numerical variable
- **Histogram**: View distribution of a single variable
  - Adjust number of bins
- **Box Plot**: Compare distributions across multiple variables
  - Select multiple columns to compare
- **Line Plot**: Show trends over time or sequence
  - Select X-axis and multiple Y-axes
- **Pair Plot**: See all pairwise relationships
  - Select 2-5 variables to compare

#### 🔍 Advanced Analysis
- **Distribution Analysis**: Detailed statistics including skewness and kurtosis
- **Outlier Detection**: Find outliers using IQR method
- **Group Statistics**: Aggregate data by categorical variables

#### 🔗 Data Merging
- Merge two datasets on a common column
- Choose merge type (inner, outer, left, right)
- Preview and download the merged dataset

## Using the Sample Data

The repository includes sample CSV files for testing:
1. `sample_data_employees.csv` - Employee information
2. `sample_data_sales.csv` - Sales data
3. `sample_data_departments.csv` - Department details

Try uploading these files to explore all features!

## Tips & Tricks

1. **Multiple Datasets**: You can analyze each dataset independently or merge them
2. **Interactive Plots**: Hover over Plotly charts to see detailed information
3. **Color Coding**: Use the color option in scatter plots to identify patterns by category
4. **Outlier Detection**: Use the Advanced Analysis tab to find unusual data points
5. **Export Data**: After merging datasets, download the result as CSV

## Customization

### Plot Styles
Change the visualization style in the sidebar under "Visualization Settings":
- default
- seaborn
- ggplot
- dark_background

### Data Requirements
Your CSV files should:
- Have headers in the first row
- Use comma (`,`) as delimiter
- Be properly formatted with consistent data types per column

## Troubleshooting

**Problem**: Error loading CSV file
- **Solution**: Check that your file is properly formatted with headers

**Problem**: Correlation heatmap shows error
- **Solution**: Ensure you have at least 2 numerical columns

**Problem**: Merge operation fails
- **Solution**: Verify that both datasets have the selected common column

**Problem**: Visualization doesn't appear
- **Solution**: Ensure you've selected appropriate columns for the chart type

## Next Steps

1. Try uploading your own CSV data
2. Explore different correlation methods
3. Create custom visualizations
4. Merge datasets for combined analysis
5. Export your findings

## Support

For issues or questions, please open an issue on the GitHub repository.
