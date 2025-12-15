# 📊 Data Analysis Environment

A comprehensive Python-based data analysis web application built with Streamlit. Upload CSV files and perform interactive data exploration, statistical analysis, and visualization.

## ✨ Features

### 📁 Data Management
- Upload up to 3 CSV files simultaneously
- Automatic data type detection
- Data preview and exploration
- Missing data analysis
- Column information and statistics

### 📈 Correlation Analysis
- Multiple correlation methods (Pearson, Spearman, Kendall)
- Interactive correlation heatmaps
- Target variable correlation analysis
- Top correlations visualization

### 📊 Visualizations
- **Scatter Plots**: Interactive with color and size encoding
- **Histograms**: Customizable distribution plots
- **Box Plots**: Compare multiple variables
- **Line Plots**: Multi-variable trend analysis
- **Pair Plots**: Comprehensive relationship analysis

### 🔍 Advanced Analysis
- Distribution statistics (mean, median, skewness, kurtosis)
- Outlier detection using IQR method
- Group statistics and aggregations
- Percentile analysis

### 🔗 Data Operations
- Merge multiple datasets
- Support for different join types (inner, outer, left, right)
- Export merged datasets

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/alexgnwuni/bachelorarbeit-analyse.git
cd bachelorarbeit-analyse
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload Data
1. Use the sidebar to upload up to 3 CSV files
2. Click "Load Datasets" to process the files
3. Each dataset will be labeled (Dataset_1, Dataset_2, Dataset_3)

### 2. Explore Data
- Navigate to the **Data Overview** tab
- View data preview, column information, and summary statistics
- Check for missing values

### 3. Analyze Correlations
- Go to the **Correlations** tab
- Select a correlation method
- View the correlation heatmap
- Analyze correlations with a target variable

### 4. Create Visualizations
- Navigate to the **Visualizations** tab
- Choose from various plot types
- Select columns and customize settings
- Interact with plots for detailed insights

### 5. Advanced Analysis
- Use the **Advanced Analysis** tab for:
  - Distribution analysis with statistical metrics
  - Outlier detection
  - Group-based statistics

### 6. Merge Datasets
- Go to the **Data Merging** tab
- Select two datasets to merge
- Choose a common column and merge type
- Download the merged dataset

## 📦 Dependencies

- **pandas**: Data manipulation and analysis
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualizations
- **streamlit**: Web application framework
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **plotly**: Interactive visualizations
- **openpyxl**: Excel file support

## 🎨 Customization

### Plot Styles
The application supports multiple matplotlib styles:
- Default
- Seaborn
- ggplot
- Dark background

Select your preferred style from the sidebar.

## 📝 Example Use Cases

1. **Sales Analysis**: Upload sales data, analyze trends, correlations between price and sales volume
2. **Scientific Research**: Compare experimental data across multiple datasets, visualize distributions
3. **Financial Analysis**: Analyze stock prices, correlations between different assets
4. **Quality Control**: Detect outliers in manufacturing data, group statistics by product type

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is available for academic and educational purposes.

## 👤 Author

Alexander Gnwuni - Bachelor Thesis Project

## 🐛 Issues

If you encounter any issues or have suggestions, please open an issue on GitHub.

## 🔮 Future Enhancements

- [ ] Support for more file formats (Excel, JSON)
- [ ] Machine learning integration
- [ ] Time series analysis
- [ ] Export reports as PDF
- [ ] Database connectivity
- [ ] Custom plot themes
