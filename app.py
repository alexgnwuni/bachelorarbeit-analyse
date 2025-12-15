"""
Streamlit Data Analysis Application
Interactive web interface for CSV data analysis and visualization.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_analyzer import DataAnalyzer
import io

# Page configuration
st.set_page_config(
    page_title="Data Analysis Environment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = DataAnalyzer()
if 'datasets_loaded' not in st.session_state:
    st.session_state.datasets_loaded = {}

# Main title
st.markdown('<p class="main-header">📊 Data Analysis Environment</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for file uploads
with st.sidebar:
    st.header("📁 Upload CSV Files")
    st.markdown("Upload up to 3 CSV files for analysis")
    
    uploaded_files = []
    for i in range(1, 4):
        file = st.file_uploader(f"Dataset {i}", type=['csv'], key=f"file_{i}")
        if file:
            uploaded_files.append((file, f"Dataset_{i}"))
    
    if uploaded_files:
        if st.button("Load Datasets", type="primary"):
            with st.spinner("Loading datasets..."):
                st.session_state.datasets_loaded = {}
                for file, name in uploaded_files:
                    try:
                        df = st.session_state.analyzer.load_csv(file, name)
                        st.session_state.datasets_loaded[name] = True
                        st.success(f"✅ {name} loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error loading {name}: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 🎨 Visualization Settings")
    plot_style = st.selectbox("Plot Style", ['default', 'seaborn', 'ggplot', 'dark_background'])
    if plot_style != 'default':
        plt.style.use(plot_style)

# Main content area
if not st.session_state.datasets_loaded:
    st.info("👆 Please upload CSV files using the sidebar to begin analysis")
    
    # Show example instructions
    st.markdown("### 📖 How to Use")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ Upload Data")
        st.write("Upload up to 3 CSV files using the sidebar")
    
    with col2:
        st.markdown("#### 2️⃣ Explore")
        st.write("View data summaries, statistics, and column information")
    
    with col3:
        st.markdown("#### 3️⃣ Analyze")
        st.write("Create visualizations, correlations, and insights")
    
else:
    # Tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Overview", 
        "📈 Correlations", 
        "📊 Visualizations",
        "🔍 Advanced Analysis",
        "🔗 Data Merging"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.markdown('<p class="sub-header">Data Overview</p>', unsafe_allow_html=True)
        
        dataset_name = st.selectbox("Select Dataset", 
                                   list(st.session_state.datasets_loaded.keys()),
                                   key="overview_dataset")
        
        if dataset_name:
            df = st.session_state.analyzer.get_dataset(dataset_name)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Numerical Columns", len(df.select_dtypes(include='number').columns))
            with col4:
                st.metric("Categorical Columns", len(df.select_dtypes(include='object').columns))
            
            st.markdown("---")
            
            # Data preview
            st.subheader("📄 Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Column information
            st.subheader("ℹ️ Column Information")
            col_info = st.session_state.analyzer.get_column_info(df)
            st.dataframe(col_info, use_container_width=True)
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            summary = st.session_state.analyzer.get_summary_statistics(df)
            st.dataframe(summary, use_container_width=True)
            
            # Missing data
            st.subheader("❓ Missing Data Analysis")
            missing_data = st.session_state.analyzer.get_missing_data_summary(df)
            if len(missing_data) > 0:
                st.dataframe(missing_data, use_container_width=True)
            else:
                st.success("No missing data found! 🎉")
    
    # Tab 2: Correlations
    with tab2:
        st.markdown('<p class="sub-header">Correlation Analysis</p>', unsafe_allow_html=True)
        
        dataset_name = st.selectbox("Select Dataset", 
                                   list(st.session_state.datasets_loaded.keys()),
                                   key="corr_dataset")
        
        if dataset_name:
            df = st.session_state.analyzer.get_dataset(dataset_name)
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            
            if len(numeric_cols) < 2:
                st.warning("⚠️ This dataset needs at least 2 numerical columns for correlation analysis")
            else:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    corr_method = st.radio("Correlation Method", 
                                          ['pearson', 'spearman', 'kendall'])
                
                with col2:
                    st.subheader("🔥 Correlation Heatmap")
                    fig = st.session_state.analyzer.plot_correlation_heatmap(df, method=corr_method)
                    st.pyplot(fig)
                    plt.close()
                
                st.markdown("---")
                
                # Correlation with target variable
                st.subheader("🎯 Target Variable Correlation")
                target_col = st.selectbox("Select Target Column", numeric_cols)
                
                if target_col:
                    corr_with_target = st.session_state.analyzer.analyze_correlation_with_target(
                        df, target_col, method=corr_method
                    )
                    st.dataframe(corr_with_target, use_container_width=True)
                    
                    # Plot top correlations
                    top_n = min(10, len(corr_with_target))
                    if top_n > 0:
                        st.subheader(f"Top {top_n} Correlations with {target_col}")
                        top_corr = corr_with_target.head(top_n)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['green' if x > 0 else 'red' for x in top_corr['Correlation']]
                        ax.barh(top_corr['Column'], top_corr['Correlation'], color=colors, alpha=0.7)
                        ax.set_xlabel('Correlation Coefficient')
                        ax.set_title(f'Top Correlations with {target_col}')
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                        ax.grid(axis='x', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
    
    # Tab 3: Visualizations
    with tab3:
        st.markdown('<p class="sub-header">Data Visualizations</p>', unsafe_allow_html=True)
        
        dataset_name = st.selectbox("Select Dataset", 
                                   list(st.session_state.datasets_loaded.keys()),
                                   key="viz_dataset")
        
        if dataset_name:
            df = st.session_state.analyzer.get_dataset(dataset_name)
            
            viz_type = st.selectbox("Select Visualization Type", 
                                   ["Scatter Plot", "Histogram", "Box Plot", 
                                    "Line Plot", "Pair Plot"])
            
            st.markdown("---")
            
            if viz_type == "Scatter Plot":
                st.subheader("📍 Scatter Plot")
                col1, col2 = st.columns(2)
                
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                all_cols = df.columns.tolist()
                
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols)
                    color_col = st.selectbox("Color by (optional)", 
                                           ["None"] + all_cols)
                
                with col2:
                    y_col = st.selectbox("Y-axis", numeric_cols)
                    size_col = st.selectbox("Size by (optional)", 
                                          ["None"] + numeric_cols)
                
                if x_col and y_col:
                    color = None if color_col == "None" else color_col
                    size = None if size_col == "None" else size_col
                    
                    fig = st.session_state.analyzer.plot_scatter(
                        df, x_col, y_col, color_col=color, size_col=size
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Histogram":
                st.subheader("📊 Histogram")
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    hist_col = st.selectbox("Select Column", numeric_cols)
                with col2:
                    bins = st.slider("Number of Bins", 10, 100, 30)
                
                if hist_col:
                    fig = st.session_state.analyzer.plot_histogram(df, hist_col, bins=bins)
                    st.pyplot(fig)
                    plt.close()
            
            elif viz_type == "Box Plot":
                st.subheader("📦 Box Plot")
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                
                selected_cols = st.multiselect("Select Columns", numeric_cols, 
                                              default=numeric_cols[:min(5, len(numeric_cols))])
                
                if selected_cols:
                    fig = st.session_state.analyzer.plot_box(df, selected_cols)
                    st.pyplot(fig)
                    plt.close()
            
            elif viz_type == "Line Plot":
                st.subheader("📈 Line Plot")
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                all_cols = df.columns.tolist()
                
                x_col = st.selectbox("X-axis", all_cols)
                y_cols = st.multiselect("Y-axis (multiple)", numeric_cols,
                                       default=numeric_cols[:min(3, len(numeric_cols))])
                
                if x_col and y_cols:
                    fig = st.session_state.analyzer.plot_line(df, x_col, y_cols)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Pair Plot":
                st.subheader("🔲 Pair Plot")
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                all_cols = df.columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    pair_cols = st.multiselect("Select Columns (2-5 recommended)", 
                                              numeric_cols,
                                              default=numeric_cols[:min(3, len(numeric_cols))])
                with col2:
                    hue_col = st.selectbox("Color by (optional)", ["None"] + all_cols)
                
                if len(pair_cols) >= 2:
                    hue = None if hue_col == "None" else hue_col
                    with st.spinner("Generating pair plot..."):
                        fig = st.session_state.analyzer.plot_pairplot(df, pair_cols, hue=hue)
                        st.pyplot(fig)
                        plt.close()
                else:
                    st.warning("Please select at least 2 columns for pair plot")
    
    # Tab 4: Advanced Analysis
    with tab4:
        st.markdown('<p class="sub-header">Advanced Analysis</p>', unsafe_allow_html=True)
        
        dataset_name = st.selectbox("Select Dataset", 
                                   list(st.session_state.datasets_loaded.keys()),
                                   key="advanced_dataset")
        
        if dataset_name:
            df = st.session_state.analyzer.get_dataset(dataset_name)
            
            analysis_type = st.selectbox("Select Analysis Type",
                                        ["Distribution Analysis", "Outlier Detection",
                                         "Group Statistics"])
            
            st.markdown("---")
            
            if analysis_type == "Distribution Analysis":
                st.subheader("📐 Distribution Analysis")
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                
                col = st.selectbox("Select Column", numeric_cols)
                
                if col:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Basic Statistics**")
                        stats_data = {
                            "Mean": df[col].mean(),
                            "Median": df[col].median(),
                            "Std Dev": df[col].std(),
                            "Min": df[col].min(),
                            "Max": df[col].max(),
                            "Skewness": df[col].skew(),
                            "Kurtosis": df[col].kurtosis()
                        }
                        st.dataframe(pd.DataFrame(stats_data.items(), 
                                                 columns=['Metric', 'Value']))
                    
                    with col2:
                        st.write("**Percentiles**")
                        percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
                        perc_data = {f"{int(p*100)}%": df[col].quantile(p) 
                                    for p in percentiles}
                        st.dataframe(pd.DataFrame(perc_data.items(),
                                                 columns=['Percentile', 'Value']))
            
            elif analysis_type == "Outlier Detection":
                st.subheader("🎯 Outlier Detection (IQR Method)")
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                
                col = st.selectbox("Select Column", numeric_cols)
                
                if col:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Outliers", len(outliers))
                    with col2:
                        st.metric("Lower Bound", f"{lower_bound:.2f}")
                    with col3:
                        st.metric("Upper Bound", f"{upper_bound:.2f}")
                    
                    if len(outliers) > 0:
                        st.write("**Outlier Records:**")
                        st.dataframe(outliers, use_container_width=True)
            
            elif analysis_type == "Group Statistics":
                st.subheader("📊 Group Statistics")
                
                categorical_cols = df.select_dtypes(include='object').columns.tolist()
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                
                if not categorical_cols:
                    st.warning("No categorical columns found for grouping")
                elif not numeric_cols:
                    st.warning("No numerical columns found for analysis")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        group_col = st.selectbox("Group By", categorical_cols)
                    with col2:
                        value_col = st.selectbox("Analyze Column", numeric_cols)
                    
                    if group_col and value_col:
                        group_stats = df.groupby(group_col)[value_col].agg([
                            'count', 'mean', 'median', 'std', 'min', 'max'
                        ]).round(2)
                        st.dataframe(group_stats, use_container_width=True)
                        
                        # Visualize grouped data
                        fig, ax = plt.subplots(figsize=(10, 6))
                        df.groupby(group_col)[value_col].mean().plot(kind='bar', ax=ax, color='steelblue')
                        ax.set_ylabel(f'Mean {value_col}')
                        ax.set_title(f'Mean {value_col} by {group_col}')
                        ax.grid(axis='y', alpha=0.3)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
    
    # Tab 5: Data Merging
    with tab5:
        st.markdown('<p class="sub-header">Data Merging</p>', unsafe_allow_html=True)
        
        if len(st.session_state.datasets_loaded) < 2:
            st.info("⚠️ Please load at least 2 datasets to use the merge functionality")
        else:
            st.write("Combine multiple datasets based on common columns")
            
            col1, col2, col3 = st.columns(3)
            
            dataset_names = list(st.session_state.datasets_loaded.keys())
            
            with col1:
                dataset1 = st.selectbox("First Dataset", dataset_names, key="merge_ds1")
            with col2:
                dataset2 = st.selectbox("Second Dataset", 
                                       [d for d in dataset_names if d != dataset1],
                                       key="merge_ds2")
            with col3:
                merge_type = st.selectbox("Merge Type", 
                                         ['inner', 'outer', 'left', 'right'])
            
            if dataset1 and dataset2:
                df1 = st.session_state.analyzer.get_dataset(dataset1)
                df2 = st.session_state.analyzer.get_dataset(dataset2)
                
                common_cols = list(set(df1.columns) & set(df2.columns))
                
                if not common_cols:
                    st.error("❌ No common columns found between datasets")
                else:
                    merge_col = st.selectbox("Merge On Column", common_cols)
                    
                    if st.button("Merge Datasets", type="primary"):
                        try:
                            merged_df = st.session_state.analyzer.merge_datasets(
                                dataset1, dataset2, on=merge_col, how=merge_type
                            )
                            
                            st.success(f"✅ Datasets merged successfully! Result has {len(merged_df)} rows")
                            
                            # Save merged dataset
                            merge_name = f"Merged_{dataset1}_{dataset2}"
                            st.session_state.analyzer.datasets[merge_name] = merged_df
                            st.session_state.datasets_loaded[merge_name] = True
                            
                            st.subheader("📄 Merged Data Preview")
                            st.dataframe(merged_df.head(20), use_container_width=True)
                            
                            # Download option
                            csv = merged_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Merged Dataset",
                                data=csv,
                                file_name=f"{merge_name}.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error merging datasets: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Built with Streamlit • Python Data Analysis Environment</p>
    </div>
""", unsafe_allow_html=True)
