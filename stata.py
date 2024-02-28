import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Default file path
DEFAULT_FILE_PATH = "US_MarketSTock.csv"

def main():
    st.title("Basic Analysis with Streamlit")

    # Upload file
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"], key="file_uploader")
    if uploaded_file is None:
        uploaded_file = DEFAULT_FILE_PATH

    # Read file into DataFrame
    if isinstance(uploaded_file, str):
        df = pd.read_excel(uploaded_file) if uploaded_file.endswith('.xlsx') else pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file.name) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file.name)

    st.subheader("Data Preview")
    st.write(df.head())

    st.subheader("Descriptive Statistics")
    numeric_columns = df.select_dtypes(include=np.number).columns
    numeric_df = df[numeric_columns]
    st.write(numeric_df.describe())

    st.subheader("Correlation Heatmap")
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    heatmap_fig, heatmap_ax = plt.subplots()
    heatmap_ax = sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(heatmap_fig)

    st.subheader("Histogram")
    selected_column = st.selectbox("Select a column for histogram", options=numeric_columns)
    hist_fig, hist_ax = plt.subplots()
    hist_ax.hist(numeric_df[selected_column].astype(float), bins='auto', color='blue', alpha=0.7)
    hist_ax.set_xlabel(selected_column)
    hist_ax.set_ylabel('Frequency')
    st.pyplot(hist_fig)

    st.subheader("Scatter Plot")
    col1, col2 = st.beta_columns(2)
    x_column = col1.selectbox("X Axis", options=numeric_columns)
    y_column = col2.selectbox("Y Axis", options=numeric_columns)
    scatter_fig, scatter_ax = plt.subplots()
    scatter_ax.scatter(numeric_df[x_column].astype(float), numeric_df[y_column].astype(float))
    scatter_ax.set_xlabel(x_column)
    scatter_ax.set_ylabel(y_column)
    st.pyplot(scatter_fig)

    st.subheader("Summary Statistics by Group")
    group_by_column = st.selectbox("Select a column for group by", options=df.columns)
    summary_by_column = st.selectbox("Select a column for summary statistics", options=numeric_columns)
    summary_df = numeric_df.groupby(group_by_column)[summary_by_column].describe()
    st.write(summary_df)

if __name__ == "__main__":
    main()
