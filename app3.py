import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Set default file path
DEFAULT_FILE_PATH = "default_data.xlsx"

# Columns to advise user to remove
COLUMNS_TO_REMOVE = ["id", "date"]

# Streamlit app
def main():
    st.title("Data Visualization and Statistics")

    # Load company logo
    logo_path = "logo/Mockup (5).jpg"
    logo = Image.open(logo_path)

    # Display company logo
    st.sidebar.image(logo, use_column_width=True)

    # Data input: Upload CSV or Excel file
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    # Use default file path if no file is uploaded
    if uploaded_file is None:
        uploaded_file = DEFAULT_FILE_PATH

    # Read data into DataFrame
    try:
        df = pd.read_excel(uploaded_file)

        # Button to go to custom data visualization
        if st.button("Go to Custom Data Visualization"):
            js_scroll_to_custom_visualization()

        # Display data summary
        st.subheader("Data Summary")
        columns_to_remove = st.multiselect("Select columns to remove", options=df.columns, default=[col for col in COLUMNS_TO_REMOVE if col in df.columns], help="Consider removing columns like 'id', 'date', etc.")
        df_summary = df.drop(columns_to_remove, axis=1).describe()
        st.write(df_summary)

        # Exploratory Data Analysis (EDA)
        st.subheader("Exploratory Data Analysis (EDA)")

        # Automatic EDA: Common plots for each column
        for column in df.columns:
            st.write(f"### {column}")
            if df[column].dtype in ['int64', 'float64']:
                st.write(f"**Histogram**")
                fig = px.histogram(df, x=column, title=f"{column} Histogram")
                st.plotly_chart(fig)
                st.write(f"**Box Plot**")
                fig = px.box(df, y=column, title=f"{column} Box Plot")
                st.plotly_chart(fig)
            elif df[column].dtype == 'object':
                st.write(f"**Count Plot**")
                fig = px.histogram(df, x=column, title=f"{column} Count Plot")
                st.plotly_chart(fig)

        # Custom data visualization section
        st.markdown('<div id="custom_viz"></div>', unsafe_allow_html=True)  # Placeholder for custom data visualization

        selected_columns = st.multiselect("Select columns to visualize", options=df.columns)
        plot_type = st.selectbox("Select plot type", options=["Histogram", "Box Plot", "Scatter Plot", "Violin Plot", "Ridgeline Plot", "Hexbin Plot", "Sunburst Plot", "TreeMap"])

        for column in selected_columns:
            st.write(f"### {column} - {plot_type}")
            if plot_type == "Histogram":
                fig = px.histogram(df, x=column, title=f"{column} Histogram")
                st.plotly_chart(fig)
            elif plot_type == "Box Plot":
                fig = px.box(df, y=column, title=f"{column} Box Plot")
                st.plotly_chart(fig)
            elif plot_type == "Scatter Plot":
                fig = px.scatter(df, x=column, title=f"{column} Scatter Plot")
                st.plotly_chart(fig)
            elif plot_type == "Violin Plot":
                fig = px.violin(df, y=column, box=True, points="all", title=f"{column} Violin Plot")
                st.plotly_chart(fig)
            elif plot_type == "Ridgeline Plot":
                sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
                fig, ax = plt.subplots()
                for _, group in df.groupby(column):
                    sns.kdeplot(data=group[column], ax=ax, fill=True)
                st.pyplot(fig)
            elif plot_type == "Hexbin Plot":
                fig = px.density_heatmap(df, x=column, y=column, title=f"{column} Hexbin Plot")
                st.plotly_chart(fig)
            elif plot_type == "Sunburst Plot":
                df_counts = df[column].value_counts().reset_index()
                df_counts.columns = ['Category', 'Count']
                fig = px.sunburst(df_counts, path=['Category'], values='Count', title=f"{column} Sunburst Plot")
                st.plotly_chart(fig)
            elif plot_type == "TreeMap":
                fig = px.treemap(df, path=[column], title=f"{column} TreeMap")
                st.plotly_chart(fig)

    except Exception as e:
        st.sidebar.error(f"Error: {e}")

def js_scroll_to_custom_visualization():
    js_scroll_to_element = "document.getElementById('custom_viz').scrollIntoView();"
    st.write(f'<script>{js_scroll_to_element}</script>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
