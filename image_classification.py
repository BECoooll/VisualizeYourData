import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
# Streamlit app
def main():
    st.title("Data Visualization and Statistics")

    # Load company logo
    logo_path = "logo/Mockup.jpg"
    logo = Image.open(logo_path)

    # Display company logo
    st.sidebar.image(logo, use_column_width=True)

    # Data input: Upload CSV or Excel file
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read data into DataFrame
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            # Display data summary
            st.subheader("Data Summary")
            st.write(df.describe())

            # Data visualization: Plot data
            st.subheader("Data Visualization")
            selected_columns = st.multiselect("Select columns to visualize", options=df.columns)
            plot_type = st.selectbox("Select plot type", options=["Violin Plot", "Ridgeline Plot", "Hexbin Plot", "Sunburst Plot"])

            for column in selected_columns:
                if plot_type == "Violin Plot":
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

        except Exception as e:
            st.sidebar.error(f"Error: {e}")

if __name__ == "__main__":
    main()
