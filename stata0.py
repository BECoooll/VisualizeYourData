import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LogisticRegression
from lifelines import KaplanMeierFitter
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.multivariate.manova import MANOVA

# Disable file uploader deprecation warning
st.set_option('deprecation.showfileUploaderEncoding', False)

# Function to load data
@st.cache
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, encoding='utf-8')  # Specify encoding as utf-8
        return data
    else:
        default_file = "default_data.csv"
        data = pd.read_csv(default_file, encoding='utf-8')  # Specify encoding as utf-8 for default file
        return data

# Function to perform descriptive statistics
def descriptive_statistics(data):
    st.write(data.describe())

    # Plot histograms for numerical columns
    numerical_cols = data.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        st.subheader(f'Histogram for {col}')
        fig, ax = plt.subplots()
        ax.hist(data[col])
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

# Function to perform inferential statistics (t-tests)
def perform_t_test(data, column, value):
    t_stat, p_value = stats.ttest_1samp(data[column], value)
    return t_stat, p_value

# Function to perform inferential statistics (ANOVA)
# Function to perform inferential statistics (ANOVA)
def perform_anova(data, groups, column):
    # Ensure column is numeric
    if data[column].dtype == 'object':
        st.error("The selected column is non-numeric. ANOVA requires numeric values for the dependent variable.")
        return None

    # Ensure groups are categorical
    if data[groups].dtype != 'category':
        st.error("The selected groups column should be categorical.")
        return None

    formula = f"{column} ~ C({groups})"
    model = ols(formula, data=data).fit()
    anova_table = anova_lm(model, typ=2)
    return anova_table


# Function to perform correlation analysis
def correlation_analysis(data):
    corr_matrix = data.corr()
    st.write(corr_matrix)

    # Plot correlation heatmap
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# Function to perform regression analysis
def perform_regression(data, x_columns, y_column):
    X = data[x_columns]
    y = data[y_column]

    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    X = sm.add_constant(X)  # Add constant for intercept

    model = sm.OLS(y, X).fit()
    return model.summary()

# Function to perform factor analysis
# Function to perform factor analysis
def factor_analysis(data, num_factors):
    # Exclude datetime columns from factor analysis
    numeric_data = data.select_dtypes(include=np.number)
    if len(numeric_data.columns) == 0:
        st.error("No numeric columns found for factor analysis.")
        return

    fa = FactorAnalysis(n_components=num_factors)
    fa.fit(numeric_data)
    st.write(fa.components_)


# Function to perform cluster analysis
def cluster_analysis(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_

    # Plot clusters for 2D data
    if len(data.columns) == 2:
        st.subheader('Cluster Plot')
        fig, ax = plt.subplots()
        ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis')
        ax.set_xlabel(data.columns[0])
        ax.set_ylabel(data.columns[1])
        st.pyplot(fig)

# Function to perform logistic regression
def logistic_regression(data, x_columns, y_column):
    X = data[x_columns]
    y = data[y_column]
    model = LogisticRegression().fit(X, y)
    return model.coef_

# Function to perform survival analysis
def survival_analysis(data, time_column, event_column):
    kmf = KaplanMeierFitter()
    kmf.fit(data[time_column], event_observed=data[event_column])
    st.write(kmf.survival_function_)

# Function to perform MANOVA
def perform_manova(data, groups, dependent_vars):
    manova = MANOVA.from_formula(dependent_vars + '~' + groups, data=data)
    st.write(manova.mv_test())

# Function to encode categorical variables
def encode_categorical(data):
    # One-hot encoding for nominal variables
    data = pd.get_dummies(data, columns=data.select_dtypes(include='object').columns)
    return data

# Main function to run the Streamlit app
def main():
    # Sidebar logo
    st.sidebar.image('logo/Mockup.jpg', use_column_width=True)

    st.sidebar.title('Data Analysis Tool')
    DEFAULT_FILE_PATH = "default_data.xlsx"
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
    if uploaded_file is None:
        uploaded_file = DEFAULT_FILE_PATH
    # Load data
    try:
        data = pd.read_excel(uploaded_file)
    except:
        data = pd.read_csv(uploaded_file)

    if data is not None:
        # Display uploaded data
        st.write('Uploaded Data:', data)

        # Encode categorical variables
        data = encode_categorical(data)

        # Options for analysis
        analysis_options = [
            "Descriptive Statistics",
            "T-Test",
            "ANOVA",
            "Correlation Analysis",
            "Regression Analysis",
            "Factor Analysis",
            "Cluster Analysis",
            "Logistic Regression",
            "Survival Analysis",
            "MANOVA"
        ]
        analysis_choice = st.sidebar.selectbox('Select Analysis', analysis_options)

        # Perform selected analysis
        if analysis_choice == "Descriptive Statistics":
            st.subheader('Descriptive Statistics')
            descriptive_statistics(data)

        elif analysis_choice == "T-Test":
            st.subheader('T-Test')
            column = st.selectbox('Select Column', data.columns)
            value = st.number_input('Enter Test Value')
            if st.button('Perform T-Test'):
                t_stat, p_value = perform_t_test(data, column, value)
                st.write('T-Statistic:', t_stat)
                st.write('P-Value:', p_value)

        elif analysis_choice == "ANOVA":
            st.subheader('ANOVA')
            groups = st.selectbox('Select Groups Column', data.columns)
            column = st.selectbox('Select Column for Analysis', data.columns)
            if st.button('Perform ANOVA'):
                anova_table = perform_anova(data, groups, column)
                st.write(anova_table)

        elif analysis_choice == "Correlation Analysis":
            st.subheader('Correlation Analysis')
            correlation_analysis(data)

        elif analysis_choice == "Regression Analysis":
            st.subheader('Regression Analysis')
            x_columns = st.multiselect('Select Independent Variables', data.select_dtypes(include=np.number).columns)
            y_column = st.selectbox('Select Dependent Variable', data.select_dtypes(include=np.number).columns)
            if st.button('Perform Regression Analysis'):
                regression_summary = perform_regression(data, x_columns, y_column)
                st.write(regression_summary)

        elif analysis_choice == "Factor Analysis":
            st.subheader('Factor Analysis')
            num_factors = st.slider('Number of Factors', min_value=1, max_value=len(data.columns))
            if st.button('Perform Factor Analysis'):
                factor_analysis(data, num_factors)

        elif analysis_choice == "Cluster Analysis":
            st.subheader('Cluster Analysis')
            num_clusters = st.slider('Number of Clusters', min_value=1, max_value=len(data))
            if st.button('Perform Cluster Analysis'):
                cluster_analysis(data, num_clusters)

        elif analysis_choice == "Logistic Regression":
            st.subheader('Logistic Regression')
            x_columns = st.multiselect('Select Independent Variables', data.select_dtypes(include=np.number).columns)
            y_column = st.selectbox('Select Dependent Variable', data.select_dtypes(include=np.number).columns)
            if st.button('Perform Logistic Regression'):
                st.write(logistic_regression(data, x_columns, y_column))

        elif analysis_choice == "Survival Analysis":
            st.subheader('Survival Analysis')
            time_column = st.selectbox('Select Time Column', data.columns)
            event_column = st.selectbox('Select Event Column', data.columns)
            if st.button('Perform Survival Analysis'):
                survival_analysis(data, time_column, event_column)

        elif analysis_choice == "MANOVA":
            st.subheader('MANOVA')
            groups = st.selectbox('Select Groups Column', data.columns)
            dependent_vars = st.multiselect('Select Dependent Variables', data.select_dtypes(include=np.number).columns)
            if st.button('Perform MANOVA'):
                perform_manova(data, groups, dependent_vars)

        # Contact Us button
        contact_us_link = "https://www.dijitatech.com/"
        st.sidebar.markdown(
            f'<a href="{contact_us_link}" style="background-color: #007bff; color: white; padding: 10px 20px; border-radius: 5px; font-size: 16px; text-decoration: none; cursor: pointer;">Contact Us Now</a>',
            unsafe_allow_html=True
        )

    def scroll_to_custom_visualization():
        st.write('<script>window.location.hash = "#custom_viz";</script>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
