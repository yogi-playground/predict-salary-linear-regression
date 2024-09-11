import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Salary Predictor Using ML Linear Regression Model")
# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Salary data.csv')
    return df

df = load_data()

# Prepare the data
X = df['YearsExperience'].values.reshape(-1, 1)
Y = df['Salary']

# Split the data and train the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Streamlit app
st.title('Salary Predictor Using ML Linear Regression Model')

# User input
years_experience = st.slider('Years of Experience', 0, 20, 5, 1)

# Predict salary
predicted_salary = regressor.predict([[years_experience]])[0]

# Display prediction

lower_bound = max(0, predicted_salary * 0.9)  # Assuming 10% variation
upper_bound = predicted_salary * 1.1

st.subheader(f"Predicted Salary Range For {years_experience} years of experience:")
st.write(f"Estimated salary range: ${lower_bound:.2f} - ${upper_bound:.2f}")

# Create visualizations
st.subheader('Salary Distribution Analysis')

# Create a figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))

# Bar plot of average salaries
experience_ranges = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20)]
avg_salaries = []
salary_data = []

for start, end in experience_ranges:
    mask = (df['YearsExperience'] >= start) & (df['YearsExperience'] < end)
    avg_salary = df.loc[mask, 'Salary'].mean()
    avg_salaries.append(avg_salary)
    salary_data.append(df.loc[mask, 'Salary'].tolist())

bars = ax1.bar([f"{start}-{end} years" for start, end in experience_ranges], avg_salaries)
ax1.set_ylabel('Average Salary ($)')
ax1.set_title('Average Salary by Years of Experience')

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:.0f}',
            ha='center', va='bottom')

# Box plot by experience range
sns.boxplot(data=salary_data, ax=ax2)
ax2.set_xticklabels([f"{start}-{end} years" for start, end in experience_ranges])
ax2.set_ylabel('Salary ($)')
ax2.set_title('Salary Distribution by Years of Experience')

# Overall box plot
sns.boxplot(x=df['Salary'], ax=ax3)
ax3.set_xlabel('Salary ($)')
ax3.set_title('Overall Salary Distribution')

# Adjust layout and display the plot
plt.tight_layout()
st.pyplot(fig)

# Display model performance
Y_test_pred = regressor.predict(X_test)
test_r2 = r2_score(Y_test, Y_test_pred)
st.subheader('Model Performance')
st.write(f"R-squared Score: {test_r2:.4f}")
st.title("Module 8: Linear Regression Assignment")
# Create two columns
col1, col2 = st.columns(2)

# Column 1: Problem Statement and Dataset
with col1:
    # Optional: Display the dataset
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(df)
    
with col2:
    

    st.write("**Problem Statement:**")
    st.write("You work in XYZ Company as a Python Data Scientist. The company officials have collected some data on salaries based on year of experience and wish for you to create a model from it.")

    st.write("**Dataset:** Salary data.csv")

    st.header("Tasks To Be Performed:")

    tasks = [
        "Load the dataset using pandas.",
        "Extract data from the years experience column into a variable named X.",
        "Extract data from the salary column into a variable named Y.",
        "Divide the dataset into two parts for training and testing in 66% and 33% proportion.",
        "Create and train a Linear Regression Model on the training set.",
        "Make predictions based on the testing set using the trained model.",
        "Check the performance by calculating the R² score of the model."
    ]

    for i, task in enumerate(tasks, 1):
        st.write(f"{i}. {task}")