import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Step 1: Create a synthetic realistic dataset
np.random.seed(42)
n_samples = 100
ages = np.random.randint(22, 60, size=n_samples)
sexes = np.random.choice(['Male', 'Female'], size=n_samples)
experience = np.clip(ages - np.random.randint(18, 25, size=n_samples), 0, None)
base_salary = 25000 + (experience * 2500) + (ages * 120) + np.where(sexes == 'Male', 2000, 0)
noise = np.random.normal(0, 5000, size=n_samples)
salaries = base_salary + noise

data = pd.DataFrame({
    'Age': ages,
    'Sex': sexes,
    'Experience': experience,
    'Salary': salaries.astype(int)
})

# Step 2: Features and target
X = data[['Age', 'Sex', 'Experience']]
y = data['Salary']

# Step 3: Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('sex', OneHotEncoder(drop='first'), ['Sex'])
], remainder='passthrough')

# Step 4: Pipeline with polynomial regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', LinearRegression())
])

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Model performance
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Step 6: User Input (with validation)
def get_valid_input(prompt, val_type, condition=lambda x: True, error_msg="Invalid input!"):
    while True:
        try:
            value = val_type(input(prompt))
            if condition(value):
                return value
            else:
                print(error_msg)
        except:
            print(error_msg)

print("\n--- Employee Salary Predictor ---")
name = input("Enter your name: ").strip().title()
age = get_valid_input("Enter your age (18-65): ", int, lambda x: 18 <= x <= 65, "Enter a valid age between 18 and 65.")
sex = ''
while sex.lower() not in ['male', 'female']:
    sex = input("Enter your sex (Male/Female): ").strip().capitalize()

experience = get_valid_input(f"Enter your years of experience (0 to {age - 18}): ", int, lambda x: 0 <= x <= (age - 18), "Invalid experience.")

# Step 7: Make prediction
user_df = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'Experience': [experience]
})

predicted_salary = model.predict(user_df)[0]

# Step 8: Display result
print(f"\n👤 Name: {name}")
print(f"📊 Predicted Salary: ₹{round(predicted_salary, 2)}")
