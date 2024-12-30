import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from PyPDF2 import PdfReader

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Load data from the specified directory
def load_data_from_directory(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            text = extract_text_from_pdf(file_path)
            if text:
                data.append(text)
    return data

# Preprocess the data (example: convert text to feature vectors)
def preprocess_data(data):
    # Example: Convert text to length of text as a feature
    features = np.array([len(text) for text in data]).reshape(-1, 1)
    # Example: Dummy target values (you should replace this with actual target values)
    targets = np.random.randint(0, 100, size=len(data))
    return features, targets

# Path to your data directory
DATA_DIRECTORY_PATH = r"G:\PythonCodes\ResumeFilter\Resumes2"

# Load and preprocess the data
data = load_data_from_directory(DATA_DIRECTORY_PATH)
features, targets = preprocess_data(data)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)

# Create and train the Linear Regression model
line_reg = LinearRegression()
line_reg.fit(x_train, y_train)

# Make predictions on the test set
pred = line_reg.predict(x_test)

# Evaluate the model
r2 = r2_score(y_test, pred)
print(f"R-squared score: {r2}")

# Optional: Print predictions and actual values for comparison
print("Predictions:", pred)
print("Actual values:", y_test)