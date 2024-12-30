import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from PyPDF2 import PdfReader

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text.lower()


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

# Path to your data directory
DATA_DIRECTORY_PATH = r"G:\PythonCodes\ResumeFilter\Resumes2"


# Load and preprocess the data
data = load_data_from_directory(DATA_DIRECTORY_PATH)
print(data)