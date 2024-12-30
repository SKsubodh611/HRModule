
import os
from PIL import Image
import pytesseract
from docx import Document
import PyPDF2
import json
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Function to extract text from an image
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text.lower()


# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text.lower()


# Function to extract text from a DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + " "
    return text.lower()


# Function to extract text from all resumes in the specified directory
def extract_text_from_resumes(folder_path):
    resumes_text = {}
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check file types and extract text accordingly
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            extracted_text = extract_text_from_image(file_path)
        elif filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith('.docx'):
            extracted_text = extract_text_from_docx(file_path)
        else:
            continue  # Skip unsupported file types
        
        resumes_text[filename] = extracted_text
    
    return resumes_text


# Function to split data into training and testing datasets
def split_data(data, train_ratio=0.8):
    filenames = list(data.keys())
    random.shuffle(filenames)
    
    split_index = int(len(filenames) * train_ratio)
    train_filenames = filenames[:split_index]
    test_filenames = filenames[split_index:]
    
    train_data = {filename: data[filename] for filename in train_filenames}
    test_data = {filename: data[filename] for filename in test_filenames}
    
    return train_data, test_data


























# Function to load JSON data
def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

# Function to preprocess text (can be customized further)
def preprocess_text(text):
    return text.lower()

# Prepare data for model training and testing
def prepare_data(data):
    texts = []
    labels = []
    for filename, content in data.items():
        texts.append(preprocess_text(content))
        labels.append(1 if "developer" in content else 0)  # Example: Label based on keyword
    return texts, labels























# Main script for user interaction
if __name__ == "__main__":
    # Get user inputs
  
    folder_path = r"G:\PythonCodes\ResumeFilter\Resumes2"  # Use raw string to handle backslashes

    # Extract text from all resumes in the specified directory
    resumes_text = extract_text_from_resumes(folder_path)
    

    train_data, test_data = split_data(resumes_text, train_ratio=0.8)

    # Print the extracted text for each resume
    for filename, text in resumes_text.items():
        print(f"Filename: {filename}")
        print(f"Extracted Text: {text}\n")


# Save the training and testing datasets as JSON files
    with open("train_data.json", "w", encoding="utf-8") as train_file:
        json.dump(train_data, train_file, ensure_ascii=False, indent=4)

    with open("test_data.json", "w", encoding="utf-8") as test_file:
        json.dump(test_data, test_file, ensure_ascii=False, indent=4)

    print("Data has been successfully split and saved into 'train_data.json' and 'test_data.json'")















 # Load training and test data
    train_data = load_data("train_data.json")
    test_data = load_data("test_data.json")

    # Prepare training data
    train_texts, train_labels = prepare_data(train_data)
    test_texts, test_labels = prepare_data(test_data)

    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Train a Logistic Regression model
    classifier = LogisticRegression()
    classifier.fit(X_train, train_labels)

    # Evaluate the model
    predictions = classifier.predict(X_test)
    print("Classification Report:")
    print(classification_report(test_labels, predictions))

    # Test the model with new data
    while True:
        new_text = input("Enter resume text to classify (or type 'exit' to quit): ")
        if new_text.lower() == 'exit':
            break
        new_text_vectorized = vectorizer.transform([preprocess_text(new_text)])
        prediction = classifier.predict(new_text_vectorized)
        print("Prediction: Developer" if prediction[0] == 1 else "Prediction: Non-Developer")
