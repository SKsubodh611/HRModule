import os
from PyPDF2 import PdfReader
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Path to your resumes folder
RESUME_FOLDER_PATH = r"G:\PythonCodes\ResumeFilter\Resumes2\INFORMATION-TECHNOLOGY"

# Step 1: Extract text from resumes
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_resumes(folder_path):
    resume_texts = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.pdf'):
            resume_texts.append(extract_text_from_pdf(file_path))
        elif file_name.endswith('.docx'):
            resume_texts.append(extract_text_from_docx(file_path))
        elif file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                resume_texts.append(file.read())
    return resume_texts

# Load resumes
resumes = load_resumes(RESUME_FOLDER_PATH)
print(f"Extracted {len(resumes)} resumes.")

# Step 2: Preprocess text
nltk.data.path.append(r'G:\PythonCodes\ResumeFilter\nltk')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(tokens)

processed_resumes = [preprocess_text(resume) for resume in resumes]

# Step 3: Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
resume_embeddings = model.encode(processed_resumes)

# Step 4: Define labels for training
# Replace these labels with your actual labels (e.g., job roles or suitability scores)
labels = [0, 1] * (len(resumes) // 2) + [0] * (len(resumes) % 2)  # Example labels

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(resume_embeddings, labels, test_size=0.2, random_state=42)

# Step 6: Train the model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 8: Save the model
model_path = r"G:\PythonCodes\ResumeFilter\resume_classifier.pkl"
joblib.dump(classifier, model_path)
print(f"Model saved to {model_path}")

# Step 9: Test the model on new data
# Test with a specific resume
TEST_RESUME_PATH = r"G:\PythonCodes\ResumeFilter\Resumes\Ridhima_Namdev.pdf"
test_resume_text = extract_text_from_pdf(TEST_RESUME_PATH)
test_resume_preprocessed = preprocess_text(test_resume_text)
test_resume_embedding = model.encode([test_resume_preprocessed])

# Compute similarity with training resumes
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(test_resume_embedding, resume_embeddings)
most_similar_index = similarities[0].argmax()

print(f"Most relevant resume in the training set: {resumes[most_similar_index]}")
