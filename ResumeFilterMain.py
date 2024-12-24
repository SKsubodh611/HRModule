import os
from PIL import Image
import pytesseract
from docx import Document
import PyPDF2

# Path to the folder containing resumes
folder_path = r'G:\PythonCodes\ResumeFilter\Resumes'

# Expanded Keywords for Python Fresher Profile                        
job_keywords = [
    "Python", "Beginner", "Programming", "OOP", "Data Structures", "Algorithms", 
    "Loops", "Functions", "Classes", "JSON", "REST API", "Git", "Database", 
    "Data Science", "Flask", "Django", "SQL", "Jupyter", "Pandas", "NumPy",
    "TensorFlow", "Keras", "PyTorch", "Selenium", "BeautifulSoup", "Requests", 
    "Scikit-learn", "Matplotlib", "OpenCV", "Pytest", "GitHub", "AWS", "Docker",
    "Machine Learning", "AI", "Data Visualization", "Problem-solving", "Project", 
    "Learning", "Communication", "Teamwork", "Leadership", "Time management", 
    "Fresher", "Internship", "CGPA", "SCPA", "BTech", "MSc", "IIT", "NIT", "BITS",
    "Gold Medal", "Dean's List", "Scholarship", "Research Paper", "Class Rank", 
    "Honors", "Software Development", "Backend Developer", "Full-stack Developer",
    "Data Analyst", "Cloud", "Azure", "Google Cloud", "Linux", "Windows"
]

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

# Function to calculate the keyword match score for a resume
def calculate_keyword_score(text, keywords):
    score = 0
    for keyword in keywords:
        if keyword.lower() in text:
            score += 1
    return score

# Function to process all resumes in the folder and rank them based on keyword match
def rank_resumes_by_keywords(folder_path, job_keywords):
    resume_scores = []
    
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
        
        # Calculate the score based on keywords
        score = calculate_keyword_score(extracted_text, job_keywords)
        resume_scores.append((filename, score))
    
    # Calculate percentage score for each resume
    total_keywords = len(job_keywords)
    resume_percentages = [(resume, (score / total_keywords) * 100) for resume, score in resume_scores]
    
    # Sort resumes by percentage (higher percentage means better fit)
    ranked_resumes = sorted(resume_percentages, key=lambda x: x[1], reverse=True)
    
    return ranked_resumes

# Get the ranked resumes with percentage
ranked_resumes = rank_resumes_by_keywords(folder_path, job_keywords)

# Print the ranked resumes with their keyword match percentages
print("Ranked Resumes (Percentage Match):")
for resume, percentage in ranked_resumes:
    print(f"Resume: {resume}, Match Percentage: {percentage:.2f}%")
