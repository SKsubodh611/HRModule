import os
from PIL import Image
import pytesseract

# Path to the folder containing resumes
folder_path = r'G:\PythonCodes\ResumeFilter\Resumes'

# Job-related keywords
job_keywords = ["Python", "developer", "Django", "Flask", "API", "SQL", "English"]

# Function to extract text from an image
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text.lower()  # Convert text to lowercase for easier matching

# Function to calculate the keyword match score for a resume
def calculate_keyword_score(text, keywords):
    score = 0
    for keyword in keywords:
        if keyword.lower() in text:  # Check if keyword is in the resume text
            score += 1
    return score

# Function to process all resumes in the folder and rank them based on keyword match
def rank_resumes_by_keywords(folder_path, job_keywords):
    resume_scores = []
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check if file is an image
            extracted_text = extract_text_from_image(file_path)
            score = calculate_keyword_score(extracted_text, job_keywords)
            resume_scores.append((filename, score))
    
    # Sort resumes by score (higher score means better fit)
    ranked_resumes = sorted(resume_scores, key=lambda x: x[1], reverse=True)
    
    return ranked_resumes

# Get the ranked resumes
ranked_resumes = rank_resumes_by_keywords(folder_path, job_keywords)

# Print the ranked resumes with their keyword match scores
print("Ranked Resumes:")
for resume, score in ranked_resumes:
    print(f"Resume: {resume}, Score: {score}")
