from PIL import Image
import pytesseract

# Path to the image
image_path = r'G:\PythonCodes\ResumeFilter\Resumes\Resume1.png'

# Open the image using PIL
image = Image.open(image_path)

# Use pytesseract to extract text
extracted_text = pytesseract.image_to_string(image)

# Print the extracted text
print(extracted_text)
