import cv2
import pytesseract

img = cv2.imread("G:\PythonCodes\ResumeFilter\resumes\resume1.jpg")
text = str(pytesseract.image_to_string(img, config='--oem 1 -psm 6'))
print(text)