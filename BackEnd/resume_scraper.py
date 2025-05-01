import csv
import os
import re
from PyPDF2 import PdfReader
from datetime import datetime

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_file}: {e}")
        return ""

# Function to save extracted information to a CSV file
def save_to_csv(user_email, text, filename='BackEnd/resumes.csv'):
    try:
        # Check if directory exists, if not create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        info = re.sub(r'\s+', ' ', text)
        
        # Write to CSV (using single column)
        with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([user_email, info])  # Note the list to create single cell
        
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def resume_already_uploaded(user_email, resume_text, filename='BackEnd/resumes.csv'):
    if not os.path.exists(filename):
        return False

    cleaned_resume = re.sub(r'\s+', ' ', resume_text).strip()

    with open(filename, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2 and row[0] == user_email and row[1].strip() == cleaned_resume:
                return True
    return False


# Main function to process resumes
def process_resumes(user_email, resume_file):
    try:
        resume_text = extract_text_from_pdf(resume_file)
        if resume_text:
            if resume_already_uploaded(user_email, resume_text):
                print(f"Resume already uploaded by {user_email}. Skipping.")
                return {"message": "Resume already exists", "skipped": True}
            else:
                save_to_csv(user_email, resume_text)
                return {"message": "Resume processed", "skipped": False}
        else:
            print("Resume text extraction failed.")
            return {"error": "Failed to extract text"}
    except Exception as e:
        print(f"Error processing resume: {e}")
        return {"error": str(e)}





if __name__ == '__main__':


    # Process the resumes
    #process_resumes(resume_file)

    #pdf_file = input("Enter the path to the file: ")

    print("Resume scraping and storage completed.")