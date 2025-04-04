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
def save_to_csv(text, filename='BackEnd/resumes.csv'):
    try:
        # Check if directory exists, if not create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        info = re.sub(r'\s+', ' ', text)
        
        # Write to CSV (using single column)
        with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([info])  # Note the list to create single cell
        
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

# Main function to process resumes
def process_resumes(resume_file):
    try:
    # Extract text from the PDF
        resume_text = extract_text_from_pdf(resume_file)  
        if resume_text:  # Only proceed if text extraction was successful
                
             # Save the extracted information to the CSV file
            save_to_csv(resume_text)
                
            print(f"Processed: {resume_file}")
        else:
            print(f"Skipping {resume_file} due to text extraction error")
    except Exception as e:
        print(f"Error processing {resume_file}: {e}")




if __name__ == '__main__':


    # Process the resumes
    #process_resumes(resume_file)

    #pdf_file = input("Enter the path to the file: ")

    print("Resume scraping and storage completed.")