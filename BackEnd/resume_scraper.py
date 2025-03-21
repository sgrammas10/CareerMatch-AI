import re
import csv
import os
from PyPDF2 import PdfReader

# Define the fields you want to extract from the resumes
FIELDS = ['Name', 'Email', 'Phone', 'Education', 'Skills', 'Experience', 'Leadership', 'Job Positions', 'Company Names']

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        print(text)
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_file}: {e}")
        return ""

# Function to extract information from a resume text
def extract_info(resume_text):
    info = {field: '' for field in FIELDS}
    
    # Extract Name
    name_match = re.search(r'#\s*([A-Za-z\s]+)', resume_text)

    if name_match:
        info['Name'] = name_match.group(1).strip()
    
    # Extract Email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', resume_text)
    if email_match:
        info['Email'] = email_match.group(0).strip()
    
    # Extract Phone
    phone_match = re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', resume_text)
    if phone_match:
        info['Phone'] = phone_match.group(0).strip()
    
    # Extract Education
    education_match = re.search(r'EDUCATION(.*?)(?=SKILLS|EXPERIENCE|LEADERSHIP|$)', resume_text, re.DOTALL)
    if education_match:
        info['Education'] = education_match.group(1).strip()
    
    # Extract Skills
    skills_match = re.search(r'SKILLS(.*?)(?=EXPERIENCE|LEADERSHIP|$)', resume_text, re.DOTALL)
    if skills_match:
        info['Skills'] = skills_match.group(1).strip()
    
    # Extract Experience
    experience_match = re.search(r'EXPERIENCE(.*?)(?=LEADERSHIP|$)', resume_text, re.DOTALL)
    if experience_match:
        experience_text = experience_match.group(1).strip()
        info['Experience'] = experience_text
        
        # Extract Job Positions and Company Names from Experience section
        job_positions = re.findall(r'([A-Za-z\s]+)\s*,\s*([A-Za-z\s]+)', experience_text)
        if job_positions:
            info['Job Positions'] = "; ".join([f"{pos[0].strip()} at {pos[1].strip()}" for pos in job_positions])
    
    # Extract Leadership
    leadership_match = re.search(r'LEADERSHIP(.*?)(?=$)', resume_text, re.DOTALL)
    if leadership_match:
        info['Leadership'] = leadership_match.group(1).strip()
    
    return info

# Function to save extracted information to a CSV file
def save_to_csv(info, filename='BackEnd/resumes.csv'):
    try:
        # Check if the file exists
        file_exists = os.path.isfile(filename)
        
        # Open the file in append mode if it exists, or write mode if it doesn't
        with open(filename, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            
            # Write the header only if the file is being created for the first time
            if not file_exists:
                writer.writeheader()
            
            # Write the row of data
            writer.writerow(info)
        
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

# Main function to process resumes
def process_resumes(resume_files):
    for resume_file in resume_files:
        try:
            # Extract text from the PDF
            resume_text = extract_text_from_pdf(resume_file)
            
            if resume_text:  # Only proceed if text extraction was successful
                # Extract information from the text
                info = extract_info(resume_text)
                
                # Save the extracted information to the CSV file
                save_to_csv(info)
                
                print(f"Processed: {resume_file}")
            else:
                print(f"Skipping {resume_file} due to text extraction error")
        except Exception as e:
            print(f"Error processing {resume_file}: {e}")

# List of resume PDF files to process
resume_files = [
    'BackEnd/5002894.pdf'
]

# Process the resumes
process_resumes(resume_files)

#pdf_file = input("Enter the path to the file: ")

print("Resume scraping and storage completed.")