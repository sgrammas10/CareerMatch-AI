import csv
from datetime import datetime
import os
from pathlib import Path
import PyPDF2

def pdf_to_text(pdf_path):
    """Extract text from PDF (basic version)"""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None

def update_resume_csv(pdf_path, csv_path="resumes.csv"):
    """
    Add a new entry to CSV with:
    - Timestamp
    - Filename
    - Extracted text
    """
    # Extract text from PDF
    resume_text = pdf_to_text(pdf_path)
    if not resume_text:
        return False
    
    # Prepare CSV data
    file_name = Path(pdf_path).name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    fieldnames = ["timestamp", "filename", "resume_text"]
    new_row = {
        "timestamp": timestamp,
        "filename": file_name,
        "resume_text": resume_text
    }
    
    # Write to CSV
    try:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(new_row)
        print(f"Successfully added {file_name} to {csv_path}")
        return True
    except Exception as e:
        print(f"Error writing to CSV: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    pdf_path = "path/to/your_resume.pdf"  # Replace with your PDF path
    success = update_resume_csv(pdf_path)
    
    if success:
        print("CSV updated successfully!")
    else:
        print("Failed to update CSV")