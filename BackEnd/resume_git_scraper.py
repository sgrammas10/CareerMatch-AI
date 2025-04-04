import re
import csv
import os
import pdfplumber
import docx2txt

# ====== GLOBAL MATCHING LISTS ======

SKILLS_LIST = [
    'Python', 'C++', 'Java', 'JavaScript', 'SQL', 'Excel', 'Machine Learning', 'Deep Learning',
    'Data Analysis', 'Embedded Systems', 'Robotics', 'HTML', 'CSS', 'C', 'Kotlin', 'Swift',
    'AWS', 'Azure', 'Docker', 'Kubernetes', 'TensorFlow', 'Pandas', 'NumPy', 'Linux', 'Git'
]

DEGREES_LIST = [
    'Bachelor', 'Bachelors', 'B.Sc', 'BSc', 'B.Tech', 'BS', 'Master', 'Masters', 'M.Sc', 'MSc',
    'M.Tech', 'MS', 'PhD', 'Doctorate', 'Diploma'
]

DESIGNATIONS_LIST = [
    'Software Engineer', 'Intern', 'Data Analyst', 'Data Scientist', 'Machine Learning Engineer',
    'Project Manager', 'Product Manager', 'Developer', 'Engineer', 'Research Assistant'
]

COLLEGES_LIST = [
    'Harvard University', 'Stanford University', 'MIT', 'Massachusetts Institute of Technology',
    'University of California', 'Carnegie Mellon University', 'Georgia Tech', 'Cornell University'
]

COMPANY_TRIGGERS = ['at', 'for', 'with', 'from']

# ====== EXTRACTION FUNCTIONS ======

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            return '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file_path.endswith('.docx'):
        return docx2txt.process(file_path)
    else:
        raise ValueError("Unsupported file type. Use .pdf or .docx")

def extract_email(text):
    match = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    return match[0] if match else 'no_data'

def extract_phone(text):
    match = re.findall(r'(\+?\d[\d\s\-\(\)]{9,}\d)', text)
    return match[0] if match else 'no_data'

def extract_name(text):
    lines = text.strip().split("\n")
    return lines[0].strip() if lines else 'no_data'

def extract_skills(text):
    found = [skill for skill in SKILLS_LIST if re.search(rf'\b{re.escape(skill)}\b', text, re.I)]
    return ', '.join(found) if found else 'no_data'

def extract_experience(text):
    match = re.findall(r'(\d+)\+?\s+years?', text, re.I)
    return f"{match[0]} years" if match else 'no_data'

def extract_college(text):
    found = [college for college in COLLEGES_LIST if re.search(re.escape(college), text, re.I)]
    return found[0] if found else 'no_data'

def extract_degree(text):
    found = [degree for degree in DEGREES_LIST if re.search(rf'\b{re.escape(degree)}\b', text, re.I)]
    return found[0] if found else 'no_data'

def extract_designation(text):
    found = [title for title in DESIGNATIONS_LIST if re.search(rf'\b{re.escape(title)}\b', text, re.I)]
    return found[0] if found else 'no_data'

def extract_company(text):
    matches = []
    for trigger in COMPANY_TRIGGERS:
        matches += re.findall(rf'{trigger}\s+([A-Z][A-Za-z&\s]+)', text)
    matches = [m.strip() for m in matches]
    return ', '.join(set(matches)) if matches else 'no_data'

def extract_resume_data(text):
    return {
        'Name': extract_name(text),
        'Email': extract_email(text),
        'Mobile': extract_phone(text),
        'Skills': extract_skills(text),
        'Experience': extract_experience(text),
        'College': extract_college(text),
        'Degree': extract_degree(text),
        'Designation': extract_designation(text),
        'Company': extract_company(text)
    }

# ====== MAIN FUNCTION ======

def main():
    file_path = input("Enter the path to the resume file (.pdf or .docx): ").strip()

    if not os.path.isfile(file_path):
        print("❌ File not found.")
        return

    try:
        text = extract_text(file_path)
        data = extract_resume_data(text)

        # Create BackEnd folder if it doesn't exist
        output_dir = 'BackEnd'
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, 'resume_output.csv')
        write_header = not os.path.exists(csv_path)

        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(data)

        print(f"✅ Resume data extracted and appended to '{csv_path}'")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
