import nltk
nltk.download('stopwords')

from pyresparser import ResumeParser
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

data = ResumeParser('resume.pdf').get_extracted_data()

print('Name:', data['name'])
print('Email:', data['email'])
print('Mobile Number:', data['mobile_number'])
print('Skills:', data['skills'])
print('College Name:', data['college_name'])
print('Degree:', data['degree'])
print('Designation:', data['designation'])
print('Company Names:', data['company_names'])
print('No Of Pages:', data['no_of_pages'])
print('Total Experience:', data['total_experience'])


'''
pip install spacy==2.3.5

pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

pip install pyresparser

!pip install nltk

import nltk
nltk.download('stopwords')

from pyresparser import ResumeParser
import warnings
'''