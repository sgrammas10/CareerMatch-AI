# CareerMatch-AI
CareerMatch AI is a cutting-edge career development platform designed to help job seekers find
the perfect fit—both in terms of job roles and company culture. By combining resumes,
performance reviews, psychometric assessments (such as DISC, MBTI, and StrengthsFinder),
and proprietary tools, CareerMatch AI creates dynamic, personalized profiles for job seekers.
These profiles go beyond just skills to capture a candidate’s values, career aspirations, and
personality, enabling a comprehensive match with potential employers.
The tool features a gamified swiping experience, similar to a dating app, where job seekers can
swipe through job opportunities and companies that align with their profile. When a candidate
expresses interest by swiping right, the company is notified, creating an engaging and interactive
process for both parties. This intuitive system ensures that job seekers are presented with
positions that truly resonate with their skills, values, and long-term goals.
CareerMatch AI also includes a virtual or live intake session, allowing job seekers to engage
with a career coach or AI-powered assistant to refine their profile. This session dives deeper into
their professional history, goals, and preferences, helping to further personalize their job search.
At the heart of CareerMatch AI is a powerful AI and machine learning algorithm that
continuously analyzes job seeker profiles and compares them with company culture and job
requirements. By leveraging this technology, CareerMatch AI identifies the best job
opportunities, enhancing the likelihood that candidates will find roles where they can thrive and
grow.


File descriptions -

    *All work in progress*

    Scripts -
    - JavaScript scripts to pull data from companies about their currently listed job descriptions
    - All scripts other than zensearch.js are not currently being used or updated
    - zensearch.js accesses the companies listed in zensearchData.csv through their nodefetch to pull
      the data about their jobs
    - all data is saved in company specific csvs in zensearchDAta

    zensearchData -
    - All jobs are filed with the following data
        - Title
        - Pay
        - Location
        - Description
        - Remote Status
        - Full, Part, Internship, etc. status
        - Experience
        - Date Posted
        - Link
    
    RecsysFiles -
    - Refer to documentation.txt in RecsysFiles
    
    FrontEnd -
    - Refer to front_end_documentation.txt in FrontEnd

    BackEnd -
    - Rudimentary sign up and log in capabilities
        - Encrypted user profiles and log in
        - Sign up sends welcome email
        - uses sqlite3 to encrypt user profiles homomorphically
    - Foundation of website is built in app.py, bringing front and back end together with Flask

    PyTorch -
    - Work in progress, not implemented
      - Aimed to refine job postings to most essential and important info