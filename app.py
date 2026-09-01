import streamlit as st
import pandas as pd
import re
import joblib

from sklearn.metrics.pairwise import cosine_similarity

from pypdf import PdfReader

# Load saved TF-IDF vectorizer
tfidf = joblib.load('tfidf_vectorizer.pkl')


# Function to clean resume text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # Remove unwanted special characters
    text = re.sub(r'[^a-z0-9+#.\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Function to extract text from PDF resume
def extract_pdf_text(pdf_file):
    try:
        reader = PdfReader(pdf_file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

        if not text.strip():
            return None

        return text

    except Exception:
        return None

# Skills used for matching
skills_list = [
    # Programming
    'python',
    'java',
    'javascript',
    'c++',
    'c#',

    # Web Development
    'html',
    'css',
    'react',
    'node.js',
    'bootstrap',
    'django',
    'flask',
    'streamlit',
    'spring boot',

    # Data & Analytics
    'sql',
    'excel',
    'pandas',
    'numpy',
    'data analysis',
    'data visualization',
    'statistics',
    'tableau',
    'power bi',

    # Machine Learning / AI
    'machine learning',
    'deep learning',
    'scikit learn',
    'tensorflow',
    'keras',
    'pytorch',
    'feature engineering',
    'regression',
    'classification',
    'cross validation',
    'model deployment',
    'nlp',
    'natural language processing',
    'tf idf',
    'computer vision',

    # Databases / Data Engineering
    'database',
    'databases',
    'mysql',
    'postgresql',
    'mongodb',
    'oracle',
    'etl',
    'data pipeline',
    'data warehouse',
    'spark',
    'hadoop',
    'airflow',

    # Cloud / DevOps
    'aws',
    'azure',
    'gcp',
    'linux',
    'git',
    'github',
    'docker',
    'kubernetes',
    'jenkins',
    'terraform',
    'ci cd',
    'automation',

    # APIs / Backend
    'api',
    'apis',
    'rest api',

    # Cybersecurity
    'cybersecurity',
    'security',
    'networking',
    'network security',
    'vulnerability assessment',
    'incident response',
    'threat analysis',

    # Testing
    'testing',
    'selenium',
    'pytest',
    'jira',

    # Mobile / UI
    'android',
    'figma',

    # Professional / Soft Skills
    'problem solving',
    'communication',
    'teamwork',
    'leadership',
    'analytical thinking',
    'critical thinking',
    'requirement analysis',
    'project management',
    'stakeholder management',
    'time management'
]


# Function to find matched skills
def get_matched_skills(resume_text, jd_text, skills):
    matched_skills = []

    for skill in skills:
        if skill in resume_text and skill in jd_text:
            matched_skills.append(skill)

    return matched_skills

# Function to find JD skills missing from resume
def get_missing_skills(resume_text, jd_text, skills):
    missing_skills = []

    for skill in skills:
        if skill in jd_text and skill not in resume_text:
            missing_skills.append(skill)

    return missing_skills


# Function to rank candidate resumes
def rank_resumes(job_description, resumes, resume_names):
    # Clean Job Description
    clean_jd = clean_text(job_description)

    # Clean resumes
    clean_resumes = [clean_text(resume) for resume in resumes]

    # Convert text into TF-IDF vectors
    resume_vectors = tfidf.transform(clean_resumes)
    jd_vector = tfidf.transform([clean_jd])

    # Calculate similarity scores
    scores = cosine_similarity(
        jd_vector,
        resume_vectors
    ).flatten() * 100

    # Store candidate results
    results = []

    for i, resume in enumerate(clean_resumes):
        matched_skills = get_matched_skills(
            resume,
            clean_jd,
            skills_list
        )

        missing_skills = get_missing_skills(
            resume,
            clean_jd,
            skills_list
        )

        # Calculate total skills detected in the Job Description
        jd_skills = [
            skill for skill in skills_list
            if skill in clean_jd
        ]

        # Calculate skill coverage percentage
        if len(jd_skills) > 0:
            skill_coverage = (
                len(matched_skills) / len(jd_skills)
            ) * 100
        else:
            skill_coverage = 0

        # Candidate recommendation based on match score
        score = scores[i]

        if score >= 60 and skill_coverage >= 80:
            recommendation = "🟢 Strong Match"

        elif score >= 45 and skill_coverage >= 60:
            recommendation = "🔵 Good Match"

        elif score >= 25 and skill_coverage >= 40:
            recommendation = "🟡 Moderate Match"

        else:
            recommendation = "🔴 Low Match"

        results.append({
            'Candidate': resume_names[i],
            'Match_Score': round(scores[i], 2),
            'Skill_Coverage': round(skill_coverage, 2),
            'Recommendation': recommendation,
            'Matched_Skills': ', '.join(matched_skills),
            'Missing_Skills': ', '.join(missing_skills),
            'Matched_Skill_Count': len(matched_skills),
            'Missing_Skill_Count': len(missing_skills),
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Rank candidates
    results_df = results_df.sort_values(
        by='Match_Score',
        ascending=False
    ).reset_index(drop=True)

    results_df['Rank'] = range(1, len(results_df) + 1)

    # Format match score for display
    results_df['Match_Score'] = results_df['Match_Score'].apply(
        lambda x: f"{x:.2f}%"
    )

    results_df['Skill_Coverage'] = results_df['Skill_Coverage'].apply(
    lambda x: f"{x:.2f}%"
    )

    # Arrange result columns
    results_df = results_df[
    [
        'Rank',
        'Candidate',
        'Match_Score',
        'Skill_Coverage',
        'Recommendation',
        'Matched_Skill_Count',
        'Matched_Skills',
        'Missing_Skill_Count',
        'Missing_Skills'
    ]
    ]

    return results_df

# Streamlit page settings
st.set_page_config(
    page_title="SwipeToHire",
    page_icon="📄",
    layout="wide"
)

# Reference profiles for job role mode
job_role_profiles = {
    "Data Scientist": """
        A Data Scientist collects, cleans, analyzes and interprets
        large datasets to discover patterns, trends and useful insights.

        The role involves data cleaning, exploratory data analysis,
        statistical analysis, feature engineering, machine learning
        and predictive modeling.

        Data Scientists work with Python, SQL, Pandas and NumPy for
        data analysis and use Scikit-learn for building and evaluating
        machine learning models.

        They may also use TensorFlow and Keras for deep learning tasks
        and Tableau or Power BI for data visualization and communicating
        analytical findings.

        The role requires analytical thinking, problem solving,
        communication and critical thinking to interpret complex data
        and support data-driven decisions.

        Data Scientists should understand regression, classification,
        model evaluation and feature engineering.
    """,

    "Data Analyst": """
        A Data Analyst collects, cleans, analyzes and interprets data
        to identify trends, patterns and useful business insights.

        The role involves data cleaning, exploratory data analysis,
        statistical analysis, reporting and data visualization.

        Data Analysts work with Python, SQL, Pandas, NumPy and Excel
        for analyzing datasets and use tools such as Tableau and
        Power BI to create dashboards and reports.

        The role requires analytical thinking, problem solving and
        communication skills to interpret findings and communicate
        insights to stakeholders.

        Data Analysts support data-driven business decision making
        by transforming raw data into meaningful information.
    """,

    "Machine Learning Engineer": """
        A Machine Learning Engineer develops, trains, evaluates and
        deploys machine learning models to solve real-world problems.

        The role involves data preprocessing, feature engineering,
        model training, model evaluation, cross validation and
        model deployment.

        Machine Learning Engineers work with Python, Pandas, NumPy
        and Scikit-learn and may use TensorFlow, Keras or PyTorch
        for deep learning applications.

        They should understand regression, classification,
        machine learning pipelines and model optimization.

        The role requires problem solving, analytical thinking,
        teamwork and communication when developing machine learning
        solutions for real-world applications.
    """,

    "Software Engineer": """
        A Software Engineer designs, develops, tests and maintains
        reliable software applications and systems.

        The role involves writing clean and efficient code,
        debugging applications, solving programming problems and
        working with databases and APIs.

        Software Engineers may work with Python, Java, C++, C# or
        JavaScript depending on the technology stack.

        They commonly use Git, GitHub, SQL, testing tools and
        software development practices.

        The role requires problem solving, teamwork, communication,
        critical thinking and time management while developing and
        maintaining software solutions.
    """,

    "Python Developer": """
        A Python Developer designs, develops, tests and maintains
        software applications using Python.

        The role involves writing efficient Python code, debugging,
        building APIs, automation and working with databases.

        Python Developers may work with Django, Flask or Streamlit
        and commonly use SQL, Git and GitHub.

        They may also use Pandas and NumPy when developing
        data-driven applications.

        The role requires problem solving, analytical thinking,
        teamwork and communication skills.
    """,

    "Web Developer": """
        A Web Developer designs, develops and maintains websites
        and web applications.

        The role involves creating responsive user interfaces,
        implementing website functionality, debugging applications
        and integrating APIs and databases.

        Web Developers commonly use HTML, CSS and JavaScript and
        may work with React, Bootstrap, Node.js, Django or Flask.

        They may also work with SQL, Git, GitHub and REST APIs.

        The role requires problem solving, teamwork, communication
        and time management.
    """,

    "Frontend Developer": """
        A Frontend Developer designs and develops the user-facing
        parts of websites and web applications.

        The role involves building responsive user interfaces,
        implementing interactive components and improving user
        experience across different devices.

        Frontend Developers commonly work with HTML, CSS,
        JavaScript, React and Bootstrap.

        They may also work with REST APIs, Git, GitHub, Figma
        and frontend testing tools.

        The role requires problem solving, communication,
        teamwork, critical thinking and attention to user needs.
    """,

    "Backend Developer": """
        A Backend Developer designs and maintains the server-side
        logic of web applications and software systems.

        The role involves building APIs, processing data,
        implementing application logic and working with databases.

        Backend Developers may work with Python, Java, JavaScript,
        Django, Flask, Spring Boot or Node.js.

        They commonly use SQL, MySQL, PostgreSQL, MongoDB,
        REST APIs, Git and GitHub.

        The role requires problem solving, analytical thinking,
        teamwork and communication.
    """,

    "Full Stack Developer": """
        A Full Stack Developer works on both frontend and backend
        components of web applications.

        The role involves creating responsive interfaces,
        implementing server-side logic, working with databases,
        integrating APIs and deploying applications.

        Full Stack Developers commonly use HTML, CSS, JavaScript,
        React and Bootstrap for frontend development and Python,
        Java, Node.js, Django, Flask or Spring Boot for backend
        development.

        They also work with SQL, databases, REST APIs, Git
        and GitHub.

        The role requires problem solving, teamwork,
        communication and time management.
    """,

    "DevOps Engineer": """
        A DevOps Engineer automates, deploys and maintains software
        applications and infrastructure.

        The role involves continuous integration, continuous
        deployment, infrastructure automation, system monitoring
        and improving software delivery processes.

        DevOps Engineers commonly work with Linux, Git, GitHub,
        Docker, Kubernetes, Jenkins, Terraform and CI CD pipelines.

        They may also work with AWS, Azure or GCP and use Python
        for automation.

        The role requires problem solving, teamwork,
        communication and analytical thinking.
    """,

    "Cloud Engineer": """
        A Cloud Engineer designs, deploys and manages applications
        and infrastructure in cloud environments.

        The role involves configuring cloud services, servers,
        storage, networking, security and monitoring.

        Cloud Engineers commonly work with AWS, Azure or GCP and
        may use Linux, Python, Docker, Kubernetes and Terraform.

        They also work with automation, Git, networking and
        infrastructure management.

        The role requires problem solving, analytical thinking,
        communication and teamwork.
    """,

    "Cybersecurity Analyst": """
        A Cybersecurity Analyst protects computer systems,
        networks and data from security threats.

        The role involves security monitoring, vulnerability
        assessment, threat analysis and incident response.

        Cybersecurity Analysts commonly work with Linux,
        networking, network security, cybersecurity and security
        monitoring tools.

        They may use Python for automation and SQL for analyzing
        security-related data.

        The role requires analytical thinking, problem solving,
        critical thinking, communication and teamwork.
    """,

    "Business Analyst": """
        A Business Analyst analyzes business processes,
        requirements and data to improve organizational
        decision making.

        The role involves requirement analysis, stakeholder
        management, reporting, documentation and business
        problem solving.

        Business Analysts commonly work with Excel, SQL,
        Tableau and Power BI.

        They may perform data analysis and data visualization
        to communicate business insights.

        The role requires communication, analytical thinking,
        problem solving, critical thinking and stakeholder
        management.
    """,

    "Database Administrator": """
        A Database Administrator manages, maintains and secures
        organizational databases.

        The role involves database configuration, monitoring,
        backup, recovery, security and performance optimization.

        Database Administrators commonly work with SQL, MySQL,
        PostgreSQL, Oracle and database management systems.

        They may also work with Linux, automation, database
        security and access management.

        The role requires problem solving, analytical thinking,
        critical thinking and time management.
    """,

    "QA Engineer / Software Tester": """
        A QA Engineer or Software Tester ensures that software
        applications work correctly and meet expected requirements.

        The role involves creating test cases, performing testing,
        identifying bugs and documenting software defects.

        QA Engineers may perform manual testing and automation
        testing using tools such as Selenium and Pytest.

        They may work with Python, Java, APIs, Jira, Git and GitHub.

        The role requires problem solving, analytical thinking,
        communication, teamwork and attention to software quality.
    """,

    "Data Engineer": """
        A Data Engineer designs, builds and maintains systems for
        collecting, processing, transforming and storing data.

        The role involves building data pipelines, ETL processes,
        data warehouses and integrating data from multiple sources.

        Data Engineers commonly work with Python, SQL, databases,
        Spark, Hadoop and Airflow.

        They may also use AWS, Azure or GCP for cloud-based
        data processing and storage.

        The role requires problem solving, analytical thinking,
        teamwork and communication.
    """,

    "AI Engineer": """
        An AI Engineer designs, develops and deploys artificial
        intelligence and machine learning solutions.

        The role involves data preprocessing, feature engineering,
        machine learning, deep learning, model evaluation and
        model deployment.

        AI Engineers commonly work with Python, Pandas, NumPy,
        Scikit-learn, TensorFlow, Keras and PyTorch.

        They may also work with NLP, natural language processing
        and computer vision applications.

        The role requires problem solving, analytical thinking,
        critical thinking, teamwork and communication.
    """,

    "NLP Engineer": """
        An NLP Engineer develops systems that analyze and process
        human language using natural language processing and
        machine learning.

        The role involves text preprocessing, feature engineering,
        TF IDF, text classification and information extraction.

        NLP Engineers commonly work with Python, Pandas, NumPy,
        Scikit-learn, TensorFlow, Keras and PyTorch.

        They may use machine learning and deep learning for
        language-based applications.

        The role requires problem solving, analytical thinking,
        critical thinking and communication.
    """,

    "Android Developer": """
        An Android Developer designs, develops, tests and maintains
        mobile applications for Android devices.

        The role involves creating user interfaces, implementing
        application functionality, integrating APIs and debugging
        mobile applications.

        Android Developers commonly work with Java, Android,
        databases, REST APIs, Git and GitHub.

        They may also perform application testing and deployment.

        The role requires problem solving, teamwork,
        communication and time management.
    """,

    "Java Developer": """
        A Java Developer designs, develops, tests and maintains
        software applications using Java.

        The role involves writing efficient Java code,
        implementing application logic, debugging applications
        and working with databases and APIs.

        Java Developers commonly work with Java, SQL,
        Spring Boot, REST APIs, Git and GitHub.

        They may develop backend, web or enterprise applications.

        The role requires problem solving, analytical thinking,
        teamwork and communication.
    """,

    "React Developer": """
        A React Developer develops modern and interactive
        web applications using React and JavaScript.

        The role involves building reusable user interface
        components, integrating REST APIs and creating responsive
        web applications.

        React Developers commonly work with React, JavaScript,
        HTML, CSS, Bootstrap, Git and GitHub.

        They may also work with Node.js, APIs and frontend
        testing tools.

        The role requires problem solving, teamwork,
        communication and critical thinking.
    """,

    "SQL Developer": """
        A SQL Developer designs, develops and maintains database
        queries and data solutions.

        The role involves writing SQL queries, managing databases,
        retrieving data and improving database performance.

        SQL Developers commonly work with SQL, MySQL,
        PostgreSQL, Oracle and databases.

        They may also work with ETL, data warehouses,
        data analysis and reporting.

        The role requires analytical thinking, problem solving,
        critical thinking and communication.
    """,

    "Data Visualization Analyst": """
        A Data Visualization Analyst transforms data into reports,
        dashboards and visual insights that support decision making.

        The role involves data analysis, data visualization,
        reporting and dashboard development.

        Data Visualization Analysts commonly work with Excel,
        SQL, Tableau and Power BI.

        They may also use Python, Pandas and statistics when
        analyzing datasets.

        The role requires analytical thinking, communication,
        problem solving and stakeholder management.
    """,

    "Automation Test Engineer": """
        An Automation Test Engineer develops automated tests
        to verify the quality and reliability of software.

        The role involves creating test cases, test automation,
        identifying defects and maintaining automated
        testing frameworks.

        Automation Test Engineers commonly work with Selenium,
        Pytest, Python, Java, APIs, Jira, Git and GitHub.

        They may perform functional testing, regression testing
        and API testing.

        The role requires problem solving, analytical thinking,
        teamwork and communication.
    """,

    "UI/UX Developer": """
        A UI/UX Developer designs and develops user-friendly
        interfaces for websites and applications.

        The role involves interface design, responsive design,
        prototyping and implementing frontend layouts.

        UI/UX Developers commonly work with Figma, HTML,
        CSS, JavaScript and Bootstrap.

        They may also work with React, Git and GitHub when
        implementing user interface designs.

        The role requires communication, problem solving,
        critical thinking and teamwork.
    """
}

# Function to suggest suitable job roles
def suggest_job_roles(resume_text):

    clean_resume = clean_text(resume_text)

    results = []

    for role, role_profile in job_role_profiles.items():

        clean_profile = clean_text(role_profile)

        # Convert resume and role profile into TF-IDF features
        resume_vector = tfidf.transform([clean_resume])
        profile_vector = tfidf.transform([clean_profile])

        # Calculate resume relevance
        relevance = cosine_similarity(
            resume_vector,
            profile_vector
        )[0][0] * 100

        # Find skills required by the role profile
        role_skills = [
            skill for skill in skills_list
            if skill in clean_profile
        ]

        # Find matched skills
        matched_skills = [
            skill for skill in role_skills
            if skill in clean_resume
        ]

        # Calculate skill coverage
        if len(role_skills) > 0:
            skill_coverage = (
                len(matched_skills) / len(role_skills)
            ) * 100
        else:
            skill_coverage = 0

        results.append({
            "Job Role": role,
            "Resume Relevance": round(relevance, 2),
            "Skill Coverage": round(skill_coverage, 2),
            "Matched Skills": ", ".join(matched_skills)
        })

    results_df = pd.DataFrame(results)

    # Rank job roles by resume relevance
    results_df = results_df.sort_values(
        by="Resume Relevance",
        ascending=False
    ).reset_index(drop=True)

    results_df.insert(
        0,
        "Rank",
        range(1, len(results_df) + 1)
    )
    results_df["Resume Relevance"] = results_df["Resume Relevance"].apply(
        lambda x: f"{x:.2f}%"
    )

    results_df["Skill Coverage"] = results_df["Skill Coverage"].apply(
        lambda x: f"{x:.2f}%"
    )
    return results_df


# App title
st.title("SwipeToHire")
st.subheader("NLP-Based Resume Screening, Matching & Job Role Recommendation System")

st.write(
    "Analyze resumes against job requirements, rank candidates by relevance, "
    "and discover suitable job roles based on resume profiles."
)

st.markdown("### How would you like to use SwipeToHire?")

input_mode = st.radio(
    "Choose an option:",
    [
        "I have a Job Description",
        "I only know the Job Role",
        "Suggest Job Roles from My Resume"
    ],
    index=None
)

# Job Description input
# Input based on selected mode
if input_mode == "I have a Job Description":

    st.markdown("### Job Description")

    job_description = st.text_area(
        "Paste the Job Description here:",
        height=200,
        placeholder="Enter the required skills, experience and responsibilities..."
    )

elif input_mode == "I only know the Job Role":

    st.markdown("### Job Role")

    job_role = st.selectbox(
        "Select the Job Role:",
        [
            "Data Scientist",
            "Data Analyst",
            "Machine Learning Engineer",
            "Software Engineer",
            "Python Developer",
            "Web Developer",
            "Frontend Developer",
            "Backend Developer",
            "Full Stack Developer",
            "DevOps Engineer",
            "Cloud Engineer",
            "Cybersecurity Analyst",
            "Business Analyst",
            "Database Administrator",
            "QA Engineer / Software Tester",
            "Data Engineer",
            "AI Engineer",
            "NLP Engineer",
            "Android Developer",
            "Java Developer",
            "React Developer",
            "SQL Developer",
            "Data Visualization Analyst",
            "Automation Test Engineer",
            "UI/UX Developer"
        ]
    )
    # Use selected role profile as the reference Job Description
    job_description = job_role_profiles[job_role]

elif input_mode == "Suggest Job Roles":

    st.markdown("### Upload Your Resume")

    st.write(
        "Upload your resume and SwipeToHire will suggest the job roles "
        "that best match your current profile."
    )

if input_mode is not None:

    # Resume upload section

    if input_mode == "Suggest Job Roles from My Resume":
        upload_heading = "### Upload Your Resume"
        upload_label = "Upload your resume in PDF format:"
        ranking_heading = "### Job Role Suggestions"
        button_label = "Suggest Job Roles"
        allow_multiple = False

    else:
        upload_heading = "### Upload Candidate Resumes"
        upload_label = "Upload resumes in PDF format:"
        ranking_heading = "### Candidate Ranking"
        button_label = "Screen & Rank Candidates"
        allow_multiple = True

    st.markdown(upload_heading)

    uploaded_files = st.file_uploader(
        upload_label,
        type=["pdf"],
        accept_multiple_files=allow_multiple
    )


    # Extract text from uploaded resumes
    resume_texts = []
    resume_names = []

    if uploaded_files:
        if input_mode == "Suggest Job Roles from My Resume":
            files_to_process = [uploaded_files]
        else:
            files_to_process = uploaded_files

        for uploaded_file in files_to_process:
            resume_text = extract_pdf_text(uploaded_file)

            if resume_text is None:
                st.warning(
                    f"Could not extract text from {uploaded_file.name}. "
                    "Please upload a valid text-based PDF."
                )
                continue

            resume_texts.append(resume_text)
            resume_names.append(uploaded_file.name)

        if resume_texts:
            st.success(f"{len(resume_texts)} resume(s) uploaded successfully.")


    # Screen resumes button
    st.markdown(ranking_heading)

    screen_button = st.button(button_label)
    
    if screen_button:
        if input_mode == "Suggest Job Roles from My Resume":

            if not resume_texts:
                st.warning("Please upload your resume.")

            else:
                suggested_roles_df = suggest_job_roles(
                    resume_texts[0]
                )

                top_role = suggested_roles_df.iloc[0]

                st.success(
                    f"🎯 Top Suggested Role: {top_role['Job Role']} | "
                    f"Resume Relevance: {top_role['Resume Relevance']} | "
                    f"Skill Coverage: {top_role['Skill Coverage']}"
                )
                st.dataframe(
                    suggested_roles_df.head(5),
                    use_container_width=True,
                    hide_index=True
                )
                st.markdown("### Understanding Job Role Suggestions")

                st.markdown("""
                    **🎯 Suggested Job Roles**  
                    Shows the job roles that most closely align with the skills and overall content identified in your resume.

                    **📊 Resume Relevance**  
                    Measures how closely your overall resume content aligns with each job role profile.

                    **🎯 Skill Coverage**  
                    Indicates the percentage of identified role-relevant skills found in your resume.

                    > **Note:** Job role suggestions are intended to support career exploration based on your current resume profile. They do not guarantee eligibility, selection, or suitability for a particular position.
                    """)
        else:       
            if not job_description.strip():
                st.warning("Please enter a Job Description.")

            elif not resume_texts:
                st.warning("Please upload at least one resume.")

            else:
                results_df = rank_resumes(
                    job_description,
                    resume_texts,
                    resume_names
                )

                display_df = results_df.rename(columns={
                    'Match_Score': 'Resume Relevance',
                    'Skill_Coverage': 'Skill Coverage',
                    'Matched_Skill_Count': 'Matched Skill Count',
                    'Missing_Skill_Count': 'Missing Skill Count',
                    'Matched_Skills': 'Matched Skills',
                    'Missing_Skills': 'Missing JD Skills'
                })

                # Display top candidate
                top_candidate = results_df.iloc[0]

                st.success(
                    f"Top Candidate: {top_candidate['Candidate']} | "
                    f"Resume Relevance: {top_candidate['Match_Score']} | "
                    f"Recommendation: {top_candidate['Recommendation']}"
                )

                st.dataframe(display_df,use_container_width=True,hide_index=True)

                csv = display_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="swipetohire_results.csv",
                    mime="text/csv"
                )

                st.markdown("### Understanding Your Results")

                st.markdown("""
                    **📊 Resume Relevance**  
                    Measures how closely a candidate's overall resume content aligns with the requirements and context of the selected job description or job role.

                    **🎯 Skill Coverage**  
                    Indicates the percentage of identified role-relevant skills found in the candidate's resume.

                    **💡 Recommendation**  
                    Provides an automated preliminary assessment based on overall resume relevance and skill alignment.

                    > **Disclaimer:** SwipeToHire provides automated preliminary assessments to support initial resume screening. Final hiring decisions should incorporate interviews, practical evaluations, experience, qualifications, and the candidate's overall suitability for the role.
                """)


