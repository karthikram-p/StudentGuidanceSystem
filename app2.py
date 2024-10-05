import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import webbrowser
import os
import google.generativeai as genai
import textwrap
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline

# Load the sentiment analysis model
nlp = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")


# Function to read the Excel file
def load_user_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Load user data from Excel
user_data = load_user_data('users.xlsx')

# Function to verify user credentials
def verify_credentials(roll_number, password):
    roll_number = str(roll_number)  # Ensure roll number is treated as a string

    for index, row in user_data.iterrows():
        if str(row['RollNumber']) == roll_number and row['Password'] == password:
            return index
    return None

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'roll_number' not in st.session_state:
    st.session_state.roll_number = None


# Check login status and switch views
if st.session_state.logged_in:
        # Use the roll number in your app
        roll_number = st.session_state.roll_number
        row_no = st.session_state.row_no
        
        # Retrieve the student details
        student_name = user_data.at[row_no, 'StudentName']
        cgpa = user_data.at[row_no, 'CGPA']
        year = user_data.at[row_no, 'Year']
        branch = user_data.at[row_no, 'Branch']
        backlogs = st.session_state.backlogs 
        print("Backlogs:", backlogs)
        df = load_user_data('users.xlsx')
        
        def get_directory(year, branch=None):
            base_directory = './pdfs/'
            if year in [1, 2, 3, 4]:
                if year == 1:
                    return os.path.join(base_directory, str(year))
                else:
                    if branch:
                        return os.path.join(base_directory, str(year), branch)
                    else:
                        raise ValueError("Branch must be provided for years 2, 3, and 4")
            else:
                raise ValueError("Year must be between 1 and 4")

        pdfs_directory1 = get_directory(year, branch if year in [2, 3, 4] else None)

        def display_pdfs(directory, title):
            if os.path.exists(directory) and os.path.isdir(directory):
                pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
                if pdf_files:
                    st.subheader(f"{title}:")
                    for pdf in pdf_files:
                        pdf_path = os.path.join(directory, pdf)
                        with open(pdf_path, "rb") as file:
                            pdf_data = file.read()
                            st.download_button(
                                label=pdf,
                                data=pdf_data,
                                file_name=pdf,
                                mime="application/pdf"
                            )
                else:
                    st.write("No PDF files found in the directory.")
            else:
                st.write("The specified directory does not exist.")

        # Function to recommend academic resources (stub function)
        def recommend_academic_resources(keyword, num_recommendations):
            # Stub function - replace with actual recommendation logic
            return [f"Resource {i+1} for {keyword}" for i in range(num_recommendations)]

        # Function to display course details (stub function)
        def display_course_details(recommendations, title):
            st.subheader(title)
            for recommendation in recommendations:
                st.write(recommendation)
        def create_pdf_recommendations(directory):
            pdf_recommendations = {}
            for pdf_file in os.listdir(directory):
                if pdf_file.endswith('.pdf'):
                    # Extract the key from the PDF file name
                    key_parts = pdf_file.replace('_', ' ').replace('.pdf', '').lower().split()
                    for key in key_parts:
                        if key not in pdf_recommendations:
                            pdf_recommendations[key] = []
                        pdf_recommendations[key].append(pdf_file)
            return pdf_recommendations
        # Load datasets
        pdfs_directory = './pdfs/'
        pdf_recommendations = create_pdf_recommendations(pdfs_directory)
        # Configure API key
        load_dotenv(find_dotenv(), override=True)
        genai.configure(api_key='AIzaSyD1k5gp6h_aT0UIe4S6UOmOjHf4bYvudmM')

# Load the GenerativeModel for text
        text_model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Load the GenerativeModel for image
        image_model = genai.GenerativeModel('gemini-pro-vision')
        @st.cache_data
        def load_datasets():
            courses_df = pd.read_csv('Online_Courses.csv')
            courses_df.drop_duplicates(subset=['Title'], inplace=True)
            courses_df.reset_index(drop=True, inplace=True)
            udemy_df = pd.read_csv('udemy_courses.csv')
            udemy_df.drop_duplicates(subset=['course_title'], inplace=True)
            udemy_df.reset_index(drop=True, inplace=True)
            df = pd.read_excel('user_courses.xlsx', sheet_name='Sheet1', engine='openpyxl')
            user_course_matrix = df.pivot_table(index='UserID', columns='CourseName', aggfunc='size', fill_value=0)
            excel_file = 'Metadata.xlsx'
            booklist_df = pd.read_excel(excel_file, sheet_name='Booklist')
            # Preprocess the data by dropping duplicates based on Title
            booklist_df.drop_duplicates(subset=['Book Title'], inplace=True)
            booklist_df.reset_index(drop=True, inplace=True)
            return courses_df, udemy_df, user_course_matrix,booklist_df

        courses_df, udemy_df, user_course_matrix, booklist_df = load_datasets()

        # Preprocess topics for FutureLearn
        def preprocess_topics(topics_str):
            if pd.isna(topics_str) or topics_str is np.nan:
                return ''
            else:
                topics_str = ' '.join(topics_str.strip().split())
                topics_list = re.split(r'/', topics_str)
                unique_topics = list(set(topics_list))
                return ' '.join(unique_topics)
            
        # Base directory where PDFs are stored
        base_pdf_path = r'C:\Users\karth\Downloads\Projects\MP\SGC\data\data'

        # Function to preprocess the Subject Classification column
        def preprocess_subject_classification(subject_classification):
            return subject_classification.replace(';', ' ')

        # Apply preprocessing to the Subject Classification column
        booklist_df['Subject Classification'] = booklist_df['Subject Classification'].apply(preprocess_subject_classification)

        # Define a function to apply basic formatting to the text
        def simple_format(text):
            """
            Applies basic formatting to the text, like indentation and bullet points.
            """
            lines = text.strip().splitlines()  # Split into lines
            formatted_text = ""
            for line in lines:
                if line.startswith("*"):  # Bullet point
                    formatted_text += "  * " + line.strip("*") + "\n"
                else:
                    formatted_text += "  " + line + "\n"  # Indent non-bullet points
            return formatted_text.rstrip()  # Remove trailing newline


        # Recommendation functions
        def recommend_coursera(input_word, n_recommendations=15):
            courses_df['combined_text'] = courses_df['Title'] + ' ' + courses_df['Skills'] + ' ' + courses_df['Category'] + ' ' + courses_df['Sub-Category']
            courses_df['combined_text'] = courses_df['combined_text'].fillna('')
            coursera_courses = courses_df[courses_df['Site'] == 'Coursera']
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(coursera_courses['combined_text'])
            query_tfidf = tfidf.transform([input_word])
            content_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
            sorted_indices = (-content_scores).argsort()[:n_recommendations]
            recommendations = coursera_courses.iloc[sorted_indices][['Title', 'URL', 'Rating', 'Number of viewers', 'Skills']]
            recommendations['Site'] = 'Coursera'
            return recommendations.reset_index(drop=True)

        def recommend_udacity(input_word, n_recommendations=7):
            courses_df['combined_text'] = courses_df['Title'] + ' ' + courses_df['Short Intro'] + ' ' + courses_df['Prequisites'] + ' ' + courses_df['What you learn']
            courses_df['combined_text'] = courses_df['combined_text'].fillna('')
            udacity_courses = courses_df[courses_df['Site'] == 'Udacity']
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(udacity_courses['combined_text'])
            query_tfidf = tfidf.transform([input_word])
            content_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
            sorted_indices = (-content_scores).argsort()[:n_recommendations]
            recommendations = udacity_courses.iloc[sorted_indices][['Title', 'URL', 'Program Type', 'Level', 'School']]
            recommendations['Site'] = 'Udacity'
            return recommendations.reset_index(drop=True)

        def recommend_futurelearn(input_word, n_recommendations=7):
            courses_df['combined_text'] = courses_df['Title'] + ' ' + courses_df['Topics related to CRM'] 
            courses_df['combined_text'] = courses_df['combined_text'].fillna('')
            courses_df['Topics related to CRM'] = courses_df['Topics related to CRM'].apply(preprocess_topics)
            courses_df['combined_text'] = courses_df['Title'] + ' ' + courses_df['Topics related to CRM'] + ' ' + courses_df['Course Short Intro']
            futurelearn_courses = courses_df[courses_df['Site'] == 'Future Learn']
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(futurelearn_courses['combined_text'])
            query_tfidf = tfidf.transform([input_word])
            content_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
            sorted_indices = (-content_scores).argsort()[:n_recommendations]
            recommendations = futurelearn_courses.iloc[sorted_indices][['Title', 'URL']]
            recommendations['Site'] = 'FutureLearn'
            return recommendations.reset_index(drop=True)

        def recommend_simplilearn(input_word, n_recommendations=7):
            courses_df['combined_text'] = courses_df['Title'] + ' ' + courses_df['Short Intro'] + ' ' + courses_df['COURSE CATEGORIES']
            courses_df['combined_text'] = courses_df['combined_text'].fillna('')
            simplilearn_courses = courses_df[courses_df['Site'] == 'Simplilearn']
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(simplilearn_courses['combined_text'])
            query_tfidf = tfidf.transform([input_word])
            content_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
            sorted_indices = (-content_scores).argsort()[:n_recommendations]
            recommendations = simplilearn_courses.iloc[sorted_indices][['Title', 'URL', 'Number of ratings']]
            recommendations['Site'] = 'SimpliLearn'
            return recommendations.reset_index(drop=True)

        def recommend_udemy(input_word, n_recommendations=15):
            udemy_df['combined_text'] = udemy_df['course_title'] + ' ' + udemy_df['subject']
            udemy_df['combined_text'] = udemy_df['combined_text'].fillna('')
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(udemy_df['combined_text'])
            query_tfidf = tfidf.transform([input_word])
            content_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
            sorted_indices = (-content_scores).argsort()[:n_recommendations]
            recommendations = udemy_df.iloc[sorted_indices][['course_title', 'url', 'is_paid', 'price', 'num_subscribers', 'num_reviews', 'num_lectures', 'level', 'content_duration']]
            return recommendations.reset_index(drop=True)

        def get_related_courses(keyword, user_course_matrix, top_n=4):
            # Find all courses that contain the keyword
            keyword_courses = [course for course in user_course_matrix.columns if keyword.lower() in course.lower()]

            # Calculate cosine similarity between users
            user_similarity = cosine_similarity(user_course_matrix)
            user_similarity_df = pd.DataFrame(user_similarity, index=user_course_matrix.index, columns=user_course_matrix.index)

            # Aggregate scores for each course
            course_scores = np.zeros(user_course_matrix.shape[1])
            for course in keyword_courses:
                # Find users who have taken this course
                users_with_course = user_course_matrix[user_course_matrix[course] > 0].index

                for user in users_with_course:
                    similar_users = user_similarity_df[user].nlargest(11).index[1:]  # Top 10 similar users, excluding the user itself
                    similar_users_courses = user_course_matrix.loc[similar_users].sum(axis=0)
                    course_scores += similar_users_courses

            # Convert course scores to a pandas Series
            course_scores = pd.Series(course_scores, index=user_course_matrix.columns)

            # Filter courses to include only those containing the keyword
            filtered_course_scores = course_scores[keyword_courses]

            # Get the top recommended courses
            top_n_courses = filtered_course_scores.nlargest(top_n).index.tolist()
            return top_n_courses

        # Function to view PDF
        def view_pdf(pdf_path):
            # Ensure the PDF opens in Google Chrome
            chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
            webbrowser.get(chrome_path).open_new_tab(pdf_path)

        # Function to display course details
        def display_course_details(recommendations, site):
            if site == 'Coursera':
                st.write("Coursera Recommendations:")
                for index, row in recommendations.iterrows():
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 20px;">
                            <h4><a href="{row['URL']}" target="_blank" style="text-decoration:none;">{row['Title']}</a></h4>
                            <p><strong>Rating:</strong> {row.get('Rating', 'N/A')}</p>
                            <p><strong>Number of Students Enrolled:</strong> {row.get('Number of viewers', 'N/A')}</p>
                            <p><strong>Skills:</strong> {row.get('Skills', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True
                    )
            # ... (rest of the display_course_details function)
        option = st.sidebar.selectbox(
            'Choose the platform or resource type for recommendations:',
            ('Home', 'Coursera Courses', 'Udacity Courses', 'FutureLearn Courses', 'SimpliLearn Courses', 'Udemy Courses', 'Popular Among Students', 'Academic PDFs', 'Youtube Channels for Skill Development', 'Youtube Channels for Academics','AI Course Recommendations', 'AI RoadMaps', 'Feedback')
        )


        def display_course_details(recommendations, site):
            if site == 'Home':
                st.write('Home')
            elif site == 'Coursera':
                st.write("Coursera Recommendations:")
                for index, row in recommendations.iterrows():
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 20px;">
                            <h4><a href="{row['URL']}" target="_blank" style="text-decoration:none;">{row['Title']}</a></h4>
                            <p><strong>Rating:</strong> {row.get('Rating', 'N/A')}</p>
                            <p><strong>Number of Students Enrolled:</strong> {row.get('Number of viewers', 'N/A')}</p>
                            <p><strong>Skills:</strong> {row.get('Skills', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True
                    )
            elif site == 'Udacity':
                st.write("Udacity Recommendations:")
                for index, row in recommendations.iterrows():
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 20px;">
                            <h4><a href="{row['URL']}" target="_blank" style="text-decoration:none;">{row['Title']}</a></h4>
                            <p><strong>Program Type:</strong> {row.get('Program Type', 'N/A')}</p>
                            <p><strong>Level:</strong> {row.get('Level', 'N/A')}</p>
                            <p><strong>School:</strong> {row.get('School', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True
                    )
            elif site == 'FutureLearn':
                st.write("FutureLearn Recommendations:")
                for index, row in recommendations.iterrows():
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 20px;">
                            <h4><a href="{row['URL']}" target="_blank" style="text-decoration:none;">{row['Title']}</a></h4>
                        </div>
                        """, unsafe_allow_html=True
                    )
            elif site == 'SimpliLearn':
                st.write("SimpliLearn Recommendations:")
                for index, row in recommendations.iterrows():
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 20px;">
                            <h4><a href="{row['URL']}" target="_blank" style="text-decoration:none;">{row['Title']}</a></h4>
                            <p><strong>Number of ratings:</strong> {row.get('Number of ratings', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True
                    )
            elif site == 'Udemy':
                st.write("Udemy Recommendations:")
                for index, row in recommendations.iterrows():
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 20px;">
                            <h4><a href="{row['url']}" target="_blank" style="text-decoration:none;">{row['course_title']}</a></h4>
                            <p><strong>Is Paid:</strong> {row.get('is_paid', 'N/A')}</p>
                            <p><strong>Price:</strong> {row.get('price', 'N/A')}</p>
                            <p><strong>Number of Subscribers:</strong> {row.get('num_subscribers', 'N/A')}</p>
                            <p><strong>Number of Reviews:</strong> {row.get('num_reviews', 'N/A')}</p>
                            <p><strong>Number of Lectures:</strong> {row.get('num_lectures', 'N/A')}</p>
                            <p><strong>Level:</strong> {row.get('level', 'N/A')}</p>
                            <p><strong>Content Duration:</strong> {row.get('content_duration', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True
                    )
            elif site == 'Related Courses':
                st.write("Popular Courses:")
                if recommendations:
                    st.write("Here are some related courses:")
                    for course in recommendations:
                        st.write(f"- {course}")
                else:
                    st.write("No related courses found.")
            elif site == 'Academic PDFs':
                st.write("Academic PDFs:")
                if not recommendations.empty:
                    for i, row in recommendations.iterrows():
                        st.write(f"{i + 1}: {row['Book Title']} by {row['Author']} (PDF: {row['File_name']})")
                        if st.button(f"Open PDF for {row['Book Title']}"):
                            view_pdf(row['PDF_path'])

        if option == 'Home':
            # Streamlit app
            st.title("Personalized Course Recommendations")
            st.write(f"Welcome, {student_name}")

        # Display student details
            st.write(f"Roll Number: {roll_number}")
            st.write(f"CGPA:{cgpa}")
            user_name = {student_name}

            academic_score = cgpa
            print(academic_score)
            wrapped_backlogs = textwrap.fill(backlogs, width=80)  # Adjust width as needed
            st.write(f"Backlogs: {wrapped_backlogs}")
            if backlogs and backlogs != "none":
                backlogs_list = [backlog.strip().lower() for backlog in backlogs.split(',')]
                # Display PDF recommendations based on backlogs
                for backlog in backlogs_list:
                    found_recommendations = False
                    for key in pdf_recommendations:
                        if backlog.lower() in key.split():
                            found_recommendations = True
                            st.subheader(f"Recommendations for {backlog}:")
                            for pdf in pdf_recommendations[key]:
                                pdf_path = os.path.join(pdfs_directory, pdf)  # Path to the PDF file
                                with open(pdf_path, "rb") as file:
                                    pdf_data = file.read()
                                    st.download_button(
                                        label=pdf,
                                        data=pdf_data,
                                        file_name=pdf,
                                        mime="application/pdf"
                                    )
                            break  # Stop searching once recommendations are found
                    if not found_recommendations:
                        st.subheader(f"Recommendations for {backlog}:")
                        st.write("PDFs not in database.")
            else:
                st.write("No backlogs found.")

        elif option == 'Coursera Courses':
            num_recommendations = st.slider('Select number of recommendations:', 1, 30, 10)
            input_word = st.text_input("Enter a keyword:", value="")
            if input_word:
                coursera_recommendations = recommend_coursera(input_word, num_recommendations)
                display_course_details(coursera_recommendations, 'Coursera')

        elif option == 'Udacity Courses':
            num_recommendations = st.slider('Select number of recommendations:', 1, 30, 10)
            input_word = st.text_input("Enter a keyword:", value="")
            if input_word:
                udacity_recommendations = recommend_udacity(input_word, num_recommendations)
                display_course_details(udacity_recommendations, 'Udacity')

        elif option == 'FutureLearn Courses':
            num_recommendations = st.slider('Select number of recommendations:', 1, 30, 10)
            input_word = st.text_input("Enter a keyword:", value="")
            if input_word:
                futurelearn_recommendations = recommend_futurelearn(input_word, num_recommendations)
                display_course_details(futurelearn_recommendations, 'FutureLearn')

        elif option == 'SimpliLearn Courses':
            num_recommendations = st.slider('Select number of recommendations:', 1, 30, 10)
            input_word = st.text_input("Enter a keyword:", value="")
            if input_word:
                simplilearn_recommendations = recommend_simplilearn(input_word, num_recommendations)
                display_course_details(simplilearn_recommendations, 'SimpliLearn')

        elif option == 'Udemy Courses':
            num_recommendations = st.slider('Select number of recommendations:', 1, 30, 10)
            input_word = st.text_input("Enter a keyword:", value="")
            if input_word:
                udemy_recommendations = recommend_udemy(input_word, num_recommendations)
                display_course_details(udemy_recommendations, 'Udemy')

        elif option == 'Popular Among Students':
            num_recommendations = st.slider('Select number of recommendations:', 1, 30, 10)
            input_word = st.text_input("Enter a keyword:", value="")
            if input_word:
                related_courses = get_related_courses(input_word, user_course_matrix, top_n=num_recommendations)
                display_course_details(related_courses, 'Related Courses')

        elif option == 'Academic PDFs':
            display_pdfs(pdfs_directory1, f'Available PDFs for year {year}')
        
        elif option == 'Youtube Channels for Skill Development':
            st.header('Youtube Channels')

            input_word = st.text_input("Enter a subject/skill for youtube channel recommendation:", value="")

    # Generate response for user input
            if st.button('Generate Recommendations'):
                response_text = text_model.generate_content(f"Give me some youtube channels with links to excel in '{input_word}' in 2024.")
                st.subheader('Course Recommendations:')
                st.write(simple_format(response_text.text))

        elif option == 'Youtube Channels for Academics':
            st.header('Youtube Channels')

            input_word = st.text_input("Enter a subject for youtube channel recommendation:", value="")

    # Generate response for user input
            if st.button('Generate Recommendations'):
                response_text = text_model.generate_content(f"Give me some youtube channels with links to excel in '{input_word}'.")
                st.subheader('Course Recommendations:')
                st.write(simple_format(response_text.text))


        elif option == 'AI Course Recommendations':
            st.header('AI Recommendations')

            input_word = st.text_input("Enter a keyword or topic for course recommendations:", value="")

    # Generate response for user input
            if st.button('Generate Recommendations'):
                response_text = text_model.generate_content(f"Recommend some top and best courses related to '{input_word}' in 2024 with links and categorize them based on their usage and level")
                st.subheader('Course Recommendations:')
                st.write(simple_format(response_text.text))

        elif option == 'AI RoadMaps':
            st.header('AI Recommendations')

            input_word = st.text_input("Enter a Role for Roadmap Generation:", value="")

    # Generate response for user input
            if st.button('Generate Recommendations'):
                response_text = text_model.generate_content(f"Give me the complete roadmap to be a '{input_word}' in 2024 with all the required skills and categorize them level wise.")
                st.subheader('Course Recommendations:')
                st.write(simple_format(response_text.text))

        elif option == 'Feedback':
            st.title("Feedback")

            feedback_text = st.text_area("Share your feedback or suggestions:", height=200)

            if st.button("Submit Feedback"):
            # Analyze sentiment of the feedback
                sentiment = nlp(feedback_text)[0]['label']
            
            # Save the feedback to a file or database
            # You can store it along with user information if needed
                with open("feedback.txt", "a") as file:
                    file.write(f"User: {roll_number}, Sentiment: {sentiment}, Feedback: {feedback_text}\n")

                st.success("Thank you for your feedback!")

else:
    st.title("Login Page")

    roll_number = st.text_input("Roll Number")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        row_no = verify_credentials(roll_number, password)
        if row_no is not None:
            st.session_state.row_no = row_no
            st.session_state.logged_in = True
            st.session_state.roll_number = roll_number
            st.session_state.student_name = user_data.at[row_no, 'StudentName']
            st.session_state.cgpa = user_data.at[row_no, 'CGPA']
            back = user_data.at[row_no, 'Backlogs']
            if back is not None:  # Check if back is not null
                st.session_state.backlogs = back  # Assign backlogs to session variable
            else:
                st.session_state.backlogs = "none"  # If back is null, set session variable to "none"
            st.experimental_rerun()  # Rerun the script to show the main app
        else:
            st.error("Invalid roll number or password")