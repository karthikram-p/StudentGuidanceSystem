# Personalized Course Recommendation System

## Overview
The **Personalized Course Recommendation System** is an interactive web application designed to assist students in making informed decisions about their academic and career paths. By utilizing advanced algorithms and machine learning techniques, this project offers personalized course recommendations, educational resources, and career guidance tailored to individual student profiles.

## Features
- **Interactive Web Interface**: Built with Streamlit for a user-friendly experience.
- **Recommendation Engine**: Utilizes TF-IDF and cosine similarity algorithms to provide tailored course recommendations based on user input.
- **AI Integration**: Incorporates Google Generative AI to generate intelligent course suggestions and educational materials.
- **User Authentication**: Implements secure user authentication to customize content access for each student.
- **Resource Library**: Offers a collection of academic resources, including PDFs, YouTube playlists, and other educational materials.
- **Career Guidance**: Combines academic resources with targeted career advice to empower students in their job search and internship applications.

## Technologies Used
- **Programming Languages**: Python
- **Frameworks**: Streamlit, Pandas, NumPy
- **Machine Learning**: Scikit-learn (TF-IDF and cosine similarity)
- **AI Integration**: Google Generative AI
- **Others**: dotenv for environment variable management, webbrowser for handling external links

## Installation
To set up this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/karthikram-p/StudentGuidanceSystem.git
   cd StudentGuidanceSystem
   
2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt

3. **Set up environment variables**: Create a .env file in the root directory and add necessary keys for Google Generative AI and other services.

4. **Run the application**:
   ```bash
   streamlit run app1.py

   
## Usage
 1. Open your web browser and go to http://localhost:8501.
 2. Enter your credentials to log in.
 3. Explore various sections to receive personalized course recommendations, view educational resources, and get career guidance.
