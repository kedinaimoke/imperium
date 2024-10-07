import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def map_to_waec_grade(score):
    if score >= 75:
        return f"{score} (A1 - Excellent)"
    elif 70 <= score < 75:
        return f"{score} (B2 - Very Good)"
    elif 65 <= score < 70:
        return f"{score} (B3 - Good)"
    elif 60 <= score < 65:
        return f"{score} (C4 - Credit)"
    elif 55 <= score < 60:
        return f"{score} (C5 - Credit)"
    elif 50 <= score < 55:
        return f"{score} (C6 - Credit)"
    elif 45 <= score < 50:
        return f"{score} (D7 - Pass)"
    elif 40 <= score < 45:
        return f"{score} (E8 - Pass)"
    else:
        return f"{score} (F9 - Fail)"

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('final_exam_model.pkl')
    scaler = joblib.load('final_exam_scaler.pkl')
    return model, scaler

def user_input_features():
    st.sidebar.header('Student Information')

    global student_type
    
    Age = st.sidebar.number_input('Age', 15, 30)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    gender = 1 if gender == 'Male' else 0
    
    ExtraCurricularActivity = st.sidebar.selectbox(
        'Extra Curricular Activity', 
        ['NIL', 'Sports', 'Literal']
    )
    activity_mapping = {'NIL': 1, 'Sports': 2, 'Literal': 3}
    ExtraCurricularActivity = activity_mapping[ExtraCurricularActivity]

    student_type = st.sidebar.selectbox(
        'Are you a Commercial or Science Student?', 
        ['Science', 'Commercial']
    )

    Maths = st.sidebar.number_input('Maths', 0, 100)
    English = st.sidebar.number_input('English', 0, 100)
    Civic = st.sidebar.number_input('Civic', 0, 100)
    Electronics = st.sidebar.number_input('Electronics', 0, 100)
    Economics = st.sidebar.number_input('Economics', 0, 100)

    if student_type == 'Science':
        Biology = st.sidebar.number_input('Biology', 0, 100)
        Physics = st.sidebar.number_input('Physics', 0, 100)
        Chemistry = st.sidebar.number_input('Chemistry', 0, 100)
        Geography = st.sidebar.number_input('Geography', 0, 100)
        Accounting = 0
        Commerce = 0
        Government = 0
        Book_Keeping = 0
    else:
        Biology = 0
        Physics = 0
        Chemistry = 0
        Geography = 0
        Accounting = st.sidebar.number_input('Accounting', 0, 100)
        Commerce = st.sidebar.number_input('Commerce', 0, 100)
        Government = st.sidebar.number_input('Government', 0, 100)
        Book_Keeping = st.sidebar.number_input('Book Keeping', 0, 100)

    SupportingParent = st.sidebar.selectbox(
        'Supporting Parent',
        ['Both', 'Father', 'Mother'])
    supportingparent_mapping = {'Both': 0, 'Father': 1, 'Mother': 2}
    SupportingParent = supportingparent_mapping[SupportingParent]

    HighestParentEducation = st.sidebar.selectbox(
        'Highest Parent Education', 
        ['Primary', 'Secondary', 'Tertiary']
    )
    highestparent_mapping = {'Primary': 1, 'Secondary': 2, 'Tertiary': 3}
    HighestParentEducation = highestparent_mapping[HighestParentEducation]
    
    Family_Size = st.sidebar.number_input('Family Size', 1, 20)

    data = {
        'gender': gender,
        'Age': Age,
        'ExtraCurricularActivity': ExtraCurricularActivity,
        'Maths': Maths,
        'English': English,
        'Civic': Civic,
        'Electronics': Electronics,
        'Economics': Economics,
        'Biology': Biology,
        'Physics': Physics,
        'Chemistry': Chemistry,
        'Geography': Geography,
        'Accounting': Accounting,
        'Commerce': Commerce,
        'Government': Government,
        'Book Keeping': Book_Keeping,
        'SupportingParent': SupportingParent,
        'HighestParentEducation': HighestParentEducation,
        'Family_Size': Family_Size
    }

    features = pd.DataFrame(data, index=[0])
    return features

def main():
    st.set_page_config(page_title="Student Performance Prediction", page_icon=":bar_chart:", layout="wide")
    
    st.title('🎓 Imperium Student Performance Prediction')
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    features = user_input_features()
    
    model, scaler = load_model_and_scaler()
    
    scaled_features = scaler.transform(features)
    
    predictions = model.predict(scaled_features)
    
    prediction_labels = ['Maths', 'English', 'Civic', 'Electronics', 'Economics', 'Biology', 'Physics', 'Chemistry', 'Geography', 'Accounting', 'Commerce', 'Government', 'Book Keeping', 'JAMB']
    predictions_df = pd.DataFrame(predictions.round().astype(int), columns=prediction_labels)  # Convert to whole numbers
    
    columns = ['Maths', 'English', 'Civic', 'Electronics', 'Economics', 'Biology',
                 'Physics', 'Chemistry', 'Geography', 'Accounting', 'Commerce', 
                 'Government', 'Book Keeping']

    for column in columns:
        predictions_df[column] = predictions_df[column].apply(map_to_waec_grade)
    
    st.write('Here is your predicted performance:')
    
    if student_type == "Science":
        relevant_columns = ['Maths', 'English', 'Civic', 'Electronics', 'Economics', 'Biology', 'Physics', 'Chemistry', 'Geography']
    else:
        relevant_columns = ['Maths', 'English', 'Civic', 'Accounting', 'Commerce', 'Government', 'Book Keeping']

    relevant_predictions_df = predictions_df[relevant_columns]

    styled_df = relevant_predictions_df.style.set_table_attributes('style="background-color: #ffffff; border-radius: 5px; padding: 10px;"')
    st.dataframe(relevant_predictions_df, use_container_width=True, hide_index=True)
    predictions_chart = predictions_df.loc[0, columns].reset_index()
    predictions_chart.columns = ['Subject', 'Score']
    predictions_chart['Score'] = predictions_chart['Score'].str.extract('(\d+)').astype(int)

    st.bar_chart(predictions_chart.set_index('Subject')['Score'], use_container_width=True)

    low_grade_threshold = 60

    if student_type == "Science":
        relevant_subjects = ['Maths', 'English', 'Civic', 'Electronics', 'Economics', 'Biology', 'Physics', 'Chemistry', 'Geography']
    else:
        relevant_subjects = ['Maths', 'English', 'Civic', 'Accounting', 'Commerce', 'Government', 'Book Keeping']

    low_grades = predictions_chart[predictions_chart['Subject'].isin(relevant_subjects) & (predictions_chart['Score'] < low_grade_threshold)]

    if not low_grades.empty:
        st.subheader('Areas for Improvement:')
        st.write("The following subjects have grades below the threshold of C4 (60):")
        st.write("Do well to improve before the final exams!")
        st.dataframe(low_grades, use_container_width=True, hide_index=True)
    else:
        st.success("Great job! No subjects need immediate improvement, make sure you don't relax though 😉.")

if __name__ == '__main__':
    main()
