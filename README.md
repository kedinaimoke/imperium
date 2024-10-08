# Imperium Student Performance Prediction App

This Streamlit application predicts the performance of students based on their input data and displays the results in a user-friendly way. The app uses a machine learning model (Random Forest with MultiOutput Regressor) to forecast scores in different subjects and maps them to the WAEC grading system. It also provides insights into subjects that need improvement.

## Features

- **Student Data Input**: Input fields for student information such as gender, extra-curricular activity, supporting parent, family size, and highest parent education level.
- **Subject Score Prediction**: Predicts subject scores based on input using a pre-trained machine learning model.
- **WAEC Grade Mapping**: The predicted scores are converted into the WAEC grading system (A1 - Excellent, B2 - Very Good, etc.).
- **Chart Visualization**: Displays a bar chart of the predicted scores for easy visual analysis.
- **Areas for Improvement**: Highlights subjects with scores below the threshold for a Credit (C4), allowing students to identify areas needing more attention.

## How to Run the App

1. Clone or download the repository.
2. Install the required Python libraries by running:
    ```bash
    pip install streamlit pandas numpy joblib matplotlib
    ```
3. Place the trained model files (`final_exam_model.pkl` and `final_exam_scaler.pkl`) in the root folder.
4. Run the Streamlit app:
    ```bash
    streamlit run imperium_exam_predictor.py
    ```
5. You can equally access the Imperium app [here](https://imperium.streamlit.app)

## Model Information

The model used for prediction is a **Random Forest** regressor wrapped with a **MultiOutput Regressor**. This allows for predicting multiple target variables (subject scores) simultaneously. The model was trained on historical student performance data and is capable of predicting outcomes for subjects including:

- Maths
- English
- Civic
- Electronics
- Economics
- Biology
- Physics
- Chemistry
- Geography
- Accounting
- Commerce
- Government
- Book Keeping
- JAMB

### Training Details:
- **Model**: Random Forest Regressor
- **Wrapper**: MultiOutput Regressor (for handling multiple subjects at once)
- **Target Variables**: Scores in the above-listed subjects
- **Input Variables**: Gender, Age, Extra Curricular Activities, Subject Scores, Family Information, Parent Support, etc.
- **Scaler**: A preprocessing step using a fitted scaler for normalization of input features.

The Random Forest model is known for its robustness and ability to handle non-linear data patterns. With the MultiOutput Regressor, it predicts all subject scores in parallel, ensuring efficient and consistent performance.

## Files and Code Structure

- **`imperium_exam_predictor.py`**: Main application file.
    - Loads the pre-trained model and scaler.
    - Allows the user to input student information.
    - Predicts scores in multiple subjects and maps them to WAEC grades.
    - Provides a summary of results with suggestions for improvement if necessary.
  
## Example Input Fields

- **Student Type**: Choose between Commercial or Science.
- **Subjects**: Enter scores for core subjects like Maths, English, Civic, and additional subjects based on the student's focus area (Science/Commercial).
- **Personal Information**: Data such as gender, age, family size, supporting parent, and parent education level.

## Example Output

- **Predicted Scores**: Shown in a table format, mapping raw scores to WAEC grades.
- **Bar Chart**: A graphical representation of the predicted performance in all subjects.
- **Improvement Suggestions**: Lists subjects with scores below a specified threshold (C4), prompting users to focus more on these areas.

## Additional Features

- **Styling**: Customized layout with Streamlit’s design features to make the app visually appealing.
- **Caching**: Uses `@st.cache_resource` to load the model and scaler efficiently.

## Credits

The app was built using:
- **Streamlit**: For the web interface.
- **Pandas and NumPy**: For data handling and manipulation.
- **Joblib**: To load the pre-trained machine learning models.
- **Matplotlib**: For chart visualization.

The machine learning model was trained using Random Forest and MultiOutput Regressor to handle multi-subject predictions.
