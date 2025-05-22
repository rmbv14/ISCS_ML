import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def load_model_and_scaler():
    try:
        model = joblib.load('recruitment_model.joblib')
        return model
    except FileNotFoundError:
        print("Error: Model file not found. Please run recruitment_model.py first to train the model.")
        return None

def get_user_input():
    print("\nEnter Applicant Details:")
    print("-" * 50)
    
    # Get input for each feature
    age = float(input("Age (20-50): "))
    gender = int(input("Gender (0 for Female, 1 for Male): "))
    education = int(input("Education Level (1-4):\n1: High School\n2: Bachelor's\n3: Master's\n4: PhD\nEnter choice (1-4): "))
    experience = float(input("Years of Experience (0-15): "))
    prev_companies = int(input("Number of Previous Companies (1-5): "))
    distance = float(input("Distance from Company (in km): "))
    interview_score = float(input("Interview Score (0-100): "))
    skill_score = float(input("Skill Score (0-100): "))
    personality_score = float(input("Personality Score (0-100): "))
    recruitment_strategy = int(input("Recruitment Strategy (1-3):\n1: Direct Application\n2: Referral\n3: Headhunter\nEnter choice (1-3): "))
    
    # Create a dictionary of the input
    applicant_data = {
        'Age': age,
        'Gender': gender,
        'EducationLevel': education,
        'ExperienceYears': experience,
        'PreviousCompanies': prev_companies,
        'DistanceFromCompany': distance,
        'InterviewScore': interview_score,
        'SkillScore': skill_score,
        'PersonalityScore': personality_score,
        'RecruitmentStrategy': recruitment_strategy
    }
    
    return pd.DataFrame([applicant_data])

def predict_applicant():
    # Load the model
    model = load_model_and_scaler()
    if model is None:
        return
    
    while True:
        # Get user input
        applicant_df = get_user_input()
        
        # Make prediction
        prediction = model.predict(applicant_df)
        probability = model.predict_proba(applicant_df)
        
        # Display results
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"Hiring Decision: {'Hired' if prediction[0] == 1 else 'Not Hired'}")
        print(f"Confidence: {probability[0][prediction[0]]*100:.2f}%")
        
        # Show probability for both classes
        print("\nDetailed Probabilities:")
        print(f"Probability of Not Hiring: {probability[0][0]*100:.2f}%")
        print(f"Probability of Hiring: {probability[0][1]*100:.2f}%")
        
        # Ask if user wants to try another applicant
        another = input("\nWould you like to predict another applicant? (yes/no): ").lower()
        if another != 'yes':
            break

if __name__ == "__main__":
    print("Welcome to the Recruitment Prediction System!")
    print("This system will predict whether an applicant will be hired based on their details.")
    predict_applicant() 