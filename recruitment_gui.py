import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import numpy as np

class RecruitmentPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Recruitment Prediction System")
        # self.root.geometry("600x800")  # Removed fixed geometry for better compatibility
        
        # Load the model
        try:
            self.model = joblib.load('recruitment_model.joblib')
        except FileNotFoundError:
            messagebox.showerror("Error", "Model file not found. Please run recruitment_model.py first.")
            root.destroy()
            return
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Recruitment Prediction System", font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Create input fields
        self.create_input_fields(main_frame)
        
        # Create prediction button
        predict_button = ttk.Button(main_frame, text="Predict", command=self.make_prediction)
        predict_button.grid(row=11, column=0, columnspan=2, pady=20)
        
        # Create results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        self.results_frame.grid(row=12, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Results labels
        self.decision_label = ttk.Label(self.results_frame, text="", font=('Helvetica', 12))
        self.decision_label.grid(row=0, column=0, columnspan=2, pady=5)
        
        self.confidence_label = ttk.Label(self.results_frame, text="")
        self.confidence_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        self.prob_not_hired_label = ttk.Label(self.results_frame, text="")
        self.prob_not_hired_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        self.prob_hired_label = ttk.Label(self.results_frame, text="")
        self.prob_hired_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        
    def create_input_fields(self, parent):
        # Input fields dictionary
        self.inputs = {}
        
        # Age
        ttk.Label(parent, text="Age (20-50):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.inputs['Age'] = ttk.Entry(parent)
        self.inputs['Age'].grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Gender
        ttk.Label(parent, text="Gender:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.inputs['Gender'] = ttk.Combobox(parent, values=['Female (0)', 'Male (1)'], state='readonly')
        self.inputs['Gender'].grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        self.inputs['Gender'].set('Female (0)')
        
        # Education Level
        ttk.Label(parent, text="Education Level:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.inputs['EducationLevel'] = ttk.Combobox(parent, values=[
            'High School (1)',
            "Bachelor's (2)",
            "Master's (3)",
            'PhD (4)'
        ], state='readonly')
        self.inputs['EducationLevel'].grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        self.inputs['EducationLevel'].set('High School (1)')
        
        # Experience
        ttk.Label(parent, text="Years of Experience (0-15):").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.inputs['ExperienceYears'] = ttk.Entry(parent)
        self.inputs['ExperienceYears'].grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Previous Companies
        ttk.Label(parent, text="Previous Companies (1-5):").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.inputs['PreviousCompanies'] = ttk.Entry(parent)
        self.inputs['PreviousCompanies'].grid(row=5, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Distance
        ttk.Label(parent, text="Distance from Company (km):").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.inputs['DistanceFromCompany'] = ttk.Entry(parent)
        self.inputs['DistanceFromCompany'].grid(row=6, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Interview Score
        ttk.Label(parent, text="Interview Score (0-100):").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.inputs['InterviewScore'] = ttk.Entry(parent)
        self.inputs['InterviewScore'].grid(row=7, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Skill Score
        ttk.Label(parent, text="Skill Score (0-100):").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.inputs['SkillScore'] = ttk.Entry(parent)
        self.inputs['SkillScore'].grid(row=8, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Personality Score
        ttk.Label(parent, text="Personality Score (0-100):").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.inputs['PersonalityScore'] = ttk.Entry(parent)
        self.inputs['PersonalityScore'].grid(row=9, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Recruitment Strategy
        ttk.Label(parent, text="Recruitment Strategy:").grid(row=10, column=0, sticky=tk.W, pady=5)
        self.inputs['RecruitmentStrategy'] = ttk.Combobox(parent, values=[
            'Direct Application (1)',
            'Referral (2)',
            'Headhunter (3)'
        ], state='readonly')
        self.inputs['RecruitmentStrategy'].grid(row=10, column=1, sticky=(tk.W, tk.E), pady=5)
        self.inputs['RecruitmentStrategy'].set('Direct Application (1)')
    
    def get_input_values(self):
        try:
            values = {}
            for key, widget in self.inputs.items():
                if type(widget) == ttk.Combobox:
                    value = widget.get().strip()
                    if value == "":
                        raise ValueError(f"{key} is empty.")
                    # Extract the number from the combobox value
                    values[key] = int(value.split('(')[1].strip(')'))
                elif type(widget) == ttk.Entry:
                    val = widget.get().strip()
                    if val == "":
                        raise ValueError(f"{key} is empty.")
                    values[key] = float(val)
            return values
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            return None
    
    def make_prediction(self):
        values = self.get_input_values()
        if values is None:
            return
        
        # Create DataFrame
        applicant_df = pd.DataFrame([values])
        
        # Make prediction
        prediction = self.model.predict(applicant_df)
        probability = self.model.predict_proba(applicant_df)
        
        # Update results
        decision = "Hired" if prediction[0] == 1 else "Not Hired"
        confidence = probability[0][prediction[0]] * 100
        prob_not_hired = probability[0][0] * 100
        prob_hired = probability[0][1] * 100
        
        # Update labels with results
        self.decision_label.config(
            text=f"Decision: {decision}",
            foreground="green" if decision == "Hired" else "red"
        )
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
        self.prob_not_hired_label.config(text=f"Probability of Not Hiring: {prob_not_hired:.2f}%")
        self.prob_hired_label.config(text=f"Probability of Hiring: {prob_hired:.2f}%")

def main():
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')  # Force a compatible theme for macOS
    app = RecruitmentPredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 