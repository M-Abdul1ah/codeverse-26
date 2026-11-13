"""
EduPulse - At-Risk Student Early Warning System
Task A4: Single Student Prediction Form & Failure Probability Estimator

Requirements:
- User enters one student's values
- Predicts Pass/Fail with exact probability (e.g., "72% risk of failing")
- Supports interactive CLI input, programmatic calls, and CLI arguments
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

class StudentPredictor:
    def __init__(self, models_dir='models'):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.join(base_dir, models_dir)
        
        self.lr_model = joblib.load(os.path.join(models_path, 'logistic_regression.joblib'))
        self.rf_model = joblib.load(os.path.join(models_path, 'random_forest.joblib'))
        self.scaler = joblib.load(os.path.join(models_path, 'scaler.joblib'))
        self.metadata = joblib.load(os.path.join(models_path, 'model_metadata.joblib'))
        self.feature_cols = self.metadata['feature_cols']

    def predict_student(self, student_dict, model_choice='Logistic Regression'):
        """
        student_dict format:
        {
            'attendance_pct': float,
            'quiz_avg': float,
            'assignment_avg': float,
            'mid_marks': float,
            'lab_marks': float,
            'study_hours_week': float,
            'prev_cgpa': float,
            'part_time_job': 'Yes' or 'No'
        }
        """
        data = student_dict.copy()
        job_val = 1 if str(data.get('part_time_job', 'No')).strip().lower() in ['yes', 'y', '1', 'true'] else 0
        
        row = [
            float(data['attendance_pct']),
            float(data['quiz_avg']),
            float(data['assignment_avg']),
            float(data['mid_marks']),
            float(data['lab_marks']),
            float(data['study_hours_week']),
            float(data['prev_cgpa']),
            job_val
        ]
        
        X_df = pd.DataFrame([row], columns=self.feature_cols)

        if model_choice == 'Logistic Regression':
            X_scaled = self.scaler.transform(X_df)
            pred = self.lr_model.predict(X_scaled)[0]
            prob_fail = self.lr_model.predict_proba(X_scaled)[0][1]
        else:
            pred = self.rf_model.predict(X_df)[0]
            prob_fail = self.rf_model.predict_proba(X_df)[0][1]

        prob_pass = 1.0 - prob_fail
        status = "Fail / At-Risk" if pred == 1 else "Pass"

        return {
            'predicted_status': status,
            'failure_risk_pct': round(prob_fail * 100, 2),
            'pass_probability_pct': round(prob_pass * 100, 2),
            'model_used': model_choice,
            'input_data': data
        }

def print_prediction_result(res):
    print("\n" + "=" * 55)
    print(" PREDICTION RESULT FOR STUDENT")
    print("=" * 55)
    print(f" Model Used            : {res['model_used']}")
    print(f" Predicted Status      : {res['predicted_status'].upper()}")
    print(f" Risk of Failing       : {res['failure_risk_pct']}%")
    print(f" Probability of Passing: {res['pass_probability_pct']}%")
    print("-" * 55)
    print(" Student Input Profile:")
    for k, v in res['input_data'].items():
        print(f"  - {k:<20}: {v}")
    print("=" * 55)

def run_interactive_form():
    print("=" * 55)
    print(" INTERACTIVE STUDENT PREDICTION FORM")
    print("=" * 55)
    try:
        att = float(input("Enter Attendance Percentage (30-100) [e.g. 72.0]: ") or 72.0)
        quiz = float(input("Enter Quiz Average (0-10) [e.g. 6.0]: ") or 6.0)
        assign = float(input("Enter Assignment Average (0-10) [e.g. 6.5]: ") or 6.5)
        mid = float(input("Enter Mid-Term Marks (0-30) [e.g. 15.0]: ") or 15.0)
        lab = float(input("Enter Lab Marks (0-20) [e.g. 13.0]: ") or 13.0)
        hours = float(input("Enter Study Hours/Week (0-40) [e.g. 7.0]: ") or 7.0)
        cgpa = float(input("Enter Previous CGPA (1.0-4.0) [e.g. 2.70]: ") or 2.70)
        job = input("Has Part-Time Job? (Yes/No) [e.g. No]: ") or "No"

        student = {
            'attendance_pct': att,
            'quiz_avg': quiz,
            'assignment_avg': assign,
            'mid_marks': mid,
            'lab_marks': lab,
            'study_hours_week': hours,
            'prev_cgpa': cgpa,
            'part_time_job': job
        }

        predictor = StudentPredictor()
        result = predictor.predict_student(student, model_choice='Logistic Regression')
        print_prediction_result(result)
    except Exception as e:
        print(f"Error during input processing: {e}")

def run_sample_predictions():
    print("=" * 60)
    print(" TASK A4: PREDICTION FORM DEMONSTRATION")
    print("=" * 60)
    predictor = StudentPredictor()

    sample_students = [
        {
            'name': 'Student A (High Risk Case)',
            'data': {
                'attendance_pct': 54.0,
                'quiz_avg': 4.2,
                'assignment_avg': 5.0,
                'mid_marks': 11.5,
                'lab_marks': 9.0,
                'study_hours_week': 3.5,
                'prev_cgpa': 2.10,
                'part_time_job': 'Yes'
            }
        },
        {
            'name': 'Student B (Borderline Case)',
            'data': {
                'attendance_pct': 72.0,
                'quiz_avg': 6.0,
                'assignment_avg': 6.5,
                'mid_marks': 16.0,
                'lab_marks': 13.5,
                'study_hours_week': 7.0,
                'prev_cgpa': 2.65,
                'part_time_job': 'Yes'
            }
        },
        {
            'name': 'Student C (Low Risk / Safe Case)',
            'data': {
                'attendance_pct': 92.5,
                'quiz_avg': 8.8,
                'assignment_avg': 9.2,
                'mid_marks': 25.5,
                'lab_marks': 18.0,
                'study_hours_week': 14.0,
                'prev_cgpa': 3.65,
                'part_time_job': 'No'
            }
        }
    ]

    for student in sample_students:
        print(f"\nEvaluating: {student['name']}")
        result = predictor.predict_student(student['data'], model_choice='Logistic Regression')
        print_prediction_result(result)

    print("\n" + "=" * 60)
    print(" TASK A4 COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EduPulse Single Student Prediction Form")
    parser.add_argument('--interactive', action='store_true', help="Run interactive prompt to enter student values manually")
    parser.add_argument('--attendance', type=float, help="Attendance Percentage (30-100)")
    parser.add_argument('--quiz', type=float, help="Quiz Average (0-10)")
    parser.add_argument('--assignment', type=float, help="Assignment Average (0-10)")
    parser.add_argument('--mid', type=float, help="Mid-Term Marks (0-30)")
    parser.add_argument('--lab', type=float, help="Lab Marks (0-20)")
    parser.add_argument('--hours', type=float, help="Study Hours per Week")
    parser.add_argument('--cgpa', type=float, help="Previous CGPA (1.0-4.0)")
    parser.add_argument('--job', type=str, choices=['Yes', 'No', 'yes', 'no'], help="Part-Time Job (Yes/No)")

    args = parser.parse_args()

    if args.interactive:
        run_interactive_form()
    elif args.attendance is not None:
        custom_student = {
            'attendance_pct': args.attendance,
            'quiz_avg': args.quiz if args.quiz is not None else 6.5,
            'assignment_avg': args.assignment if args.assignment is not None else 7.0,
            'mid_marks': args.mid if args.mid is not None else 18.0,
            'lab_marks': args.lab if args.lab is not None else 15.0,
            'study_hours_week': args.hours if args.hours is not None else 9.0,
            'prev_cgpa': args.cgpa if args.cgpa is not None else 2.80,
            'part_time_job': args.job if args.job is not None else 'No'
        }
        predictor = StudentPredictor()
        result = predictor.predict_student(custom_student)
        print_prediction_result(result)
    else:
        run_sample_predictions()
        print("\nTip: Run with '--interactive' to enter custom values manually, or use '--attendance 75 --mid 18' flags!")
