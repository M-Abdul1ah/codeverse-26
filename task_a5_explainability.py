"""
EduPulse - At-Risk Student Early Warning System
Task A5: Explainability, Factor Breakdown & Plain-Language Actionable Advice

Requirements:
- Show top global factors (feature importance for Random Forest and coefficients for Logistic Regression)
- Provide per-student factor diagnostic (compares individual metrics against safe thresholds)
- Generate plain-language advice messages tailored to the student's weaknesses
- e.g., "Attendance is low - improve to above 75%"
- Save visual feature importance chart to charts/
"""

import os
# Configure local cache to prevent sandbox permission warnings
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.matplotlib_cache')

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ResultExplainer:
    def __init__(self, models_dir='models', charts_dir='charts'):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_path = os.path.join(base_dir, models_dir)
        self.charts_path = os.path.join(base_dir, charts_dir)
        os.makedirs(self.charts_path, exist_ok=True)

        self.lr_model = joblib.load(os.path.join(self.models_path, 'logistic_regression.joblib'))
        self.rf_model = joblib.load(os.path.join(self.models_path, 'random_forest.joblib'))
        self.metadata = joblib.load(os.path.join(self.models_path, 'model_metadata.joblib'))
        self.feature_names = self.metadata['feature_cols']

    def get_global_feature_importance(self, save_chart=True):
        """
        Calculates and ranks feature importance for both Logistic Regression (coefficients)
        and Random Forest (Gini importances).
        """
        lr_coefs = self.lr_model.coef_[0]
        rf_importances = self.rf_model.feature_importances_

        df_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'LR_Coefficient': lr_coefs,
            'LR_Abs_Impact': np.abs(lr_coefs),
            'RF_Importance': rf_importances
        }).sort_values(by='LR_Abs_Impact', ascending=False)

        if save_chart:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot 1: Logistic Regression Coefficients
            sorted_lr = df_importance.sort_values(by='LR_Coefficient', ascending=True)
            colors = ['#27ae60' if c < 0 else '#c0392b' for c in sorted_lr['LR_Coefficient']]
            ax1.barh(sorted_lr['Feature'], sorted_lr['LR_Coefficient'], color=colors, alpha=0.85, edgecolor='black')
            ax1.axvline(0, color='black', linestyle='--', linewidth=1)
            ax1.set_title('Logistic Regression Feature Coefficients\n(Green = Reduces Risk, Red = Increases Risk)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Standardized Coefficient (Log-Odds Impact)', fontsize=11)

            # Plot 2: Random Forest Relative Feature Importance
            sorted_rf = df_importance.sort_values(by='RF_Importance', ascending=True)
            ax2.barh(sorted_rf['Feature'], sorted_rf['RF_Importance'], color='#2980b9', alpha=0.85, edgecolor='black')
            ax2.set_title('Random Forest Feature Importance\n(Overall Predictive Contribution)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Relative Importance (Gini)', fontsize=11)

            plt.tight_layout()
            chart_path = os.path.join(self.charts_path, 'task_a5_feature_importance.png')
            plt.savefig(chart_path, dpi=300)
            plt.close()
            print(f"[+] Saved Feature Importance Chart -> {chart_path}")

        return df_importance

    def get_student_risk_breakdown(self, student_dict):
        """
        Per-Student Explainability:
        Compares individual student metrics against institutional benchmarks to show
        WHY this specific student is predicted at-risk or safe.
        """
        att = float(student_dict.get('attendance_pct', 77.75))
        mid = float(student_dict.get('mid_marks', 18.0))
        lab = float(student_dict.get('lab_marks', 15.3))
        quiz = float(student_dict.get('quiz_avg', 6.5))
        assign = float(student_dict.get('assignment_avg', 7.1))
        hours = float(student_dict.get('study_hours_week', 9.0))
        cgpa = float(student_dict.get('prev_cgpa', 2.78))
        has_job = str(student_dict.get('part_time_job', 'No')).strip().lower() in ['yes', 'y', '1', 'true']

        breakdown = [
            {
                'Metric': 'Mid-Term Marks',
                'Student Value': f"{mid:.1f} / 30",
                'Safe Standard': '>= 15.0 / 30',
                'Status': 'Critically Low' if mid < 15.0 else 'Safe',
                'Risk Factor': 'HIGH' if mid < 15.0 else 'LOW'
            },
            {
                'Metric': 'Attendance Rate',
                'Student Value': f"{att:.1f}%",
                'Safe Standard': '>= 75.0%',
                'Status': 'Critically Low' if att < 75.0 else ('Borderline' if att < 80.0 else 'Safe'),
                'Risk Factor': 'HIGH' if att < 75.0 else ('MODERATE' if att < 80.0 else 'LOW')
            },
            {
                'Metric': 'Prior CGPA',
                'Student Value': f"{cgpa:.2f} / 4.0",
                'Safe Standard': '>= 2.50',
                'Status': 'Vulnerable' if cgpa < 2.50 else 'Safe',
                'Risk Factor': 'HIGH' if cgpa < 2.50 else 'LOW'
            },
            {
                'Metric': 'Lab Marks',
                'Student Value': f"{lab:.1f} / 20",
                'Safe Standard': '>= 12.0 / 20',
                'Status': 'At-Risk' if lab < 12.0 else 'Safe',
                'Risk Factor': 'MODERATE' if lab < 12.0 else 'LOW'
            },
            {
                'Metric': 'Quiz Average',
                'Student Value': f"{quiz:.1f} / 10",
                'Safe Standard': '>= 6.0 / 10',
                'Status': 'Below Standard' if quiz < 6.0 else 'Safe',
                'Risk Factor': 'MODERATE' if quiz < 6.0 else 'LOW'
            },
            {
                'Metric': 'Assignment Avg',
                'Student Value': f"{assign:.1f} / 10",
                'Safe Standard': '>= 6.0 / 10',
                'Status': 'Below Standard' if assign < 6.0 else 'Safe',
                'Risk Factor': 'LOW'
            },
            {
                'Metric': 'Study Hours/Wk',
                'Student Value': f"{hours:.1f} hrs",
                'Safe Standard': '>= 8.0 hrs/wk',
                'Status': 'Insufficient' if hours < 8.0 else 'Safe',
                'Risk Factor': 'MODERATE' if hours < 8.0 else 'LOW'
            },
            {
                'Metric': 'Part-Time Job',
                'Student Value': 'Yes' if has_job else 'No',
                'Safe Standard': 'No Job or Balanced Workload',
                'Status': 'Workload Strain' if has_job and (att < 75.0 or hours < 8.0) else 'Normal',
                'Risk Factor': 'MODERATE' if has_job and (att < 75.0 or hours < 8.0) else 'LOW'
            }
        ]

        return pd.DataFrame(breakdown)

    def generate_advice(self, student_dict, failure_risk_pct=None):
        """
        Generates dynamic, plain-language actionable advice for students and advisors.
        """
        advice_list = []
        att = float(student_dict.get('attendance_pct', 80.0))
        mid = float(student_dict.get('mid_marks', 20.0))
        lab = float(student_dict.get('lab_marks', 15.0))
        quiz = float(student_dict.get('quiz_avg', 7.0))
        assign = float(student_dict.get('assignment_avg', 7.0))
        hours = float(student_dict.get('study_hours_week', 8.0))
        cgpa = float(student_dict.get('prev_cgpa', 3.0))
        has_job = str(student_dict.get('part_time_job', 'No')).strip().lower() in ['yes', 'y', '1', 'true']

        # Rule 1: Attendance
        if att < 75.0:
            deficit = round(75.0 - att, 1)
            advice_list.append(f"Attendance is critically low ({att}%). Increase attendance by at least {deficit}% to meet the mandatory 75% institutional requirement.")
        elif att < 80.0:
            advice_list.append(f"Attendance is borderline ({att}%). Regular class attendance directly protects against failing.")

        # Rule 2: Mid-Term Examination
        if mid < 15.0:
            advice_list.append(f"Mid-term exam score is low ({mid}/30). Book office hours with the instructor or TA for concept reinforcement before the final exam.")

        # Rule 3: Lab Marks
        if lab < 12.0:
            advice_list.append(f"Lab practical score is below passing standard ({lab}/20). Complete missing lab reports and attend makeup lab sessions.")

        # Rule 4: Quizzes & Assignments
        if quiz < 6.0:
            advice_list.append(f"Quiz average is weak ({quiz}/10). Review weekly lecture notes and practice self-quizzing.")
        if assign < 6.0:
            advice_list.append(f"Assignment average is below target ({assign}/10). Seek peer study group support or writing center assistance.")

        # Rule 5: Study Hours
        if hours < 8.0:
            advice_list.append(f"Weekly independent study time ({hours} hrs/week) is low. Dedicate at least 10 hrs/week for comprehensive revision.")

        # Rule 6: Prior CGPA
        if cgpa < 2.50:
            advice_list.append(f"Prior CGPA is vulnerable ({cgpa}/4.0). Seek foundational academic coaching through the student success center.")

        # Rule 7: Part-Time Job Workload
        if has_job and (hours < 8.0 or att < 75.0):
            advice_list.append("Work-study balance: Managing a part-time job appears to impact study hours. Consider academic workload adjustment or time-management advising.")

        # If no flags triggered
        if not advice_list:
            advice_list.append("Student performance is in good standing across all metrics! Keep up the consistent effort and active participation.")

        return advice_list

def explain_and_advise(student_dict, failure_risk_pct=None):
    explainer = ResultExplainer()
    print("=" * 65)
    print(" TASK A5: MODEL EXPLAINABILITY & STUDENT ADVICE REPORT")
    print("=" * 65)
    
    # Global Importance
    print("\n[+] Top Global Predictive Factors Ranking:")
    importance_df = explainer.get_global_feature_importance(save_chart=True)
    for idx, row in importance_df.iterrows():
        impact_dir = "Reduces failure risk" if row['LR_Coefficient'] < 0 else "Increases failure risk"
        print(f"  {row['Feature']:<22} | Coeff: {row['LR_Coefficient']:+.3f} ({impact_dir}) | RF Importance: {row['RF_Importance']:.3f}")

    # Per-Student Factor Breakdown
    print("\n" + "-" * 65)
    print(" PER-STUDENT RISK FACTOR BREAKDOWN:")
    print("-" * 65)
    breakdown_df = explainer.get_student_risk_breakdown(student_dict)
    print(breakdown_df.to_string(index=False))

    # Plain Language Advice
    print("\n" + "-" * 65)
    print(" TAILORED PLAIN-LANGUAGE ADVICE FOR STUDENT:")
    print("-" * 65)
    advice_items = explainer.generate_advice(student_dict, failure_risk_pct)
    for i, item in enumerate(advice_items, 1):
        print(f"  {i}. {item}")
    print("=" * 65)
    print(" TASK A5 COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == '__main__':
    sample_at_risk_student = {
        'attendance_pct': 58.5,
        'quiz_avg': 4.5,
        'assignment_avg': 5.2,
        'mid_marks': 12.0,
        'lab_marks': 10.5,
        'study_hours_week': 4.0,
        'prev_cgpa': 2.20,
        'part_time_job': 'Yes'
    }
    explain_and_advise(sample_at_risk_student, failure_risk_pct=94.5)
