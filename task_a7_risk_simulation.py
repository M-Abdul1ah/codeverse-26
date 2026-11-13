"""
EduPulse - At-Risk Student Early Warning System
Task A7: Risk Levels Badging & Interactive "What-If" Counterfactual Simulator

Requirements:
- Risk levels (Low / Medium / High) with colour badges / indicators
- "What-if" simulation engine (e.g. "What if attendance goes up 10%?", "What if mid-term improves?")
- Computes dynamic risk change delta (accurate "Risk Reduction" or "Risk Increase" labels)
- Generates attendance sensitivity curve chart
"""

import os
# Configure local cache to prevent sandbox permission warnings
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.matplotlib_cache')

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_risk_badge(risk_pct):
    """
    Returns visual badge text and color metadata.
    """
    if risk_pct >= 65.0:
        return {
            'level': 'HIGH RISK',
            'color': '#e74c3c',
            'badge': '[HIGH RISK - CRITICAL]',
            'action': 'Urgent intervention and mandatory advising required.'
        }
    elif risk_pct >= 35.0:
        return {
            'level': 'MEDIUM RISK',
            'color': '#f39c12',
            'badge': '[MEDIUM RISK - MONITOR]',
            'action': 'Targeted academic support and progress check recommended.'
        }
    else:
        return {
            'level': 'LOW RISK',
            'color': '#27ae60',
            'badge': '[LOW RISK - ON TRACK]',
            'action': 'Student is performing adequately.'
        }

def format_risk_delta(delta_pct):
    """
    Formats the delta string with mathematically accurate description.
    """
    if delta_pct < -0.05:
        return f"{abs(delta_pct):.1f}% (Risk Reduction)"
    elif delta_pct > 0.05:
        return f"+{delta_pct:.1f}% (Risk Increase)"
    else:
        return "0.0% (No Change)"

class WhatIfSimulator:
    def __init__(self, models_dir='models', charts_dir='charts'):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_path = os.path.join(base_dir, models_dir)
        self.charts_path = os.path.join(base_dir, charts_dir)
        os.makedirs(self.charts_path, exist_ok=True)

        self.lr_model = joblib.load(os.path.join(self.models_path, 'logistic_regression.joblib'))
        self.scaler = joblib.load(os.path.join(self.models_path, 'scaler.joblib'))
        self.metadata = joblib.load(os.path.join(self.models_path, 'model_metadata.joblib'))
        self.feature_cols = self.metadata['feature_cols']

    def predict_risk(self, student_dict):
        job_val = 1 if str(student_dict.get('part_time_job', 'No')).strip().lower() in ['yes', 'y', '1', 'true'] else 0
        row = [
            float(student_dict['attendance_pct']),
            float(student_dict['quiz_avg']),
            float(student_dict['assignment_avg']),
            float(student_dict['mid_marks']),
            float(student_dict['lab_marks']),
            float(student_dict['study_hours_week']),
            float(student_dict['prev_cgpa']),
            job_val
        ]
        X_df = pd.DataFrame([row], columns=self.feature_cols)
        X_scaled = self.scaler.transform(X_df)
        prob_fail = self.lr_model.predict_proba(X_scaled)[0][1]
        return prob_fail * 100.0

    def simulate_interventions(self, baseline_student, attendance_boost=10.0, mid_marks_boost=3.0, study_hours_boost=3.0, save_chart=True):
        """
        Simulates individual and combined interventions and calculates risk reduction.
        """
        base_risk = self.predict_risk(baseline_student)
        base_badge = get_risk_badge(base_risk)

        scenarios = []

        # Scenario 1: Baseline
        scenarios.append({
            'scenario': 'Baseline (Current State)',
            'attendance': baseline_student['attendance_pct'],
            'mid_marks': baseline_student['mid_marks'],
            'study_hours': baseline_student['study_hours_week'],
            'risk_pct': base_risk,
            'badge': base_badge['badge'],
            'delta_pct': 0.0,
            'delta_desc': 'Baseline'
        })

        # Scenario 2: Attendance Increase
        scen_att = baseline_student.copy()
        new_att = min(100.0, baseline_student['attendance_pct'] + attendance_boost)
        scen_att['attendance_pct'] = new_att
        risk_att = self.predict_risk(scen_att)
        delta_att = risk_att - base_risk
        scenarios.append({
            'scenario': f'What-If: Attendance +{attendance_boost}% ({new_att:.1f}%)',
            'attendance': new_att,
            'mid_marks': baseline_student['mid_marks'],
            'study_hours': baseline_student['study_hours_week'],
            'risk_pct': risk_att,
            'badge': get_risk_badge(risk_att)['badge'],
            'delta_pct': delta_att,
            'delta_desc': format_risk_delta(delta_att)
        })

        # Scenario 3: Mid-Term Improvement (Tutoring)
        scen_mid = baseline_student.copy()
        new_mid = min(30.0, baseline_student['mid_marks'] + mid_marks_boost)
        scen_mid['mid_marks'] = new_mid
        risk_mid = self.predict_risk(scen_mid)
        delta_mid = risk_mid - base_risk
        scenarios.append({
            'scenario': f'What-If: Mid-Term +{mid_marks_boost} marks ({new_mid:.1f}/30)',
            'attendance': baseline_student['attendance_pct'],
            'mid_marks': new_mid,
            'study_hours': baseline_student['study_hours_week'],
            'risk_pct': risk_mid,
            'badge': get_risk_badge(risk_mid)['badge'],
            'delta_pct': delta_mid,
            'delta_desc': format_risk_delta(delta_mid)
        })

        # Scenario 4: Combined Comprehensive Intervention
        scen_all = baseline_student.copy()
        scen_all['attendance_pct'] = new_att
        scen_all['mid_marks'] = new_mid
        scen_all['study_hours_week'] = min(35.0, baseline_student['study_hours_week'] + study_hours_boost)
        risk_all = self.predict_risk(scen_all)
        delta_all = risk_all - base_risk
        scenarios.append({
            'scenario': 'Combined Intervention (Attendance + Mid + Study Hours)',
            'attendance': new_att,
            'mid_marks': new_mid,
            'study_hours': scen_all['study_hours_week'],
            'risk_pct': risk_all,
            'badge': get_risk_badge(risk_all)['badge'],
            'delta_pct': delta_all,
            'delta_desc': format_risk_delta(delta_all)
        })

        # Sensitivity Curve over Attendance Range (30% to 100%)
        att_range = np.linspace(30, 100, 71)
        sim_risks = []
        for a in att_range:
            temp = baseline_student.copy()
            temp['attendance_pct'] = a
            sim_risks.append(self.predict_risk(temp))

        if save_chart:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(att_range, sim_risks, color='#2980b9', linewidth=3, label='Failure Risk Curve')
            ax.axhline(50, color='#e74c3c', linestyle='--', linewidth=1.5, label='50% Fail Risk Boundary')
            ax.axvline(baseline_student['attendance_pct'], color='#7f8c8d', linestyle=':', linewidth=2, 
                       label=f"Current Attendance ({baseline_student['attendance_pct']}%)")
            
            # Highlight current point
            ax.scatter([baseline_student['attendance_pct']], [base_risk], color='#c0392b', s=120, zorder=5, 
                       label=f"Current Risk: {base_risk:.1f}%")

            # Highlight simulated point
            ax.scatter([new_att], [risk_att], color='#27ae60', s=120, zorder=5, 
                       label=f"+{attendance_boost}% Attendance Risk: {risk_att:.1f}%")

            ax.set_title(f"Task A7: What-If Attendance Sensitivity Curve", fontsize=14, fontweight='bold', pad=12)
            ax.set_xlabel('Simulated Attendance Percentage (%)', fontsize=12)
            ax.set_ylabel('Predicted Risk of Failing (%)', fontsize=12)
            ax.set_ylim(-5, 105)
            ax.legend(loc='upper right', frameon=True)

            plt.tight_layout()
            curve_path = os.path.join(self.charts_path, 'task_a7_what_if_attendance_curve.png')
            plt.savefig(curve_path, dpi=300)
            plt.close()
            print(f"[+] Saved What-If Sensitivity Curve -> {curve_path}")

        return pd.DataFrame(scenarios)

def run_task_a7():
    print("=" * 65)
    print(" TASK A7: RISK TIERS, BADGES & WHAT-IF INTERVENTION SIMULATOR")
    print("=" * 65)

    simulator = WhatIfSimulator()
    sample_student = {
        'attendance_pct': 62.0,
        'quiz_avg': 5.0,
        'assignment_avg': 5.5,
        'mid_marks': 13.5,
        'lab_marks': 12.0,
        'study_hours_week': 4.5,
        'prev_cgpa': 2.35,
        'part_time_job': 'Yes'
    }

    print("\n[+] Target Student Baseline Profile:")
    for k, v in sample_student.items():
        print(f"    - {k:<20}: {v}")

    results_df = simulator.simulate_interventions(sample_student, attendance_boost=12.0, mid_marks_boost=4.0)

    print("\n" + "=" * 65)
    print(" WHAT-IF COUNTERFACTUAL SIMULATION OUTCOMES:")
    print("=" * 65)
    for idx, row in results_df.iterrows():
        print(f" Scenario: {row['scenario']}")
        print(f"   -> Risk: {row['risk_pct']:.1f}% ({row['delta_desc']}) | Badge: {row['badge']}")
    print("=" * 65)
    print(" TASK A7 COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == '__main__':
    run_task_a7()
