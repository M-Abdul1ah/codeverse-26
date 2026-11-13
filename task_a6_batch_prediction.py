"""
EduPulse - At-Risk Student Early Warning System
Task A6: Batch Mode - Class CSV Processing & At-Risk Prioritized Export

Requirements:
- Upload/process CSV of a whole class
- Validate input columns and report missing/mismatched features
- Support both Logistic Regression and Random Forest with consistent preprocessing
- Compute risk probability for each student
- Sort students in descending order of failure risk
- Download/export a prioritized CSV of at-risk students
"""

import os
import joblib
import pandas as pd
import numpy as np

def validate_and_preprocess_batch(df_raw, feature_cols, training_medians):
    """
    Validates uploaded CSV columns and imputes missing features using training medians.
    Returns: cleaned DataFrame, list of missing columns detected.
    """
    df_batch = df_raw.copy()
    df_batch.columns = df_batch.columns.str.strip()

    missing_cols = [c for c in feature_cols if c not in df_batch.columns and c != 'part_time_job_numeric']

    # Map part_time_job
    if 'part_time_job' in df_batch.columns:
        df_batch['part_time_job_numeric'] = df_batch['part_time_job'].astype(str).str.strip().str.lower().isin(['yes', 'y', '1', 'true']).astype(int)
    elif 'part_time_job_numeric' not in df_batch.columns:
        df_batch['part_time_job_numeric'] = 0
        missing_cols.append('part_time_job')

    # Impute missing features
    for col in feature_cols:
        if col not in df_batch.columns:
            df_batch[col] = training_medians.get(col, 0.0)
        col_med = df_batch[col].median() if not pd.isna(df_batch[col].median()) else training_medians.get(col, 0.0)
        df_batch[col] = pd.to_numeric(df_batch[col], errors='coerce').fillna(col_med).fillna(training_medians.get(col, 0.0))

    return df_batch, missing_cols

def process_batch_class(input_csv_path, output_at_risk_path=None, output_full_path=None, model_choice='Logistic Regression'):
    print("=" * 65)
    print(f" TASK A6: BATCH CLASS PROCESSING & AT-RISK EXPORT (Model: {model_choice})")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(base_dir, 'models')

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input batch file not found: {input_csv_path}")

    # Load Model, Scaler, and Metadata
    lr_model = joblib.load(os.path.join(models_path, 'logistic_regression.joblib'))
    rf_model = joblib.load(os.path.join(models_path, 'random_forest.joblib'))
    scaler = joblib.load(os.path.join(models_path, 'scaler.joblib'))
    metadata = joblib.load(os.path.join(models_path, 'model_metadata.joblib'))
    feature_cols = metadata['feature_cols']
    training_medians = metadata.get('training_medians', {
        'attendance_pct': 77.75, 'quiz_avg': 6.5, 'assignment_avg': 7.1,
        'mid_marks': 18.0, 'lab_marks': 15.3, 'study_hours_week': 9.05,
        'prev_cgpa': 2.78, 'part_time_job_numeric': 0
    })

    df_raw = pd.read_csv(input_csv_path)
    print(f" Loaded batch class roster: {len(df_raw)} students from '{input_csv_path}'")

    # Column Validation & Preprocessing
    df_batch, missing_cols = validate_and_preprocess_batch(df_raw, feature_cols, training_medians)
    if missing_cols:
        print(f"[!] Warning: Input CSV was missing columns: {missing_cols}. Filled using training medians.")

    X_batch = df_batch[feature_cols].copy()

    # Predict Risk Probability with appropriate preprocessing for selected model
    if model_choice == 'Logistic Regression':
        X_scaled = scaler.transform(X_batch)
        pred_labels = lr_model.predict(X_scaled)
        prob_fails = lr_model.predict_proba(X_scaled)[:, 1]
    else:
        pred_labels = rf_model.predict(X_batch)
        prob_fails = rf_model.predict_proba(X_batch)[:, 1]

    # Add output columns
    df_result = df_raw.copy()
    df_result['failure_risk_pct'] = np.round(prob_fails * 100, 2)
    df_result['predicted_outcome'] = np.where(pred_labels == 1, 'Fail / At-Risk', 'Pass')

    # Assign Risk Level (Task A7 integration)
    conditions = [
        df_result['failure_risk_pct'] >= 65.0,
        (df_result['failure_risk_pct'] >= 35.0) & (df_result['failure_risk_pct'] < 65.0),
        df_result['failure_risk_pct'] < 35.0
    ]
    tiers = ['HIGH RISK', 'MEDIUM RISK', 'LOW RISK']
    df_result['risk_level'] = np.select(conditions, tiers, default='LOW RISK')

    # Sort descending by risk (Highest Risk First)
    df_result_sorted = df_result.sort_values(by='failure_risk_pct', ascending=False).reset_index(drop=True)
    df_result_sorted['advisor_priority_rank'] = df_result_sorted.index + 1

    # Filter At-Risk Cohort (Risk >= 50% or Predicted Fail)
    at_risk_df = df_result_sorted[df_result_sorted['failure_risk_pct'] >= 50.0].copy()

    # Save to disk
    if output_at_risk_path:
        at_risk_df.to_csv(output_at_risk_path, index=False)
        print(f"[+] Saved Prioritized At-Risk List -> '{output_at_risk_path}'")
    if output_full_path:
        df_result_sorted.to_csv(output_full_path, index=False)
        print(f"[+] Saved Full Class Risk Scores  -> '{output_full_path}'")

    # Print Summary
    print("\n" + "=" * 65)
    print(" BATCH PROCESSING SUMMARY REPORT")
    print("=" * 65)
    print(f" Total Students Evaluated       : {len(df_result_sorted)}")
    print(f" High Risk Students (>= 65%)    : {sum(df_result_sorted['risk_level'] == 'HIGH RISK')}  [Immediate Intervention]")
    print(f" Medium Risk Students (35%-64%) : {sum(df_result_sorted['risk_level'] == 'MEDIUM RISK')}  [Targeted Monitoring]")
    print(f" Low Risk Students (< 35%)      : {sum(df_result_sorted['risk_level'] == 'LOW RISK')}  [On Track]")
    print("-" * 65)
    print("\nTOP 10 MOST CRITICAL AT-RISK STUDENTS (ADVISOR PRIORITY):")
    cols_to_show = ['advisor_priority_rank', 'student_id', 'failure_risk_pct', 'risk_level', 'attendance_pct', 'mid_marks', 'lab_marks', 'prev_cgpa']
    display_cols = [c for c in cols_to_show if c in df_result_sorted.columns]
    print(df_result_sorted[display_cols].head(10).to_string(index=False))
    print("=" * 65)
    print(" TASK A6 COMPLETED SUCCESSFULLY!")
    print("=" * 65)

    return df_result_sorted, at_risk_df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'cleaned_student_data.csv')
    at_risk_file = os.path.join(base_dir, 'at_risk_students_prioritized.csv')
    full_file = os.path.join(base_dir, 'full_class_predictions.csv')
    process_batch_class(input_file, at_risk_file, full_file)
