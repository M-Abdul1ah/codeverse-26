"""
EduPulse - At-Risk Student Early Warning System
Task A1: Data Cleaning, Imputation & Summary Report

Requirements:
- Load the CSV
- Remove duplicates
- Handle missing values (Median imputation for numeric, Mode for categorical)
- Strip column names and string whitespace
- Show a complete data summary (rows, pass/fail count, class balance)
- Save cleaned dataset
"""

import os
import json
import pandas as pd
import numpy as np

def clean_data(input_csv_path, output_csv_path=None):
    print("=" * 65)
    print(" TASK A1: DATA LOADING, CLEANING & SUMMARY REPORT")
    print("=" * 65)

    # 1. Load CSV
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)
    initial_rows = len(df)
    print(f" Loaded CSV: '{input_csv_path}'")
    print(f" Initial shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # Clean whitespace in column names
    df.columns = df.columns.str.strip()

    # Clean whitespace in string/object columns
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})

    # 2. Check and remove duplicates
    duplicate_count = df.duplicated().sum()
    print(f"[-] Duplicate rows detected: {duplicate_count}")
    df_cleaned = df.drop_duplicates().copy()
    rows_after_dedup = len(df_cleaned)
    print(f"[+] Rows after removing duplicates: {rows_after_dedup}\n")

    # 3. Check and handle missing values
    missing_summary = df_cleaned.isnull().sum()
    print("[-] Missing values before imputation:")
    for col, count in missing_summary.items():
        if count > 0:
            print(f"    - {col:<18}: {count} missing ({count / rows_after_dedup * 100:.2f}%)")

    # Numeric features for median imputation
    numeric_cols = [
        'attendance_pct', 'quiz_avg', 'assignment_avg', 
        'mid_marks', 'lab_marks', 'study_hours_week', 'prev_cgpa'
    ]

    imputation_values = {}
    for col in numeric_cols:
        if col in df_cleaned.columns:
            median_val = round(float(df_cleaned[col].median()), 2)
            imputation_values[col] = median_val
            df_cleaned[col] = df_cleaned[col].fillna(median_val)

    # Categorical features for mode imputation
    categorical_cols = ['part_time_job', 'final_result']
    for col in categorical_cols:
        if col in df_cleaned.columns:
            mode_val = df_cleaned[col].mode()[0]
            imputation_values[col] = mode_val
            df_cleaned[col] = df_cleaned[col].fillna(mode_val)

    # Verify no missing values remain
    remaining_missing = df_cleaned.isnull().sum().sum()
    print(f"\n[+] Missing values successfully imputed:")
    for col, val in imputation_values.items():
        print(f"    - {col:<18} -> Imputed with {val}")
    print(f"[+] Total remaining missing values across all columns: {remaining_missing}\n")

    # 4. Data Summary (Rows, Pass/Fail count)
    pass_fail_counts = df_cleaned['final_result'].value_counts()
    pass_count = pass_fail_counts.get('Pass', 0)
    fail_count = pass_fail_counts.get('Fail', 0)
    total_students = len(df_cleaned)

    print("=" * 65)
    print(" DATA SUMMARY REPORT (FINAL CLEANED DATASET)")
    print("=" * 65)
    print(f" Total Unique Students: {total_students}")
    print(f" Pass Count           : {pass_count} ({pass_count / total_students * 100:.2f}%)")
    print(f" Fail Count (At-Risk) : {fail_count} ({fail_count / total_students * 100:.2f}%)")
    print(f" Pass-to-Fail Ratio   : {pass_count / fail_count:.2f}:1")
    print("-" * 65)
    print("\nFeature Descriptive Statistics (Cleaned Data):")
    print(df_cleaned[numeric_cols].describe().round(2))

    # 5. Save Cleaned Dataset (with fallback if open in another program)
    if output_csv_path:
        try:
            df_cleaned.to_csv(output_csv_path, index=False)
            print(f"\n[+] Cleaned dataset successfully saved to: '{output_csv_path}'")
        except PermissionError:
            fallback_path = output_csv_path.replace('.csv', '_latest.csv')
            df_cleaned.to_csv(fallback_path, index=False)
            print(f"\n[!] Note: '{output_csv_path}' is currently open in an external application (e.g. Excel).")
            print(f"[+] Cleaned dataset successfully saved to fallback file: '{fallback_path}'")

    print("=" * 65)
    print(" TASK A1 COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    return df_cleaned

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, 'student_data.csv')
    cleaned_path = os.path.join(base_dir, 'cleaned_student_data.csv')
    clean_data(raw_path, cleaned_path)
