# EduPulse: At-Risk Student Early Warning System
# Vibe Jam: AI Sprint | 2-Hour CodeVerse26 Competition

EduPulse is an AI-powered early-warning decision support system designed for university academic advisors and course instructors. By evaluating mid-semester indicators (attendance, quiz averages, assignment scores, mid-term marks, lab marks, weekly study hours, prior CGPA, and employment status), EduPulse identifies vulnerable students before final examinations and provides plain-language intervention recommendations.

---

## 📁 Project Directory Structure

```text
code verse/
│
├── student_data.csv                    # Original raw dataset with missing values & duplicates
├── cleaned_student_data.csv            # Cleaned & imputed dataset (600 unique records)
│
├── task_a1_data_cleaning.py            # [Task A1] Load CSV, deduplicate, median imputation, data summary
├── task_a2_dashboard_charts.py         # [Task A2] Generates 5 analytics charts & visual insights
├── task_a3_model_training.py           # [Task A3] Trains Logistic Regression, Random Forest, Decision Tree
├── task_a4_prediction.py               # [Task A4] Single student risk prediction form (CLI & flags support)
├── task_a5_explainability.py           # [Task A5] Feature importance ranking & plain-language advisor advice
├── task_a6_batch_prediction.py         # [Task A6 - Bonus] Whole class batch processing & prioritized export
├── task_a7_risk_simulation.py          # [Task A7 - Bonus] Risk badges & 'What-if' counterfactual simulator
│
├── app.py                              # All-in-One Interactive Streamlit Web Application
├── README.md                           # Documentation and execution guide
│
├── charts/                             # Saved high-resolution figures for submission
│   ├── chart1_attendance_vs_result.png
│   ├── chart2_cgpa_distribution.png
│   ├── chart3_part_time_job_effect.png
│   ├── chart4_correlation_heatmap.png
│   ├── chart5_mid_vs_lab_marks.png
│   ├── task_a3_confusion_matrices.png
│   ├── task_a5_feature_importance.png
│   └── task_a7_what_if_attendance_curve.png
│
├── models/                             # Serialized trained model artifacts
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── decision_tree.joblib
│   ├── scaler.joblib
│   └── model_metadata.joblib
│
├── at_risk_students_prioritized.csv    # Export of vulnerable students sorted by failure risk
└── full_class_predictions.csv          # Complete cohort risk scores & tier classifications
```

---

## 🚀 How to Run in VS Code / Terminal

### Prerequisites
Make sure dependencies are installed:
```bash
pip install pandas numpy scikit-learn matplotlib streamlit joblib
```

---

### Step-by-Step Task Execution

#### 1. Task A1: Data Cleaning & Summary
```bash
python task_a1_data_cleaning.py
```
- Removes 6 duplicate records.
- Handles missing values using robust **Median Imputation** (and Mode for categorical).
- Displays summary statistics: 600 total unique students (495 Pass, 105 Fail / At-Risk).
- Outputs: `cleaned_student_data.csv`.

#### 2. Task A2: Dashboard Charts & EDA
```bash
python task_a2_dashboard_charts.py
```
- Generates 5 analytical charts saved in `charts/`:
  - Attendance vs Outcome Boxplot
  - CGPA Distribution Histogram
  - Part-time Employment Risk Breakdown
  - Academic Feature Correlation Heatmap
  - Mid-Term vs Lab Marks Interaction Scatter

#### 3. Task A3: Machine Learning Model Benchmarking
```bash
python task_a3_model_training.py
```
- Evaluates **Logistic Regression**, **Random Forest**, and **Decision Tree** using stratified 80/20 train/test split.
- Compares Accuracy, Recall, Precision, and F1-Score for both Pass and Fail classes.
- Generates and saves confusion matrices to `charts/task_a3_confusion_matrices.png`.
- Saves model artifacts into `models/`.

#### 4. Task A4: Single Student Prediction Form
Run sample test cases:
```bash
python task_a4_prediction.py
```
Run interactively (prompts you to type student metrics):
```bash
python task_a4_prediction.py --interactive
```
Or pass specific student parameters via CLI flags:
```bash
python task_a4_prediction.py --attendance 68.0 --mid 13.5 --quiz 5.0 --assignment 5.5 --lab 11.0 --hours 4.0 --cgpa 2.25 --job Yes
```

#### 5. Task A5: Feature Importance & Plain-Language Advice
```bash
python task_a5_explainability.py
```
- Identifies the top factors driving failure risk (e.g. low mid-term marks, low attendance, low CGPA).
- Produces customized, actionable advice messages for advisors and students.

#### 6. Task A6: Batch Class Processing (Bonus Task)
```bash
python task_a6_batch_prediction.py
```
- Ingests whole-class CSV roster.
- Ranks students in descending order of failure risk.
- Generates:
  - `at_risk_students_prioritized.csv`
  - `full_class_predictions.csv`

#### 7. Task A7: Risk Levels & What-If Simulator (Bonus Task)
```bash
python task_a7_risk_simulation.py
```
- Applies color-coded Risk Tiers (`[HIGH RISK]`, `[MEDIUM RISK]`, `[LOW RISK]`).
- Simulates counterfactual improvements (e.g. +12% attendance, +4 mid-term marks) and displays risk reduction delta.
- Plots attendance sensitivity curve to `charts/task_a7_what_if_attendance_curve.png`.

---

### 🌐 Launch the Interactive Web Application (Streamlit)
To launch the full interactive web application in your browser from VS Code:
```bash
streamlit run app.py
```
The app will open automatically at `http://localhost:8501`.

#### Features inside the Web App:
1. **Overview & Architecture**: Live cohort KPIs and system overview.
2. **A1 Data Cleaning**: Interactive dataset table, missing value analysis, and balance stats.
3. **A2 Visual Dashboard**: Interactive selector for all 5 analytical charts with takeaways.
4. **A3 ML Evaluation**: Performance metrics table and confusion matrices.
5. **A4 & A5 Prediction & Advice**: Real-time interactive sliders for student metrics, risk gauge, and instant actionable guidance.
6. **A6 Batch Processing**: File uploader for class CSV with live priority ranking and CSV download button.
7. **A7 What-If Simulator**: Live intervention sliders with real-time risk reduction and dynamic sensitivity curves.

---

## 🏆 Key Analytical Findings & Hackathon Results

1. **Primary Failure Drivers**:
   - **Mid-Term Examination Score** ($\beta = -1.343$): Scoring below 15/30 is the single strongest indicator of semester failure.
   - **Attendance Percentage** ($\beta = -1.290$): Passing students median attendance is **78.6%**, compared to **70.9%** for at-risk students.
   - **Prior CGPA** ($\beta = -1.276$): Students entering below 2.50 require immediate baseline monitoring.
2. **Optimal ML Model Choice**:
   - In an academic early warning setting, **Recall is the critical metric** (failing to identify an at-risk student has severe consequences).
   - **Logistic Regression achieved 100% Recall** on the test set, successfully flagging all 21 at-risk students with zero false negatives.
