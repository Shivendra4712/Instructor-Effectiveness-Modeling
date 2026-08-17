# Instructor Effectiveness Modeling (EdTech Context)

An end-to-end Data Science and Machine Learning pipeline developed for an EdTech platform to analyze, define, and predict **Instructor Effectiveness Tiers** based on learner outcomes, engagement analytics, and feedback metrics.

---

## 💻 Live Project Links
Aap is project ka live execution code aur core source elements yahan click karke direct dekh sakte hain:  
👉 **Live Google Colab Notebook:** https://google.com  
👉 **Live Dataset Download:** https://drive.google.com/file/d/1NgAGEEGVC9ygYnXswVawZv7M5zNk8Kc3/view?usp=sharing

---

## 📌 Project Overview
In an EdTech ecosystem, instructors handle multiple batches across different courses over time. This project implements a robust framework to:
1. **Simulate a realistic 500-row, 12-column synthetic dataset** adhering strictly to operational business distributions (Beta & Normal distributions).
2. **Perform Exploratory Data Analysis (EDA)**, outlier treatment, and median-based missing value imputation.
3. **Formulate a normalized Instructor Effectiveness Score** and split performance into balanced classification tiers (Low, Medium, High) using multi-class quantile splitting.
4. **Train and evaluate a tuned Random Forest Classifier** to predict instructor performance tiers with robust feature scaling.

---

## 📊 Dataset Schema (12 Columns)

| Category | Column Name | Description | Range / Type |
|---|---|---|---|
| **Identifiers** | `batch_id` | Unique identifier for a course batch | String |
| | `instructor_id` | Unique instructor identifier | String |
| | `course_id` | Course identifier | String |
| **Learner Outcomes** | `completion_rate` | Fraction of learners who completed the course | 0.0 – 1.0 |
| | `dropout_rate` | Fraction of learners who dropped out | 0.0 – 1.0 |
| | `avg_score_improvement` | Average improvement from pre- to post-assessment | Continuous |
| | `avg_quiz_score` | Average quiz score for the batch | 0.0 – 100.0 |
| **Engagement** | `avg_watch_time` | Normalized average video watch time | 0.0 – 1.0 |
| | `assignment_submission_rate` | Fraction of learners submitting assignments | 0.0 – 1.0 |
| | `forum_activity_rate` | Fraction of learners active on discussion forums | 0.0 – 1.0 |
| **Feedback** | `avg_feedback_score` | Average learner feedback rating | 1.0 – 5.0 |
| | `feedback_response_rate` | Fraction of learners who submitted feedback | 0.0 – 1.0 |

---

## ⚙️ Core Architecture & Pipeline Phases

### 1. Robust Data Cleaning
* Handles missing feedback markers using median-based imputation strategies to prevent model drift.
* Corrects directional anomalies (e.g., negative score improvements) via localized distribution bounds.

### 2. Strategic Feature Engineering
* Features are normalized using min-max mapping scaling matrices to establish mathematical parity before composite indexing.
* **Effectiveness Score Formula**: 
  $$\text{Score} = 0.30 \times \text{Completion} + 0.20 \times \text{Feedback} + 0.20 \times \text{Quiz} + 0.20 \times \text{Improvement} + 0.10 \times \text{Watch Time}$$

### 3. Balanced Tier Quantilization
* Avoids class imbalance issues by using **Quantile Splitting (33rd and 66th percentiles)** to enforce equal distribution among Low (0), Medium (1), and High (2) tiers.

### 4. Predictive Modeling
* Utilizes a **Random Forest Classifier** with specialized depth constraints (`max_depth=5`) to guarantee maximum generalization capability across batches.

---

## 🛠️ Installation & Usage

### Prerequisites
Make sure you have the required libraries installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Pipeline Locally
Simply clone this repository and execute the script:
```bash
python instructor_effectiveness_modeling.py
```

---

## 🧠 Strategic Evaluation Insights

* **Top Drivers**: Course completion rates combined with student feedback distributions act as the primary operational anchors influencing classification metrics.
* **Systemic Failures**: The model assumes uniform distribution of course complexity. It might inadvertently penalize elite instructors managing high-difficulty/advanced engineering tracks.
* **Ethical Usage**: This framework is designed strictly as an internal developmental diagnostic tool and should not be used as an absolute metric for high-stakes payroll decisions.
