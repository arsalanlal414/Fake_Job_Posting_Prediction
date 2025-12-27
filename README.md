# Fake Job Postings Prediction Project

## 📋 Project Overview
This project aims to detect fraudulent job postings using machine learning techniques. With the rise of online job platforms, identifying fake job postings has become crucial to protect job seekers from scams.

## 🎯 Objective
Build a binary classification model to predict whether a job posting is fraudulent (1) or legitimate (0).

## 📊 Dataset
- **Training Data**: 9,999 job postings
- **Test Data**: 7,882 job postings
- **Features**: 18+ columns including text and categorical data

### Key Features:
- **Text Features**: title, location, company_profile, description, requirements, benefits
- **Categorical Features**: employment_type, required_experience, required_education, industry, function
- **Binary Features**: telecommuting, has_company_logo, has_questions
- **Numerical Features**: salary_range (needs parsing)
- **Target**: fraudulent (0 = legitimate, 1 = fake)

## 📁 Project Structure
```
fake_job_prediction_project/
├── data/                          # Dataset files
│   ├── fake_job_postings_train.csv
│   ├── fake_job_postings_test.csv
│   └── submit_example.csv
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
├── src/                          # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── utils.py
├── models/                       # Saved models
├── results/                      # Model outputs and predictions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Technologies
- Python 3.8+
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- NLTK/spaCy for text processing
- XGBoost/LightGBM (optional)

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Usage
1. **Data Exploration**: Start with `notebooks/01_exploratory_data_analysis.ipynb`
2. **Preprocessing**: Clean and prepare data
3. **Feature Engineering**: Create meaningful features
4. **Model Training**: Train and evaluate models
5. **Prediction**: Generate predictions for test data

## 📈 Methodology
1. **Data Analysis**: Understand patterns in fraudulent vs legitimate postings
2. **Data Cleaning**: Handle missing values, outliers
3. **Feature Engineering**: 
   - Text features (TF-IDF, word counts, sentiment)
   - Categorical encoding
   - New features (has_salary, description_length, etc.)
4. **Model Selection**: Try multiple algorithms
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost
   - Neural Networks (advanced)
5. **Evaluation**: Precision, Recall, F1-Score, AUC-ROC
6. **Handle Imbalance**: SMOTE, class weights, or ensemble methods

## 📊 Expected Deliverables
- [ ] Comprehensive EDA report
- [ ] Cleaned and processed dataset
- [ ] Trained classification models
- [ ] Model comparison analysis
- [ ] Final predictions on test set
- [ ] Documentation and presentation

## Contributers
Muhammad Arsalan
Qazi Naveed Ur Rehman
Mannan Aleem
Prashant Lamichhane

## 📅 Timeline
- Week 1: Data Exploration & Preprocessing
- Week 2: Feature Engineering
- Week 3: Model Training & Evaluation
- Week 4: Final Testing & Documentation

## 📝 Notes
- Fraudulent job postings are typically rare (imbalanced dataset)
- Focus on Precision and Recall balance
- Text features are likely very important
- Consider ensemble methods for better performance
