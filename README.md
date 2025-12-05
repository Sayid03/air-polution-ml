**Air Quality Prediction Using Machine Learning
Machine Learning & Data Analytics Coursework — WIUT**  
📌 Project Overview

This project presents an end-to-end machine learning pipeline for analysing and predicting air pollutant concentrations using the UCI Air Quality Dataset.
It includes data exploration, preprocessing, training multiple machine learning models, evaluating their performance, and deploying an interactive Streamlit web application.

The work consists of:

- A detailed Jupyter Notebook (solution.ipynb) with full analysis
- A Streamlit app (app.py) demonstrating dataset exploration, preprocessing, model selection, and predictions
- A cleaned and structured project director
- Complete reproducibility using requirements.txt

The goal of the project is to build a reliable prediction model for pollutant concentrations based on sensor data, and to support it with clear analysis and visual exploration.

📂 Project Structure
project/
│
├── dataset/
│   ├── AirQualityUCI.csv
│   └── AirQualityUCI.xlsx
│
├── models/               # (optional: trained models can be saved here)
│
├── app.py                # Streamlit application
├── solution.ipynb        # Full ML analysis notebook
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT/Apache (as per coursework requirement)
├── .gitignore
└── README.md             # Project documentation

📊 Dataset Information

Source: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/dataset/360/air+quality

Description:
The dataset contains hourly measurements of various gas pollutants and sensor responses collected from an array of chemical sensors installed in an Italian city from March 2004 to April 2005.

The target pollutant used for prediction in this project is CO(GT), though the notebook allows extending the analysis to other pollutants.

🧠 Machine Learning Workflow

The entire workflow is implemented in solution.ipynb and includes:

1. Exploratory Data Analysis
- Dataset shape, structure, and variable types
- Summary statistics
- Correlation analysis
- Justified visualizations (histograms, boxplots, scatterplots)

2. Data Preparation
- Removal of invalid values (e.g., -200)
- Handling missing values using imputation
- Outlier treatment
- Feature scaling
- Feature engineering and cleaning
- Train/test split (80/20)

3. Model Training
Three models were trained and compared:
- Linear Regression
- Support Vector Regression (SVR)
- Random Forest Regression

4. Evaluation Metrics
Using the test dataset:
- MAE
- MSE
- RMSE
- R²
Random Forest achieved the highest predictive accuracy.

5. Deployment
- The app.py Streamlit application includes:
- Dataset preview and basic exploration
- Preprocessing pipeline
- Model training and inference interface
- Evaluation results displayed interactively


▶️ How to Run the Project
1. Clone the Repository
```git clone <your-repo-url>```
```cd <your-project-folder>```

2. Create and Activate Virtual Environment (Recommended)

_Windows_
```python -m venv venv```
```venv\Scripts\activate```


_macOS/Linux_
```python3 -m venv venv```
```source venv/bin/activate```

3. Install Dependencies
```pip install -r requirements.txt```

4. Run the Streamlit Application
streamlit run app.py


This will launch the app in your browser.

5. Open the Jupyter Notebook

If you want to reproduce the analysis:

jupyter notebook solution.ipynb

📦 Requirements

All required libraries are listed in requirements.txt and include:

pandas

numpy

scikit-learn

matplotlib

seaborn

streamlit

The project is fully reproducible and follows the coursework specification for environment setup.

📜 License

This project uses an open-source license (MIT/Apache) as required by the coursework.
See the LICENSE file for details.

🧑‍💻 Author

This project was completed as part of the Machine Learning & Data Analytics module at the Westminster International University in Tashkent (WIUT).

Student ID: [Your ID goes here]
Module Code: 6COSC017C-n
Lecturer: Mukhammadmuso Abduzhabbarov
