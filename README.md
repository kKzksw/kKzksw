# Project Title
**Bank Credit Scoring Prediction**

## Project Introduction
This project aims to predict customer credit scores using a machine learning model (CatBoost). It includes data preprocessing, model training, hyperparameter tuning, and an intuitive demonstration interface.

## Features
- **Data Preprocessing:** Data cleaning, feature engineering, and transformations.
- **Model Training:** Training and validation using the CatBoost classifier.
- **Result Visualization:** Clear presentation of prediction results and customer features.
- **User Interface:** A user-friendly application built with Streamlit.

## File Structure
- **README.md:** Project overview and instructions.
- **data_preprocessing.ipynb:** Data preprocessing code.
- **machinelearning.ipynb:** Machine learning model training code.
- **demo.py:** Streamlit application for demonstrating credit score predictions.
- **dataset/**: Dataset files, including raw and preprocessed data.
- **eclyon/**: Helper modules for data transformations and visualizations.
- **catboost_model.pk:** Trained CatBoost model file.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/kKzksw/kKzksw.git
```bash
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the demo application:
```bash
streamlit run demo.py
```
## Usage

1. Open the application.
2. Select a customer index to view credit score predictions.
3. Review customer details and the model’s prediction probabilities.

## Contributing

Suggestions and pull requests to improve the project are highly welcome!

## License

This project is licensed under the MIT License.
