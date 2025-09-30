# Diabetes Prediction

This repository contains an end-to-end machine learning project aimed at predicting the likelihood of diabetes based on user-provided health data. The project demonstrates the full machine learning pipeline from data gathering to model deployment .

## Project Overview

The goal of this project is to create a seamless process for predicting diabetes by building a machine learning model that analyzes various health parameters. The web application takes user input, processes the data through the model, and provides the prediction result on a new page.

## Project Objectives

The project follows these key steps:

1. **Data Gathering**: Collected relevant medical data from various sources, including public datasets.
2. **Descriptive Analysis**: Explored the dataset to understand the underlying patterns and trends.
3. **Data Visualizations**: Created insightful visualizations to represent key relationships in the data.
4. **Data Preprocessing**: Cleaned and transformed the data for use in the machine learning model.
5. **Data Modelling**: Trained a machine learning model using scikit-learn to predict diabetes.
6. **Model Evaluation**: Assessed the model's performance using various metrics to ensure accuracy.
7. **Model Deployment**: Deployed the model as a web application using Flask, hosted on Heroku.

## Technical Aspects

### Machine Learning Model
- **Library**: scikit-learn
- **Algorithms Used**: Logistic Regression, Decision Trees, Random Forests 
- **Input Features**: The following fields are taken from the user:
  - Number of Pregnancies
  - Insulin Level
  - Age
  - Body Mass Index (BMI)
  - Blood Pressure
  - Glucose Level
  - Skin Thickness
  - Diabetes Pedigree Function
- **Output**: The model predicts whether the person is likely to have diabetes (Yes/No).

### Prerequisites
- Python 3.x
- scikit-learn
- Pandas

## Acknowledgments

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Heroku Documentation](https://devcenter.heroku.com/)

---
