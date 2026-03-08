# Oil Production Forecasting with Machine Learning

## Problem
Predict crude oil production trends using historical OECD data.

##Problem Definition

Crude oil production forecasting is important for energy planning, operational optimization, and policy analysis. Governments, energy companies, and planners rely on production forecasts to anticipate supply trends and make informed decisions about infrastructure, investment, and resource management.

This project aims to build a machine learning model that predicts crude oil production trends using historical OECD production data. The goal is to evaluate whether machine learning models can capture patterns in production changes across countries and time.

The primary users of this analysis are energy analysts, engineers, and strategic planners who need data-driven insights into production behavior.

The project will deliver production forecasts, model evaluation metrics (RMSE and R²), and visual analysis of key factors influencing production trends. The final results will also discuss model limitations and uncertainties, ensuring the predictions are interpreted responsibly.

## Dataset
OECD crude oil production dataset from Kaggle.

## Approach
1. Data exploration
2. Feature engineering
3. Train regression models
4. Evaluate performance

## Models Tested
- Linear Regression
- Random Forest
- Gradient Boosting

## How to Run
pip install -r requirements.txt
python src/train.py
