# Oil Production Forecasting & Natural Language Q&A

A capstone project for the **Skills for Hire Atlantic — Advanced Data + AI Program**.

This project combines machine learning forecasting with a Generative AI natural language interface to analyze and query global oil production data covering 139 countries from 1971 to 2017.

---

## Project Overview

| | |
|---|---|
| **Dataset** | OECD Oil Production (139 countries, 1971–2017) |
| **Best Model** | Random Forest Regressor (R² = 0.987) |
| **GenAI Layer** | Google Gemini 2.5 Flash — Prompt Engineering |
| **Tools** | Python, pandas, scikit-learn, matplotlib, seaborn |

---

## Repository Structure

```
├── data/
│   ├── raw/                  # Original OECD dataset
│   └── processed/            # Cleaned dataset with lag features
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory data analysis & feature engineering
│   ├── 02_ML.ipynb           # Model training, evaluation & comparison
│   └── 03_GenAI.ipynb        # Gemini-powered natural language Q&A
├── output/                   # Saved charts and visualisations
├── .env.example              # Template for environment variables
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn google-genai python-dotenv
```

### 3. Set up your API key

Copy the example environment file and add your Gemini API key:

```bash
cp .env.example .env
```

Then open `.env` and replace the placeholder:

```
GEMINI_API_KEY=your-actual-key-here
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com).

---

## Running the Project

Run the notebooks in order:

| Step | Notebook | Description |
|---|---|---|
| 1 | `01_EDA.ipynb` | Load, clean, and explore the data. Creates lag features and saves processed CSV. |
| 2 | `02_ML.ipynb` | Train and compare forecasting models. Evaluates on held-out test data (2014–2017). |
| 3 | `03_GenAI.ipynb` | Run the natural language Q&A interface powered by Google Gemini. |

### Interactive Q&A (Terminal)

For a better interactive experience, run the Q&A tool from the terminal:

```bash
cd notebooks
python qa_tool.py
```

Type any question about the oil production data and receive a grounded, data-backed answer. Type `quit` to exit.

---

## Key Results

### Model Comparison (Test Set: 2014–2017)

| Model | R² | MAE (KTOE) | RMSE (KTOE) |
|---|---|---|---|
| Linear Regression | 0.9959 | 1,609 | 5,228 |
| Random Forest | 0.9867 | 2,292 | 9,398 |
| Gradient Boosting | 0.9705 | 2,806 | 14,001 |

> Lag features (previous 1–3 years of production) are the dominant predictors. Without them, Linear Regression R² collapses from 0.996 to 0.006.

### GenAI Guardrail

The Q&A system includes a prompt guardrail that prevents hallucination. When asked a question with a false premise:

> *"Why did Angola's production drop in 1990?"*

The model correctly responds:

> *"Angola's production did not drop in 1990. It increased from 22,765 to 23,827 thousand barrels/day. The data does not contain information explaining reasons for production changes."*

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key |

Never commit your `.env` file. It is listed in `.gitignore`.

---

## Limitations

- Dataset covers 1971–2017 only — no recent data
- Forecasting beyond one year ahead requires iterative prediction (using predicted values as lag inputs), which introduces compounding error
- The GenAI Q&A layer answers based on summarised statistics, not the full row-level dataset

---

## License

This project was created for educational purposes as part of the Skills for Hire Atlantic Advanced Data + AI Program.
