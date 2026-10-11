# Shiv Patel — Projects

## NBAPredictor

NBAPredictor is a machine learning system that predicts NBA All-Star selections using a dataset of 5,386 player-seasons spanning 2015–2025. The core classifier achieves an AUC-ROC of 0.9933, indicating extremely strong separation between All-Star and non-All-Star seasons.

Beyond raw prediction, the project includes SHAP explainability so every prediction can be broken down feature-by-feature (e.g., points per game, efficiency, team performance) to show *why* a player was flagged as an All-Star candidate. It also includes a player recommendation engine and a "compare" tool built with K-Means clustering and cosine similarity to find statistically similar players, plus a "Guess Who" game built on top of the same player-similarity data. The whole system is deployed as a full-stack Flask web application.

**Tech stack:** Python, XGBoost, K-Means clustering, SHAP, Flask, cosine similarity.

## F1 Race Outcome Prediction

F1 Race Outcome Prediction is a hybrid machine learning pipeline built on 26,760 Formula 1 race entries from 2008–2024. It predicts podium finishes with an AUC-ROC of 0.930, validated using 5-fold cross-validation for robustness.

The pipeline combines XGBoost for the predictive model with K-Means clustering to group drivers into performance-based archetypes (e.g., consistent points-scorers vs. high-variance overtakers), and uses SHAP to explain which features (grid position, team performance, driver form, etc.) drive each prediction.

**Tech stack:** Python, XGBoost, K-Means, SHAP, Pandas, scikit-learn.

## ResumeIQ

ResumeIQ is an AI-powered resume screener that scores a resume against a job description using NLP techniques. It's built with a Flask backend, spaCy for natural language processing, SQLite for storage, and a JavaScript front end.

Key features include ATS (Applicant Tracking System) simulation to show how a resume would score against automated screeners, a keyword heatmap highlighting matched/missing job-description keywords, gap analysis, an AI cover letter generator, an interview question predictor based on the job description and resume content, and a resume version tracker to compare edits over time.

**Tech stack:** Python, Flask, spaCy, SQLite, NLP, JavaScript.

## FIFA-VAR AI Failure Audit

FIFA-VAR AI Failure Audit is an AI safety research project analyzing 14 VAR (Video Assistant Referee) incidents from the 2026 FIFA World Cup. It applies established AI governance frameworks — the NIST AI Risk Management Framework (AI RMF) and CISA guidelines — to formally audit where and how AI-assisted officiating decisions failed or introduced risk.

The project includes a statistical z-score bias screening analysis to detect potential systematic bias in VAR decisions, plus a full written research report documenting findings, methodology, and framework-aligned recommendations.

**Tech stack:** Python, Pandas, XGBoost, NIST AI RMF, CISA, Jupyter.

## StockSentiment

StockSentiment scrapes financial news headlines and scores their sentiment using two independent NLP engines (VADER and TextBlob), then correlates that sentiment with subsequent stock price movements. An XGBoost model uses the combined sentiment signal to predict price direction.

The project includes a backtesting dashboard to evaluate strategy performance over historical data, and visualizations comparing sentiment trends against actual price movements over time. The web interface is built with Flask.

**Tech stack:** Python, Flask, VADER, TextBlob, XGBoost, Pandas.

## HealthPredictor

HealthPredictor predicts hospital readmission risk using the UCI "Diabetes 130-US Hospitals" dataset, covering 101,766 patient encounters. The project starts with rigorous data profiling and data quality checks, followed by SQL-based exploratory analysis to understand readmission patterns.

The predictive model is a calibrated XGBoost classifier (calibrated so its predicted probabilities are meaningful, not just its rankings), served through a Flask web interface. Data is stored and queried via SQLite.

**Tech stack:** Python, SQL, XGBoost, Pandas, SQLite, Flask.

## RAGPortfolio

RAGPortfolio is this very project: a Retrieval Augmented Generation (RAG) chatbot that answers questions about Shiv's projects, resume, and experience. It indexes his project documentation and resume into a ChromaDB vector database using local sentence-transformers embeddings, retrieves the most relevant chunks for a given question, and generates grounded answers using a Llama 3 model served through Groq.

**Tech stack:** Python, LangChain, ChromaDB, Groq, sentence-transformers, Flask.
