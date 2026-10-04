# AI-Driven Dynamic Tech Index (iFX Hack 2026)
An AI-powered portfolio risk manager built for the HKEX ATMXJ tech basket, designed to detect market contagion and dynamically trigger a cash-allocation circuit breaker.

### Core Architecture
* **AI Smoke Detector:** Random Forest model predicting short-term market volatility.
* **Contagion Graph:** NetworkX matrix mapping super-spreader risk across equities.
* **Deployment Roadmap:** Designed for Amazon S3 (Data Lake) and SageMaker (Model Retraining).

### Run Locally
This project uses `uv` for dependency management.
```bash
uv run streamlit run app.py