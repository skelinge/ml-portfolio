### 2. Telco Customer Churn Prediction ⭐
**[GitHub](https://github.com/skelinge/churn-prediction)** | LightGBM, Optuna, SHAP
- Предсказание оттока клиентов (7K записей, дисбаланс 27%)
- PR-AUC 0.65 через Bayesian optimization (Optuna, 50 trials)
- SHAP выявил ключевые драйверы: Contract_Two_year, tenure, Fiber optic
- Feature engineering: ChargesPerMonth, IsNewCustomer
