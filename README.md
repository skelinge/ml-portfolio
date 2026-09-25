### 1. Credit Card Fraud Detection ⭐
**[GitHub](https://github.com/skelinge/credit-card-fraud-detection)** | LightGBM, Optuna, SHAP

Обнаружение мошеннических транзакций в условиях экстремального дисбаланса классов (0.172% fraud).

**Ключевые результаты:**
| Модель | PR-AUC ↑ | ROC-AUC | F1-score |
|--------|----------|---------|----------|
| Logistic Regression | ~0.75 | ~0.93 | ~0.60 |
| Random Forest | ~0.78 | ~0.96 | ~0.65 |
| **LightGBM + Optuna** | **~0.82** | **~0.97** | **~0.70** |

**Технические решения:**
- Борьба с дисбалансом 1:578: stratified split, `is_unbalance=True`, оптимизация по PR-AUC (не Accuracy)
- Предобработка: RobustScaler для `Amount` (устойчив к выбросам до $25K), конвертация `Time` → часы суток
- Оптимизация: Optuna (Bayesian TPE, 50 trials) + Early Stopping (patience=100)
- Интерпретируемость: SHAP TreeExplainer для объяснения предсказаний бизнесу

**Стек:** Python, LightGBM, Optuna, SHAP, Scikit-learn, Pandas, Matplotlib

### 2. Telco Customer Churn Prediction ⭐
**[GitHub](https://github.com/skelinge/churn-prediction)** | LightGBM, Optuna, SHAP
- Предсказание оттока клиентов (7K записей, дисбаланс 27%)
- PR-AUC 0.65 через Bayesian optimization (Optuna, 50 trials)
- SHAP выявил ключевые драйверы: Contract_Two_year, tenure, Fiber optic
- Feature engineering: ChargesPerMonth, IsNewCustomer
