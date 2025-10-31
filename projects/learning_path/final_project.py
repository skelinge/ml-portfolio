# -*- coding: utf-8 -*-
import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ФИНАЛЬНЫЙ ПРОЕКТ
# Анализ датасета и построение модели для предсказания
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve

# Настройка
np.random.seed(42)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

print("="*70)
print("ФИНАЛЬНЫЙ ПРОЕКТ: Анализ данных клиентов интернет-магазина")
print("="*70)

# =========================
# 1. СОЗДАНИЕ ДАТАСЕТА
# =========================
print("\n📍 Этап 1: Создание синтетического датасета...")

n_samples = 1000
np.random.seed(42)

data = {
    'age': np.random.randint(18, 70, n_samples),
    'annual_income': np.random.normal(50000, 15000, n_samples),
    'spending_score': np.random.randint(1, 100, n_samples),
    'num_orders': np.random.poisson(10, n_samples),
    'avg_order_value': np.random.normal(150, 50, n_samples),
    'days_since_last_order': np.random.exponential(30, n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples),
    'city': np.random.choice(['Moscow', 'SPB', 'Kazan', 'Other'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
}

df = pd.DataFrame(data)

# Создаем целевую переменную (churn - уход клиента)
# Клиент уходит если давно не заказывал И низкая оценка трат
df['churn'] = (
    (df['days_since_last_order'] > 45) & 
    (df['spending_score'] < 30)
).astype(int)

# Добавим немного случайности
churn_prob = 0.05
np.random.seed(42)
df.loc[np.random.random(n_samples) < churn_prob, 'churn'] = 1

print(f"Датасет создан: {df.shape[0]} строк, {df.shape[1]-1} признаков")
print(f"Целевой класс: churn (уход клиента)")
print(f"\nРаспределение классов:")
print(df['churn'].value_counts())
print(f"Доля ушедших клиентов: {(df['churn'] == 1).mean()*100:.1f}%")

# =========================
# 2. EXPLORATORY DATA ANALYSIS
# =========================
print("\n📍 Этап 2: Исследовательский анализ данных...")

print("\nПервые строки:")
print(df.head())

print("\nБазовая статистика:")
print(df.describe())

print("\nПропущенные значения:")
print(df.isnull().sum())

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Гистограммы
df['age'].hist(bins=20, ax=axes[0, 0], edgecolor='black')
axes[0, 0].set_title('Распределение возраста')
axes[0, 0].set_xlabel('Возраст')

df['annual_income'].hist(bins=30, ax=axes[0, 1], edgecolor='black', color='skyblue')
axes[0, 1].set_title('Распределение дохода')
axes[0, 1].set_xlabel('Доход (руб)')

df['spending_score'].hist(bins=20, ax=axes[0, 2], edgecolor='black', color='lightgreen')
axes[0, 2].set_title('Оценка трат')
axes[0, 2].set_xlabel('Оценка')

# Boxplots
df.boxplot(column='spending_score', by='churn', ax=axes[1, 0])
axes[1, 0].set_title('Оценка трат vs Churn')
axes[1, 0].set_xlabel('Churn')

df.boxplot(column='days_since_last_order', by='churn', ax=axes[1, 1])
axes[1, 1].set_title('Дней с последнего заказа vs Churn')
axes[1, 1].set_xlabel('Churn')

# Корреляционная матрица
correlation_cols = ['age', 'annual_income', 'spending_score', 'num_orders', 'avg_order_value', 'days_since_last_order']
corr_matrix = df[correlation_cols + ['churn']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1, 2], square=True)
axes[1, 2].set_title('Корреляционная матрица')

plt.tight_layout()
plt.savefig('final_project_eda.png', dpi=150)
print("✅ EDA графики сохранены: final_project_eda.png")
plt.show()

# =========================
# 3. FEATURE ENGINEERING
# =========================
print("\n📍 Этап 3: Создание новых признаков...")

# Копируем датафрейм
df_processed = df.copy()

# Интерактивные признаки
df_processed['income_per_order'] = df_processed['annual_income'] / (df_processed['num_orders'] + 1)
df_processed['value_per_day'] = df_processed['avg_order_value'] / (df_processed['days_since_last_order'] + 1)
df_processed['total_spent'] = df_processed['num_orders'] * df_processed['avg_order_value']

# Биннинг
df_processed['age_group'] = pd.cut(df_processed['age'], bins=[0, 30, 45, 60, 100], 
                                    labels=['Young', 'Adult', 'Middle', 'Senior'])
df_processed['income_group'] = pd.cut(df_processed['annual_income'], bins=3, 
                                       labels=['Low', 'Medium', 'High'])

# Encoding
le_gender = LabelEncoder()
le_city = LabelEncoder()
le_age = LabelEncoder()
le_income = LabelEncoder()

df_processed['gender_encoded'] = le_gender.fit_transform(df_processed['gender'])
df_processed['city_encoded'] = le_city.fit_transform(df_processed['city'])
df_processed['age_group_encoded'] = le_age.fit_transform(df_processed['age_group'])
df_processed['income_group_encoded'] = le_income.fit_transform(df_processed['income_group'])

print("Новые признаки созданы")
print(f"Всего признаков: {df_processed.shape[1]}")

# =========================
# 4. ПОДГОТОВКА ДАННЫХ
# =========================
print("\n📍 Этап 4: Подготовка данных для модели...")

# Выбираем численные признаки
feature_cols = [
    'age', 'annual_income', 'spending_score', 'num_orders', 
    'avg_order_value', 'days_since_last_order',
    'gender_encoded', 'city_encoded', 'age_group_encoded', 'income_group_encoded',
    'income_per_order', 'value_per_day', 'total_spent'
]

X = df_processed[feature_cols]
y = df_processed['churn']

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 5. ОБУЧЕНИЕ МОДЕЛИ
# =========================
print("\n📍 Этап 5: Обучение модели...")

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ Модель обучена")

# =========================
# 6. ОЦЕНКА МОДЕЛИ
# =========================
print("\n📍 Этап 6: Оценка качества модели...")

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nТочность (Accuracy): {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred, target_names=['Не ушел', 'Ушел']))

print("\nМатрица ошибок:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Важность признаков
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nТоп-5 самых важных признаков:")
print(feature_importance.head(5))

# =========================
# 7. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_xlabel('Предсказанный')
axes[0].set_ylabel('Реальный')
axes[0].set_title('Матрица ошибок')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', label='Случайный')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC кривая')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Feature Importance
top_features = feature_importance.head(10)
axes[2].barh(range(len(top_features)), top_features['importance'], color='coral')
axes[2].set_yticks(range(len(top_features)))
axes[2].set_yticklabels(top_features['feature'])
axes[2].set_xlabel('Важность')
axes[2].set_title('Важность признаков')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('final_project_results.png', dpi=150)
print("\n✅ Графики результатов сохранены: final_project_results.png")
plt.show()

print("\n" + "="*70)
print("🎉 ПРОЕКТ ЗАВЕРШЕН!")
print("="*70)
print("\nВы успешно прошли весь путь от базового Python до полноценного проекта Data Science!")
print("\n📈 Следующие шаги:")
print("1. Попробуйте другие алгоритмы (XGBoost, LightGBM)")
print("2. Настройте гиперпараметры")
print("3. Поработайте с реальными данными (Kaggle, UCI ML Repository)")
print("4. Изучите Deep Learning для более сложных задач")
print("\nУдачи в Data Science! 🚀")

