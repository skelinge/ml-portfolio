# -*- coding: utf-8 -*-
# =============================================================================
# ML ENGINEER ШПАРГАЛКА: Быстрое исправление типичных ошибок
# =============================================================================

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("⚡ ML ENGINEER: Быстрое исправление типичных ошибок")
print("="*70)

# =============================================================================
# ОШИБКА #1: Data Leakage в масштабировании
# =============================================================================
print("\n" + "="*70)
print("🔴 ОШИБКА #1: Data Leakage")
print("="*70)
print("""
❌ НЕПРАВИЛЬНО:
    X_scaled = scaler.fit_transform(X)  # fit на всех данных!
    X_train, X_test = train_test_split(X_scaled, ...)

✅ ПРАВИЛЬНО:
    X_train, X_test = train_test_split(X, ...)  # Сначала split
    X_train_scaled = scaler.fit_transform(X_train)  # fit на train
    X_test_scaled = scaler.transform(X_test)        # transform на test
""")

# =============================================================================
# ОШИБКА #2: Категориальные признаки
# =============================================================================
print("="*70)
print("🔴 ОШИБКА #2: Категории без encoding")
print("="*70)
print("""
❌ НЕПРАВИЛЬНО:
    model.fit(X)  # X содержит ['A', 'B', 'C'] - упадет!

✅ ПРАВИЛЬНО:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X['category_encoded'] = le.fit_transform(X['category'])
    # Или One-Hot Encoding:
    X = pd.get_dummies(X, columns=['category'])
""")

# =============================================================================
# ОШИБКА #3: Пропуски (NaN)
# =============================================================================
print("="*70)
print("🔴 ОШИБКА #3: Пропуски не обработаны")
print("="*70)
print("""
❌ НЕПРАВИЛЬНО:
    X.dropna()  # Удалили все строки - потеряли данные!

✅ ПРАВИЛЬНО:
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')  # или 'mean', 'most_frequent'
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
""")

# =============================================================================
# ОШИБКА #4: Дисбаланс классов
# =============================================================================
print("="*70)
print("🔴 ОШИБКА #4: Игнорирование дисбаланса")
print("="*70)
print("""
❌ НЕПРАВИЛЬНО:
    model = RandomForestClassifier()  # Игнорирует дисбаланс

✅ ПРАВИЛЬНО:
    model = RandomForestClassifier(class_weight='balanced')
    # Или вычислить веса:
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', 
                                         classes=np.unique(y_train),
                                         y=y_train)
    # Или SMOTE для oversampling минорного класса
""")

# =============================================================================
# ОШИБКА #5: Неправильная оценка
# =============================================================================
print("="*70)
print("🔴 ОШИБКА #5: Только accuracy")
print("="*70)
print("""
❌ НЕПРАВИЛЬНО:
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")  # Часто вводит в заблуждение!

✅ ПРАВИЛЬНО:
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score
    
    # Детальный отчет
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    # Кросс-валидация
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Правильные метрики
    # - Классификация: Precision, Recall, F1, ROC-AUC
    # - Регрессия: MSE, RMSE, MAE, R²
""")

# =============================================================================
# ОШИБКА #6: Нет сохранения модели
# =============================================================================
print("="*70)
print("🔴 ОШИБКА #6: Модель не сохранена")
print("="*70)
print("""
❌ НЕПРАВИЛЬНО:
    model.fit(X_train, y_train)
    # Модель потеряна после завершения скрипта!

✅ ПРАВИЛЬНО:
    import joblib
    
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    
    # Для продакшена:
    model = joblib.load('model.pkl')
    predictions = model.predict(scaler.transform(new_data))
""")

# =============================================================================
# ОШИБКА #7: Переобучение
# =============================================================================
print("="*70)
print("🔴 ОШИБКА #7: Переобучение")
print("="*70)
print("""
❌ НЕПРАВИЛЬНО:
    model = RandomForestClassifier(n_estimators=1000, 
                                   max_depth=None)  # Запоминает данные!

✅ ПРАВИЛЬНО:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,              # Ограничение глубины
        min_samples_split=20,      # Минимум образцов для split
        min_samples_leaf=10,       # Минимум в листе
        max_features='sqrt'        # Подвыборка признаков
    )
    
    # Или добавить регуляризацию для других моделей
    # LogisticRegression: C=0.1 (меньше C = больше регуляризация)
    # XGBoost: reg_alpha, reg_lambda
""")

# =============================================================================
# ОШИБКА #8: Нет Feature Selection
# =============================================================================
print("="*70)
print("🔴 ОШИБКА #8: Все признаки в модели")
print("="*70)
print("""
❌ НЕПРАВИЛЬНО:
    model.fit(X)  # 1000 признаков, большинство шум!

✅ ПРАВИЛЬНО:
    # Анализ важности
    model.fit(X_train, y_train)
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Выбрать топ признаки
    top_features = importances.head(20)['feature'].tolist()
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]
    
    # Или автоматически
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=20)
    X_train_selected = selector.fit_transform(X_train, y_train)
""")

# =============================================================================
# ПРАВИЛЬНЫЙ ПАЙПЛАЙН (Шаблон)
# =============================================================================
print("="*70)
print("✅ ШАБЛОН ПРАВИЛЬНОГО ПАЙПЛАЙНА")
print("="*70)
print("""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Загрузка данных
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# 2. ⚠️ СНАЧАЛА РАЗДЕЛЕНИЕ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Обработка пропусков
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 4. Кодирование категорий
# (если нужно, делаем до imputation)
le = LabelEncoder()
X_train_cat = le.fit_transform(X_train['category'])
X_test_cat = le.transform(X_test['category'])

# 5. Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 6. Обучение с правильными параметрами
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 7. Кросс-валидация
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 8. Оценка
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# 9. Сохранение
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(le, 'encoder.pkl')

print("✅ Модель готова к продакшену!")
""")

# =============================================================================
# ТОП-10 ЧЕКЛИСТ ПЕРЕД ПРОДАКШЕНОМ
# =============================================================================
print("\n" + "="*70)
print("📋 ЧЕКЛИСТ: 10 вопросов перед продакшеном")
print("="*70)

checklist = [
    "1. Нет Data Leakage? (split → fit → transform)",
    "2. Все NaN обработаны? (imputation/fill)",
    "3. Категории закодированы? (Label/One-Hot)",
    "4. Дисбаланс учтен? (class_weight/SMOTE)",
    "5. Выбрана правильная метрика? (не только accuracy)",
    "6. Есть кросс-валидация? (5-fold CV)",
    "7. Модель не переобучена? (train/test близки)",
    "8. Важные признаки проанализированы? (feature importance)",
    "9. Модель сохранена? (joblib/pickle)",
    "10. Код задокументирован? (комментарии, docstrings)"
]

for item in checklist:
    print(f"  ☐ {item}")

print("\n" + "="*70)
print("💡 Если все ☑️ - модель готова к продакшену!")
print("="*70)

