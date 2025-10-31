# -*- coding: utf-8 -*-
# =============================================================================
# ML ENGINEER TASK: Исправление ошибок в пайплайне машинного обучения
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ML ENGINEER: Исправление ошибок в ML пайплайне")
print("="*70)

# =============================================================================
# ЗАДАЧА: Создан ML пайплайн, но в нем есть критические ошибки
# Нужно найти и исправить все ошибки
# =============================================================================

print("\n📊 Этап 1: Загрузка данных\n")

# Симуляция загрузки данных
np.random.seed(42)
n_samples = 1000

# Создаем синтетический датасет
data = {
    'feature_1': np.random.normal(50, 15, n_samples),
    'feature_2': np.random.normal(100, 25, n_samples),
    'feature_3': np.random.randint(0, 10, n_samples),
    'feature_4': np.random.choice(['A', 'B', 'C'], n_samples),
    'target': np.random.choice([0, 1], n_samples)
}

df = pd.DataFrame(data)
print(f"Датасет загружен: {df.shape[0]} строк, {df.shape[1]-1} признаков")
print(f"\nПервые строки:\n{df.head()}")

# =============================================================================
# ОШИБКА 1: Неправильная обработка категориальных признаков
# =============================================================================
print("\n" + "="*70)
print("❌ ОШИБКА 1: Неправильная обработка категориальных признаков")
print("="*70)

print("\n🔍 Проблема:")
print("- Признак 'feature_4' содержит категории ('A', 'B', 'C')")
print("- Модели работают только с численными данными")
print("- Нужно применить encoding (например, Label Encoding или One-Hot)")

from sklearn.preprocessing import LabelEncoder

# Исправляем: кодируем категориальные признаки
le = LabelEncoder()
df['feature_4_encoded'] = le.fit_transform(df['feature_4'])

print("\n✅ Исправлено: Применен Label Encoding")
print(f"Маппинг категорий: {dict(zip(le.classes_, range(len(le.classes_))))}")

# =============================================================================
# ОШИББКА 2: Утечка данных (Data Leakage)
# =============================================================================
print("\n" + "="*70)
print("❌ ОШИБКА 2: Утечка данных - масштабирование до разделения")
print("="*70)

print("\n🔍 Проблема:")
print("- Многие делают StandardScaler.fit() НА ВСЕХ данных")
print("- Это позволяет модели 'увидеть' тестовые данные при обучении")
print("- Результат: завышенные метрики на тесте, плохая работа на новых данных")

# Правильный порядок:
# 1. Разделить данные
# 2. fit() только на train
# 3. transform() на train и test

# Выбираем признаки для модели
features = ['feature_1', 'feature_2', 'feature_3', 'feature_4_encoded']
X = df[features]
y = df['target']

# ✅ ИСПРАВЛЕНО: Сначала разделяем данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n✅ Исправлено: Данные разделены правильно")
print(f"Train: {X_train.shape[0]} строк")
print(f"Test: {X_test.shape[0]} строк")

# Теперь масштабируем
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit ТОЛЬКО на train
X_test_scaled = scaler.transform(X_test)        # transform на test

print("\n✅ Исправлено: Масштабирование применено корректно")
print("fit() на train, transform() на train и test")

# =============================================================================
# ОШИБКА 3: Отсутствие баланса классов
# =============================================================================
print("\n" + "="*70)
print("❌ ОШИБКА 3: Игнорирование дисбаланса классов")
print("="*70)

print("\n🔍 Проблема:")
print(f"- Распределение классов в train:\n{y_train.value_counts().sort_index()}")
print(f"- Распределение классов в test:\n{y_test.value_counts().sort_index()}")

# Если классы сильно несбалансированы, модель может предсказывать только
# мажоритарный класс и иметь высокую accuracy, но плохо работать на минорных

# Решения: class_weight, SMOTE, undersampling, изменение метрики
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print(f"\n✅ Исправлено: Вычислены веса классов")
print(f"Веса: {class_weight_dict}")

# =============================================================================
# ОШИБКА 4: Неправильная оценка модели
# =============================================================================
print("\n" + "="*70)
print("❌ ОШИБКА 4: Неправильная оценка модели")
print("="*70)

print("\n🔍 Проблема:")
print("- Часто оценивают модель только на accuracy")
print("- Нужно использовать кросс-валидацию")
print("- Нужно смотреть на несколько метрик")

# Обучаем модель с учетом весов классов
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight=class_weight_dict  # Добавили веса
)
model.fit(X_train_scaled, y_train)

# Предсказания
y_pred = model.predict(X_test_scaled)

# Метрики
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Исправлено: Добавлены веса классов в модель")
print(f"\n📈 Метрики на тестовой выборке:")
print(f"Accuracy: {accuracy:.4f}")

# Детальный отчет
print("\n" + "="*50)
print("Детальный отчет о классификации:")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Класс 0', 'Класс 1']))

# Кросс-валидация для более надежной оценки
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model, 
    X_train_scaled, 
    y_train, 
    cv=5,
    scoring='accuracy'
)

print(f"\n✅ Кросс-валидация (5 folds):")
print(f"Средняя точность: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# =============================================================================
# ОШИБКА 5: Игнорирование важности признаков
# =============================================================================
print("\n" + "="*70)
print("❌ ОШИБКА 5: Не анализируют важность признаков")
print("="*70)

print("\n🔍 Проблема:")
print("- Модель обучена, но не понятно что важно")
print("- Нет feature selection/engineering")

# Получаем важность признаков
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n✅ Исправлено: Проанализирована важность признаков")
print("\nТоп признаков:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Можно удалить неважные признаки для улучшения модели

# =============================================================================
# ОШИБКА 6: Отсутствие сохранения модели
# =============================================================================
print("\n" + "="*70)
print("❌ ОШИБКА 6: Не сохраняют модель для продакшена")
print("="*70)

import joblib

# Сохраняем модель и скейлер
model_filename = 'trained_model.pkl'
scaler_filename = 'scaler.pkl'

joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)
joblib.dump(le, 'label_encoder.pkl')

print("\n✅ Исправлено: Модель сохранена")
print(f"  - Модель: {model_filename}")
print(f"  - Scaler: {scaler_filename}")
print(f"  - Label Encoder: label_encoder.pkl")

# Пример загрузки (для продакшена)
print("\n📦 Пример загрузки модели для продакшена:")
print("```python")
print("import joblib")
print("model = joblib.load('trained_model.pkl')")
print("scaler = joblib.load('scaler.pkl')")
print("predictions = model.predict(scaler.transform(new_data))")
print("```")

# =============================================================================
# ИТОГОВЫЙ РЕЗЮМЕ
# =============================================================================
print("\n" + "="*70)
print("📝 РЕЗЮМЕ: Ключевые ошибки, которые нужно избегать")
print("="*70)

mistakes = [
    "1. Обработка категориальных признаков без encoding",
    "2. Data Leakage (масштабирование до разделения train/test)",
    "3. Игнорирование дисбаланса классов",
    "4. Оценка только по accuracy без кросс-валидации",
    "5. Не анализ важности признаков",
    "6. Отсутствие сохранения модели для продакшена",
    "7. Неправильная работа с пропусками (NaN)",
    "8. Переобучение (нет регуляризации)",
    "9. Неправильный выбор метрик для задачи",
    "10. Отсутствие мониторинга модели в продакшене"
]

for mistake in mistakes:
    print(f"  ❌ {mistake}")

print("\n" + "="*70)
print("✅ Все ошибки исправлены! Пайплайн готов к продакшену.")
print("="*70)

print("\n🎯 Следующие шаги:")
print("  1. Добавить логирование")
print("  2. Настроить CI/CD пайплайн")
print("  3. Добавить мониторинг модели")
print("  4. A/B тестирование")
print("  5. Постепенное внедрение")

