# 🎯 ML Engineer Interview: Пример решенной задачи

## 📋 Частая задача на собеседовании

**Ситуация:** Молодой DS написал ML пайплайн, но есть критическая ошибка - модель плохо работает на новых данных. Нужно найти и исправить.

---

## ❌ НЕПРАВИЛЬНЫЙ КОД (с ошибкой)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# ❌ ОШИБКА 1: Масштабирование ДО разделения
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fit на ВСЕХ данных!

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Обучение
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# ❌ ОШИБКА 2: Не сохранили модель
```

---

## ✅ ПРАВИЛЬНОЕ РЕШЕНИЕ

```python
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Загрузка данных
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# ✅ ИСПРАВЛЕНИЕ: Сначала разделяем данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ ИСПРАВЛЕНИЕ: Масштабирование ПОСЛЕ разделения
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit ТОЛЬКО на train
X_test_scaled = scaler.transform(X_test)        # transform на test

# ✅ ДОПОЛНЕНИЕ: Кодирование категориальных признаков
# (если нужно)
le = LabelEncoder()
if X_train.select_dtypes(include=['object']).shape[1] > 0:
    X_train_scaled = ...  # encoding логика

# Обучение
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'  # ✅ ДОПОЛНЕНИЕ: для дисбаланса
)
model.fit(X_train_scaled, y_train)

# ✅ ИСПРАВЛЕНИЕ: Кросс-валидация для надежной оценки
cv_scores = cross_val_score(
    model, X_train_scaled, y_train, cv=5, scoring='accuracy'
)
print(f"CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Оценка на тесте
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# ✅ ИСПРАВЛЕНИЕ: Сохранение модели для продакшена
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

---

## 🔑 Ключевые ошибки

| # | Ошибка | Последствие | Решение |
|---|--------|-------------|---------|
| 1 | **Data Leakage** - масштабирование до разделения | Завышенные метрики, плохая работа в продакшене | Сначала split, потом fit() |
| 2 | Неправильная обработка категорий | Модель не учится | Label Encoding / One-Hot |
| 3 | Игнорирование дисбаланса классов | Модель предсказывает только мажоритарный класс | class_weight, SMOTE |
| 4 | Оценка только по accuracy | Не видно где модель слабая | Precision, Recall, F1, ROC-AUC |
| 5 | Нет кросс-валидации | Переобучение не замечено | cross_val_score |
| 6 | Не сохранили модель | Модель нельзя использовать | joblib.dump() |
| 7 | Нет feature selection | Лишние признаки мешают | Анализ важности |
| 8 | Неправильная работа с NaN | Модель падает | fillna(), SimpleImputer |

---

## 🎤 Как отвечать на собеседовании

### Вопрос: "Найдите ошибку в этом коде"

**Ответ:**
> "Вижу критическую проблему - это Data Leakage. Масштабирование происходит до разделения на train/test. Из-за этого модель видит статистику тестовых данных во время обучения, что приводит к завышенным метрикам и плохой работе на новых данных.
>
> Правильный порядок:
> 1. Сначала split на train/test
> 2. fit_transform() только на train
> 3. transform() на test
>
> Также добавлю кросс-валидацию для более надежной оценки и сохраню модель для продакшена."

---

## 📊 Дополнительные улучшения

```python
# ✅ Feature Engineering
X['feature1_x_feature2'] = X['feature1'] * X['feature2']
X['feature1_squared'] = X['feature1'] ** 2

# ✅ Обработка пропусков
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ✅ Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# ✅ Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# ✅ Production-ready: Логирование и мониторинг
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Model trained with accuracy: {accuracy}")
```

---

## 🎯 Чеклист для проверки ML пайплайна

- [ ] Нет Data Leakage (правильный порядок операций)
- [ ] Категориальные признаки закодированы
- [ ] Пропуски обработаны корректно
- [ ] Используются веса классов / handling дисбаланса
- [ ] Кросс-валидация для оценки
- [ ] Правильные метрики для задачи
- [ ] Модель сохранена (joblib/pickle)
- [ ] Есть feature selection/engineering
- [ ] Логирование важных шагов
- [ ] Код задокументирован
- [ ] Есть тесты (unit tests)
- [ ] Модель работает с новыми данными

---

## 💼 Для резюме

**Опыт решения:** Исправлял критические ошибки в ML пайплайнах (Data Leakage, preprocessing, evaluation)

**Навыки:** Scikit-learn, Pandas, NumPy, MLOps best practices, production-ready code

