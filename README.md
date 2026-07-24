# Project Context: gram-event-quant

**Project Name**: gram_quant  
**Repository**: gram-event-quant  
**Branch**: feature/event-window-slicer (або main)

## Мета проєкту
Дослідницька платформа для аналізу впливу подій GRAM/TON (крипто / blockchain events) на різні метрики.

## Технологічний стек (фіксований)
- **Python 3.11+**
- **DuckDB** — основний OLAP / аналітичний движок
- **Parquet** — data lake / зберігання сирих даних
- **Polars** — основна бібліотека для DataFrame операцій (швидше і легше pandas)
- **SQLModel / Pydantic** — для схем і валідації
- **uv** — менеджер залежностей

## Архітектурні принципи (дотримуватися строго)
- Чітке розділення шарів: ingestion → storage → processing → analytics
- DuckDB як центральне сховище (не SQLite, не PostgreSQL)
- Максимальне використання SQL через DuckDB для аналітики
- Polars для трансформацій даних
- Мінімалізм: без зайвих абстракцій, якщо можна вирішити через DuckDB + Polars

## Поточний стан (на момент контексту)
- Реалізовано DuckDBStore з підтримкою context manager
- Реєстрація Parquet файлів як views у DuckDB
- Базова інфраструктура ingestion
- Частково написані unit-тести

## Що НЕ робити
- Не вводити нові фреймворки (SQLAlchemy, Django тощо)
- Не переходити на pandas, якщо можна Polars
- Не створювати складні ORM без обговорення
- Не рефакторити працюючі модулі без явної потреби
- Не вигадувати файли, яких немає в PROJECT_FILES.txt

## Поточне завдання
Довести до ладу storage-шар, завершити unit-тести для DuckDBStore, виправити/дописати event window slicer.

---

**Інструкція для AI**: 
Спочатку прочитай PROJECT_TREE.txt, PROJECT_FILES.txt, GEMINI_CONTEXT.md, pyproject.toml та GIT_HISTORY.txt. 
Тільки після цього давай аналіз поточного стану. Не пропонуй зміни коду, поки я не скажу.
