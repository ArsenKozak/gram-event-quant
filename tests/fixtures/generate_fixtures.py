import polars as pl

# Створюємо тестовий датасет із різними сценаріями (валідні та невалідні)
data = {
    "time": ["2024-05-10T15:30:00", "2024-05-11T18:00:00", "invalid-date"],
    "author": ["Pavel Durov", "Pavel Durov", "Unknown"],
    "text": ["GRAM is booming!", "TON ecosystem news", "Broken post"],
    "importance": [5, 3, 10],  # 10 - невалідна важливість (повинна впасти у Pydantic)
    "price_before": [0.015, 0.020, -0.05],  # -0.05 - невалідна ціна
    "btc_before": [62000.0, 63000.0, 0.0],
}

df = pl.DataFrame(data)
df.write_excel("tests/fixtures/sample_events.xlsx")
print("Fixture sample_events.xlsx generated successfully.")
