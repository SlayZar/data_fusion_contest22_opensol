# data_fusion_contest22_opensol
Открытое решение для [Data Fusion Contest 2022](https://ods.ai/tracks/data-fusion-2022-competitions)

* Результаты на публичном лидерборде 

| Task | R1 | MRR@100 | Precision@100 |
| :---: | :---: | :---: | :---: |
| 1 | 0.234219 | 0.179605 | 0.336561 |
| 2 | 0.012731 | 0.006671 | 0.138875 |

Решение для отправки в первую задачу - папка [catboost_baseline_subm](catboost_baseline_subm)

Решение для отправки во вторую задачу - файл [puzzle_solution.csv](puzzle_solution.csv)

Время от чтения данных до обученной модели по первой задаче - менее 6 минут.

Инференс в первой задаче - 8 минут

Инференс во второй задаче - 23 минуты

### Краткое описание:

Решение основано на подходе из бейзлайна, однако помимо эмбеддингов категорий считаются эмбеддинги по часам транзакций и кликстримов.

После этого обучается простая модель катбуст на всех данных на дефолтных параметрах.

Данный простой подход позволяет за 6 минут считать данные и получить обученную модель, которая на текущий момент позволяет попасть в топ-10 по задаче Matching.
