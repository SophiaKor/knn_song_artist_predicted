## Определение исполнителя по тексту песни

Программа выполняет задачу классификации, предсказывая имя исполнителя на основе таких признаков, как длина текста песни,
количество использованных в нем знаков препинания и год выхода песни.

Использованные классификаторы:
- KNeighborsClassifier
- DecisionTreeClassifier.

Настройка KNeighborsClassifier осуществляется при помощи Grid Search with Cross-Validation (GridSearchCV) на основе
заданных параметров, для DecisionTreeClassifier использованы параметры по умолчанию.

Оценить результаты предсказания позволяют получаемые в результате выполнения программы отчет о классификации
(classification_report) и матрица ошибок (confusion matrix).
