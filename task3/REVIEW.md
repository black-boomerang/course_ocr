# Описание решения

Задача разбивается на 2 этапа. На первом происходит сегментация штрих-кода 
HRNet. После этого вокруг карты сегментации описывается прямоугольник
минимального размера. Он и является результатом детекции. После этого к
вырезанному изображению шрих-кода применяется алгоритм, похожий на
[используемый](https://opencv.org/recognizing-one-dimensional-barcode-using-opencv/)
в OpenCV. Для каждой строки

Подготовка данных происходит в ноутбуке `prepare_data.ipynb` (для
воспроизведения нужно добавить в папку с тестовыми данными `markup.csv`),
обучение и использование модели - в `solution.ipynb`.
