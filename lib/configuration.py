import os

GOLD_ENCODING = "latin-1"
EXTRA_GOLD_ENCODING = "utf-8"

MODEL = "es_core_news_sm"

X_FILENAME = "x.txt"
Y_PREDICTION_FILENAME = "y_prediction.txt"
Y_FILENAME = "y.conll"

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
X_PATH = os.path.join(PATH, X_FILENAME)
Y_TRUE_PATH = os.path.join(PATH, "esp.testb")
Y_PREDICTION_PATH = os.path.join(PATH, Y_PREDICTION_FILENAME)
Y_PATH = os.path.join(PATH, Y_FILENAME)

EXTRA_PATH = os.path.join(PATH, "extra")
EXTRA_X_PATH = os.path.join(EXTRA_PATH, X_FILENAME)
EXTRA_Y_TRUE_PATH = os.path.join(EXTRA_PATH, "y_true.conll")
EXTRA_Y_PREDICTION_PATH = os.path.join(EXTRA_PATH, Y_PREDICTION_FILENAME)
EXTRA_Y_PATH = os.path.join(EXTRA_PATH, Y_FILENAME)
