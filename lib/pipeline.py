import spacy
from spacy.tokenizer import Tokenizer

from lib.configuration import MODEL
from lib.conlleval import evaluate_conll_file
from lib.utilities import combine_tags, get_iob, get_plain_text


def run(x_path, y_true_path, y_prediction_path, y_path, gold_encoding):
    plain_text = get_plain_text(y_true_path, encoding=gold_encoding)

    with open(x_path, "w", encoding="utf-8") as file:
        file.write(plain_text)
        print(f"Plain text written to '{x_path}'")

    nlp = spacy.load(MODEL)
    nlp.tokenizer = Tokenizer(nlp.vocab)

    with (
        open(x_path, "r", encoding="utf-8") as x_file,
        open(y_prediction_path, "w", encoding="utf-8") as y_prediction_file,
    ):
        y_prediction = get_iob(nlp(x_file.read()))
        y_prediction_file.write(y_prediction)
        print(f"Predicted IOB tags written to '{y_prediction_path}'")

    with open(y_path, "w", encoding="utf-8") as file:
        y = combine_tags(y_true_path, y_prediction_path, gold_encoding=gold_encoding)
        file.writelines(y)
        print(f"Combined tags written to '{y_path}'")

    result = evaluate_conll_file(y)
    print(result)
