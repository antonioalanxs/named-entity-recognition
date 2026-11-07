def get_plain_text(path, encoding="utf-8"):
    """
    Reconstruct plain text from a file with tags.

    `line.split()` separates each line into fields, and `[0]` selects
    the first field (the word). The selected fields are joined with spaces.

    Example input file:
        La B-LOC
        Coruña I-LOC
        , O
        23 O
        may O

    Example output:
        La Coruña , 23 may

    Args:
        path (str or Path): Path to the input file.
        encoding (str): Input file encoding.

    Returns:
        str: The reconstructed plain text.
    """
    plain_text = []

    with open(path, "r", encoding=encoding) as file:
        lines = file.readlines()

    for line in lines:
        if line.strip():
            word = line.split()[0]
            plain_text.append(word)
        else:
            plain_text.append("")

    plain_text = " ".join(plain_text)
    return plain_text


def get_iob(doc):
    """
    Convert a spaCy Doc object to IOB tagged format.

    Args:
        doc (spacy.tokens.Doc): A spaCy Doc object.

    Returns:
        str: A string containing the IOB tagged format of the input Doc.
    """
    output = []

    for token in doc:
        iob = token.ent_iob_
        if iob in ("B", "I"):
            output.append(f"{token.text} {iob}-{token.ent_type_}")
        elif token.text == " ":
            output.append(" ")
        else:
            output.append(f"{token.text} O")

    output = "\n".join(output)
    return output


def combine_tags(y_true_path, y_prediction_path, gold_encoding="latin-1"):
    """
    Combine gold and predicted tags into a single output.

    Example input files:
        y_true_path:
            La B-LOC
            Coruña I-LOC
            , O
        y_prediction_path:
            La B-LOC
            Coruña I-LOC
            , O

    Example output:
        La B-LOC B-LOC
        Coruña I-LOC I-LOC
        , O O

    Args:
        y_true_path (str or Path): Path to the file with gold tags.
        y_prediction_path (str or Path): Path to the file with predicted tags.
        gold_encoding (str): Encoding of the file with gold tags.

    Returns:
        list[str]: Lines containing the word, gold tag and predicted tag.
    """
    output = []

    with (
        open(y_true_path, encoding=gold_encoding) as y_true_file,
        open(y_prediction_path, encoding="utf-8") as y_prediction_file,
    ):
        for index, (true_line, prediction_line) in enumerate(
            zip(y_true_file, y_prediction_file, strict=True), start=1
        ):
            gold = true_line.split()
            prediction = prediction_line.split()

            if not gold and not prediction:
                output.append("\n")
                continue

            if len(gold) != 2 or len(prediction) != 2 or gold[0] != prediction[0]:
                raise ValueError(
                    f"Unaligned gold and prediction at line {index}"
                )

            output.append(f"{gold[0]} {gold[1]} {prediction[1]}\n")

    return output
