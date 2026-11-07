from lib.configuration import (
    GOLD_ENCODING,
    EXTRA_GOLD_ENCODING,
    X_PATH,
    Y_PATH,
    Y_PREDICTION_PATH,
    Y_TRUE_PATH,
    EXTRA_X_PATH,
    EXTRA_Y_PATH,
    EXTRA_Y_PREDICTION_PATH,
    EXTRA_Y_TRUE_PATH,
)
from lib.pipeline import run


if __name__ == "__main__":
    options = [
        (
            "CoNLL-2002 (esp.testb file)",
            (
                X_PATH,
                Y_TRUE_PATH,
                Y_PREDICTION_PATH,
                Y_PATH,
                GOLD_ENCODING
            ),
        ),
        (
            """Champions League 
   (
    https://es.uefa.com/uefachampionsleague/news/02ab-21fe48f0b7da-351f3a5a0318-1000--resumen-del-miercoles-de-la-jornada-4-de-la-champions-leag/,
    https://www.fcbarcelona.es/es/noticias/4603860/fc-barcelona-aston-villa-tres-goles-tres-puntos-3-0
   )""",
            (
                EXTRA_X_PATH,
                EXTRA_Y_TRUE_PATH,
                EXTRA_Y_PREDICTION_PATH,
                EXTRA_Y_PATH,
                EXTRA_GOLD_ENCODING,
            ),
        ),
    ]

    for number, (text, _) in enumerate(options, start=1):
        print(f"{number}. {text}")

    option = input(f"Select a corpus ({1} - {len(options)})\n").strip()
    while not option.isdecimal() or not 1 <= int(option) <= len(options):
        option = input(f"Invalid option, select a corpus ({1} - {len(options)})\n").strip()

    _, args = options[int(option) - 1]

    run(*args)
