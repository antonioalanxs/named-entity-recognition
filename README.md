# Spanish Named Entity Recognition with spaCy

This project examines named entity recognition in Spanish texts from the [CoNLL-2002](https://aclanthology.org/W02-2024/) corpus. Its objective is to identify people, organisations, locations and other named entities and assess the quality of automatic annotation against a reference. It uses [Python](https://www.python.org/) and [spaCy](https://spacy.io/)'s pretrained Spanish model, with particular attention to text reconstruction, annotation in IOB format and analysis of the resulting errors.

The same procedure is also applied to three manually annotated texts about the Champions League. The results are compared with those obtained on the reference corpus.

## Table of Contents

- [Running the project](#running-the-project)
- [Chapter 1. CoNLL-2002](#chapter-1-conll-2002)
  - [Task description](#task-description)
  - [Submitted systems](#submitted-systems)
  - [Results and relevance to this project](#results-and-relevance-to-this-project)
- [Chapter 2. Named entity recognition with spaCy](#chapter-2-named-entity-recognition-with-spacy)
  - [Reconstructing text from IOB annotations](#reconstructing-text-from-iob-annotations)
  - [Text processing and automatic annotation](#text-processing-and-automatic-annotation)
  - [Evaluation against the reference annotations](#evaluation-against-the-reference-annotations)
    - [Evaluation metrics](#evaluation-metrics)
    - [CoNLL-2002 results](#conll-2002-results)
- [Chapter 3. Error analysis and proposed improvements](#chapter-3-error-analysis-and-proposed-improvements)
  - [Error analysis](#error-analysis)
  - [Proposed improvements](#proposed-improvements)
    - [Using context to resolve ambiguity](#using-context-to-resolve-ambiguity)
    - [Comparing predictions with another model](#comparing-predictions-with-another-model)
    - [Checking corpus integrity](#checking-corpus-integrity)
    - [Supplementing the model with dictionaries and rules](#supplementing-the-model-with-dictionaries-and-rules)
- [Chapter 4. Additional experiment with Champions League reports](#chapter-4-additional-experiment-with-champions-league-reports)
  - [Text selection and sources](#text-selection-and-sources)
  - [Texts used](#texts-used)
    - [Text 1. FC Barcelona vs Aston Villa](#text-1-fc-barcelona-vs-aston-villa)
    - [Text 2. AEK Athens vs Real Madrid](#text-2-aek-athens-vs-real-madrid)
    - [Text 3. Borussia Dortmund vs Real Betis and Fenerbahçe vs Liverpool](#text-3-borussia-dortmund-vs-real-betis-and-fenerbahçe-vs-liverpool)
  - [Reference annotation and processing](#reference-annotation-and-processing)
  - [Champions League results](#champions-league-results)

## Running the project

```bash
sudo apt update
sudo apt install python3
python -m pip install -r requirements.txt
python main.py
```

Select **1** for CoNLL-2002 or **2** for the Champions League corpus. The program prints the evaluation and writes [x.txt](./data/x.txt), [y_prediction.txt](./data/y_prediction.txt) and [y.conll](./data/y.conll) in the selected dataset's directory.

## Chapter 1. CoNLL-2002

### Task description

CoNLL-2002 was a shared task on named entity recognition (NER) in Spanish and Dutch. The task consisted of locating the words that form each entity and assigning one of four categories: person (PER), organisation (ORG), location (LOC) or miscellaneous (MISC).

Participants were given training, development and test sets. The Spanish corpus consisted of news articles from the EFE news agency, while the Dutch corpus came from four editions of the Belgian newspaper *De Morgen*. This project uses [esp.testb](./data/esp.testb), the Spanish test set. Its IOB annotations distinguish the beginning of an entity (`B`), its continuation (`I`) and words outside an entity (`O`).

The evaluation required a match in both entity boundaries and category. For example, if the reference marked “La Coruña” as a location, a prediction containing only “Coruña” counted as an error. Precision measures the proportion of predicted entities that are correct, while recall measures the proportion of annotated entities recovered by the system. F1 combines the two as their harmonic mean.

### Submitted systems

Twelve systems were submitted. Their methods included decision trees, support vector machines, maximum entropy models and transformation-based learning. All twelve outperformed the baseline, which searched the text for entities associated with a single category in the training data. When matches overlapped, the baseline prioritised the longest match.

Carreras, Màrquez and Padró obtained the best result in both languages. Their system separated the detection of entity boundaries from the assignment of entity categories. For both stages it combined fixed-depth decision trees through AdaBoost and used features such as neighbouring words, part-of-speech tags and lists of known names. Wu and colleagues also combined decision trees with a variant of AdaBoost, while Florian used transformation-based learning to identify entities.

### Results and relevance to this project

| System | Spanish F1 | Dutch F1 |
|---|---:|---:|
| Carreras, Màrquez and Padró | 81.39% | 77.05% |
| Wu et al. | 76.61% | 75.36% |
| Florian | 79.05% | 74.99% |

CoNLL-2002 established a common procedure for comparing systems on the same test texts, categories and scoring criteria. This project uses that framework to evaluate a pretrained spaCy model on [esp.testb](./data/esp.testb). The shared-task scores provide historical context. Participants could train their systems on the competition data, whereas this project applies spaCy without further training. 

> [CoNLL-2002 shared-task paper](https://aclanthology.org/W02-2024/)

## Chapter 2. Named entity recognition with spaCy

### Reconstructing text from IOB annotations

Each line of [esp.testb](./data/esp.testb) contains a word or punctuation mark followed by its IOB tag. To process the file with spaCy, the text is first reconstructed from the first column. The original tags remain in the reference file for comparison with the model's predictions.

The `get_plain_text` function in [lib/utilities.py](./lib/utilities.py) performs this reconstruction. The pipeline opens [esp.testb](./data/esp.testb) as Latin-1 and the function processes its lines in order. On each non-empty line, `line.split()` separates the fields and `[0]` selects the first one. For example, given "Coruña I-LOC", the expression `line.split()[0]` returns "Coruña".

```python
for line in lines:
    if line.strip():
        word = line.split()[0]
        plain_text.append(word)
    else:
        plain_text.append("")

plain_text = " ".join(plain_text)
```

Blank lines are represented by empty strings. When the elements are joined with `" ".join()`, these strings produce additional spaces that represent the breaks in the original file. A space is also retained between each word and punctuation mark. For example, the lines "La B-LOC", "Coruña I-LOC" and ", O" become "La Coruña ,". The code in [lib/pipeline.py](./lib/pipeline.py) saves the reconstructed text as UTF-8 in [x.txt](./data/x.txt) for use as model input.

```python
plain_text = get_plain_text(y_true_path, encoding=gold_encoding)

with open(x_path, "w", encoding="utf-8") as file:
    file.write(plain_text)
```

### Text processing and automatic annotation

Once [x.txt](./data/x.txt) has been generated, its contents are processed with `es_core_news_sm`, spaCy's pretrained Spanish model. The tokenizer is configured in [lib/pipeline.py](./lib/pipeline.py) to respect the whitespace segmentation of the reconstructed text. This keeps the words and punctuation aligned with the annotated lines in [esp.testb](./data/esp.testb).

```python
nlp = spacy.load(MODEL)
nlp.tokenizer = Tokenizer(nlp.vocab)
```

Calling `nlp(x_file.read())` returns a document containing the model's predictions. The `get_iob` function in [lib/utilities.py](./lib/utilities.py) then walks through that document and converts the predictions into the same line format as the reference.

```python
def get_iob(doc):
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
```

The `token.ent_iob_` attribute indicates whether the token begins an entity (`B`), continues one (`I`) or falls outside one (`O`). In the first two cases, the function adds the entity category from `token.ent_type_`. If spaCy recognises “La Coruña” as a location, the function produces "La B-LOC" and "Coruña I-LOC". A comma outside the entity is written as ", O".

When the token consists of exactly one space, the function preserves it as an unlabelled line to retain the breaks in the source. Finally, `"\n".join(output)` combines all the lines into one string. The following code reads the input text, runs the model and writes the result to [y_prediction.txt](./data/y_prediction.txt) in UTF-8.

```python
with (
    open(x_path, "r", encoding="utf-8") as x_file,
    open(y_prediction_path, "w", encoding="utf-8") as y_prediction_file,
):
    y_prediction = get_iob(nlp(x_file.read()))
    y_prediction_file.write(y_prediction)
```

### Evaluation against the reference annotations

After generating [y_prediction.txt](./data/y_prediction.txt), the pipeline compares spaCy's predicted labels with the reference annotations in [esp.testb](./data/esp.testb). The `combine_tags` function in [lib/utilities.py](./lib/utilities.py) reads both files in parallel and checks that words and punctuation align line by line.

```python
def combine_tags(y_true_path, y_prediction_path, gold_encoding="latin-1"):
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
```

The result is written to [y.conll](./data/y.conll). Its three columns contain the original token, the reference tag and the predicted tag. The example below uses the intended spelling of “Coruña”.

```text
La B-LOC O
Coruña I-LOC O
, O O
```

The reference marks “La Coruña” as a location, while the model assigns `O` to both words. The evaluator counts one missed entity. The comma receives the correct tag, which counts towards token-level accuracy.

The following excerpt from [lib/pipeline.py](./lib/pipeline.py) writes the combined lines and passes them to `evaluate_conll_file` in [lib/conlleval.py](./lib/conlleval.py).

```python
y = combine_tags(y_true_path, y_prediction_path, gold_encoding=gold_encoding)

with open(y_path, "w", encoding="utf-8") as file:
    file.writelines(y)

result = evaluate_conll_file(y)
print(result)
```

#### Evaluation metrics

Model performance is measured using precision, recall and F1. A true positive (TP) is a predicted entity that matches the reference in both its boundaries and category. A false positive (FP) is a prediction without an exact match in the reference. A false negative (FN) is a reference entity that the model fails to recover correctly.

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = (2 × Precision × Recall) / (Precision + Recall)
```

Precision is the proportion of predicted entities that are correct. Recall is the proportion of reference entities recovered by the model. F1 is the harmonic mean of precision and recall and falls when either component is low. True negatives (TN) are outside these three formulas.

The same definitions apply to a binary decision about membership in a class. For NER, the evaluator reconstructs complete entity spans, also called chunks, from the IOB tags and distinguishes the four entity categories. `B` marks the start of a span, `I` its continuation and `O` a token outside an entity. The span ends when its sequence of associated tags ends.

The evaluator calculates the scores in three steps.

1. **Count entities**. For each category, count the reference entities, the predicted entities and the exact matches.
2. **Calculate scores for each category**. For category C, divide the exact matches by the predictions to obtain precision and by the reference entities to obtain recall. Calculate F1 from that category's precision and recall.

   ```text
   Precision(C) = correct entities in C / predicted entities in C
   Recall(C)    = correct entities in C / reference entities in C
   F1(C)        = (2 × Precision(C) × Recall(C)) / (Precision(C) + Recall(C))
   ```

3. **Micro-average the global results**. Sum TP, FP and FN across categories, then apply the original formulas to those totals. Each entity contributes equally to the counts, so more frequent categories have a greater influence on the overall scores. Per-category results can expose weak performance that the micro-average hides.

#### CoNLL-2002 results

The evaluator processed 53,049 tokens and counted 3,559 reference entities. SpaCy predicted 3,955 entities, of which 1,746 matched the reference in both boundaries and category.

```text
processed 53049 tokens with 3559 phrases; found: 3955 phrases; correct: 1746.
accuracy:  48.14%; (non-O)
accuracy:  90.43%; precision:  44.15%; recall:  49.06%; FB1:  46.47
              LOC: precision:  45.75%; recall:  59.59%; FB1:  51.76  1412
             MISC: precision:   9.33%; recall:  21.76%; FB1:  13.06   793
              ORG: precision:  70.21%; recall:  40.07%; FB1:  51.02   799
              PER: precision:  48.90%; recall:  63.27%; FB1:  55.16   951
(44.146649810366625, 49.0587243607755, 46.473249933457545)
```

The model obtained **44.15%** precision, **49.06%** recall and **46.47%** F1. The precision uses the 3,955 predictions as its denominator. Recall relates the same 1,746 matches to the 3,559 reference entities, indicating that the model recovered about half of the annotated entities. The F1 leaves room for improvement in both reducing incorrect predictions and recovering missed entities.

PER achieved the highest F1 (55.16%) and recall (63.27%), while ORG had the highest precision (70.21%). LOC achieved an F1 of 51.76%, slightly above ORG's 51.02%. MISC had the lowest F1 (13.06%) and a precision of 9.33%. Fewer than one in ten of the 793 MISC predictions matched the reference.

## Chapter 3. Error analysis and proposed improvements

### Error analysis

Reviewing the generated [y.conll](./data/y.conll) file reveals three kinds of error.

- **Missed entities**. The reference marks “La Coruña” as a location, while spaCy assigns `O` to both words.
- **Category errors**. SpaCy detects "EFECOM" as an entity and assigns MISC. The reference classifies it as an organisation (ORG).
- **Boundary and category errors**. The reference annotates “Atlético de Madrid” as a single organisation. The model detects only “Madrid” and classifies it as a location. This counts as an incorrect prediction and a missed reference entity.

The category breakdown reveals different difficulties. MISC precision was 9.33%, meaning fewer than one in ten predictions matched the reference. ORG precision was substantially higher at 70.21%, but its recall of 40.07% means about six in ten annotated organisations were missed.

### Proposed improvements

#### Using context to resolve ambiguity

“Madrid” can refer to a location or appear within the name of an organisation. An additional review step could examine the surrounding words and the complete expression before changing a prediction. This would help distinguish an independent mention of the city from a mention within the name of a club.

#### Comparing predictions with another model

A second Spanish NER model could be evaluated on [esp.testb](./data/esp.testb) using the same exact-match criterion. If the two models make different errors, their predictions could be combined using rules determined on the development set. Comparing overall and per-category F1 against both individual models would show whether the combination improves performance.

#### Checking corpus integrity

The included copy of [esp.testb](./data/esp.testb) contains replacement characters in names such as “Coruña” and “Atlético”. Reading the file as Latin-1 passes those altered sequences into the model input. An initial improvement would be to obtain an intact copy of the corpus, verify its encoding and rerun the same model. The change in results would help quantify the effect of the damaged text.

#### Supplementing the model with dictionaries and rules

A dictionary of full organisation names and acronyms could help recognise entities such as "EFECOM" and recover the full boundaries of names such as “Atlético de Madrid”. Rules could prioritise a match with the full name. The dictionary would be built from external sources or training data, keeping the test set separate from rule development.

## Chapter 4. Additional experiment with Champions League reports

### Text selection and sources

The same spaCy model was evaluated on three Spanish texts about the fourth matchday of the 2026/27 Champions League, played on 3 and 4 November 2026. The texts were written for the project using match reports and results published by UEFA, FC Barcelona and Real Madrid. This topic brings together names of players, coaches, clubs, cities, stadiums and competitions within the same genre.

- [FC Barcelona - Aston Villa: Tres goles, tres puntos (3-0)](https://www.fcbarcelona.es/es/noticias/4603860/fc-barcelona-aston-villa-tres-goles-tres-puntos-3-0/)
- [Resumen y vídeos del miércoles de la jornada 4 de la Champions League: pleno para el Madrid y exhibición del Como | UEFA Champions League | UEFA.com](https://es.uefa.com/uefachampionsleague/news/02ab-21fe48f0b7da-351f3a5a0318-1000--resumen-del-miercoles-de-la-jornada-4-de-la-champions-leag/)


### Texts used

The three texts are reproduced in full in English below, with conventional punctuation spacing for readability. The original Spanish texts, which are the inputs evaluated by the model, are preserved in [data/extra/y_true.conll](data/extra/y_true.conll). When option 2 runs, it generates [data/extra/x.txt](./data/extra/x.txt) with spaces between words and punctuation marks to preserve the segmentation of the reference annotations.

#### Text 1. FC Barcelona vs Aston Villa

On 3 November 2026, FC Barcelona hosted Aston Villa at the Spotify Camp Nou and won 3–0 on the fourth matchday of the Champions League. Lamine Yamal opened the scoring before the tenth minute. The forward, who was born in Mataró, won the ball back in the move that led to the first goal. Hansi Flick had changed his starting line-up from the previous match against Alavés, bringing in Gerard Martín and Gordon. In the first half, Manzambi tested Joan Garcia, who preserved the hosts' lead until half-time. Raphinha extended the lead in the second half with a shot that deflected off Pau Torres before beating Suzuki. The Brazilian scored again after a pass from Cancelo, matching the 94 goals Ronaldinho had scored for Barça. With the result all but settled, Frenkie de Jong came off the bench to make his 300th appearance for the club. Flick's team ended the evening with nine points in the league phase.

#### Text 2. AEK Athens vs Real Madrid

Real Madrid secured their fourth consecutive Champions League win on 4 November 2026. José Mourinho's side beat AEK Athens 3–2 at the Allwyn Arena in Athens after falling behind twice. Koita put the hosts ahead in the 12th minute. Early in the match, Courtois had to intervene to keep the Greek side from extending their lead. The visitors responded before half-time. Marc Cucurella crossed and Jude Bellingham equalised with a backheel finish. The English midfielder was making his 60th appearance in the competition. In the second half, Rota cleared a Cucurella effort off the line, but Pilios put AEK back in front. Kylian Mbappé equalised again and scored for the sixth consecutive match in the competition. In stoppage time, Federico Valverde settled the match with a shot from outside the area. UEFA named Bellingham player of the match.

#### Text 3. Borussia Dortmund vs Real Betis and Fenerbahçe vs Liverpool

Real Betis suffered their first defeat of the Champions League league phase, losing 2–1 to Borussia Dortmund in Germany. In the first half, Serhou Guirassy shot wide and Álvaro Valles denied Daniel Svensson with a save on the line. After half-time, Fábio Silva capitalised on a rebound to put the German side ahead. Aitor Ruibal equalised after receiving a pass from Abde, but Konstantinos Karetsas settled the match with a shot from the edge of the area. In another match that round, Fenerbahçe beat Liverpool 3–2. Trey Nyoni opened the scoring for the English club, and Mason Greenwood played a part in the equaliser with a free kick that took a deflection before beating Alisson Becker. Alexander Isak restored Liverpool's lead in the 69th minute. The hosts responded with goals from Archie Brown and Vedat Muriqi, who secured the win with a header in the 79th minute.

### Reference annotation and processing

The three texts were combined in [data/extra/y_true.conll](./data/extra/y_true.conll) and annotated using the same IOB format as the CoNLL-2002 experiment. Each line contains a word or punctuation mark and its tag. The prefix `B` marks the start of an entity and `I` marks its continuation. `O` is assigned to tokens outside entities. Players and coaches were tagged as PER, clubs and organisations as ORG, cities and stadiums as LOC, and the competition name as MISC.

The following excerpt shows the annotation of two clubs and a stadium.

```text
FC B-ORG
Barcelona I-ORG
recibió O
al O
Aston B-ORG
Villa I-ORG
en O
el O
Spotify B-LOC
Camp I-LOC
Nou I-LOC
```

Selecting **Champions League** in [main.py](./main.py) runs the same procedure as for CoNLL-2002. The pipeline reconstructs the text in [x.txt](./data/x.txt), processes it with `es_core_news_sm` and saves the automatic labels in [y_prediction.txt](./data/y_prediction.txt). The `combine_tags` function then writes the token, reference tag and predicted tag to [y.conll](./data/y.conll). An entity counts as correct when its boundaries and category both match. All the files for this experiment are read and written in UTF-8 under [data/extra](./data/extra).

### Champions League results

The evaluator processed 555 tokens and counted 58 reference entities. Of the 64 entities predicted by spaCy, 37 matched the reference. Precision was **57.81%**, recall **63.79%**, and F1 **60.66%**. Token-level accuracy was 89.37% overall and 71.58% in the evaluator's non-O calculation.

```text
processed 555 tokens with 58 phrases; found: 64 phrases; correct: 37.
accuracy:  71.58%; (non-O)
accuracy:  89.37%; precision:  57.81%; recall:  63.79%; FB1:  60.66
              LOC: precision:  27.27%; recall:  60.00%; FB1:  37.50  11
             MISC: precision:  17.65%; recall: 100.00%; FB1:  30.00  17
              ORG: precision:  50.00%; recall:  30.77%; FB1:  38.10   8
              PER: precision:  96.43%; recall:  72.97%; FB1:  83.08  28
(57.8125, 63.793103448275865, 60.65573770491803)
```

PER obtained the best result. The model correctly identified 27 of its 28 predicted person mentions and recovered 27 of the 37 annotated mentions. In MISC it recovered all three reference entities, but made 17 predictions, giving 100% recall alongside 17.65% precision. In LOC it correctly identified 3 of 11 predictions, while in ORG it recovered 4 of the 13 annotated entities.

| Corpus | Reference entities | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| CoNLL-2002 | 3,559 | 44.15% | 49.06% | 46.47% |
| Champions League | 58 | 57.81% | 63.79% | 60.66% |

The football corpus scored 14.19 percentage points higher in global F1. PER rose from 55.16% to 83.08%, while LOC fell from 51.76% to 37.50% and ORG fell from 51.02% to 38.10%. Person mentions account for 37 of the 58 reference entities in the football corpus and have a substantial influence on its micro-average. The small size and different distribution of categories limit the comparison between corpora.
