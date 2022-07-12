# Housing Classification with Transformers

This repo aims to perform the classification of housing data into Apartment or House. It fine-tunes and evaluates two pre-trained transformer models on the CASAFARI dataset.

---

### Contents

* [Installation](#installation)
* [Data](#Data)
* [Train](#Train)
* [Evaluation](#Evaluation)
* [Experimentation](#Experimentation)

---

## Installation
```console
$ virtualenv venv -p python3
$ source venv/bin/activate
$ pip install -r requirements.txt
$ pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

## Data

### Preprocessing text file

Pre-process external files to generate training, development and test sets.

```console
$ python -m src.data.make_dataset <dataset_file>
```
Parameters:
* **dataset_file**: Housing dataset (.json) + binary labels (Apartment, House), e.g. "assessment_NLP.json".

The files must be inside:
```console
$./data/raw/
```

Output:
```console
$./data/processed/
```

## Train
Fine-tune pre-trained transformer models on training and development data.

```console
$ python -m src.models.train_model <model_name>
```
Parameters:
* **model_name**: pre-trained transformer model, e.g. "distilbert-base-uncased" or "bert-base-uncased".

Output:
```console
$./model/
```

## Evaluation

Evaluate transformer models on test data.

```console
$ python -m src.models.evaluate_model <model_name>
```
Parameters:
* **model_name**: transformer model fine-tuned on <dataset_file>, e.g. "checkpoint-32924".

The file must be inside:
```console
$./model/
```

## Experimentation

Perform evaluation of Housing Classification models on CASAFARI data.

```console
$ cd notebooks/
$ jupyter notebook
```
