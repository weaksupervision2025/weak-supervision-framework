# iterative-weak-labelling

This project explores iterative generation of weak labelling functions using LLM prompts and [Snorkel](https://snorkel.ai/) for training label models.

## Setup

1. Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

2. Set config.py

## Data folder structure

All data is stored in the `data/` directory. Each dataset is placed under its
own folder:

```
 data/
   <dataset_name>/
     source/               # raw dataset files
       train.json
       validation.json
       test.json
     iter_<n>/             # outputs of iterative pipeline runs
       criteria.jsonl
       classified/
         train.jsonl
         dev.jsonl
         test.jsonl
       weak_labels/
         train.jsonl
         dev.jsonl
         test.jsonl
       models/
         label_model.pkl
       metrics/
         metrics.json
       filtered_lfs.json
```

The `source/` directory must contain splits in the JSON format expected by
`iterative_pipeline.py` (see `load_dataset_df` for details). Each iteration
creates a new `iter_<n>` directory with generated criteria, weak labels and the
trained label model.

## Running the pipeline

The main entry point is `iterative_pipeline.py`. The minimal invocation is:

```bash
python iterative_pipeline.py --dataset banking77 --max_iter 1
```

Arguments:

- `--dataset` – name of the dataset folder inside `data/`.
- `--output_dir` – where to place pipeline outputs (default: `data`).
- `--max_iter` – number of iterations to run.
- `--dev_size` – number of items in dev.
- `--accuracy_threshold` – LF accuracy cut-off during filtering.
- `--num_workers` – number of parallel threads for generation and classification.

## Pipeline logic

`iterative_pipeline.py` performs the following steps in each iteration:

1. **Load dataset splits** – training, test and development splits are loaded
   from `data/<dataset>/source/`.
2. **Generate criteria** – `CriteriaGenerator` uses an LLM prompt to propose new
   labeling criteria for the dataset. Criteria from previous iterations are
   reused and deduplicated.
3. **Classify texts** – the generated criteria are applied to the dev set with
   `DialogueCriteriaClassifier` producing boolean predictions for each LF.
4. **Filter labeling functions** – labeling functions are evaluated on the dev
   set with Snorkel. Those with empirical accuracy above the threshold are kept.
5. **Create weak labels** – the filtered LFs classify train, dev and test splits;
   the resulting matrices are saved under `weak_labels/`.
6. **Train label model** – a Snorkel `LabelModel` is trained on the weak labels
   and saved to `models/`.
7. **Evaluate and gather errors** – the label model predictions on the dev set
   are compared with the true labels and metrics are written to
   `metrics/metrics.json`. Misclassified texts are collected to drive the next
   iteration.

The loop repeats until `max_iter` iterations have completed. Outputs from each
iteration are stored in separate `iter_<n>` directories allowing inspection of
intermediate results.

## Criteria generation helper

You can also generate or extend criteria manually using `generate_criteria.py`:

```bash
python generate_criteria.py --dataset DATASET_NAME --output criteria.jsonl [--existing old.jsonl]
```

This produces a JSONL file where every line has `criterion`, `description` and
`class` fields. 