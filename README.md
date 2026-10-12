# Robot Failure Prediction and Maintenance Intelligence

## 1. Project Overview

This project is an integrated predictive-maintenance system developed as part of the Green Circle AI Course capstone (Modules 1–30). It uses machine learning, PostgreSQL, data analysis, clustering, natural language processing (NLP), text retrieval, and parameter-efficient fine-tuning (PEFT) to explore machine-failure prediction and maintenance-related information.

The project focuses on identifying machine failures from sensor measurements, comparing classification and regression models, analyzing machine operating patterns, and retrieving relevant maintenance information.

**Primary objective:** Develop a reproducible data pipeline and machine-learning system that supports machine-failure analysis and provides a foundation for maintenance decision support.

### Intended end users

- Maintenance engineers investigating machine operating conditions.
- Reliability engineers studying failure patterns.
- Data analysts evaluating predictive-maintenance models.
- Technical teams exploring how structured sensor data and unstructured maintenance text can be combined.

## 2. Dataset and Data Quality

The project uses the AI4I 2020 Predictive Maintenance dataset, containing 10,000 machine records and sensor-related measurements.

The dataset includes:

- Unique record identifier (`UDI`)
- Product identifier and machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Overall machine-failure label
- Individual failure-mode indicators

The dataset is designed for predictive-maintenance research and contains simulated/synthetic industrial data. Results should therefore be interpreted as experimental findings on this dataset, not as proof of performance on live factory equipment.

### Data quality and label consistency

The source dataset contains an important distinction between the overall `Machine failure` target and the individual failure-mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, and `RNF`).

Some records have an individual cause flag set to 1 while the overall machine-failure label is 0. These fields are not interchangeable. The project preserves this distinction when interpreting records and reporting results.

## 3. Class Imbalance

The classification target is highly imbalanced:

| Target | Records | Percentage |
|---|---:|---:|
| No machine failure (0) | 9,661 | 96.61% |
| Machine failure (1) | 339 | 3.39% |
| **Total** | **10,000** | **100%** |

Only 3.39% of records belong to the failure class. A model that predicts no failures for every record could achieve 96.61% accuracy while detecting no actual failures.

For this reason, accuracy alone is not a sufficient evaluation metric. The project considers precision, recall, F1-score, ROC-AUC, and especially average precision / PR-AUC.

The modeling pipeline excludes the individual failure-mode indicators from the input features because they are closely related to the target and could cause target leakage. It also removes record and product identifiers from model inputs.

### Imbalance experiment

A separate Random Forest experiment compared class-weighting strategies:

| Strategy | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| No class weighting | 0.9649 | 0.8088 | 0.8800 | 0.8954 | 0.9730 |
| `balanced` | 0.9623 | 0.7500 | 0.8430 | 0.8717 | 0.9780 |
| `balanced_subsample` | 0.9623 | 0.7500 | 0.8430 | 0.8790 | 0.9742 |

In this experiment, the unweighted Random Forest achieved the highest recall, F1-score, and PR-AUC among the three tested configurations. Class weighting did not improve every metric, demonstrating why imbalance-handling strategies must be evaluated rather than assumed to be better.

These results belong to a particular train/test split and should not be directly compared with results from experiments that use different splits.

### Threshold experiment

A separate experiment used a 60% training, 20% validation, and 20% test split. The threshold was investigated using validation data before the final test evaluation.

At the 0.50 threshold, the reported test results were:

- Precision: 0.9636
- Recall: 0.7794
- F1-score: 0.8618
- PR-AUC: 0.8858
- ROC-AUC: 0.9696
- False positives: 2
- False negatives: 15

A threshold of 0.48 produced the same reported test metrics, so there was no demonstrated benefit from lowering it in that experiment.

## 4. System Architecture

The project is organized into separate modules for data ingestion, database access, analysis, modeling, clustering, NLP, and retrieval.

```text
AI4I 2020 Dataset
        |
        v
Data Ingestion and Preprocessing
        |
        +--------------------------+
        |                          |
        v                          v
PostgreSQL Database          ML Pipeline
        |                    - Classification
        |                    - Regression
        |                    - Feature Engineering
        |                          |
        v                          v
SQL Queries and Views        Champion Model
        |
        +--------------------------+
        |
        v
Maintenance Text Processing
        |
        v
TF-IDF Text Index and Retrieval Evaluation
        |
        v
Command-Line Interface (main.py)
        |
        v
User Questions and Analysis
```

The intended final architecture combines structured database/model outputs with retrieved maintenance text and language-model-generated responses. The current `ask` implementation provides retrieval-based question answering; full generative RAG integration remains a development item until implemented and tested.

## 5. Database Design

PostgreSQL is used to store normalized machine, sensor, failure-event, cause, and maintenance-note data.

The database includes these tables:

- `product_types`
- `machines`
- `sensor_readings`
- `failure_events`
- `failure_causes`
- `event_causes`
- `maintenance_notes`

The database work includes schema creation, data ingestion, primary and foreign key relationships, integrity constraints, analytical SQL queries, views, and a stored procedure for adding maintenance notes.

The stored procedure `add_maintenance_note(p_machine_id, p_note_text)` validates the supplied text and checks that the machine exists before inserting the note.

Analytical SQL queries examine machine counts by type, average sensor measurements, failure events, cause assignments, sensor differences between event groups, tool wear, and temperature differences.

The recorded event and cause tables must be interpreted according to their own definitions rather than treated as identical to the original dataset's overall failure label.

## 6. Data Analysis and Feature Engineering

The analysis pipeline examines dataset dimensions, data types, missing values, target distribution, and numerical summaries.

The classification pipeline includes reusable preprocessing:

- Median imputation for numeric features.
- Standard scaling for numeric features.
- Most-frequent imputation for categorical features.
- One-hot encoding for categorical variables.
- Removal of identifiers and failure-mode fields that could introduce leakage.

Additional engineered features include:

1. **Temperature difference:** Process temperature minus air temperature.
2. **Mechanical power proxy:** Rotational speed multiplied by torque.
3. **Tool wear–torque interaction:** Tool wear multiplied by torque.

These features are intended to represent operating conditions and possible interactions between mechanical measurements. They are experimental predictors, not proof of causal relationships.

## 7. Machine-Learning Models

The classification suite includes:

- Dummy baseline
- Perceptron
- Logistic regression
- Linear Support Vector Machine
- RBF-kernel Support Vector Machine
- Decision tree
- Random Forest
- AdaBoost

Models are compared using precision, recall, F1-score, ROC-AUC, and PR-AUC. PR-AUC receives particular attention because failures are uncommon.

The project also includes regression experiments using linear regression, Random Forest regression, and polynomial regression with regularization.

### Reported regression results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 59.36 rpm | 85.92 rpm | 0.7560 |
| Random Forest Regressor | 41.92 rpm | 53.10 rpm | 0.9068 |

The advanced polynomial regression experiment reported a best test RMSE of 59.30 and R² of 0.8838.

The regression target is rotational speed. These results describe prediction error for that target; they do not measure machine-failure classification performance.

### Five-fold cross-validation

An additional five-fold stratified cross-validation experiment was conducted using a Random Forest with 150 trees and no class weighting.

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Precision | 0.9826 | 0.0156 |
| Recall | 0.8231 | 0.0274 |
| F1-score | 0.8955 | 0.0177 |
| ROC-AUC | 0.9751 | 0.0019 |
| PR-AUC | 0.8978 | 0.0097 |

The standard deviations summarize variation across the five folds. These results provide a robustness estimate, not a guarantee of real-world performance. They should be reported separately from the held-out test results because the evaluation protocols differ.

Cross-validation results are saved in `reports/cross_validation_results.csv`.

## 8. Clustering

K-Means clustering is used as an unsupervised exploration of machine operating patterns.

The reported silhouette scores were:

| Number of clusters | Silhouette score |
|---|---:|
| 2 | 0.205180 |
| 3 | 0.208171 |
| 4 | 0.210314 |
| 5 | 0.201138 |
| 6 | 0.199497 |

Four clusters were selected among the tested values. Their recorded sizes were 2,523, 2,076, 2,673, and 2,728 records.

The selected cluster labels are stored in the PostgreSQL `machine_clusters` table. The silhouette score is relatively modest, so the groups should be interpreted as exploratory segments rather than naturally distinct or operationally validated machine categories.

## 9. NLP and Maintenance Text Retrieval

The NLP component explores text preprocessing and retrieval over generated machine descriptions and maintenance-related text.

The current TF-IDF representation has:

- Matrix shape: 10,000 records × 1,214 features
- Non-zero entries: 219,450

The project includes a saved TF-IDF index and a retrieval evaluation script. Retrieval performance is assessed separately from machine-failure prediction. The evaluation has shown that query phrasing and the definition of relevant records affect retrieval results; retrieval quality should not be inferred from classification metrics.

The saved index and supporting reports include:

- `models/maintenance_tfidf_index.joblib`
- `reports/nlp_model_comparison.csv`
- `reports/nlp_notes.txt`
- `reports/retrieval_evaluation.csv`

The retrieval evaluation should include a small hand-labeled query set and report recall@k to meet the capstone's RAG evaluation requirement. Results should reflect the actual labeled queries and retrieved records.

## 10. LoRA Fine-Tuning Experiment

A small PEFT experiment was completed using LoRA and the local-compatible Hugging Face model `sshleifer/tiny-gpt2`.

The experiment used 250 examples, split into 200 training examples and 50 validation examples, and ran for one epoch on CPU.

| Metric | Base model | Fine-tuned model |
|---|---:|---:|
| Validation loss | 10.8295 | 10.8294 |
| Perplexity | 50,487.5366 | 50,484.6626 |

The fine-tuned adapter was saved under `models/lora_maintenance_adapter/`, and the comparison report is saved at `reports/lora_comparison.txt`.

The change in validation loss and perplexity was negligible. This experiment demonstrates that the LoRA training workflow ran and produced an adapter; it does not demonstrate meaningful improvement in maintenance-language generation, factual accuracy, or real-world usefulness. The small model and synthetic-template dataset limit the conclusions that can be drawn.

## 11. Current RAG and Integration Status

The current application includes a retrieval-based `ask` workflow. A retrieval system finds relevant text records, but retrieval alone is not the same as retrieval-augmented generation.

The capstone requires a working RAG pipeline with a vector index, a hand-labeled retrieval evaluation, and integration of structured database/model results with an LLM-generated answer.

**Remaining integration work:** connect retrieved text and structured results to a local language model, use a documented prompt template that treats retrieved text as untrusted evidence, and test whether generated answers are supported by the retrieved records. Do not describe generative RAG as completed until this workflow has been implemented and verified.

## 12. Project Structure

The repository is organized into modules, SQL files, reports, model artifacts, and the command-line entry point.

```text
robot_failure_project/
├── data/
│   └── ai4i2020.csv
├── models/
│   ├── champion_model.joblib
│   ├── maintenance_tfidf_index.joblib
│   └── lora_maintenance_adapter/
├── reports/
│   ├── analysis_report.txt
│   ├── model_comparison.csv
│   ├── model_report.txt
│   ├── cross_validation_results.csv
│   ├── retrieval_evaluation.csv
│   └── lora_comparison.txt
├── sql/
│   ├── schema.sql
│   ├── queries.sql
│   └── analytical_queries.sql
├── src/
│   ├── analysis.py
│   ├── clustering.py
│   ├── config.py
│   ├── database.py
│   ├── ingestion.py
│   ├── modeling.py
│   ├── regression.py
│   ├── regression_advanced.py
│   ├── retrieval.py
│   ├── retrieval_evaluation.py
│   ├── text_features.py
│   └── lora_finetune.py
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

This is a representative structure; ensure the final listing matches the files actually included in the submitted repository.

## 13. Setup and Usage

### Prerequisites

- Python and the packages listed in `requirements.txt`
- PostgreSQL
- A local project configuration in `.env` consistent with `src/config.py`

Do not commit real passwords, API keys, or other credentials.

### Installation

Create and activate a virtual environment using the instructions appropriate for your operating system, then install dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Configure the database connection using the variable names expected by the project's configuration module. Create the PostgreSQL database if required by your setup.

### Main application commands

Run the commands from the project root:

```bash
python main.py --help
python main.py setup
python main.py analyze
python main.py train
python main.py regression
python main.py regression-advanced
python main.py cluster
python main.py index
```

Example question:
### Example 1: Operating Conditions Associated with Machine Failures

**Question**
```text
Which operating conditions are associated with machine failures?
```

**Actual output (from `python main.py ask`)**
```text
Question: Which operating conditions are associated with machine failures?

Database statistics
Machines in database: 10000
Machine failure events: 339
Recorded machine-failure rate: 3.39%
Cause-flag event counts (may overlap):
  HDF — Heat dissipation failure: 115
  OSF — Overstrain failure: 98
  PWF — Power failure: 95
  TWF — Tool wear failure: 46
  RNF — Random failure: 1

Most relevant machine records

Machine UDI 7851 (product M22710, type M)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 36.8, 'tool wear': 0.0, 'air temperature': 300.3, 'process temperature': 311.7, 'rotational speed': 1447.0}
  Retrieval score: 0.1022

Machine UDI 8362 (product L55541, type L)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 44.3, 'tool wear': 7.0, 'air temperature': 298.6, 'process temperature': 309.5, 'rotational speed': 1450.0}
  Retrieval score: 0.1018

Machine UDI 6085 (product M20944, type M)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 36.2, 'tool wear': 0.0, 'air temperature': 300.8, 'process temperature': 310.6, 'rotational speed': 1505.0}
  Retrieval score: 0.1017

Machine UDI 9180 (product H38593, type H)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 44.7, 'tool wear': 8.0, 'air temperature': 297.9, 'process temperature': 309.1, 'rotational speed': 1435.0}
  Retrieval score: 0.1015

Machine UDI 3198 (product L50377, type L)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 40.6, 'tool wear': 9.0, 'air temperature': 300.0, 'process temperature': 309.1, 'rotational speed': 1465.0}
  Retrieval score: 0.1015

Data-quality note: cause flags and the overall machine-failure label are separate source columns. Inconsistent combinations are reported as recorded, not treated as proof that the overall label is positive.
Retrieval ranks records by text and, for explicit high/low sensor questions, dataset-relative numeric criteria. It does not establish causation.
```

**Interpretation:** The CLI combines database-level statistics with relevant retrieved machine records. This particular query returned records labeled as non-failures, so the retrieved examples alone do not establish which operating conditions cause failures. Cause-flag counts may overlap and should not be summed as unique failure events.

### Example 2: Finding Machines with High Torque

**Question**
```text
Which machines have the highest torque?
```

**Actual output (from `python main.py ask`)**
```text
Question: Which machines have the highest torque?

Database statistics
Machines in database: 10000
Machine failure events: 339
Recorded machine-failure rate: 3.39%
Cause-flag event counts (may overlap):
  HDF — Heat dissipation failure: 115
  OSF — Overstrain failure: 98
  PWF — Power failure: 95
  TWF — Tool wear failure: 46
  RNF — Random failure: 1

Most relevant machine records

Machine UDI 82 (product L47261, type L)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 47.5, 'tool wear': 9.0, 'air temperature': 298.9, 'process temperature': 308.9, 'rotational speed': 1450.0}
  Retrieval score: 0.8589

Machine UDI 1425 (product L48604, type L)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 46.8, 'tool wear': 4.0, 'air temperature': 298.7, 'process temperature': 309.7, 'rotational speed': 1462.0}
  Retrieval score: 0.8589

Machine UDI 9909 (product L57088, type L)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 47.7, 'tool wear': 0.0, 'air temperature': 298.6, 'process temperature': 309.0, 'rotational speed': 1456.0}
  Retrieval score: 0.8589

Machine UDI 5914 (product H35327, type H)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 52.3, 'tool wear': 9.0, 'air temperature': 300.7, 'process temperature': 310.2, 'rotational speed': 1479.0}
  Retrieval score: 0.8588

Machine UDI 2868 (product H32281, type H)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 48.5, 'tool wear': 7.0, 'air temperature': 300.7, 'process temperature': 309.6, 'rotational speed': 1390.0}
  Retrieval score: 0.8588

Data-quality note: cause flags and the overall machine-failure label are separate source columns. Inconsistent combinations are reported as recorded, not treated as proof that the overall label is positive.
Retrieval ranks records by text and, for explicit high/low sensor questions, dataset-relative numeric criteria. It does not establish causation.
```

**Interpretation:** The system retrieves records with high torque readings and shows their associated sensor values and recorded failure labels. The retrieved examples are not necessarily confirmed failures; torque alone does not establish that a machine will fail.


### Example 3: Failure Statistics and Common Causes

**Question**
```text
How many machine failures were recorded, and which failure causes were most common?
```

**Actual output (from `python main.py ask`)**
```text
Question: How many machine failures were recorded, and which failure causes were most common?

Database statistics
Machines in database: 10000
Machine failure events: 339
Recorded machine-failure rate: 3.39%
Cause-flag event counts (may overlap):
  HDF — Heat dissipation failure: 115
  OSF — Overstrain failure: 98
  PWF — Power failure: 95
  TWF — Tool wear failure: 46
  RNF — Random failure: 1

Most relevant machine records

Machine UDI 7851 (product M22710, type M)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 36.8, 'tool wear': 0.0, 'air temperature': 300.3, 'process temperature': 311.7, 'rotational speed': 1447.0}
  Retrieval score: 0.1296

Machine UDI 8362 (product L55541, type L)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 44.3, 'tool wear': 7.0, 'air temperature': 298.6, 'process temperature': 309.5, 'rotational speed': 1450.0}
  Retrieval score: 0.1291

Machine UDI 6085 (product M20944, type M)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 36.2, 'tool wear': 0.0, 'air temperature': 300.8, 'process temperature': 310.6, 'rotational speed': 1505.0}
  Retrieval score: 0.1291

Machine UDI 9180 (product H38593, type H)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 44.7, 'tool wear': 8.0, 'air temperature': 297.9, 'process temperature': 309.1, 'rotational speed': 1435.0}
  Retrieval score: 0.1288

Machine UDI 3198 (product L50377, type L)
  Overall machine failure label: 0
  Individual cause flags: none
  Sensor values: {'torque': 40.6, 'tool wear': 9.0, 'air temperature': 300.0, 'process temperature': 309.1, 'rotational speed': 1465.0}
  Retrieval score: 0.1287

Data-quality note: cause flags and the overall machine-failure label are separate source columns. Inconsistent combinations are reported as recorded, not treated as proof that the overall label is positive.
Retrieval ranks records by text and, for explicit high/low sensor questions, dataset-relative numeric criteria. It does not establish causation.
```

**Interpretation:** The database contains 10,000 machine records and 339 overall machine-failure events, representing 3.39% of the dataset. Heat dissipation failure (HDF) has the highest recorded cause-flag count (115), followed by overstrain failure (OSF, 98) and power failure (PWF, 95).

Cause flags may overlap, so these counts must not be added together as if they represented distinct failure events. The retrieved machine examples are contextual records and do not necessarily represent failed machines.


The exact question syntax should match the CLI implementation. `setup`, `train`, and `index` are the principal commands for rebuilding the database, champion model, and text index, respectively.

### Additional experiments

```bash
python -m src.cross_validation
python -m src.lora_finetune
```

These commands run the separate cross-validation and LoRA experiments. The LoRA run may require a Hugging Face model download the first time it is executed.

## 14. Generated Artifacts

Important outputs include:

- `reports/analysis_report.txt` — dataset analysis.
- `reports/model_comparison.csv` — classification model comparison.
- `reports/model_report.txt` — champion model and test-set evaluation.
- `reports/cross_validation_results.csv` — five-fold cross-validation results.
- `reports/random_forest_imbalance_comparison.csv` — imbalance experiment.
- `reports/regression_comparison.csv` — regression comparison.
- `reports/advanced_regression_comparison.csv` — advanced regression comparison.
- `reports/cluster_report.txt` — clustering results.
- `reports/retrieval_evaluation.csv` — retrieval evaluation output.
- `reports/lora_comparison.txt` — LoRA before-and-after comparison.
- `models/champion_model.joblib` — saved champion classification pipeline.
- `models/maintenance_tfidf_index.joblib` — saved TF-IDF retrieval index.
- `models/lora_maintenance_adapter/` — saved LoRA adapter and tokenizer files.

Verify that the listed files are present before submission.

## 15. Limitations and Future Improvements

1. **Synthetic dataset:** the AI4I data is not a substitute for validation on real industrial sensor streams.
2. **Class imbalance:** failure records are rare, so precision, recall, F1, and PR-AUC are more informative than accuracy alone.
3. **Label consistency:** individual failure-mode flags and the overall machine-failure label are distinct fields and must not be conflated.
4. **Model generalization:** cross-validation and held-out test results do not establish performance on unseen factories, machines, or operating environments.
5. **Clustering interpretation:** the selected four-cluster solution has a modest silhouette score and needs domain validation.
6. **Text retrieval:** retrieval quality depends on the query and the relevance labels used for evaluation.
7. **LoRA performance:** the tiny-model experiment showed negligible improvement and is a technical demonstration rather than a useful production model.
8. **Generative RAG integration:** the structured and unstructured components still need to be connected through a tested generation step if that integration is not yet present in the code.
9. **Operational use:** the project is an educational prototype. Predictions should not be treated as definitive maintenance decisions without independent validation.

## 16. Conclusion

This project brings together database engineering, exploratory analysis, feature engineering, classification, regression, clustering, NLP, text retrieval, and PEFT fine-tuning in a predictive-maintenance setting.

It emphasizes evaluation under class imbalance, leakage prevention, comparison of multiple models, reproducible artifacts, and honest documentation of experimental limitations. The next priority is to complete and validate the generative RAG integration and ensure the `ask` command provides a coherent answer grounded in both structured results and retrieved maintenance text.
