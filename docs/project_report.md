# Robot Failure Prediction System

## 1. Project Overview

This project develops a machine-learning system for analyzing industrial machine operating conditions and predicting machine failures. It combines data analysis, PostgreSQL storage, feature engineering, supervised learning, clustering, and a small NLP experiment.

## 2. Dataset

The project uses the AI4I 2020 Predictive Maintenance dataset.

- Records: 10,000
- Original columns: 14
- Failure cases: 339 (3.39%)
- Non-failure cases: 9,661 (96.61%)
- Missing values: None in the original dataset

The dataset includes machine type, air and process temperatures, rotational speed, torque, tool wear, the machine-failure target, and failure-cause indicators.

## 3. Database

The project loads the data into PostgreSQL. The database contains tables for machines, sensor readings, failure events, event causes, and machine clusters.

The database supports structured storage and analysis of the predictive-maintenance data.

## 4. Data Analysis and Feature Engineering

Exploratory analysis examines the dataset's structure, distributions, and failure-class imbalance.

Three additional features were engineered:

- Temperature difference: process temperature minus air temperature.
- Mechanical power proxy: rotational speed multiplied by torque.
- Tool-wear–torque interaction: tool wear multiplied by torque.

The target variable, failure-cause indicators, and identifier columns are excluded from the model inputs to reduce target leakage and avoid using identifiers as predictive measurements. Numeric and categorical preprocessing is handled in the modeling pipeline.

## 5. Supervised Learning

Several classifiers were compared, including logistic regression, linear and RBF support vector machines, a decision tree, random forest, AdaBoost, and a dummy baseline.

The main training run selected Random Forest as its champion model.

| Metric | Random Forest |
|---|---:|
| F1-score | 0.8430 |
| Recall | 0.7500 |
| PR-AUC | 0.8717 |
| ROC-AUC | 0.9780 |

On the 2,000-row test set, the reported confusion matrix was:

- True negatives: 1,930
- False positives: 2
- False negatives: 17
- True positives: 51

Because failures represent only 3.39% of the original dataset, accuracy alone would be insufficient to evaluate this task. Recall and precision-recall performance are also important. The reported results are experimental results on this dataset, not a guarantee of real-world performance.

## 6. Feature-Engineering Experiment

A separate experiment compared the original features with the original features plus all three engineered features.

| Experiment | F1-score | Recall | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|
| Original features | 0.6078 | 0.4559 | 0.7762 | 0.9636 |
| Original + engineered features | 0.8333 | 0.7353 | 0.8721 | 0.9781 |

In this experiment, the engineered features improved the reported metrics. These results belong to the separate feature experiment and should not be mixed with the main model-comparison results.

## 7. Unsupervised Learning: K-Means

K-Means clustering was evaluated for K values from 2 through 6. The highest reported silhouette score was 0.2103 at K = 4.

The resulting four clusters describe different operating profiles, including differences in tool wear, rotational speed, torque, and temperature.

These clusters identify patterns in machine operating conditions. They do not, by themselves, establish that one cluster has a higher failure risk.

## 8. NLP Experiment

Machine descriptions were generated from sensor measurements and represented using TF-IDF. A text-based logistic regression model was compared with a sensor-feature logistic regression model.

| Model | F1-score | Recall | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|
| Sensor features | 0.2978 | 0.8824 | 0.4297 | 0.9385 |
| TF-IDF text | 0.4000 | 0.6765 | 0.4277 | 0.8975 |

The text model had a higher F1-score in this experiment, while the sensor model had higher recall, PR-AUC, and ROC-AUC. The text descriptions were generated from the same sensor measurements, so they do not provide independent information.

## 9. Generated Artifacts

The project generates analysis reports, model comparisons, feature-experiment results, clustering results, NLP comparison results, and a serialized champion model.

Key output files include:

- `reports/analysis_report.txt`
- `reports/cluster_report.txt`
- `reports/feature_experiment.csv`
- `reports/feature_ablation.csv`
- `reports/nlp_model_comparison.csv`
- `models/model_comparison.csv`
- `models/model_report.txt`
- `models/champion_model.joblib`

## 10. How to Run

Run these commands from the project root in an environment with the project's requirements installed and PostgreSQL configured:

```powershell
python main.py --help
python main.py analyze
python main.py train
python main.py cluster
python -m src.nlp
```

Run `python main.py setup` only when the database setup or data-loading step is required.

## 11. Limitations and Future Work

- Failure cases are a small minority of the dataset.
- Evaluation on this benchmark does not establish performance on real factory equipment.
- Clustering describes operating patterns and does not directly predict cluster-specific risk.
- Generated text is derived from sensor measurements rather than independent text observations.
- Further validation on independent operational data would be needed before practical deployment.

## Conclusion

The project brings together a PostgreSQL data pipeline, exploratory analysis, engineered sensor features, supervised failure classification, unsupervised operating-profile discovery, and an NLP comparison. The results demonstrate an end-to-end predictive-maintenance workflow while highlighting the need for careful evaluation under class imbalance.
