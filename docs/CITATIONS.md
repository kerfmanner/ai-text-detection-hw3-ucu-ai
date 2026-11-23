# Citations

No code was copied from external sources; everything in `src/` and `notebooks/` was written for this project. The ideas, papers, models and tools used are listed below.

## Data

- Schaaf, Bönisch & Mehler. *GhostWriter* (LREC 2026). Processed subset provided by the competition, CC BY-NC-SA 4.0, course use only.

## Methods and ideas

| used in | source |
|---|---|
| hashed TF-IDF features (`tfidf.py`) | Weinberger, Dasgupta, Langford, Smola, Attenberg. *Feature Hashing for Large Scale Multitask Learning.* ICML 2009 |
| linear SVM text classifier (`tfidf.py`) | Joachims. *Text Categorization with Support Vector Machines.* ECML 1998; Fan, Chang, Hsieh, Wang, Lin. *LIBLINEAR.* JMLR 9, 2008 |
| Complement Naive Bayes (`tfidf.py`) | Rennie, Shih, Teevan, Karger. *Tackling the Poor Assumptions of Naive Bayes Text Classifiers.* ICML 2003 |
| near-duplicate grouping by TF-IDF cosine and shared openings (`split.py`), word 3-gram overlap (`eda.py`) | Broder. *On the Resemblance and Containment of Documents.* SEQUENCES 1997; Manning, Raghavan, Schütze. *Introduction to Information Retrieval*, ch. 6 and 19, 2008 |
| distinctive words by log-odds z-score (`eda.py`) | Monroe, Colaresi, Quinn. *Fightin' Words.* Political Analysis 16(4), 2008 (simplified: add-one smoothing, no informative prior) |
| multilingual encoder (`encoder.py`) | Conneau et al. *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R). ACL 2020 |
| window-and-aggregate handling of long texts (`encoder.py`) | Pappagari et al. *Hierarchical Transformers for Long Document Classification.* ASRU 2019; Sun, Qiu, Xu, Huang. *How to Fine-Tune BERT for Text Classification?* CCL 2019 |
| AdamW, warmup + linear decay, mixed precision (`encoder.py`) | Loshchilov & Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019; Micikevicius et al. *Mixed Precision Training.* ICLR 2018 |
| token log-probability statistics (`lm.py`) | Gehrmann, Strobelt, Rush. *GLTR: Statistical Detection and Visualization of Generated Text.* ACL 2019 demo |
| observer/performer cross-perplexity ratio (`lm.py`) | Hans et al. *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text.* ICML 2024 |
| segment-level likelihood spread for partly rewritten text (`lm.py`) | Wang et al. *SeqXGPT: Sentence-Level AI-Generated Text Detection.* EMNLP 2023 (idea only) |
| stacking with out-of-fold predictions (`ensemble.py`: X4, S1) | Wolpert. *Stacked Generalization.* Neural Networks 5(2), 1992 |
| error overlap / diversity of ensemble members (`runs.error_overlap`) | Kuncheva & Whitaker. *Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy.* Machine Learning 51, 2003 |
| label cascade, coarse then fine decision (`tfidf.run_cascade`) | Silla & Freitas. *A Survey of Hierarchical Classification Across Different Application Domains.* Data Mining and Knowledge Discovery 22, 2011 |
| character n-gram features for style (X6) | Stamatatos. *A Survey of Modern Authorship Attribution Methods.* JASIST 60(3), 2009 |
| text distortion / masked n-grams (X8) | Stamatatos. *Authorship Attribution Using Text Distortion.* EACL 2017 |
| histogram gradient boosting (`lm.py`, `ensemble.py`) | Ke et al. *LightGBM.* NeurIPS 2017 (scikit-learn `HistGradientBoostingClassifier` is based on it) |

## Pretrained weights (open-weight, via Hugging Face)

- `FacebookAI/xlm-roberta-base` (MIT)
- `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0). Qwen Team. *Qwen2.5 Technical Report.* arXiv:2412.15115, 2024

## Software

- scikit-learn: Pedregosa et al. JMLR 12, 2011 (`StratifiedGroupKFold`, `HashingVectorizer`, `LinearSVC`, `HistGradientBoostingClassifier`, metrics)
- PyTorch: Paszke et al. NeurIPS 2019
- Hugging Face Transformers: Wolf et al. EMNLP 2020 demo
- pandas, NumPy, SciPy, matplotlib, PyArrow, joblib, tqdm, PyYAML
- uv (Astral) for the environment; Kaggle CLI for the data download
