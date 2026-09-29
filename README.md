# Native-Language Label Sensitivity in English Grammatical Error Correction

*A Counterfactual Audit of Qwen 3.5 and Gemma 3*

This project investigates whether **changing only the stated native-language (L1) label** affects how local Large Language Models (LLMs) correct the same English learner sentence. The study performs a controlled counterfactual audit using **Qwen 3.5 4B** and **Gemma 3 4B** running locally through **Ollama**.

> Research Project – Trends in Natural Language Processing (Summer 2026), Universität Trier

---

## Authors

- Eldho Saji
- Jeevan George Mathew

---

## Overview

Grammatical Error Correction (GEC) systems often receive learner metadata such as a user's native language. This project asks a simple but important question:

> **If the English sentence never changes, can changing only the stated first-language label change the model's correction?**

To answer this, we keep every experimental condition constant except the L1 label and compare the corrections produced by two local LLMs.

---

## Research Objectives

- Measure whether LLM corrections are sensitive to native-language labels.
- Compare sensitivity across different sentence types.
- Examine whether different models become sensitive on the same sentences.
- Provide a reproducible local evaluation pipeline.

---

## Experimental Design

### Controlled Variables

- English sentence
- Correction prompt
- Decoding settings (Temperature = 0)
- Thinking mode disabled
- Model configuration

### Changed Variable

Only the **stated L1 label** changes.

L1 conditions:

- German
- Malayalam
- Arabic
- Unknown

### Models

- Qwen 3.5 4B
- Gemma 3 4B

Both models are executed locally using **Ollama**.

---

## Dataset

The experiment uses sentences sampled from the **BEA-2019 W&I + LOCNESS** B-level development dataset.

Filtering process:

| Stage | Count |
|-------|------:|
| Original M2 blocks | 1,290 |
| Eligible sentences | 963 |
| Manually reviewed | 150 |
| Final sample | **60** |

Final composition:

- 40 clear errors
- 10 ambiguous errors
- 10 correct controls

---

## Methodology

For each sentence:

1. Keep the English sentence unchanged.
2. Change only the stated L1 label.
3. Send the same prompt to the model.
4. Collect corrections for all four L1 conditions.
5. Normalize formatting.
6. Compare the resulting correction strings.

A sentence is considered **label-sensitive** when the four normalized outputs contain **at least two distinct correction strings**.

### Experimental Scale

- 60 sentences
- 4 L1 labels
- 2 models

**Total requests:** **480**

---

## Project Workflow

```text
English Sentence
       │
       ▼
Change only L1 Label
(DE / ML / AR / Unknown)
       │
       ▼
Local LLM (Ollama)
 ├── Qwen 3.5 4B
 └── Gemma 3 4B
       │
       ▼
Normalize Output
       │
       ▼
Compare Correction Strings
       │
       ▼
Label-Sensitive?
```

---

## Project Structure

```text
L1_label_sensitivity/
│
├── data/
│   ├── input_sentences.csv
│   ├── sampled_sentences.csv
│   └── model_outputs.csv
│
├── prompts/
│   └── correction_prompt.txt
│
├── scripts/
│   ├── run_experiment.py
│   ├── normalize_outputs.py
│   ├── analyze_results.py
│   └── annotate_results.py
│
├── results/
│   ├── qwen_results.csv
│   ├── gemma_results.csv
│   ├── figures/
│   └── tables/
│
├── poster/
│   └── research_poster.pdf
│
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/EldhoSaji55/L1_label_sensitivity.git
cd L1_label_sensitivity
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install Ollama

Download Ollama from:

<https://ollama.com>

Pull the required models:

```bash
ollama pull qwen3.5:4b
ollama pull gemma3:4b
```

---

## Running the Experiment

Run the correction pipeline:

```bash
python scripts/run_experiment.py
```

Normalize outputs:

```bash
python scripts/normalize_outputs.py
```

Generate analysis:

```bash
python scripts/analyze_results.py
```

---

## Output Format

Example input:

| sentence_id | language | sentence |
|-------------|----------|----------|
| 001 | German | I goes to school yesterday. |

Example output:

| sentence_id | model | language | qwen_answer |
|-------------|--------|----------|-------------|
| 001 | Qwen 3.5 | German | I went to school yesterday. |

---

## Results Summary

| Metric | Qwen | Gemma |
|--------|------:|------:|
| Overall label sensitivity | 43.3% | 43.3% |
| Sensitive sentences | 26/60 | 26/60 |

### Category Findings

- **Qwen** showed the highest sensitivity on ambiguous errors.
- **Gemma** showed higher sensitivity on clear errors.
- Only part of the sensitive sentence set overlaps between models, indicating model-specific behavior.

---

## Key Findings

- Native-language labels can influence grammatical corrections even when the English sentence remains identical.
- Different LLMs respond differently to the same metadata.
- Equal overall sensitivity rates can conceal different sentence-level behaviors.

---

## Limitations

- 60-sentence sample
- Four L1 labels
- Two 4B local models
- One prompt configuration
- Textual differences alone do not necessarily indicate errors or bias.

---

## Technologies Used

- Python
- Ollama
- Qwen 3.5 4B
- Gemma 3 4B
- CSV processing
- Unicode normalization
- Counterfactual evaluation

---

## Citation

If you use this repository, please cite the accompanying research poster:

> Eldho Saji & Jeevan George Mathew (2026). *Native-Language Label Sensitivity in English Grammatical Error Correction: A Counterfactual Audit of Qwen 3.5 and Gemma 3.* Trends in Natural Language Processing, Universität Trier.

---

## License

This project is intended for academic and research purposes. Please ensure that any use of the BEA-2019 W&I + LOCNESS dataset complies with its original licensing terms.
