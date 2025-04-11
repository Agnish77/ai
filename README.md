
# Vyorius AI Moderation & Automation Intern Task

## Objective
This project is a Python application that reads user comments from a local file (CSV/JSON), uses an open-source generative AI model to detect offensive or inappropriate content, and generates a report of flagged comments.

---

## Features
- Load and process comment data
- Detect offensive content using a pre-trained model (`unitary/toxic-bert`)
- Classify offense types (e.g., toxic, threat, obscene, insult, identity hate)
- Generate reports and charts
- Export analyzed data to CSV
- Command-line interface support

---

## Setup Instructions

### 1. Clone the repo (if applicable) or download the script.

### 2. Install dependencies:
```bash
pip install transformers pandas torch matplotlib
```

### 3. Run the script:
```bash
python open_source_moderation.py --file comments.csv --output results.csv
```

---

## Input File Format

Input file should be in CSV or JSON format with the following fields:
- `comment_id`
- `username`
- `comment_text`

Example:
```csv
comment_id,username,comment_text
1,user123,"You’re an idiot, go away."
2,peaceful_one,"I love how this was handled."
```

---

## Output
- CSV file with flagged comments and explanations
- Summary in terminal
- Bar chart of offense type distribution saved as `offense_distribution_open_source.png`

---

## Sample Files
- ✅ `comments.csv`: Sample input file  
- ✅ `results_open_source.csv`: Output with moderation results  
- ✅ `offense_distribution_open_source.png`: Offense type breakdown

---

## Offense Types
Detected categories include:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

---

## Author
Agnish Paul
