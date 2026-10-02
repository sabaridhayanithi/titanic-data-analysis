# Can Data Predict Human Fate?

A reproducible exploratory data analysis project for the Kaggle Titanic passenger dataset. The presentation frames the analysis as an investigation: which passenger characteristics were associated with survival?

## Project contents

- `src/analyze.py` loads the Kaggle training CSV, reports data quality and descriptive statistics, runs the planned hypothesis tests, and saves editable PNG charts and a JSON results file.
- `slides/storyboard.md` records the ten-slide, five-minute narrative and four appendix slides.
- `output/Titanic_Survival_Analysis_Classroom.pptx` is the editable presentation, with embedded chart data and speaker notes.
- `requirements.txt` lists the Python dependencies.

## Get the data

Download `train.csv` from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data) after accepting the competition rules. Place it at `data/train.csv`. The script does not redistribute the dataset.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/analyze.py --data data/train.csv --out outputs
```

The script writes `outputs/summary.json` and charts. It uses complete observed values for each comparison, reports sample sizes, and avoids claiming causation. Missing age values are not imputed for age tests; missing embarkation values are excluded from the relevant count. Cabin missingness is reported as a data-quality feature.

## Analysis scope

The project covers dataset shape and types, missingness and duplicates, survival rates by sex/class/age/fare, age/fare descriptive statistics, a correlation matrix, a chi-square test of passenger class versus survival, and a Welch t-test comparing age by survival outcome. Statistical tests describe associations in this sample; they do not establish cause. The Kaggle training file is a sample and should not be treated as a complete passenger manifest.

## Story

The central question is not simply “who survived?” but “what pattern of access and circumstances appears in the records?” The deck should lead with the question, introduce the evidence, reveal differences by sex and class, then qualify the story with data limitations and statistical tests. Replace every provisional statement in the storyboard with the generated results before presenting.
