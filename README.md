# UN Speech Sentiment and Palestine Voting Behavior

This repository contains the implementation of the analysis of Palestine-related rhetoric in United Nations General Debate speeches and its relationship with voting behavior on Palestine-related United Nations General Assembly resolutions. The preprocessing pipeline, exploratory analysis, and predictive modeling form the basis of the accompanying report, [`Palestine Rhetoric_and_Resolution.pdf`](Palestine_Rhetoric_and_Resolution.pdf).

## Project Overview

This project combines United Nations General Debate speeches, General Assembly voting records, country-region metadata, and Palestine recognition data to compare diplomatic rhetoric with voting behavior. It was developed in relation to the [2025 United Nations General Assembly theme, “Better together: 80 years and more for peace, development and human rights”](https://www.un.org/en/ga/80/), and is connected to [Sustainable Development Goal 16: Peace, Justice and Strong Institutions](https://www.un.org/sustainabledevelopment/peace-justice/).

The project investigates whether Palestine-related rhetoric in United Nations General Debate speeches can help explain or predict country-level voting behavior on Palestine-related United Nations General Assembly resolutions.

The analysis focuses on two main questions:

1. Are there patterns in Palestine-related mentions, sentiment, geography, and recognition status that relate to countries’ voting behavior?
2. Can speech-based and contextual features predict whether a country-year shows majority support for Palestine-related resolutions?

In this project, a country is treated as supporting Palestine-related resolutions in a given year when it voted `Yes` in more than 50% of the Palestine-related resolutions for that year. The predictive task was formulated as a binary classification problem, where the model predicted whether a country in a specific year showed majority support for Palestine-related resolutions.

## Key Findings
* Palestine-related mentions and geographic region were stronger predictors of voting behavior than sentiment polarity.
* Countries in Africa, Asia, and Latin America generally showed higher support for Palestine-related resolutions.
* Recognition of Palestine was related to voting behavior, but had lower predictive importance than mentions and region.
* The fine-tuned Random Forest classifier achieved an F1-score of 53% for the imbalanced `No/Other` class and an overall accuracy of 83.5%.


## Data

- `Data/TXT/`
  - Raw [UN General Debate speech](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y) text files, organized by General Assembly session and year.

- `Data/2025_9_19_ga_voting.csv`
  - [UN General Assembly voting dataset](https://digitallibrary.un.org/record/4060887/files/2025_09_19_ga_voting_md.md) covering resolutions from 1946 to 2025.

- `Data/UNSD — Methodology.csv`
  - [UN Statistics Division](https://unstats.un.org/unsd/methodology/m49/overview/) country and region metadata, used to add country names, ISO codes, and regional groupings to the speech and recognition datasets.

- `Data/countries-that-recognize-palestine-2025.csv`
  - Dataset of countries that recognize Palestine and their recognition dates from [World Population Review](https://worldpopulationreview.com/country-rankings/countries-that-recognize-palestine).

- `Data/Senticnet/senticnet.xlsx`
  - [SenticNet](https://sentic.net/downloads/) sentiment lexicon, used to extend VADER sentiment scoring with additional words found through TF-IDF in the Palestine-related speech corpus.


## Preprocessing Steps

The preprocessing pipeline prepares the datasets used in the exploratory analysis and predictive modeling.

The main steps are:

- Consolidate the raw UN General Debate speech text files into a tabular dataset.

- Extract Palestine-related speech content using keywords and count Palestine mentions for each country speech.

- Identify informative Palestine-related terms using TF-IDF.

- Extend the VADER lexicon with polarity scores from SenticNet for relevant missing terms.

- Compute sentiment scores for Palestine-related sentences and classify each speech as positive, neutral, or negative.

- Filter UN General Assembly voting records to Palestine-related resolutions using keywords in resolution titles and agenda items.

- Keep resolutions where Israel voted `No`, so a `Yes` vote can be interpreted as support for the Palestine-related resolution.

- Merge speech, voting, recognition, and regional metadata into a country-year analysis dataset.


## Running the Project

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```
Download the required raw data from the links provided above and place the files under a `Data` folder.

To rebuild `Data/AllSpeeches.csv` from the raw UN General Debate text files, run:

```bash
python scripts/build_all_speeches_csv.py
```
Then run the preprocessing pipeline:

```bash
python scripts/preprocessing_datasets.py
```

This creates the required processed datasets:
`Data/analysis_dataset.csv`
`Data/palestine_voting_filtered.csv`


## Analysis and Outputs

After preprocessing, the analysis notebook can be run:
`UN Palestine Research.ipynb`

The notebook performs the exploratory analysis, visualizations, feature engineering, model training, model evaluation, feature-importance analysis, and ROC/AUC analysis.
The predictive task is formulated as a binary classification problem. For each country-year, the target variable indicates whether the country voted `Yes` in more than 50% of Palestine-related resolutions.


## Contributors
- [Christoforos Fragkiadakis](https://github.com/cfragiadakis)
- [Konstantinos Koutris](https://github.com/kkoutris)
- [Max Johnston](https://github.com/MaximusJ08)
- [Berk Bahcetepe](https://github.com/berkbahcetepe6)

1st Assignment for Fundamentals of Data Science 2025-2026 (Information Studies - Data Science track, UvA)
