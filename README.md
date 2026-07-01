# UN Speech Sentiment and Palestine Voting Behavior

This project analyzes United Nations General Debate speeches and General Assembly voting records to study how rhetoric about Palestine relates to voting behavior on Palestine-related resolutions.

## Datasets

- `Data/TXT/`
  - Raw UN General Debate speech text files, organized by General Assembly session and year.
  - Each file contains one country speech and uses the country code in the filename.
  - Source: [UN General Debate Corpus on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y).


- `Data/2025_9_19_ga_voting.csv`
  - [UN General Assembly voting dataset](https://digitallibrary.un.org/record/4060887/files/2025_09_19_ga_voting_md.md) covering resolutions from 1946 to 2025.
  - The notebook filters this data to Palestine-related resolutions using title and agenda keywords.
  - Votes from 2025 are excluded because corresponding speeches are not available in the speech dataset.

- `Data/UNSD — Methodology.csv`
  - UN Statistics Division country and region metadata.
  - Used to add country names, ISO codes, and regional groupings to the speech and recognition datasets.
  - Source: [UNSD M49 methodology overview](https://unstats.un.org/unsd/methodology/m49/overview/).

- `Data/countries-that-recognize-palestine-2025.csv`
  - Dataset of countries that recognize Palestine and their recognition dates.
  - Used to create a time-aware `recognized_palestine` feature for each country-year.
  - Source: [World Population Review country ranking](https://worldpopulationreview.com/country-rankings/countries-that-recognize-palestine).

- `Data/Senticnet/senticnet.xlsx`
  - [SenticNet](https://sentic.net/downloads/) sentiment lexicon.
  - Used to extend VADER sentiment scoring with additional words found through TF-IDF in the Palestine-related speech corpus.


## What the Notebook Does

- Loads the consolidated UN speech dataset from `Data/AllSpeeches.csv`.

- Extracts Palestine-related speech content using keywords such as `gaza`, `west bank`, `palestine`, `palestinian`, `two-state`, `east jerusalem`, and `plo`.

- Counts Palestine-related keyword mentions for each country speech.

- Uses TF-IDF to identify important words in Palestine-related sentences.

- Extends VADER sentiment scoring with SenticNet polarity values for relevant words that are missing from VADER.

- Computes sentiment scores for Palestine-related sentences and classifies each speech as positive, neutral, or negative.

- Merges speech data with UNSD country and region metadata.

- Filters UN General Assembly voting records to Palestine-related resolutions.

- Keeps resolutions where Israel voted `No`, so a `Yes` vote can be interpreted as support for the Palestine-related resolution.

- Aggregates each country's yearly Palestine-related voting behavior into a `yes_pcg` value.

- Adds Palestine recognition timing and creates a yearly `recognized_palestine` feature.

- Performs exploratory data analysis on:
  - Palestine mentions over time.
  - Sentiment trends over time.
  - Voting behavior by country and year.
  - Voting behavior by Palestine recognition status.
  - Regional and geographic patterns.

- Builds classification models to predict whether a country-year has a majority `Yes` voting pattern on Palestine-related resolutions.

- Compares Random Forest, Gaussian Naive Bayes, and Logistic Regression models.

- Performs grid search for the Random Forest model.

- Evaluates the final model using accuracy, classification reports, confusion matrix, feature importance, and ROC/AUC analysis.

## Outputs

Generated figures are saved in `figs/`, including:

- Sentiment and Palestine keyword trends.
- Palestine recognition and voting behavior plots.
- Choropleth maps for mentions and voting.
- Mention distribution boxplots.
- Random Forest feature importance.
- Confusion matrix.

## Rebuilding the Speech Dataset

To rebuild `Data/AllSpeeches.csv` from the raw text files:

```bash
python scripts/build_all_speeches_csv.py
```

To write the generated CSV to a different path:

```bash
python scripts/build_all_speeches_csv.py --output path/to/AllSpeeches.csv
```

## Environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```
