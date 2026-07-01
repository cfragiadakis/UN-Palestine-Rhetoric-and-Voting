# UN Speech Sentiment and Palestine Voting Behavior

This repository contains the implementation of Assignment 1 of the course Fundamentals of Data Science, completed as part of the [MSc Information Studies](https://www.uva.nl/en/programmes/masters/information-studies/information-studies.html) programme at the [University of Amsterdam](https://www.uva.nl/en). It contains the code of the results and analysis that are reported in `Rhetoric_and_Resolution__Using_UN_Speech_Sentiment_to_Predict_Voting_Behavior_on_Palestine.pdf`

This project analyzes United Nations General Debate speeches and General Assembly voting records to compare countries’ Palestine-related rhetoric with their voting behavior on Palestine-related resolutions.

## Datasets used

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

The preprocessing pipeline prepares the dataset used for the analysis.

- Consolidates the raw UN General Debate speech text files into a tabular dataset.

- Extracts Palestine-related speech content using keywords and counts Palestine mentions for each country speech.

- Uses TF-IDF to identify important words in Palestine-related sentences.

- Extends VADER sentiment scoring with SenticNet polarity values for relevant words that are missing from VADER lexicon.

- Computes sentiment scores for Palestine-related sentences and classifies each speech as positive, neutral, or negative.

- Filters UN General Assembly voting records to Palestine-related resolutions using keywords in resolution titles and agenda items.

- Keeps resolutions where Israel voted `No`, so a `Yes` vote can be interpreted as support for the Palestine-related resolution.


## Analysis and Outputs

After preprocessing, the analysis notebook `UN Palestine Research.ipynb` can be run.

The notebook uses the preprocessed datasets to study how Palestine-related rhetoric, voting behavior, and recognition status relate to each other over time. The exploratory analysis first examines trends in Palestine-related mentions and sentiment, then compares these patterns with country-year voting behavior on Palestine-related resolutions.

The analysis also considers differences between countries that recognize Palestine and those that do not, as well as regional and geographic variation in mentions, sentiment, recognition, and voting patterns. Finally, a classification model is implemented to predict whether a country-year has a majority `Yes` voting pattern on Palestine-related resolutions. A Random Forest fine-tuned with grid search is created, evaluated with accuracy, classification reports, a confusion matrix, feature importance, and ROC/AUC analysis.


## Rebuilding the Speech Dataset

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

To rebuild `Data/AllSpeeches.csv` from the raw text files:

Download the required raw data from the links provided and place them under a `Data` folder. 

```bash
python scripts/build_all_speeches_csv.py
```


Contributors:
* Christoforos Fragkiadakis
* Konstantinos Koutris
* Max Johnston
* Berk Bahcetepe
