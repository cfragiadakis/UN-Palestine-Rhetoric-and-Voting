"""
This script performs the preprocessing steps: Palestine keyword extraction, sentiment scoring, country metadata
joins, Palestine-related voting aggregation, and Palestine recognition features.

Run from the project root with:
    python scripts/preprocessing_datasets.py
"""

from argparse import ArgumentParser
from pathlib import Path
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


PALESTINE_RELATED_KEYWORDS = [
    "gaza",
    "west bank",
    "palestine",
    "palestinian",
    "palestinians",
    "two-state",
    "two state",
    "east jerusalem",
    "plo",
]


def extract_palestine_sentences(speech: str, keywords: list[str]) -> list[str]:
    sentences = sent_tokenize(speech)
    return [sentence for sentence in sentences if any(k in sentence.lower() for k in keywords)]


def count_palestine_keywords(text: str, keywords: list[str]) -> int:
    if not isinstance(text, str):
        return 0

    text_lower = text.lower()
    return sum(text_lower.count(keyword) for keyword in keywords)


def tfidf_words(df: pd.DataFrame, stop_words_list: set[str], top_n: int = 200) -> pd.DataFrame:
    documents = []

    for sentences in df["palestine_sentences"].dropna():
        text = " ".join(sentences)
        text = re.sub(r"[^\w\s]", "", text).lower()
        text = re.sub(r"\d+", "", text)
        documents.append(text)

    vectorizer = TfidfVectorizer(stop_words=list(stop_words_list))
    tfidf_matrix = vectorizer.fit_transform(documents)

    scores = []
    for i, word in enumerate(vectorizer.get_feature_names_out()):
        word_scores = tfidf_matrix[:, i].toarray().reshape(-1)
        scores.append((word, word_scores.mean()))

    return (
        pd.DataFrame(scores, columns=["word", "score"])
        .sort_values("score", ascending=False)
        .head(top_n)
    )


def build_sentiment_analyzer(df_speech: pd.DataFrame, senticnet_path: Path) -> SentimentIntensityAnalyzer:
    senticnet = pd.read_excel(senticnet_path)
    stop_words = set(stopwords.words("english"))

    additional_stopwords = ["also", "ha", "middle", "east", "united", "nations"]
    palestine_keyword_words = [
        word for phrase in PALESTINE_RELATED_KEYWORDS for word in phrase.split()
    ]
    stop_words.update(additional_stopwords)
    stop_words.update(palestine_keyword_words)

    top_words = tfidf_words(df_speech, stop_words, top_n=200)
    sentiment_analyzer = SentimentIntensityAnalyzer()

    exclude_words = [
        "world",
        "general",
        "bank",
        "government",
        "assembly",
        "countries",
        "organization",
        "order",
        "situation",
        "conditions",
        "president",
        "east",
        "fully",
        "long",
        "need",
        "international",
        "internationally",
        "deployment",
    ]

    assigned_words = 0
    for word in top_words["word"]:
        if word in exclude_words or word in sentiment_analyzer.lexicon:
            continue

        senticnet_word = senticnet[senticnet["CONCEPT"] == word]
        if senticnet_word.empty:
            continue

        polarity_score = senticnet_word.iloc[0]["POLARITY INTENSITY"]
        sentiment_analyzer.lexicon[word] = 4 * polarity_score
        assigned_words += 1

    print(f"Enhanced VADER with {assigned_words} SenticNet words.")
    return sentiment_analyzer


def sentiment_score(sentences: list[str], analyzer: SentimentIntensityAnalyzer) -> float | None:
    if not sentences:
        return None

    scores = [analyzer.polarity_scores(sentence)["compound"] for sentence in sentences]
    return sum(scores) / len(scores)


def classify_sentiment(compound_score: float | None) -> str:
    if pd.isna(compound_score):
        return "NEUTRAL"
    if compound_score > 0.05:
        return "POSITIVE"
    if compound_score < -0.05:
        return "NEGATIVE"
    return "NEUTRAL"


def build_palestine_voting(voting_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    voting = pd.read_csv(voting_path, low_memory=False)
    voting["date"] = pd.to_datetime(voting["date"])
    voting["Year"] = voting["date"].dt.year
    voting = voting.dropna(subset="agenda_title")

    palestine_voting = voting[
        voting["title"].str.contains("Palest|Gaza|East Jerusalem", na=False)
        | voting["agenda_title"].str.contains("Palest|Gaza|East Jerusalem", na=False)
    ].copy()
    palestine_voting = palestine_voting.sort_values("date")

    palestine_voting = palestine_voting[
        [
            "undl_id",
            "ms_code",
            "ms_name",
            "ms_vote",
            "date",
            "session",
            "resolution",
            "title",
            "agenda_title",
            "total_yes",
            "total_no",
            "total_abstentions",
            "total_non_voting",
            "total_ms",
            "Year",
        ]
    ]

    israel_no_resolutions = set(
        palestine_voting[
            (palestine_voting["ms_code"] == "ISR") & (palestine_voting["ms_vote"] == "N")
        ]["resolution"]
    )
    palestine_voting = palestine_voting[
        palestine_voting["resolution"].isin(israel_no_resolutions)
    ]
    palestine_voting = palestine_voting[
        (palestine_voting["Year"] < 2025) & (palestine_voting["Year"] >= 1969)
    ].copy()

    yes_percentage = (
        palestine_voting.groupby(["ms_code", "Year"])
        .agg(
            total_votes=("ms_vote", "count"),
            yes_votes=("ms_vote", lambda votes: (votes == "Y").sum()),
        )
        .reset_index()
    )
    yes_percentage["yes_pcg"] = yes_percentage["yes_votes"] / yes_percentage["total_votes"]

    return palestine_voting, yes_percentage


def add_recognition_features(
    df_speech: pd.DataFrame,
    recognition_path: Path,
    unsd: pd.DataFrame,
) -> pd.DataFrame:
    recognize_palestine = pd.read_csv(recognition_path)
    recognize_palestine = recognize_palestine.merge(
        unsd[["Region Name", "ISO-alpha2 Code", "ISO-alpha3 Code", "Country or Area"]],
        left_on="flagCode",
        right_on="ISO-alpha2 Code",
    )
    recognize_palestine = recognize_palestine.drop(
        columns=["flagCode", "Country or Area"]
    ).rename(columns={"countriesThatRecognizePalestine_dateRecog": "recognition_date"})

    recognize_palestine["recognition_date"] = pd.to_datetime(
        recognize_palestine["recognition_date"],
        format="%m-%d-%Y",
    )
    recognize_palestine["recognition_year"] = recognize_palestine["recognition_date"].dt.year

    df_speech = df_speech.merge(
        recognize_palestine[["recognition_year", "ISO-alpha3 Code"]],
        left_on="ms_code",
        right_on="ISO-alpha3 Code",
        how="left",
    )
    df_speech = df_speech.drop(columns="ISO-alpha3 Code")

    df_speech["recognized_palestine"] = (
        df_speech["recognition_year"].notna()
        & (df_speech["Year"] >= df_speech["recognition_year"])
    )

    return df_speech


def build_analysis_dataset(
    data_dir: Path,
    analysis_output_path: Path,
    voting_output_path: Path,
) -> tuple[Path, Path]:
    df_speech = pd.read_csv(data_dir / "AllSpeeches.csv")

    df_speech["palestine_sentences"] = df_speech["Speech"].apply(
        lambda speech: extract_palestine_sentences(speech, PALESTINE_RELATED_KEYWORDS)
    )
    df_speech["palestine_sentences_count"] = df_speech["palestine_sentences"].apply(len)
    df_speech["palestine_keywords_count"] = df_speech["Speech"].apply(
        lambda speech: count_palestine_keywords(speech, PALESTINE_RELATED_KEYWORDS)
    )

    analyzer = build_sentiment_analyzer(df_speech, data_dir / "Senticnet" / "senticnet.xlsx")
    df_speech["Speech_Compound"] = df_speech["palestine_sentences"].apply(
        lambda sentences: sentiment_score(sentences, analyzer)
    )
    df_speech["Sentiment"] = df_speech["Speech_Compound"].apply(classify_sentiment)

    unsd = pd.read_csv(data_dir / "UNSD — Methodology.csv", sep=";")
    df_speech = df_speech.merge(
        unsd[["Country or Area", "ISO-alpha3 Code", "Region Name"]],
        on="ISO-alpha3 Code",
    )

    palestine_voting, yes_percentage = build_palestine_voting(
        data_dir / "2025_9_19_ga_voting.csv"
    )

    df_speech = df_speech.reset_index().rename(
        columns={"index": "id", "ISO-alpha3 Code": "ms_code"}
    )
    df_speech = df_speech.merge(yes_percentage, on=["ms_code", "Year"], how="left")
    df_speech = add_recognition_features(
        df_speech,
        data_dir / "countries-that-recognize-palestine-2025.csv",
        unsd,
    )

    analysis_columns = [
        "id",
        "Session",
        "Year",
        "ms_code",
        "Country or Area",
        "Region Name",
        "palestine_sentences_count",
        "palestine_keywords_count",
        "Speech_Compound",
        "Sentiment",
        "total_votes",
        "yes_votes",
        "yes_pcg",
        "recognition_year",
        "recognized_palestine",
    ]

    analysis_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_speech[analysis_columns].to_csv(analysis_output_path, index=False)
    palestine_voting.to_csv(voting_output_path, index=False)

    print(f"Saved analysis dataset to {analysis_output_path}")
    print(f"Saved filtered Palestine voting dataset to {voting_output_path}")
    return analysis_output_path, voting_output_path


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "Data",
        help="Directory containing the project data files.",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=project_root / "Data" / "analysis_dataset.csv",
        help="Path for the generated EDA/modeling dataset.",
    )
    parser.add_argument(
        "--voting-output",
        type=Path,
        default=project_root / "Data" / "palestine_voting_filtered.csv",
        help="Path for the generated filtered voting dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_analysis_dataset(args.data_dir, args.analysis_output, args.voting_output)
