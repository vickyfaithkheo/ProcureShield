import re
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fuzzy_match(string1: str, string2: str) -> int:
    if string1 == "" or string2 == "":
        return 0

    string1 = "".join(string1.lower().split(" "))
    string2 = "".join(string2.lower().split(" "))
    string1 = re.sub(r"[^a-zA-Z0-9\s]", "", string1)
    string2 = re.sub(r"[^a-zA-Z0-9\s]", "", string2)

    return fuzz.token_sort_ratio(string1, string2)


def is_similar(string_1: str, string_2: str, threshold: int = 80) -> bool:
    return fuzzy_match(string_1, string_2) > threshold


def build_pairwise_similarity_table(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    work_df = df.copy()
    texts = work_df["Full Text"].fillna("").tolist()

    if len(texts) < 2:
        return pd.DataFrame(columns=["File A", "File B", "Similarity", "Flag"])

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(matrix)

    rows = []
    for i in range(len(work_df)):
        for j in range(i + 1, len(work_df)):
            score = float(sim_matrix[i, j])
            rows.append({
                "File A": work_df.loc[i, "Filename"],
                "File B": work_df.loc[j, "Filename"],
                "Similarity": round(score, 4),
                "Flag": "TRUE" if score >= threshold else "FALSE"
            })

    pairwise_df = pd.DataFrame(rows)
    if not pairwise_df.empty:
        pairwise_df = pairwise_df.sort_values("Similarity", ascending=False)

    return pairwise_df