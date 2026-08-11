import pandas as pd


def derive_risk_flags(pdf_data: pd.DataFrame) -> pd.DataFrame:
    df = pdf_data.copy()

    def safe_prefix(val):
        val = "" if pd.isna(val) else str(val)
        return val[:10]

    df["Modified After Creation"] = df.apply(
        lambda row: str(not (
            safe_prefix(row.get("Created Date", "")) ==
            safe_prefix(row.get("Modified Date", ""))
        )).upper(),
        axis=1
    )

    df_author = df[df["Author"].notna() & (df["Author"] != "")]
    counts = df_author["Author"].value_counts()
    author_checks = counts[counts > 1].index
    df["Same Author Across Different Documents"] = df["Author"].isin(
        author_checks
    ).astype(str).str.upper()

    df_author_vendor = df[
        df["Has Identifiers"].notna() & (df["Has Identifiers"] != "") &
        df["Vendor ID"].notna() & (df["Vendor ID"] != "") &
        df["Author"].notna() & (df["Author"] != "")
    ]
    counts = df_author_vendor.groupby("Author")["Vendor ID"].nunique()
    author_vendor_check = counts[counts > 1].index

    df["Same Author Across Different Vendors"] = df.apply(
        lambda x: (
            "TRUE" if x["Has Identifiers"] == "TRUE" and
            x["Author"] in author_vendor_check else "FALSE"
        ),
        axis=1
    )

    df_appln_vendor = df[
        df["Has Identifiers"].notna() & (df["Has Identifiers"] != "") &
        df["Vendor ID"].notna() & (df["Vendor ID"] != "") &
        df["PDF Application"].notna() & (df["PDF Application"] != "")
    ]
    counts = df_appln_vendor.groupby("PDF Application")["Vendor ID"].nunique()
    appln_vendor_check = counts[counts > 1].index
    df["Same PDF Application Across Different Vendors"] = df.apply(
        lambda x: (
            "TRUE" if x["Has Identifiers"] == "TRUE" and
            x["PDF Application"] in appln_vendor_check else "FALSE"
        ),
        axis=1
    )

    df_producer_vendor = df[
        df["Has Identifiers"].notna() & (df["Has Identifiers"] != "") &
        df["Vendor ID"].notna() & (df["Vendor ID"] != "") &
        df["PDF Producer"].notna() & (df["PDF Producer"] != "")
    ]
    counts = df_producer_vendor.groupby("PDF Producer")["Vendor ID"].nunique()
    producer_vendor_check = counts[counts > 1].index
    df["Same PDF Producer Across Different Vendors"] = df.apply(
        lambda x: (
            "TRUE" if x["Has Identifiers"] == "TRUE" and
            x["PDF Producer"] in producer_vendor_check else "FALSE"
        ),
        axis=1
    )

    return df


def add_overall_risk_score(df: pd.DataFrame, pairwise_df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()

    work_df["Metadata Flags Count"] = (
        (work_df["Modified After Creation"] == "TRUE").astype(int)
        + (work_df["Same Author Across Different Documents"] == "TRUE").astype(int)
        + (work_df["Same Author Across Different Vendors"] == "TRUE").astype(int)
        + (work_df["Same PDF Application Across Different Vendors"] == "TRUE").astype(int)
        + (work_df["Same PDF Producer Across Different Vendors"] == "TRUE").astype(int)
    )

    work_df["High Similarity Pair Count"] = 0 if pairwise_df.empty else int((pairwise_df["Flag"] == "TRUE").sum())

    def risk_label(row):
        score = row["Metadata Flags Count"] + row["High Similarity Pair Count"]
        if score >= 4:
            return "High"
        if score >= 2:
            return "Medium"
        return "Low"

    work_df["Overall Risk"] = work_df.apply(risk_label, axis=1)
    return work_df


def process_uploaded_pdfs(pdf_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> pd.DataFrame:
    return add_overall_risk_score(pdf_df, pairwise_df)