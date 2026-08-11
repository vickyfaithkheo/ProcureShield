from io import BytesIO
from pathlib import Path
from datetime import datetime
import pandas as pd
import pytz
from pypdf import PdfReader
from pypdf.errors import PdfReadError


def parse_pdf_date(dt_str: str) -> str:
    if not dt_str:
        return ""

    try:
        val = str(dt_str).strip().replace("D:", "").replace("'", "")
        if "Z" in val:
            val = val.split("Z")[0]

        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            return ""

        if dt.tzinfo is not None:
            dt = dt.tz_convert("Asia/Singapore")
        else:
            dt = dt.tz_localize("Asia/Singapore")

        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def parse_file_id(name_of_file: str) -> dict:
    result_fields = ["Has Identifiers", "Tender ID", "Vendor ID", "Reference Name"]
    identifiers = name_of_file.split("_")[:3]

    if len(identifiers) < 3 or all(x == "" for x in identifiers[:2]):
        return dict(zip(result_fields, ["FALSE"] + [""] * 3))

    return dict(zip(result_fields, ["TRUE"] + [x for x in identifiers]))


def extract_metadata(reader: PdfReader) -> dict:
    doc_metadata = reader.metadata or {}
    field_mapping = {
        "Author": "/Author",
        "PDF Application": "/Creator",
        "PDF Producer": "/Producer",
        "Created Date": "/CreationDate",
        "Modified Date": "/ModDate",
        "Title": "/Title",
        "Subject": "/Subject",
        "Keywords": "/Keywords",
    }

    data = {k: doc_metadata.get(v, "") for k, v in field_mapping.items()}

    for k in ("Created Date", "Modified Date"):
        data[k] = parse_pdf_date(data[k])

    return data


def extract_annotations(reader: PdfReader) -> dict:
    buffer = []

    if reader.pages:
        for page in reader.pages:
            annots = getattr(page, "annotations", None)
            if not annots:
                continue

            for annot in annots:
                if annot.get("/Subtype") in ("/Link", "/Popup"):
                    continue

                buffer.append({
                    "Annotation Type": annot.get("/Subtype", "unknown"),
                    "Author": annot.get("/T"),
                    "Position": annot.get("/Rect"),
                    "Created Date": annot.get("/CreationDate"),
                    "Modified Date": annot.get("/M"),
                })

    annot_summary = set(x.get("Annotation Type") for x in buffer) if buffer else set()

    return {
        "Annotations Summary": str(annot_summary),
        "Number of Annotations": len(buffer),
        "Annotation Details": str(buffer) if buffer else "",
    }


def extract_pdf_info_from_upload(uploaded_file) -> dict:
    try:
        reader = PdfReader(BytesIO(uploaded_file.read()))
    except PdfReadError:
        return {
            "Filename": uploaded_file.name,
            "Number of Pages": 0,
            "Has Identifiers": "",
            "Tender ID": "",
            "Vendor ID": "",
            "Reference Name": "",
            "Author": "",
            "PDF Application": "",
            "PDF Producer": "",
            "Created Date": "",
            "Modified Date": "",
            "Title": "",
            "Subject": "",
            "Keywords": "",
            "PDF Version": "",
            "Annotations Summary": "",
            "Number of Annotations": "",
            "Annotation Details": "",
            "Full Text": "",
        }

    full_text = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            full_text.append(page_text)

    return {
        "Filename": uploaded_file.name,
        "Number of Pages": len(reader.pages),
        **parse_file_id(Path(uploaded_file.name).stem),
        **extract_metadata(reader),
        "PDF Version": reader.pdf_header,
        **extract_annotations(reader),
        "Full Text": "\n".join(full_text).strip(),
    }