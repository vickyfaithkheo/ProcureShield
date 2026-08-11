from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd
import fitz  # PyMuPDF


def _rects_intersect(a: fitz.Rect, b: fitz.Rect) -> bool:
    return not (
        a.x1 <= b.x0 or
        b.x1 <= a.x0 or
        a.y1 <= b.y0 or
        b.y1 <= a.y0
    )


def _is_white_fill(fill) -> bool:
    if not fill:
        return False
    try:
        rgb = list(fill[:3])
        return all(c >= 0.95 for c in rgb)
    except Exception:
        return False


def analyze_pdf_tamper(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Page-level tamper screening.
    This flags suspicious visual editing patterns, not proof of fraud.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    rows: List[Dict[str, Any]] = []

    for page_index in range(len(doc)):
        page = doc[page_index]

        text = page.get_text("text").strip()
        images = page.get_images(full=True)
        drawings = page.get_drawings()

        text_boxes: List[fitz.Rect] = []
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    try:
                        text_boxes.append(fitz.Rect(span["bbox"]))
                    except Exception:
                        pass

        white_overlay_count = 0
        white_overlay_detected = False

        for d in drawings:
            rect = d.get("rect")
            if not rect:
                continue

            try:
                draw_rect = fitz.Rect(rect)
            except Exception:
                continue

            if not _is_white_fill(d.get("fill")):
                continue

            if any(_rects_intersect(draw_rect, tb) for tb in text_boxes):
                white_overlay_count += 1
                white_overlay_detected = True

        image_only = (len(text) == 0 and len(images) > 0)
        drawing_heavy = (len(drawings) >= 5 and len(text) > 0)

        score = 0
        if white_overlay_detected:
            score += 50
        if image_only:
            score += 35
        if drawing_heavy:
            score += 15

        if score >= 60:
            risk = "High"
        elif score >= 30:
            risk = "Medium"
        else:
            risk = "Low"

        rows.append({
            "Filename": filename,
            "Page": page_index + 1,
            "Text Length": len(text),
            "Image Count": len(images),
            "Drawing Count": len(drawings),
            "Image Only Page": "TRUE" if image_only else "FALSE",
            "White Overlay Over Text": "TRUE" if white_overlay_detected else "FALSE",
            "White Overlay Count": white_overlay_count,
            "Drawing Heavy Page": "TRUE" if drawing_heavy else "FALSE",
            "Tamper Score": score,
            "Tamper Risk": risk,
        })

    return pd.DataFrame(rows)