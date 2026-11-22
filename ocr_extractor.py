"""EasyOCR + regex pipeline that extracts PAN, Name and DOB from a PAN card."""
from __future__ import annotations

import re
from datetime import datetime

import cv2
import numpy as np

import config

_reader = None

PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
DOB_REGEX = re.compile(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})")

# OCR confusions fixed by position in the PAN pattern (LLLLLDDDDL)
_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
_TO_DIGIT = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6", "D": "0"}

_NOISE = (
    "INCOME", "TAX", "DEPARTMENT", "GOVT", "GOVERNMENT", "INDIA", "PERMANENT",
    "ACCOUNT", "NUMBER", "SIGNATURE", "CARD", "FATHER", "DATE", "BIRTH", "NAME",
)


def get_reader():
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(config.OCR_LANGS, gpu=False, verbose=False)
    return _reader


# --------------------------------------------------------------------------
# OCR passes
# --------------------------------------------------------------------------
def _preprocess_variants(card_bgr: np.ndarray):
    """Several enhanced versions of the card so a failure on one doesn't kill OCR."""
    gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    adaptive = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return [card_bgr, clahe, adaptive]


def _group_lines(results, y_tol: float = 0.6):
    """Cluster EasyOCR boxes into text lines (top-to-bottom, left-to-right)."""
    items = []
    for box, text, conf in results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"x": min(xs), "yc": sum(ys) / 4, "h": max(ys) - min(ys), "t": text.strip()})
    items.sort(key=lambda i: i["yc"])
    lines, current = [], []
    for it in items:
        if current and abs(it["yc"] - current[-1]["yc"]) > y_tol * max(it["h"], 1):
            lines.append(current)
            current = []
        current.append(it)
    if current:
        lines.append(current)
    return [" ".join(i["t"] for i in sorted(l, key=lambda i: i["x"])) for l in lines]


def run_ocr(card_bgr: np.ndarray):
    """Return (list_of_line_lists, raw_text). One entry per preprocessing variant."""
    reader = get_reader()
    all_lines = []
    for variant in _preprocess_variants(card_bgr):
        try:
            res = reader.readtext(variant, detail=1, paragraph=False)
            lines = _group_lines(res)
            if lines:
                all_lines.append(lines)
        except Exception:  # fault tolerance: skip a failing pass
            continue
    raw = "\n\n--- pass ---\n".join("\n".join(l) for l in all_lines)
    return all_lines, raw


# --------------------------------------------------------------------------
# Field parsers
# --------------------------------------------------------------------------
def _fix_pan_candidate(token: str) -> str | None:
    token = re.sub(r"[^A-Z0-9]", "", token.upper())
    if len(token) != 10:
        return None
    chars = list(token)
    for i in range(10):
        if i < 5 or i == 9:  # letter positions
            if chars[i].isdigit():
                chars[i] = _TO_LETTER.get(chars[i], chars[i])
        else:  # digit positions
            if chars[i].isalpha():
                chars[i] = _TO_DIGIT.get(chars[i], chars[i])
    fixed = "".join(chars)
    return fixed if PAN_REGEX.match(fixed) else None


def extract_pan(lines: list[str]) -> str | None:
    text = " ".join(lines).upper()
    # 1) strict match on raw text
    m = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text)
    if m:
        return m.group(0)
    # 2) tolerant match: any 10-char alphanumeric run, corrected by position
    for line in lines:
        compact = re.sub(r"[^A-Z0-9]", "", line.upper())
        for i in range(0, max(len(compact) - 9, 0)):
            fixed = _fix_pan_candidate(compact[i:i + 10])
            if fixed:
                return fixed
    return None


def extract_dob(lines: list[str]) -> str | None:
    text = " ".join(lines)
    for d, m, y in DOB_REGEX.findall(text):
        try:
            dt = datetime(int(y), int(m), int(d))
            if 1900 <= dt.year <= datetime.now().year:
                return dt.strftime("%d/%m/%Y")
        except ValueError:
            continue
    return None


def _is_name_line(line: str) -> bool:
    s = line.strip()
    if len(s) < 3 or re.search(r"\d", s):
        return False
    letters = re.sub(r"[^A-Za-z ]", "", s)
    if len(letters.replace(" ", "")) < 3:
        return False
    up = letters.upper()
    return not any(n in up.split() for n in _NOISE) and not any(
        n in up for n in ("INCOME TAX", "GOVT", "DEPARTMENT", "PERMANENT")
    )


def extract_name(lines: list[str]) -> tuple[str | None, str | None]:
    """Return (name, father_name) using PAN-card layout cues, with a heuristic fallback."""
    name = father = None

    for i, line in enumerate(lines):
        up = line.upper()
        if "FATHER" in up:
            for nxt in lines[i + 1:i + 3]:
                if _is_name_line(nxt):
                    father = nxt.strip().upper()
                    break
        elif re.search(r"\bNAME\b", up) and name is None:
            for nxt in lines[i + 1:i + 3]:
                if _is_name_line(nxt):
                    name = nxt.strip().upper()
                    break

    if name is None:
        candidates = [l.strip().upper() for l in lines if _is_name_line(l)]
        candidates = [c for c in candidates if c.replace(" ", "").isalpha()]
        if candidates:
            name = candidates[0]
            if father is None and len(candidates) > 1:
                father = candidates[1]
    return name, father


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def extract_pan_details(card_bgr: np.ndarray) -> dict:
    """Run OCR over several preprocessed variants and merge the best field values."""
    passes, raw = run_ocr(card_bgr)
    out = {"pan": None, "name": None, "father_name": None, "dob": None, "raw_text": raw}

    for lines in passes:  # first pass that yields each field wins
        if out["pan"] is None:
            out["pan"] = extract_pan(lines)
        if out["dob"] is None:
            out["dob"] = extract_dob(lines)
        if out["name"] is None or out["father_name"] is None:
            n, f = extract_name(lines)
            out["name"] = out["name"] or n
            out["father_name"] = out["father_name"] or f
    return out


def is_valid_pan(pan: str | None) -> bool:
    return bool(pan and PAN_REGEX.match(pan))
