import re
import statistics
from dataclasses import dataclass
from typing import List, Optional


RAW_OCR = """
[233,40] 100.02% INVOICE
[59,88] 94.82% 67h, Martin street
[78,71] 96.57% Ad4tech Material LLC
[35,112] 100.02% 576832
[53,100] 97.59% Alexander road
[385,102] 99.99% Logo
[65,125] 99.71% Mobile: +123456789
[90,138] 98.66% Email: ad4example@gmail.com
[34,166] 99.96% Bill To
[67,182] 97.13% Green1 Materials LLC
[295,167] 96.83% Invoice No :
[298,186] 98.99% Invoice Date:
[412,185] 99.82% Jun 22, 2021
[419,167] 100.01% INV-005
[38,207] 98.64% City park
[52,194] 99.28% #34, Car street
[43,219] 99.72% Honk Kong
[305,201] 93.21% Due Date :
[412,201] 99.49% Jun 27, 2021
[58,242] 94.70% SI. Description
[82,259] 98.46% Desktop furniture
[265,242] 100.01% Qty
[269,259] 100.02% 1
[337,258] 100.02% $232.00
[347,241] 100.01% Rate
[422,258] 96.83% $232.00
[425,242] 99.97% Amount
[30,294] 100.02% 3
[94,294] 100.01% Water tank repair works
[107,276] 99.32% 2 Plumbing and electrical services
[269,294] 100.02% 2
[269,277] 100.02% 2
[337,276] 93.25% $514.00
[337,294] 100.02% $152.00
[419,276] 100.02% $1,028.00
[422,294] 99.82% $304.00
[40,354] 99.92% John Doe
[51,339] 99.81% Pay Cheque to
[68,321] 99.96% Payment Instructions
[320,322] 94.41% Subtotal
[421,323] 100.02% $1,564.00
[294,366] 97.96% Paid (Jun 22, 2021)
[328,350] 100.02% Total
[421,350] 99.96% $1,564.00
[310,382] 100.00% Balance Due
[421,381] 100.00% $1,332.00
[424,366] 100.02% $232.00
[371,431] 69.18% a
[338,462] 98.30% Authorized Signatory
""".strip()


@dataclass
class Token:
    x: float
    y: float
    text: str
    conf: float


@dataclass
class Line:
    tokens: List[Token]
    y_mean: float


def parse_raw_ocr(raw: str) -> List[Token]:
    """
    Parse lines of the form:
    [x,y] conf% text
    into Token objects.
    """
    tokens: List[Token] = []
    pat = re.compile(
        r"\[(?P<x>-?\d+(?:\.\d+)?),(?P<y>-?\d+(?:\.\d+)?)\]\s+"
        r"(?P<conf>\d+(?:\.\d+)?)%\s+"
        r"(?P<text>.+)"
    )
    for line in raw.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        tokens.append(
            Token(
                x=float(m.group("x")),
                y=float(m.group("y")),
                conf=float(m.group("conf")),
                text=m.group("text").strip(),
            )
        )
    return tokens


def group_tokens_into_lines(tokens: List[Token]) -> List[Line]:
    """
    Group tokens into visual lines based on Y proximity.
    """
    tokens = sorted(tokens, key=lambda t: t.y)
    if len(tokens) > 1:
        y_diffs = [tokens[i+1].y - tokens[i].y for i in range(len(tokens)-1)]
        y_diffs = [d for d in y_diffs if d > 0]
        typical = statistics.median(y_diffs) if y_diffs else 10.0
    else:
        typical = 10.0

    tol = typical * 0.6  # Y tolerance within same line

    lines_raw: List[List[Token]] = []
    cur: List[Token] = []
    cy: Optional[float] = None

    for t in tokens:
        if cy is None:
            cur = [t]
            cy = t.y
            continue
        if abs(t.y - cy) <= tol:
            cur.append(t)
            cy = (cy * (len(cur)-1) + t.y) / len(cur)
        else:
            lines_raw.append(cur)
            cur = [t]
            cy = t.y
    if cur:
        lines_raw.append(cur)

    lines: List[Line] = []
    for lt in lines_raw:
        y_mean = sum(t.y for t in lt) / len(lt)
        lt_sorted = sorted(lt, key=lambda tt: tt.x)
        lines.append(Line(tokens=lt_sorted, y_mean=y_mean))

    return sorted(lines, key=lambda l: l.y_mean)


def segment_line_tokens(line: Line, gap_factor: float = 1.3) -> List[List[Token]]:
    """
    Split a line into 'columns' based on big X gaps.
    gap_factor: smaller → more aggressive splitting.
    """
    toks = line.tokens
    if not toks:
        return []
    if len(toks) == 1:
        return [toks[:]]

    gaps = [toks[i+1].x - toks[i].x for i in range(len(toks)-1)]
    median_gap = statistics.median(gaps) if gaps else 1.0
    if median_gap <= 0:
        median_gap = 1.0

    threshold = median_gap * gap_factor

    segments: List[List[Token]] = []
    cur = [toks[0]]
    for i in range(1, len(toks)):
        dx = toks[i].x - toks[i-1].x
        if dx > threshold:
            segments.append(cur)
            cur = [toks[i]]
        else:
            cur.append(toks[i])
    segments.append(cur)
    return segments


def to_line_column_csv(tokens: List[Token], gap_factor: float = 1.3) -> str:
    """
    Convert tokens into CSV: line_id, column_id, text
    using Y-grouping for lines and X-gap splitting for columns.
    """
    lines = group_tokens_into_lines(tokens)

    rows = []
    for line_id, ln in enumerate(lines):
        segments = segment_line_tokens(ln, gap_factor=gap_factor)
        for col_id, seg in enumerate(segments):
            text = " ".join(t.text for t in seg)
            rows.append((line_id, col_id, text))

    output_lines = ["line_id,column_id,text"]
    for line_id, col_id, text in rows:
        safe_text = '"' + text.replace('"', '""') + '"'
        output_lines.append(f"{line_id},{col_id},{safe_text}")
    return "\n".join(output_lines)


if __name__ == "__main__":
    tokens = parse_raw_ocr(RAW_OCR)
    csv_result = to_line_column_csv(tokens, gap_factor=1.3)
    print(csv_result)
