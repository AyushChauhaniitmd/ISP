"""Convert a Markdown research plan to a readable PDF without Pandoc.

Usage: python scripts/build_pdf.py RESEARCH_GRADE_PLAN.md reports/RESEARCH_GRADE_PLAN.pdf
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


def inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", escaped)
    escaped = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"<link href='\2' color='blue'>\1</link>", escaped)
    return escaped


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line):
            continue
        cells = [inline_markdown(cell.strip()) for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def build(markdown_path: Path, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("PlanTitle", parent=styles["Title"], fontSize=20, leading=25, alignment=TA_CENTER, spaceAfter=18)
    heading = {
        1: ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, leading=20, spaceBefore=14, spaceAfter=8),
        2: ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=12, spaceAfter=6),
        3: ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, leading=14, spaceBefore=10, spaceAfter=5),
    }
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=6)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=14, firstLineIndent=-9)
    code = ParagraphStyle("Code", parent=styles["Code"], fontName="Courier", fontSize=7.5, leading=9, leftIndent=8)
    story = []
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    index = 0
    in_code = False
    code_lines: list[str] = []
    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), code))
                story.append(Spacer(1, 5))
                code_lines = []
            in_code = not in_code
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue
        if line.startswith("|") and "|" in line[1:]:
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            rows = parse_table(table_lines)
            if rows:
                width = 18.0 * cm / len(rows[0])
                table = Table([[Paragraph(cell, body) for cell in row] for row in rows], colWidths=[width] * len(rows[0]), repeatRows=1)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9BAAB8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.extend([table, Spacer(1, 8)])
            continue
        match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if match:
            level = len(match.group(1))
            text = inline_markdown(match.group(2))
            story.append(Paragraph(text, title if level == 1 else heading[level]))
        elif line.startswith("> "):
            quote = ParagraphStyle("Quote", parent=body, leftIndent=16, borderColor=colors.HexColor("#4B82A8"), borderWidth=1, borderPadding=6)
            story.append(Paragraph(inline_markdown(line[2:]), quote))
        elif re.match(r"^\s*(?:[-*]|\d+\.)\s+", line):
            item = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line)
            story.append(Paragraph("- " + inline_markdown(item), bullet))
        elif line.strip():
            story.append(Paragraph(inline_markdown(line), body))
        else:
            story.append(Spacer(1, 3))
        index += 1
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm)
    document.build(story)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: build_pdf.py INPUT.md OUTPUT.pdf")
    build(Path(sys.argv[1]), Path(sys.argv[2]))
