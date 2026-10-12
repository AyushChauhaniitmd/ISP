"""Generate the FA-ready DP-ForgetBench presentation."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


OUT = Path("reports/DP_ForgetBench_FA_Presentation_ATTACK_METRICS.pptx")

NAVY = RGBColor(10, 20, 38)
INK = RGBColor(18, 32, 52)
MUTED = RGBColor(91, 110, 133)
CYAN = RGBColor(38, 198, 218)
BLUE = RGBColor(45, 109, 220)
GREEN = RGBColor(35, 170, 120)
ORANGE = RGBColor(239, 145, 64)
RED = RGBColor(214, 83, 93)
WHITE = RGBColor(255, 255, 255)
BG = RGBColor(246, 249, 252)
CARD = RGBColor(255, 255, 255)
LINE = RGBColor(220, 229, 238)


def make_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    def rect(slide, x, y, w, h, fill, radius=True, line=None):
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
            Inches(x), Inches(y), Inches(w), Inches(h),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
        if line:
            shape.line.color.rgb = line
            shape.line.width = Pt(1)
        else:
            shape.line.fill.background()
        return shape

    def text(slide, x, y, w, h, value, size=16, color=INK, bold=False, align=None, font="Aptos"):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.clear()
        tf.word_wrap = True
        tf.margin_left = Inches(0.02)
        tf.margin_right = Inches(0.02)
        p = tf.paragraphs[0]
        p.text = value
        p.font.name = font
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
        if align:
            p.alignment = align
        return box

    def bullets(slide, x, y, w, h, items, size=15, color=INK, gap=8):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.clear()
        tf.word_wrap = True
        tf.margin_left = Inches(0.02)
        tf.margin_right = Inches(0.02)
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = f"• {item}"
            p.font.name = "Aptos"
            p.font.size = Pt(size)
            p.font.color.rgb = color
            p.space_after = Pt(gap)
        return box

    def base(title, kicker, dark=False):
        slide = prs.slides.add_slide(blank)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = NAVY if dark else BG
        if not dark:
            rect(slide, 0, 0, 13.333, 0.12, BLUE, False)
        text(slide, 0.75, 0.36, 11.8, 0.25, kicker.upper(), 9, CYAN if dark else BLUE, True)
        text(slide, 0.75, 0.70, 11.9, 0.55, title, 25, WHITE if dark else INK, True)
        return slide

    def card_title(slide, x, y, w, title_value, accent=BLUE):
        text(slide, x, y, w, 0.3, title_value, 13, accent, True)

    def metric_card(slide, x, y, w, label, value, color=BLUE, note=None):
        rect(slide, x, y, w, 1.25, CARD, True, LINE)
        rect(slide, x, y, 0.08, 1.25, color, False)
        text(slide, x + 0.22, y + 0.18, w - 0.35, 0.25, label.upper(), 9, MUTED, True)
        text(slide, x + 0.22, y + 0.45, w - 0.35, 0.42, value, 25, color, True)
        if note:
            text(slide, x + 0.22, y + 0.91, w - 0.35, 0.2, note, 9, MUTED)

    def bar_chart(slide, x, y, w, h, labels, values, colors, maximum=50, suffix="%", title=None):
        if title:
            text(slide, x, y, w, 0.28, title, 13, INK, True)
            y += 0.45
            h -= 0.45
        chart_x = x + 0.35
        chart_w = w - 0.55
        chart_h = h - 0.45
        for tick in (0, maximum / 2, maximum):
            gy = y + chart_h - chart_h * tick / maximum
            rect(slide, chart_x, gy, chart_w, 0.01, LINE, False)
            text(slide, x, gy - 0.08, 0.3, 0.18, f"{tick:g}", 8, MUTED, False, PP_ALIGN.RIGHT)
        count = len(values)
        gap = chart_w / count
        bar_w = min(0.62, gap * 0.58)
        for i, (label, value, color) in enumerate(zip(labels, values, colors)):
            bx = chart_x + gap * i + (gap - bar_w) / 2
            bh = chart_h * value / maximum
            by = y + chart_h - bh
            rect(slide, bx, by, bar_w, bh, color, True)
            text(slide, bx - 0.15, by - 0.28, bar_w + 0.3, 0.2, f"{value:.1f}{suffix}", 10, color, True, PP_ALIGN.CENTER)
            text(slide, bx - 0.28, y + chart_h + 0.12, bar_w + 0.56, 0.28, label, 9, INK, True, PP_ALIGN.CENTER)

    def footer(slide, n):
        text(slide, 0.75, 7.16, 9, 0.18, "DP-FORGETBENCH  |  RESEARCH STATUS  |  18 SEP 2026", 8, MUTED)
        text(slide, 12.1, 7.16, 0.45, 0.18, str(n), 8, MUTED, True, PP_ALIGN.RIGHT)

    # 1
    slide = base("DP-ForgetBench", "Federated learning × client-level privacy × unlearning", True)
    text(slide, 0.78, 1.65, 11.4, 0.7, "When is explicit client unlearning redundant under differentially private federated learning?", 23, CYAN, True)
    rect(slide, 0.78, 2.75, 11.75, 0.02, CYAN, False)
    text(slide, 0.8, 3.18, 5.3, 1.6, "A privacy-aligned benchmark for deciding whether to do nothing, unlearn, or retrain after a client deletion request.", 22, WHITE, True)
    rect(slide, 7.1, 3.05, 5.25, 2.2, RGBColor(20, 42, 70), True, CYAN)
    text(slide, 7.45, 3.4, 4.55, 0.35, "THE CENTRAL COMPARISON", 11, CYAN, True)
    text(slide, 7.45, 3.9, 4.4, 0.8, "M_DP  vs  M_U  vs  M_R", 28, WHITE, True, PP_ALIGN.CENTER)
    text(slide, 7.45, 4.85, 4.4, 0.25, "DP-only  |  explicit unlearning  |  exact retrain", 10, RGBColor(190, 210, 230), False, PP_ALIGN.CENTER)
    text(slide, 0.8, 6.35, 11.5, 0.35, "Not a claim that DP is deletion. A test of when explicit unlearning adds material benefit.", 13, RGBColor(190, 210, 230))
    footer(slide, 1)

    # 2
    slide = base("The research question", "01  |  motivation")
    rect(slide, 0.8, 1.45, 5.8, 4.85, CARD, True, LINE)
    card_title(slide, 1.1, 1.75, 5.0, "A CLIENT REQUESTS DELETION", ORANGE)
    bullets(slide, 1.1, 2.25, 5.0, 3.4, [
        "Exact retraining is the clean reference, but expensive.",
        "Approximate unlearning may change utility or expose retained data.",
        "A DP model may already have a small client influence.",
        "The wrong comparison can make a useless model look forgotten.",
    ], 16)
    rect(slide, 7.0, 1.45, 5.55, 4.85, NAVY, True, BLUE)
    text(slide, 7.35, 1.8, 4.8, 0.25, "OUR TEST", 11, CYAN, True)
    text(slide, 7.35, 2.25, 4.7, 1.35, "Does explicit unlearning provide meaningful benefit beyond leaving the client-level DP model unchanged?", 22, WHITE, True)
    text(slide, 7.35, 4.25, 4.65, 0.9, "A benefit must be measured against retraining variability, not one arbitrary retrain.", 16, RGBColor(190, 210, 230))
    footer(slide, 2)

    # 3
    slide = base("What is genuinely novel?", "02  |  contribution")
    items = [
        ("Privacy-scope alignment", "Client deletion is evaluated under client-level central DP, not example-level DP-SGD.", CYAN),
        ("Redundancy frontier", "Map redundant, beneficial, harmful, and inconclusive regions over privacy and deletion factors.", BLUE),
        ("Retrain variability", "Independent M_R runs define the natural stochastic reference band.", GREEN),
        ("Tri-population + cost", "Forgotten, retained, unseen privacy and full request-lifecycle cost are visible.", ORANGE),
    ]
    coords = [(0.8,1.45),(6.9,1.45),(0.8,4.0),(6.9,4.0)]
    for (heading, body, color), (x,y) in zip(items, coords):
        rect(slide, x, y, 5.65, 2.1, CARD, True, LINE)
        rect(slide, x, y, 0.1, 2.1, color, False)
        text(slide, x+0.3, y+0.3, 4.95, 0.35, heading, 16, color, True)
        text(slide, x+0.3, y+0.85, 4.95, 0.85, body, 14, INK)
    footer(slide, 3)

    # 4
    slide = base("The privacy contract", "03  |  mechanism")
    steps = [
        ("01", "Poisson sample clients"), ("02", "Clip complete client updates"),
        ("03", "Add Gaussian noise to sum"), ("04", "Normalize by fixed q×N"),
        ("05", "Account with RDP"),
    ]
    for i, (num, label) in enumerate(steps):
        x = 0.85 + (i % 3) * 4.15
        y = 1.55 + (i // 3) * 2.0
        rect(slide, x, y, 3.6, 1.3, CARD, True, LINE)
        text(slide, x+0.25, y+0.2, 0.6, 0.4, num, 20, CYAN, True)
        text(slide, x+1.0, y+0.25, 2.3, 0.65, label, 15, INK, True)
    rect(slide, 0.85, 5.75, 11.7, 0.75, RGBColor(232, 243, 250), True, RGBColor(170, 210, 230))
    text(slide, 1.15, 5.97, 11.1, 0.25, "Adjacency = one complete client dataset  |  delta = 1e-5  |  secure aggregation not implemented", 12, INK, True, PP_ALIGN.CENTER)
    footer(slide, 4)

    # 5
    slide = base("We prevent vacuous unlearning claims", "04  |  validity gate")
    rect(slide, 0.85, 1.55, 3.65, 4.7, RGBColor(255, 241, 242), True, RGBColor(245, 190, 195))
    text(slide, 1.15, 1.9, 3.0, 0.35, "UTILITY COLLAPSE", 13, RED, True)
    text(slide, 1.15, 2.5, 3.0, 0.7, "≈10%", 34, RED, True)
    text(slide, 1.15, 3.45, 2.95, 1.4, "Low leakage here does not prove forgetting. The model did not learn enough for deletion to matter.", 16, INK)
    rect(slide, 4.85, 1.55, 3.65, 4.7, RGBColor(255, 248, 235), True, RGBColor(245, 210, 160))
    text(slide, 5.15, 1.9, 3.0, 0.35, "DP-DOMINATED", 13, ORANGE, True)
    text(slide, 5.15, 2.5, 3.0, 0.7, "25–40%", 30, ORANGE, True)
    text(slide, 5.15, 3.45, 2.95, 1.4, "The model learns, but DP may mask the client effect relative to retrain variability.", 16, INK)
    rect(slide, 8.85, 1.55, 3.65, 4.7, RGBColor(235, 249, 243), True, RGBColor(170, 225, 200))
    text(slide, 9.15, 1.9, 3.0, 0.35, "HEALTHY", 13, GREEN, True)
    text(slide, 9.15, 2.5, 3.0, 0.7, "≥40%", 34, GREEN, True)
    text(slide, 9.15, 3.45, 2.95, 1.4, "Only here is a meaningful DP-versus-unlearning comparison possible.", 16, INK)
    footer(slide, 5)

    # 6
    slide = base("What we built", "05  |  completed work")
    left = ["Client-level DP + RDP ledger", "Deterministic deletion manifests", "Exact + independent retraining", "Fine-tuning + cached reconstruction", "FedEraser-style baseline"]
    right = ["CIFAR-10 multiclass backend", "GroupNorm CNN / ResNet path", "MUB + validity gates", "MIA + tri-population probes", "GPU/Colab reproducibility"]
    rect(slide, 0.85, 1.5, 5.65, 4.9, CARD, True, LINE)
    rect(slide, 6.85, 1.5, 5.65, 4.9, CARD, True, LINE)
    card_title(slide, 1.15, 1.85, 4.8, "BENCHMARK CORE", BLUE)
    card_title(slide, 7.15, 1.85, 4.8, "EVALUATION SURFACE", CYAN)
    bullets(slide, 1.15, 2.35, 4.9, 3.5, left, 16)
    bullets(slide, 7.15, 2.35, 4.9, 3.5, right, 16)
    footer(slide, 6)

    # 7
    slide = base("Utility gate: a non-vacuous frozen regime", "06  |  result")
    metric_card(slide, 0.85, 1.55, 3.65, "Non-private mean", "44.82%", GREEN, "3 seeds")
    metric_card(slide, 4.85, 1.55, 3.65, "Frozen DP-only", "42.38%", BLUE, "epsilon ≈ 32")
    metric_card(slide, 8.85, 1.55, 3.65, "DP / non-DP", "94.6%", CYAN, "observed mean ratio")
    text(slide, 0.9, 3.25, 11.5, 0.35, "Frozen configuration", 16, BLUE, True)
    text(slide, 0.9, 3.72, 11.5, 0.45, "SmallGroupNormCNN  |  100 clients  |  q=0.5  |  40 rounds  |  5 local epochs  |  alpha=0.5", 18, INK, True)
    rect(slide, 0.9, 4.7, 11.5, 1.1, RGBColor(232, 243, 250), True, RGBColor(175, 210, 235))
    text(slide, 1.2, 5.02, 10.9, 0.45, "The current model is not broken. The experiment has reached a useful finite-DP regime where unlearning is no longer a vacuous question.", 17, INK, True, PP_ALIGN.CENTER)
    footer(slide, 7)

    # 8: privacy-utility curve
    slide = base("The privacy–utility curve", "07  |  measured frontier")
    rect(slide, 0.85, 1.45, 7.25, 5.15, CARD, True, LINE)
    bar_chart(slide, 1.15, 1.85, 6.65, 4.25, ["ε=8", "ε=16", "ε=32", "non-DP"], [19.45, 28.79, 36.50, 44.82], [RED, ORANGE, BLUE, GREEN], 50, "%", "Mean test accuracy across 3 seeds")
    rect(slide, 8.45, 1.45, 4.05, 5.15, NAVY, True, BLUE)
    text(slide, 8.8, 1.85, 3.3, 0.3, "READING THE CURVE", 12, CYAN, True)
    bullets(slide, 8.8, 2.35, 3.1, 2.8, [
        "ε=8 is utility-invalid.",
        "ε=16 is marginal but above chance.",
        "ε=32 is the strongest finite-DP point.",
        "Non-DP establishes the learning ceiling for this federated setup.",
    ], 14, WHITE, 11)
    text(slide, 8.8, 5.55, 3.1, 0.55, "The benchmark only moves to unlearning once the model is non-vacuous.", 14, CYAN, True)
    footer(slide, 8)

    # 9: local epoch effect
    slide = base("One controlled change unlocked the frozen regime", "08  |  diagnostic ablation")
    rect(slide, 0.85, 1.45, 7.25, 5.15, CARD, True, LINE)
    bar_chart(slide, 1.15, 1.85, 6.65, 4.25, ["2 epochs", "4 epochs", "5 epochs"], [32.12, 40.15, 42.38], [ORANGE, BLUE, GREEN], 50, "%", "ε=32 · 40 rounds · mean accuracy across 3 seeds")
    rect(slide, 8.45, 1.45, 4.05, 5.15, RGBColor(235,249,243), True, RGBColor(170,225,200))
    text(slide, 8.8, 1.85, 3.25, 0.3, "+10.26 pp", 26, GREEN, True)
    text(slide, 8.8, 2.35, 3.1, 0.5, "from 2 → 5 local epochs", 16, INK, True)
    bullets(slide, 8.8, 3.15, 3.0, 2.1, [
        "Same model.",
        "Same ε=32.",
        "Same 40 rounds.",
        "Only local learning changed.",
    ], 14, INK, 8)
    text(slide, 8.8, 5.55, 3.1, 0.55, "Frozen candidate: 5 local epochs; 42.38% mean DP utility.", 14, GREEN, True)
    footer(slide, 9)

    # 10
    slide = base("First frozen unlearning comparison", "07  |  results")
    headers = ["METHOD", "TEST ACC", "FORGOTTEN ACC"]
    rows = [
        ("DP-only / no action", "42.38%", "51.00%", BLUE),
        ("Retained fine-tuning", "44.05%", "50.83%", GREEN),
        ("Cached reconstruction", "41.47%", "40.50%", ORANGE),
        ("FedEraser-style", "36.79%", "42.83%", RED),
        ("Exact retrain", "41.73%", "43.50%", INK),
    ]
    x0, y0 = 0.95, 1.55
    widths = [5.8, 2.4, 3.0]
    for x, w, h in zip([x0, x0+widths[0], x0+widths[0]+widths[1]], widths, headers):
        rect(slide, x, y0, w, 0.55, NAVY, False)
        text(slide, x+0.18, y0+0.16, w-0.3, 0.2, h, 11, WHITE, True, PP_ALIGN.CENTER)
    for i, (method, acc, forgotten, color) in enumerate(rows):
        y = y0 + 0.65 + i*0.85
        rect(slide, x0, y, sum(widths), 0.7, CARD if i%2==0 else RGBColor(239,244,248), True, LINE)
        rect(slide, x0, y, 0.1, 0.7, color, False)
        text(slide, x0+0.28, y+0.2, widths[0]-0.4, 0.25, method, 15, INK, method=="DP-only / no action")
        text(slide, x0+widths[0], y+0.2, widths[1], 0.25, acc, 17, color, True, PP_ALIGN.CENTER)
        text(slide, x0+widths[0]+widths[1], y+0.2, widths[2], 0.25, forgotten, 17, color, True, PP_ALIGN.CENTER)
    rect(slide, 0.95, 6.15, 11.5, 0.55, RGBColor(255,248,235), True, RGBColor(245,210,160))
    text(slide, 1.15, 6.3, 11.1, 0.22, "Forgotten accuracy is a behavior signal, not a privacy guarantee; MIA and tri-population results are required.", 11, INK, True, PP_ALIGN.CENTER)
    footer(slide, 10)

    # 11: detailed comparison chart
    slide = base("Utility versus forgotten-client behavior", "09  |  detailed result")
    rect(slide, 0.85, 1.45, 7.25, 5.15, CARD, True, LINE)
    bar_chart(slide, 1.15, 1.82, 6.65, 4.35, ["DP", "FT", "CR", "FE", "R"], [42.38, 44.05, 41.47, 36.79, 41.73], [BLUE, GREEN, ORANGE, RED, INK], 50, "%", "Test accuracy")
    rect(slide, 8.45, 1.45, 4.05, 5.15, CARD, True, LINE)
    card_title(slide, 8.8, 1.85, 3.3, "WHAT CHANGED?", BLUE)
    bullets(slide, 8.8, 2.35, 3.05, 2.8, [
        "Fine-tuning is best for utility.",
        "Cached reconstruction is close to retrain on forgotten accuracy, but overshoots alignment.",
        "FedEraser loses 5.59 pp versus DP-only.",
        "DP-only is only 0.65 pp above retrain in mean test accuracy.",
    ], 13, INK, 8)
    text(slide, 8.8, 5.55, 3.1, 0.55, "No single method dominates every endpoint.", 15, ORANGE, True)
    footer(slide, 11)

    # 12
    slide = base("Audited attack results", "08  |  privacy audit")
    metric_card(slide, 0.85, 1.5, 3.65, "DP-only forgotten AUC", "0.535", BLUE, "held-out evaluation")
    metric_card(slide, 4.85, 1.5, 3.65, "Fine-tune forgotten AUC", "0.526", GREEN, "held-out evaluation")
    metric_card(slide, 8.85, 1.5, 3.65, "Retrain forgotten AUC", "0.516", INK, "held-out evaluation")
    rect(slide, 0.85, 3.25, 11.65, 2.7, CARD, True, LINE)
    card_title(slide, 1.15, 3.55, 10.4, "AUDIT PROTOCOL", CYAN)
    bullets(slide, 1.15, 4.0, 10.7, 1.45, [
        "Label-matched forgotten/retained members versus unseen data.",
        "Threshold selected on calibration split; advantage evaluated on disjoint held-out split.",
        "AUC is threshold-independent; bootstrap intervals are reported.",
        "Tri-population probe separates forget, retain, and unseen populations.",
    ], 14)
    text(slide, 0.9, 6.2, 11.5, 0.35, "Current interpretation: no method shows a reliable forgotten-set privacy improvement over DP-only with n=3 seeds.", 15, ORANGE, True, PP_ALIGN.CENTER)
    footer(slide, 12)

    # 13: attack variability
    slide = base("The attack signal is weak and variable", "10  |  uncertainty")
    rect(slide, 0.85, 1.45, 7.25, 5.15, CARD, True, LINE)
    bar_chart(slide, 1.15, 1.85, 6.65, 4.25, ["DP", "FT", "CR", "FE", "R"], [0.535, 0.526, 0.516, 0.525, 0.516], [BLUE, GREEN, ORANGE, RED, INK], 0.60, "", "Forgotten-set MIA AUC (held-out)")
    rect(slide, 8.45, 1.45, 4.05, 5.15, RGBColor(255,248,235), True, RGBColor(245,210,160))
    text(slide, 8.8, 1.85, 3.25, 0.3, "n = 3 seeds", 23, ORANGE, True)
    bullets(slide, 8.8, 2.45, 3.0, 2.6, [
        "Intervals are wide.",
        "AUC values are near 0.5.",
        "No explicit method shows a reliable privacy win over DP-only.",
        "Tri-population aggregation remains a required next check.",
    ], 14, INK, 9)
    text(slide, 8.8, 5.55, 3.1, 0.55, "Current answer: suggestive, not decisive.", 15, RED, True)
    footer(slide, 13)

    # 14: attack metric definitions
    slide = base("Attack metrics: what each number means", "11  |  metric dictionary")
    metric_defs = [
        ("AUC", "Area under the ROC curve. Threshold-independent ranking quality; 0.5 is chance, 1.0 is perfect separation.", CYAN),
        ("Advantage", "Held-out TPR − held-out FPR at the threshold selected on the calibration split. Positive means member-like separation.", BLUE),
        ("TPR @ 1% FPR", "Fraction of members detected while allowing only 1% false alarms. Highlights precision-sensitive attacks.", GREEN),
        ("TPR @ 0.1% FPR", "Stricter low-false-positive operating point for high-confidence attacks.", ORANGE),
        ("Tri-probe accuracy", "Three-class classification accuracy for forget, retain, and unseen populations using pre/post loss-confidence features.", RED),
        ("Macro OVR AUC", "One-vs-rest AUC averaged across forget, retain, and unseen classes; 0.5 is chance-like ranking.", INK),
    ]
    coords = [(0.8,1.45),(6.9,1.45),(0.8,3.15),(6.9,3.15),(0.8,4.85),(6.9,4.85)]
    for (heading, body, color), (x, y) in zip(metric_defs, coords):
        rect(slide, x, y, 5.65, 1.35, CARD, True, LINE)
        rect(slide, x, y, 0.09, 1.35, color, False)
        text(slide, x+0.28, y+0.2, 4.95, 0.25, heading, 14, color, True)
        text(slide, x+0.28, y+0.57, 4.95, 0.55, body, 10.5, INK)
    footer(slide, 14)

    # 15: populations and protocol
    slide = base("Attack populations and calibration protocol", "12  |  audit design")
    rect(slide, 0.8, 1.45, 3.7, 4.9, RGBColor(255,241,242), True, RGBColor(245,190,195))
    rect(slide, 4.82, 1.45, 3.7, 4.9, RGBColor(235,249,243), True, RGBColor(170,225,200))
    rect(slide, 8.84, 1.45, 3.7, 4.9, RGBColor(232,243,250), True, RGBColor(175,210,235))
    card_title(slide, 1.1, 1.8, 3.0, "FORGOTTEN", RED)
    text(slide, 1.1, 2.35, 3.0, 1.0, "Examples belonging to the deleted client.", 17, INK, True)
    text(slide, 1.1, 4.0, 3.0, 1.2, "Question: does the released model still expose member-like signal for this population?", 14, INK)
    card_title(slide, 5.12, 1.8, 3.0, "RETAINED", GREEN)
    text(slide, 5.12, 2.35, 3.0, 1.0, "Examples belonging to clients that remain.", 17, INK, True)
    text(slide, 5.12, 4.0, 3.0, 1.2, "Question: does unlearning create collateral leakage or utility damage for non-requesting clients?", 14, INK)
    card_title(slide, 9.14, 1.8, 3.0, "UNSEEN", BLUE)
    text(slide, 9.14, 2.35, 3.0, 1.0, "Held-out examples never used by clients.", 17, INK, True)
    text(slide, 9.14, 4.0, 3.0, 1.2, "Reference non-member population, label-matched against members.", 14, INK)
    rect(slide, 1.0, 5.75, 11.25, 0.42, NAVY, True, CYAN)
    text(slide, 1.2, 5.86, 10.85, 0.18, "Loss-MIA: 50/50 calibration/evaluation split · threshold from calibration · AUC and advantage on held-out evaluation", 10.5, WHITE, True, PP_ALIGN.CENTER)
    footer(slide, 15)

    # 16: interpretation guide
    slide = base("How to interpret attack results", "13  |  reading the evidence")
    rect(slide, 0.85, 1.45, 5.65, 4.95, CARD, True, LINE)
    rect(slide, 6.85, 1.45, 5.65, 4.95, NAVY, True, BLUE)
    card_title(slide, 1.15, 1.85, 4.8, "A LOW SCORE DOES NOT AUTOMATICALLY MEAN UNLEARNING", ORANGE)
    bullets(slide, 1.15, 2.35, 4.85, 3.2, [
        "AUC near 0.5 means the attack cannot rank members well.",
        "Advantage near 0 means little held-out operating-point separation.",
        "A chance-level model can also produce low attack signal.",
        "Always check utility and validity first.",
    ], 14, INK, 10)
    card_title(slide, 7.15, 1.85, 4.8, "WHAT COUNTS AS EVIDENCE?", CYAN)
    bullets(slide, 7.15, 2.35, 4.85, 3.2, [
        "Compare M_DP, M_U, and M_R on the same frozen populations.",
        "Use retrain variability and confidence intervals.",
        "Check forgotten and retained privacy together.",
        "Report cost and post-request data access alongside privacy.",
    ], 14, WHITE, 10)
    footer(slide, 16)

    # 17
    slide = base("Marginal benefit versus practical cost", "09  |  decision surface")
    headers = ["METHOD", "ACC MUB", "JS MUB", "MIA MUB", "DECISION"]
    widths = [4.2,1.6,1.6,1.6,3.0]
    x = 0.75
    for w,h in zip(widths,headers):
        rect(slide, x, 1.45, w, 0.55, NAVY, False)
        text(slide, x+0.05, 1.62, w-0.1, 0.18, h, 10, WHITE, True, PP_ALIGN.CENTER)
        x += w
    data = [("Fine-tuning", "+0.0167", "+0.0061", "-0.0026", "utility gain; privacy unclear", GREEN), ("Cached reconstruction", "-0.0092", "-0.0077", "-0.0060", "overshoots; history cost", ORANGE), ("FedEraser", "-0.0559", "-0.0027", "-0.0226", "harmful utility tradeoff", RED)]
    for i,row in enumerate(data):
        y=2.15+i*1.05; x=0.75
        rect(slide,x,y,sum(widths),0.82,CARD if i%2==0 else RGBColor(239,244,248),True,LINE)
        for j,(value,w) in enumerate(zip(row[:-1],widths)):
            text(slide,x+0.05,y+0.25,w-0.1,0.25,str(value),14,INK if j==0 else (GREEN if str(value).startswith('+') else RED),j==0,PP_ALIGN.CENTER if j>0 else None)
            x += w
    rect(slide,0.85,5.65,5.6,0.7,RGBColor(232,243,250),True,RGBColor(175,210,235))
    text(slide,1.05,5.88,5.2,0.22,"DP-only: no post-request action",14,BLUE,True,PP_ALIGN.CENTER)
    rect(slide,6.9,5.65,5.6,0.7,RGBColor(255,241,242),True,RGBColor(245,190,195))
    text(slide,7.1,5.88,5.2,0.22,"FedEraser: ≈497 MB history + ≈965 MB communication",12,RED,True,PP_ALIGN.CENTER)
    footer(slide, 17)

    # 18
    slide = base("Transparency: what is and is not claimed", "10  |  scientific boundary")
    rect(slide,0.85,1.45,5.65,4.95,RGBColor(235,249,243),True,RGBColor(170,225,200))
    rect(slide,6.85,1.45,5.65,4.95,RGBColor(255,241,242),True,RGBColor(245,190,195))
    card_title(slide,1.15,1.8,4.8,"ESTABLISHED",GREEN)
    bullets(slide,1.15,2.3,4.9,3.4,["Client-level DP mechanism and ledger run correctly.","CNN learns multiclass CIFAR-10 above chance.","Frozen DP regime is useful and non-vacuous.","Explicit baselines and audited attacks run end to end.","FedEraser has a measurable utility/storage tradeoff."],15)
    card_title(slide,7.15,1.8,4.8,"NOT YET ESTABLISHED",RED)
    bullets(slide,7.15,2.3,4.9,3.4,["DP perfectly deletes client data.","A universal epsilon redundancy threshold.","Certified or legal deletion.","A confirmed frontier across datasets and request streams.","A statistically decisive winner with only three seeds."],15)
    footer(slide, 18)

    # 19
    slide = base("Next stage: discover the frontier", "11  |  roadmap")
    roadmap = [("1", "Complete tri-population aggregation", CYAN), ("2", "Confirm transition / harmful cells with 10 seeds", BLUE), ("3", "Add heterogeneity, deletion size, and request sequences", GREEN), ("4", "Extend to natural client-partitioned data", ORANGE), ("5", "Publish reproducible artifacts and paper", INK)]
    for i,(num,label,color) in enumerate(roadmap):
        y=1.45+i*0.9
        rect(slide,0.95,y,0.65,0.55,color,True)
        text(slide,0.95,y+0.14,0.65,0.22,num,15,WHITE,True,PP_ALIGN.CENTER)
        rect(slide,1.85,y,10.55,0.55,CARD,True,LINE)
        text(slide,2.15,y+0.14,9.8,0.22,label,15,INK,True)
    rect(slide,0.95,6.15,11.45,0.55,NAVY,True,CYAN)
    text(slide,1.15,6.31,11.05,0.2,"Goal: (ε, α, deletion size, request count) → {redundant, beneficial, harmful, inconclusive}",13,WHITE,True,PP_ALIGN.CENTER)
    footer(slide, 19)

    # 20
    slide = base("The one-line takeaway", "12  |  close", True)
    text(slide,0.9,1.6,11.5,0.8,"We are not proposing another unlearning algorithm.",24,WHITE,True,PP_ALIGN.CENTER)
    text(slide,1.35,2.75,10.65,1.5,"We are building the evidence needed to determine when client-level DP already suppresses a client's influence enough that explicit federated unlearning adds no meaningful benefit — and when it still fails to do so.",25,CYAN,True,PP_ALIGN.CENTER)
    rect(slide,1.25,5.05,10.85,0.03,CYAN,False)
    text(slide,1.3,5.45,10.7,0.4,"Current status: benchmark validated · frozen study complete · frontier confirmation next",14,RGBColor(190,210,230),False,PP_ALIGN.CENTER)
    footer(slide, 20)
    return prs


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    make_deck().save(OUT)
    print(f"Created {OUT}")
