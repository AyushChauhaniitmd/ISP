import sys
import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()

    # Set slide dimensions to widescreen 16:9
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme colors
    NAVY = RGBColor(15, 23, 42)        # #0f172a
    SLATE = RGBColor(30, 41, 59)       # #1e293b
    ACCENT_BLUE = RGBColor(37, 99, 235) # #2563eb
    CYAN = RGBColor(14, 165, 233)     # #0ea5e9
    LIGHT_BG = RGBColor(248, 250, 252) # #f8fafc
    CARD_BG = RGBColor(255, 255, 255)  # #ffffff
    TEXT_DARK = RGBColor(15, 23, 42)
    TEXT_MUTED = RGBColor(100, 116, 139)
    WHITE = RGBColor(255, 255, 255)
    BORDER_COLOR = RGBColor(226, 232, 240)

    def set_slide_background(slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, title_text, category_text="DP-FORGETBENCH RESEARCH AUDIT"):
        # Header category
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.3))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE
        p_cat.font.name = "Calibri"

        # Main Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.6))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_DARK
        p_title.font.name = "Calibri"

    def create_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=BORDER_COLOR):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1)
        else:
            shape.line.fill.background()
        return shape

    # ==========================================
    # SLIDE 1: Title & Problem Statement (PS)
    # ==========================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1, NAVY)

    # Decorative title accent box
    accent_bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.15), Inches(2.2))
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = CYAN
    accent_bar.line.fill.background()

    title_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.1), Inches(11.0), Inches(2.3))
    tf = title_box.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "DP-ForgetBench: Privacy-Aligned Audit of Federated Unlearning"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.text = "When is explicit client unlearning redundant under differentially private federated learning?"
    p2.font.size = Pt(16)
    p2.font.color.rgb = CYAN
    p2.space_before = Pt(10)

    # Card 1: Problem Statement
    c1 = create_card(slide1, Inches(0.8), Inches(3.7), Inches(5.6), Inches(3.2), bg_color=SLATE, border_color=CYAN)
    tf_c1 = c1.text_frame
    tf_c1.word_wrap = True
    tf_c1.margin_left = Inches(0.25)
    tf_c1.margin_top = Inches(0.25)
    tf_c1.margin_right = Inches(0.25)

    p = tf_c1.paragraphs[0]
    p.text = "PROBLEM STATEMENT (PS)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN

    bullets_ps = [
        "Federated Learning (FL) models can leak sensitive client data via membership attacks.",
        "Exact Retraining from scratch upon client removal request is computationally prohibited at scale.",
        "Existing Federated Unlearning (FU) methods rely on heuristic updates, lack standardized privacy accounting, and often ignore Differential Privacy (DP).",
        "Scope Mismatch: Literature frequently conflates example-level DP-SGD with client-level deletion requests."
    ]
    for b in bullets_ps:
        p = tf_c1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(11)
        p.font.color.rgb = WHITE
        p.space_before = Pt(6)

    # Card 2: What Are We Doing?
    c2 = create_card(slide1, Inches(6.8), Inches(3.7), Inches(5.7), Inches(3.2), bg_color=SLATE, border_color=ACCENT_BLUE)
    tf_c2 = c2.text_frame
    tf_c2.word_wrap = True
    tf_c2.margin_left = Inches(0.25)
    tf_c2.margin_top = Inches(0.25)
    tf_c2.margin_right = Inches(0.25)

    p = tf_c2.paragraphs[0]
    p.text = "WHAT ARE WE DOING?"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN

    bullets_doing = [
        "Building a reproducible research-grade benchmark (DP-ForgetBench) for client deletion.",
        "Rigorous Privacy Matching: Evaluating central client-level DP (Poisson sampling + Gaussian noise + RDP accountant).",
        "Redundancy Testing: Determining where leaving a DP model unchanged is functionally equivalent to exact retraining.",
        "Gold-Standard Retraining Reference: Benchmarking methods against independent target retrains (MR vs MR') to account for natural training variability."
    ]
    for b in bullets_doing:
        p = tf_c2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(11)
        p.font.color.rgb = WHITE
        p.space_before = Pt(6)

    # ==========================================
    # SLIDE 2: Published Works Summary
    # ==========================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2, LIGHT_BG)
    add_header(slide2, "Prior Published Works & Their Core Contributions")

    published_works = [
        ("Gu, He, & Chen (arXiv 2025)", "Auditing Approximate Machine Unlearning for DP Models", "Demonstrated that post-hoc approximate unlearning on DP models can inadvertently increase membership privacy leakage on the retained set. Highlighted the necessity of retained-set privacy audits."),
        ("Fu et al. (arXiv 2026)", "Revisiting Privacy Leakage in Machine Unlearning", "Introduced tri-population membership inference (TC-UMIA) to evaluate leakage across Forgotten, Retained, and Unseen populations simultaneously rather than inspecting forgotten data in isolation."),
        ("Liu et al. (IEEE TIFS 2024)", "Privacy-Preserving FU with Certified Client Removal", "Proposed certified client removal (e.g., Starfish) using theoretical upper bounds on residual gradient influence, but required heavy cryptographic/2PC system assumptions and custom architectures."),
        ("Zhao et al. (arXiv 2023)", "Exploring Federated Unlearning (OpenFederatedUnlearning)", "Constructed an empirical benchmark for baseline FU (FedEraser, fine-tuning), but lacked client-level central DP integration and formal retrain-variability equivalence bands."),
        ("Cadet et al. (NeurIPS 2024)", "Deep Unlearn: Benchmarking Machine Unlearning", "Proved that unlearning evaluation without multiple random seeds, calibrated attacks, and gold-standard retrains leads to false positive unlearning claims.")
    ]

    top_pos = 1.4
    for authors, title, summary in published_works:
        card = create_card(slide2, Inches(0.8), Inches(top_pos), Inches(11.7), Inches(1.05))
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.2)
        tf.margin_top = Inches(0.12)
        tf.margin_right = Inches(0.2)

        p1 = tf.paragraphs[0]
        p1.text = f"{authors} — "
        p1.font.bold = True
        p1.font.size = Pt(12)
        p1.font.color.rgb = ACCENT_BLUE

        run1 = p1.add_run()
        run1.text = f'"{title}"'
        run1.font.bold = True
        run1.font.color.rgb = TEXT_DARK

        p2 = tf.add_paragraph()
        p2.text = summary
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(3)

        top_pos += 1.15

    # ==========================================
    # SLIDE 3: Where Do We Come In & Our Novelty
    # ==========================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3, LIGHT_BG)
    add_header(slide3, "Where Do We Come In? Our Core Novelty & Contributions")

    novelties = [
        ("1. Privacy-Scope Alignment Audit", "Directly resolves the prevalent literature mismatch by enforcing client-level central DP (client updates clipped, Gaussian noise added to server sum, Poisson RDP ledger) for client deletion requests, rather than per-example DP-SGD."),
        ("2. Empirical Redundancy Frontier Map", "Establishes a rigorous map over (ε, heterogeneity α, deletion size, request count) discovering where DP-only protection is statistically equivalent to retraining target, rendering post-hoc unlearning redundant."),
        ("3. Retrain Variability Equivalence Band", "Evaluates unlearning methods against independent target retrains (MR vs MR') rather than demanding zero parameter distance, preventing false conclusions driven by standard stochastic optimization noise."),
        ("4. Tri-Population & Cost-Aware Ledger", "Simultaneously evaluates Forgotten, Retained, and Unseen MIA risks while tracking communication bytes, GPU training hours, and sensitive per-client update storage overhead.")
    ]

    positions = [
        (Inches(0.8), Inches(1.4), Inches(5.6), Inches(2.6)),
        (Inches(6.8), Inches(1.4), Inches(5.7), Inches(2.6)),
        (Inches(0.8), Inches(4.3), Inches(5.6), Inches(2.6)),
        (Inches(6.8), Inches(4.3), Inches(5.7), Inches(2.6)),
    ]

    for (title, desc), (l, t, w, h) in zip(novelties, positions):
        card = create_card(slide3, l, t, w, h)
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.25)
        tf.margin_top = Inches(0.2)
        tf.margin_right = Inches(0.25)

        p = tf.paragraphs[0]
        p.text = title
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = ACCENT_BLUE

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_DARK
        p2.space_before = Pt(8)

    # ==========================================
    # SLIDE 4: Previous Weeks' Work & Results
    # ==========================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4, LIGHT_BG)
    add_header(slide4, "Phase 1 Results & Empirical Identifications (CIFAR-10 Real Data)")

    # Left Column: Table of Results
    tbl_card = create_card(slide4, Inches(0.8), Inches(1.4), Inches(7.5), Inches(5.5))
    tf_tbl = tbl_card.text_frame
    tf_tbl.word_wrap = True
    tf_tbl.margin_left = Inches(0.15)
    tf_tbl.margin_top = Inches(0.15)

    p_tbl = tf_tbl.paragraphs[0]
    p_tbl.text = "Phase 1 Benchmark Grid (12 Clients, 1 Full-Client Deletion, 5 Seeds/Cell)"
    p_tbl.font.bold = True
    p_tbl.font.size = Pt(11)
    p_tbl.font.color.rgb = ACCENT_BLUE

    # Add PowerPoint Table
    rows, cols = 10, 6
    left, top, width, height = Inches(0.9), Inches(1.85), Inches(7.3), Inches(4.9)
    table_shape = slide4.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    headers = ["ε", "α", "Method", "Test Acc", "JS to Retrain", "Forgot MIA AUC"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(9.5)
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER

    data = [
        ["2.0", "0.2", "DP-Only (No Action)", "0.549 ± 0.046", "0.286 ± 0.103", "0.510 ± 0.035"],
        ["2.0", "0.2", "Retained Fine-Tune", "0.585 ± 0.037", "0.287 ± 0.090", "0.508 ± 0.034"],
        ["2.0", "0.2", "Cached Reconstruction", "0.547 ± 0.046", "0.286 ± 0.104", "0.510 ± 0.035"],
        ["2.0", "0.2", "Independent Retrain (MR')", "0.519 ± 0.128", "0.304 ± 0.065", "0.503 ± 0.019"],
        ["8.0", "0.2", "DP-Only (No Action)", "0.613 ± 0.040", "0.187 ± 0.065", "0.505 ± 0.033"],
        ["8.0", "0.2", "Retained Fine-Tune", "0.649 ± 0.023", "0.179 ± 0.047", "0.497 ± 0.035"],
        ["8.0", "0.2", "Cached Reconstruction", "0.603 ± 0.037", "0.189 ± 0.067", "0.505 ± 0.035"],
        ["∞", "0.2", "DP-Only (No Action)", "0.776 ± 0.010", "0.0014 ± 0.0007", "0.471 ± 0.015"],
        ["∞", "0.2", "Retained Fine-Tune", "0.782 ± 0.011", "0.0016 ± 0.0006", "0.472 ± 0.014"],
    ]

    for row_idx, row_data in enumerate(data, start=1):
        for col_idx, val in enumerate(row_data):
            cell = table.cell(row_idx, col_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_BG if row_idx % 2 == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(8.5)
                p.font.color.rgb = TEXT_DARK
                if col_idx in [0, 1, 3, 4, 5]:
                    p.alignment = PP_ALIGN.CENTER

    # Right Column: Summary of Identifications
    sum_card = create_card(slide4, Inches(8.5), Inches(1.4), Inches(4.0), Inches(5.5), bg_color=CARD_BG, border_color=ACCENT_BLUE)
    tf_sum = sum_card.text_frame
    tf_sum.word_wrap = True
    tf_sum.margin_left = Inches(0.2)
    tf_sum.margin_top = Inches(0.2)
    tf_sum.margin_right = Inches(0.2)

    p_s = tf_sum.paragraphs[0]
    p_s.text = "KEY IDENTIFICATIONS FROM RESULTS"
    p_s.font.bold = True
    p_s.font.size = Pt(12)
    p_s.font.color.rgb = ACCENT_BLUE

    points = [
        ("1. Redundancy at Low Epsilon", "Across all cells (ε=2, 8), explicit unlearning (fine-tuning/reconstruction) produced NO practically material JS improvement over DP-only (Mean marginal JS benefit ≈ -0.001 to +0.013)."),
        ("2. Noise Dominance", "DP noise dominates model updates at strong privacy, rendering post-hoc unlearning redundant relative to retrain variability (q90 ≈ 0.259 - 0.370)."),
        ("3. Storage & Exposure Risk", "Cached update reconstruction requires storing 65 KiB/run of sensitive client update history, creating separate privacy exposure without functional benefit."),
        ("4. Baseline Sanity", "In the non-private control (ε=∞), test JS divergence to retrain is minimal (~0.0014), confirming baseline evaluator correctness.")
    ]

    for title, desc in points:
        p = tf_sum.add_paragraph()
        p.text = title
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        p.space_before = Pt(8)

        p2 = tf_sum.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(9)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(2)

    # ==========================================
    # SLIDE 5: Workflow, Current Standing & Next Steps
    # ==========================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5, LIGHT_BG)
    add_header(slide5, "Research Workflow, Current Standing & Next Roadmap")

    phases = [
        ("Phase 0: Feasibility & Protocol", "COMPLETE", CYAN, "Built deterministic simulator, Poisson RDP ledger, deletion manifests, retrain variability checks, and validation script."),
        ("Phase 1: Exploratory Benchmark", "COMPLETE", CYAN, "Executed 30 real CIFAR-10 binary feature runs across 6 cells (ε={2,8,∞}, α={0.2,0.75}). Discovered initial redundancy signal."),
        ("Phase 2: Core Factorial Study", "CURRENT GATE", ACCENT_BLUE, "Scale to multi-class CIFAR-10/100 and FEMNIST with ResNet-18/GroupNorm CNN. Grid: ε={1,2,4,8,∞}, α={0.1,0.3,1.0}, 25% partial deletion."),
        ("Phase 3: Confirmation & Attacks", "UPCOMING", TEXT_MUTED, "Run 10 confirmation seeds on transition cells, implement full TC-UMIA tri-population MIA attacks, and test 4-request sequential deletion streams."),
        ("Phase 4: Artifact Release & Paper", "UPCOMING", TEXT_MUTED, "Publish frozen deletion manifests, privacy ledgers, reproducible benchmark codebase, and finalize paper submission.")
    ]

    top_p = 1.4
    for name, status, st_color, detail in phases:
        card = create_card(slide5, Inches(0.8), Inches(top_p), Inches(11.7), Inches(1.05))
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.25)
        tf.margin_top = Inches(0.15)
        tf.margin_right = Inches(0.25)

        p = tf.paragraphs[0]
        p.text = f"{name}  |  "
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_DARK

        st_run = p.add_run()
        st_run.text = f"[{status}]"
        st_run.font.bold = True
        st_run.font.size = Pt(12)
        st_run.font.color.rgb = st_color

        p2 = tf.add_paragraph()
        p2.text = detail
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_before = Pt(4)

        top_p += 1.15

    # ==========================================
    # SLIDE 6 (APPENDIX 1): Formulations & DP Ledger
    # ==========================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6, LIGHT_BG)
    add_header(slide6, "Appendix A: Mathematical Formulations & Central DP Mechanism")

    # Card 1: Client Sampling & Clipping
    c_a1 = create_card(slide6, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.5))
    tf_a1 = c_a1.text_frame
    tf_a1.word_wrap = True
    tf_a1.margin_left = Inches(0.25)
    tf_a1.margin_top = Inches(0.2)
    tf_a1.margin_right = Inches(0.25)

    p = tf_a1.paragraphs[0]
    p.text = "1. CLIENT-LEVEL DP MECHANISM"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_BLUE

    eqs1 = [
        ("Poisson Client Selection:", "Client i ∈ St ~ Bernoulli(q),  where q = client_sample_rate"),
        ("Local Update Computation:", "Δi(t) = LocalSGD(w(t), Di) - w(t)"),
        ("L2 Norm Clipping:", "Δ̄i(t) = Δi(t) · min( 1, C / ||Δi(t)||₂ )"),
        ("Server Aggregation & Noise Addition:", "w(t+1) = w(t) + (1 / (q·N)) · [ Σ_{i∈St} Δ̄i(t) + N(0, σ² C² I) ]"),
        ("Public Normalizer Note:", "Divided by fixed public q·N (expected sample size), NEVER realized |St|, strictly preserving RDP guarantee.")
    ]
    for lbl, eq in eqs1:
        p_l = tf_a1.add_paragraph()
        p_l.text = lbl
        p_l.font.bold = True
        p_l.font.size = Pt(10)
        p_l.font.color.rgb = TEXT_DARK
        p_l.space_before = Pt(8)

        p_e = tf_a1.add_paragraph()
        p_e.text = eq
        p_e.font.size = Pt(9.5)
        p_e.font.color.rgb = SLATE
        p_e.space_before = Pt(2)

    # Card 2: RDP Accounting Formula
    c_a2 = create_card(slide6, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.5))
    tf_a2 = c_a2.text_frame
    tf_a2.word_wrap = True
    tf_a2.margin_left = Inches(0.25)
    tf_a2.margin_top = Inches(0.2)
    tf_a2.margin_right = Inches(0.25)

    p = tf_a2.paragraphs[0]
    p.text = "2. RÉNYI DP (RDP) ACCOUNTING"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_BLUE

    eqs2 = [
        ("Composed RDP Orders:", "α ∈ {2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 48, 64, 96, 128, 256}"),
        ("Elementary Event:", "E = PoissonSampledDpEvent( q, GaussianDpEvent(σ) )"),
        ("Self-Composition over T Rounds:", "E_total = SelfComposedDpEvent( E, T )"),
        ("Epsilon Conversion at Target δ:", "ε(δ) = min_{α > 1} [ D_α(E_total) + ln(1/δ) / (α - 1) ]"),
        ("Privacy Adjacency Contract:", "Add/remove one complete client dataset D_i. Protects released final model checkpoint.")
    ]
    for lbl, eq in eqs2:
        p_l = tf_a2.add_paragraph()
        p_l.text = lbl
        p_l.font.bold = True
        p_l.font.size = Pt(10)
        p_l.font.color.rgb = TEXT_DARK
        p_l.space_before = Pt(8)

        p_e = tf_a2.add_paragraph()
        p_e.text = eq
        p_e.font.size = Pt(9.5)
        p_e.font.color.rgb = SLATE
        p_e.space_before = Pt(2)

    # ==========================================
    # SLIDE 7 (APPENDIX 2): Evaluation Metrics & Params
    # ==========================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7, LIGHT_BG)
    add_header(slide7, "Appendix B: Metric Definitions & Hyperparameter Settings")

    # Card 1: Metrics Equations
    c_b1 = create_card(slide7, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.5))
    tf_b1 = c_b1.text_frame
    tf_b1.word_wrap = True
    tf_b1.margin_left = Inches(0.25)
    tf_b1.margin_top = Inches(0.2)
    tf_b1.margin_right = Inches(0.25)

    p = tf_b1.paragraphs[0]
    p.text = "1. EVALUATION METRIC FORMULAS"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_BLUE

    eqs3 = [
        ("Marginal Unlearning Benefit (MUB):", "MUB_j = distance_j( M_DP, M_R ) - distance_j( M_U, M_R )"),
        ("Jensen-Shannon (JS) Divergence:", "D_JS(P || Q) = 1/2 D_KL(P || (P+Q)/2) + 1/2 D_KL(Q || (P+Q)/2)\nwhere P = Softmax(M_U(x)), Q = Softmax(M_R(x))"),
        ("MIA Advantage Score:", "Advantage = max_τ [ TPR(τ) - FPR(τ) ]"),
        ("Retrain Equivalence Threshold:", "Equivalence declared if |MUB| < q90( ||M_R - M_R'|| )")
    ]
    for lbl, eq in eqs3:
        p_l = tf_b1.add_paragraph()
        p_l.text = lbl
        p_l.font.bold = True
        p_l.font.size = Pt(10)
        p_l.font.color.rgb = TEXT_DARK
        p_l.space_before = Pt(8)

        p_e = tf_b1.add_paragraph()
        p_e.text = eq
        p_e.font.size = Pt(9.5)
        p_e.font.color.rgb = SLATE
        p_e.space_before = Pt(2)

    # Card 2: Exact Hyperparameter Settings Table
    c_b2 = create_card(slide7, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.5))
    tf_b2 = c_b2.text_frame
    tf_b2.word_wrap = True
    tf_b2.margin_left = Inches(0.25)
    tf_b2.margin_top = Inches(0.2)
    tf_b2.margin_right = Inches(0.25)

    p = tf_b2.paragraphs[0]
    p.text = "2. EXPERIMENT PARAMETER VALUES"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ACCENT_BLUE

    params = [
        ("Population Size (N):", "12 clients (Phase 1) / 50-200 (Phase 2)"),
        ("Client Sample Rate (q):", "0.25 (Poisson independent sampling)"),
        ("L2 Clip Norm (C):", "1.0"),
        ("Noise Multiplier (σ):", "0.5 (ε≈8.0), 2.0 (ε≈2.0), 0.0 (ε=∞)"),
        ("Target Delta (δ):", "1.0e-5"),
        ("Federated Rounds (T):", "300 rounds"),
        ("Local Training:", "5 epochs, LR = 0.01, Batch Size = 32"),
        ("Dirichlet Skew (α):", "0.2 (high heterogeneity), 0.75 (low)"),
        ("Deletion Granularity:", "1 full client (100% data) / 25% partial")
    ]
    for param_name, param_val in params:
        p_p = tf_b2.add_paragraph()
        p_p.text = f"{param_name} "
        p_p.font.bold = True
        p_p.font.size = Pt(9.5)
        p_p.font.color.rgb = TEXT_DARK
        p_p.space_before = Pt(4)

        r_v = p_p.add_run()
        r_v.text = param_val
        r_v.font.bold = False
        r_v.font.color.rgb = TEXT_MUTED

    # Ensure output directory exists
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/DP_ForgetBench_Presentation.pptx"
    prs.save(output_path)
    print(f"Successfully generated PowerPoint presentation at: {output_path}")

if __name__ == "__main__":
    create_presentation()
