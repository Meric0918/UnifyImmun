from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import os

result_dir = "/home/mclab/mjp/unifyimmun/source/results/tcr_experiment/tcr_eval_report_20260323_203715"
plots_dir = os.path.join(result_dir, "plots")

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Slide 1: Overview and Metrics
slide_layout = prs.slide_layouts[6]
slide1 = prs.slides.add_slide(slide_layout)

title_box = slide1.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12), Inches(0.8))
title_frame = title_box.text_frame
title_para = title_frame.paragraphs[0]
title_para.text = "TCR Model Evaluation Report - Lung Cancer Dataset"
title_para.font.size = Pt(32)
title_para.font.bold = True
title_para.alignment = PP_ALIGN.CENTER

subtitle_box = slide1.shapes.add_textbox(
    Inches(0.5), Inches(1.2), Inches(6), Inches(0.5)
)
subtitle_frame = subtitle_box.text_frame
subtitle_para = subtitle_frame.paragraphs[0]
subtitle_para.text = "Model: TCR_2 | Dataset: LungCancer_TCR.csv | Samples: 383"
subtitle_para.font.size = Pt(16)

metrics_title = slide1.shapes.add_textbox(
    Inches(0.5), Inches(1.8), Inches(6), Inches(0.4)
)
metrics_title.text_frame.paragraphs[0].text = "Key Performance Metrics"
metrics_title.text_frame.paragraphs[0].font.size = Pt(20)
metrics_title.text_frame.paragraphs[0].font.bold = True

metrics_box = slide1.shapes.add_textbox(
    Inches(0.5), Inches(2.3), Inches(5.5), Inches(4.5)
)
tf = metrics_box.text_frame
tf.word_wrap = True

metrics_data = [
    ("ROC-AUC", "0.710", "Moderate discriminative ability"),
    ("AUPR", "0.576", "Precision-Recall performance"),
    ("Accuracy", "49.3%", "Overall classification accuracy"),
    ("Sensitivity", "4.92%", "True positive rate (low)"),
    ("Specificity", "90.0%", "True negative rate (high)"),
    ("Precision", "31.0%", "Positive predictive value"),
    ("F1-Score", "0.085", "Harmonic mean of P and R"),
    ("MCC", "-0.096", "Slight negative correlation"),
]

for i, (name, value, desc) in enumerate(metrics_data):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = f"{name}: {value}"
    p.font.size = Pt(16)
    p.font.bold = True
    p.space_after = Pt(4)

    p2 = tf.add_paragraph()
    p2.text = f"  → {desc}"
    p2.font.size = Pt(12)
    p2.font.color.rgb = RGBColor(100, 100, 100)
    p2.space_after = Pt(12)

confusion_title = slide1.shapes.add_textbox(
    Inches(6.5), Inches(1.8), Inches(6), Inches(0.4)
)
confusion_title.text_frame.paragraphs[0].text = "Confusion Matrix"
confusion_title.text_frame.paragraphs[0].font.size = Pt(20)
confusion_title.text_frame.paragraphs[0].font.bold = True

cm_box = slide1.shapes.add_textbox(Inches(6.5), Inches(2.3), Inches(5.5), Inches(2.5))
cm_tf = cm_box.text_frame
cm_tf.word_wrap = True

cm_p = cm_tf.paragraphs[0]
cm_p.text = "                    Predicted"
cm_p.font.size = Pt(14)
cm_p.font.bold = True

cm_p2 = cm_tf.add_paragraph()
cm_p2.text = "              Negative    Positive"
cm_p2.font.size = Pt(14)

cm_p3 = cm_tf.add_paragraph()
cm_p3.text = "Actual Neg     TN=180      FP=20"
cm_p3.font.size = Pt(14)

cm_p4 = cm_tf.add_paragraph()
cm_p4.text = "Actual Pos     FN=174      TP=9"
cm_p4.font.size = Pt(14)

cm_p5 = cm_tf.add_paragraph()
cm_p5.text = ""
cm_p5.font.size = Pt(8)

cm_p6 = cm_tf.add_paragraph()
cm_p6.text = "True Negatives: 180 | False Positives: 20"
cm_p6.font.size = Pt(12)

cm_p7 = cm_tf.add_paragraph()
cm_p7.text = "False Negatives: 174 | True Positives: 9"
cm_p7.font.size = Pt(12)

analysis_box = slide1.shapes.add_textbox(
    Inches(6.5), Inches(5.0), Inches(6), Inches(2.0)
)
analysis_tf = analysis_box.text_frame
analysis_tf.word_wrap = True

ap = analysis_tf.paragraphs[0]
ap.text = "Analysis:"
ap.font.size = Pt(16)
ap.font.bold = True

ap2 = analysis_tf.add_paragraph()
ap2.text = "• High specificity (90%) but low sensitivity (4.9%)"
ap2.font.size = Pt(12)

ap3 = analysis_tf.add_paragraph()
ap3.text = "• Model is conservative - predicts negative more often"
ap3.font.size = Pt(12)

ap4 = analysis_tf.add_paragraph()
ap4.text = "• ROC-AUC (0.71) shows moderate discrimination"
ap4.font.size = Pt(12)

ap5 = analysis_tf.add_paragraph()
ap5.text = "• Class imbalance: 200 neg vs 183 pos samples"
ap5.font.size = Pt(12)

# Slide 2: ROC and PR Curves
slide2 = prs.slides.add_slide(slide_layout)

title_box2 = slide2.shapes.add_textbox(
    Inches(0.5), Inches(0.3), Inches(12), Inches(0.6)
)
title_frame2 = title_box2.text_frame
title_para2 = title_frame2.paragraphs[0]
title_para2.text = "ROC and Precision-Recall Curves"
title_para2.font.size = Pt(28)
title_para2.font.bold = True
title_para2.alignment = PP_ALIGN.CENTER

roc_path = os.path.join(plots_dir, "roc_custom_fold_1.png")
pr_path = os.path.join(plots_dir, "pr_custom_fold_1.png")

if os.path.exists(roc_path):
    slide2.shapes.add_picture(roc_path, Inches(0.5), Inches(1.2), width=Inches(5.8))

    roc_label = slide2.shapes.add_textbox(
        Inches(0.5), Inches(5.5), Inches(5.8), Inches(1.5)
    )
    roc_tf = roc_label.text_frame
    roc_tf.word_wrap = True
    rp = roc_tf.paragraphs[0]
    rp.text = "ROC Curve (AUC = 0.710)"
    rp.font.size = Pt(14)
    rp.font.bold = True
    rp2 = roc_tf.add_paragraph()
    rp2.text = "Shows trade-off between TPR and FPR"
    rp2.font.size = Pt(11)
    rp3 = roc_tf.add_paragraph()
    rp3.text = "AUC > 0.7 indicates moderate predictive power"
    rp3.font.size = Pt(11)
else:
    missing_box = slide2.shapes.add_textbox(
        Inches(0.5), Inches(2), Inches(5.8), Inches(1)
    )
    missing_box.text_frame.paragraphs[0].text = "ROC curve image not found"

if os.path.exists(pr_path):
    slide2.shapes.add_picture(pr_path, Inches(6.8), Inches(1.2), width=Inches(5.8))

    pr_label = slide2.shapes.add_textbox(
        Inches(6.8), Inches(5.5), Inches(5.8), Inches(1.5)
    )
    pr_tf = pr_label.text_frame
    pr_tf.word_wrap = True
    pp = pr_tf.paragraphs[0]
    pp.text = "Precision-Recall Curve (AUPR = 0.576)"
    pp.font.size = Pt(14)
    pp.font.bold = True
    pp2 = pr_tf.add_paragraph()
    pp2.text = "Shows trade-off between Precision and Recall"
    pp2.font.size = Pt(11)
    pp3 = pr_tf.add_paragraph()
    pp3.text = "Useful for imbalanced datasets"
    pp3.font.size = Pt(11)
else:
    missing_box2 = slide2.shapes.add_textbox(
        Inches(6.8), Inches(2), Inches(5.8), Inches(1)
    )
    missing_box2.text_frame.paragraphs[0].text = "PR curve image not found"

footer = slide2.shapes.add_textbox(Inches(0.5), Inches(7.0), Inches(12), Inches(0.4))
footer.text_frame.paragraphs[
    0
].text = "Dataset: LungCancer_TCR.csv | Model: TCR_2 | Threshold: 0.5 | Device: CUDA"
footer.text_frame.paragraphs[0].font.size = Pt(10)
footer.text_frame.paragraphs[0].font.color.rgb = RGBColor(128, 128, 128)
footer.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

output_path = "/home/mclab/mjp/unifyimmun/source/TCR_Evaluation_Report.pptx"
prs.save(output_path)
print(f"PPT saved to: {output_path}")
