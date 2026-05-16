"""
Generate PPT for ESM2 Embedding Integration Project Summary.
4 pages with charts and diagrams.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

# Create output directory
output_dir = "/home/mclab/mjp/unifyimmun/docs"
os.makedirs(output_dir, exist_ok=True)

# Create presentation
prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)

def add_title_slide(prs, title, subtitle):
    """Add title slide"""
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(12.33), Inches(1.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 51, 102)
    p.alignment = PP_ALIGN.CENTER

    # Subtitle
    sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(12.33), Inches(1))
    tf = sub_box.text_frame
    p = tf.paragraphs[0]
    p.text = subtitle
    p.font.size = Pt(24)
    p.font.color.rgb = RGBColor(102, 102, 102)
    p.alignment = PP_ALIGN.CENTER

    return slide

def add_content_slide(prs, title, content_items, image_path=None):
    """Add content slide with bullet points and optional image"""
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.33), Inches(0.8))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 51, 102)

    # Content
    if image_path:
        # Left side: content
        content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(6), Inches(5.5))
        # Right side: image
        slide.shapes.add_picture(image_path, Inches(7), Inches(1.2), width=Inches(5.5))
    else:
        content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(12.33), Inches(5.5))

    tf = content_box.text_frame
    tf.word_wrap = True

    for i, item in enumerate(content_items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()

        if isinstance(item, tuple):
            # Tuple: (text, level)
            p.text = item[0]
            p.level = item[1]
        else:
            p.text = "• " + item
            p.level = 0

        p.font.size = Pt(18)
        p.font.color.rgb = RGBColor(51, 51, 51)
        p.space_after = Pt(12)

    return slide

# Generate charts
print("Generating charts...")

# Chart 1: Original vs New Embedding Comparison
fig, ax = plt.subplots(figsize=(10, 6))
methods = ['Original\nnn.Embedding', 'ESM2\nPretrained']
vocab_sizes = [21, 33]
embed_dims = [64, 1280]
output_dims = [64, 64]

x = np.arange(len(methods))
width = 0.25

bars1 = ax.bar(x - width, vocab_sizes, width, label='Vocab Size', color='#3498db')
bars2 = ax.bar(x, embed_dims, width, label='Embedding Dim', color='#e74c3c')
bars3 = ax.bar(x + width, output_dims, width, label='Output Dim', color='#2ecc71')

ax.set_ylabel('Size', fontsize=14)
ax.set_title('Embedding Method Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 1400)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout()
chart1_path = os.path.join(output_dir, 'chart1_embedding_comparison.png')
plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Model Architecture Flow
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis('off')

# Draw boxes
boxes = [
    (0.5, 3, 2, 1.5, 'Input\nSequence', '#f39c12'),
    (3, 3, 2, 1.5, 'ESM2\nTokenizer', '#3498db'),
    (5.5, 3, 2, 1.5, 'ESM2 Model\n(650M)', '#9b59b6'),
    (8, 3, 2, 1.5, 'Projection\n1280→64', '#e74c3c'),
    (10.5, 3, 1, 1.5, 'Output\n64-d', '#2ecc71'),
]

for x, y, w, h, text, color in boxes:
    rect = plt.Rectangle((x, y), w, h, fill=True, facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, fontweight='bold')

# Draw arrows
arrows = [(2.5, 3.75, 0.5, 0), (5, 3.75, 0.5, 0), (7.5, 3.75, 0.5, 0), (10, 3.75, 0.5, 0)]
for x, y, dx, dy in arrows:
    ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Add info boxes
info_texts = [
    (3, 1.5, 'Frozen: 1.3B params'),
    (8, 1.5, 'Trainable: 98K params'),
]
for x, y, text in info_texts:
    ax.text(x, y, text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

ax.set_title('ESM2 Embedding Architecture', fontsize=16, fontweight='bold', y=4.5)
plt.tight_layout()
chart2_path = os.path.join(output_dir, 'chart2_architecture.png')
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Training Flow Diagram
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# Phase boxes
phase1_boxes = [
    (1, 4.5, 2.5, 1.2, 'Phase 1', '#3498db', 'white'),
    (0.5, 2.5, 2, 1, 'HLA_ESM2', '#2980b9', 'white'),
    (3.5, 2.5, 2, 1, 'TCR_ESM2', '#2980b9', 'white'),
]

phase2_boxes = [
    (7, 4.5, 2.5, 1.2, 'Phase 2', '#e74c3c', 'white'),
    (6.5, 2.5, 2, 1, 'HLA_ESM2_2', '#c0392b', 'white'),
    (9.5, 2.5, 2, 1, 'TCR_ESM2_2', '#c0392b', 'white'),
]

# Draw phase labels
for x, y, w, h, text, color, tc in phase1_boxes + phase2_boxes:
    rect = plt.Rectangle((x, y), w, h, fill=True, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=12, fontweight='bold', color=tc)

# Draw arrows for encoder transfer
arrow_style = dict(arrowstyle='->', color='#27ae60', lw=2)
ax.annotate('', xy=(3.5, 3), xytext=(2.5, 3), arrowprops=arrow_style)  # HLA1 to TCR1
ax.annotate('', xy=(6.5, 3), xytext=(5.5, 3), arrowprops=arrow_style)  # TCR1 to HLA2
ax.annotate('', xy=(9.5, 3), xytext=(8.5, 3), arrowprops=arrow_style)  # HLA2 to TCR2

# Add labels
ax.text(3, 3.5, 'encoder_P', fontsize=9, ha='center', color='#27ae60')
ax.text(6, 3.5, 'encoder_P', fontsize=9, ha='center', color='#27ae60')
ax.text(9, 3.5, 'encoder_P', fontsize=9, ha='center', color='#27ae60')

# Training info
ax.text(1.5, 1.2, '30 epochs\n5 folds\nbatch=64', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#ecf0f1'))
ax.text(7.5, 1.2, '30 epochs\n5 folds\nbatch=64', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#ecf0f1'))

ax.set_title('Training Pipeline Flow', fontsize=16, fontweight='bold', y=5.5)
plt.tight_layout()
chart3_path = os.path.join(output_dir, 'chart3_training_flow.png')
plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
plt.close()

# Chart 4: File Structure and Statistics
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Files pie chart
files_data = {'New Files': 6, 'Modified': 1}
colors = ['#3498db', '#e74c3c']
axes[0].pie(files_data.values(), labels=files_data.keys(), colors=colors,
            autopct='%1.0f%%', startangle=90, textprops={'fontsize': 14})
axes[0].set_title('Project Files Changes', fontsize=14, fontweight='bold')

# Right: Parameter bar chart
params_data = {
    'Frozen (ESM2)': 1304707882,
    'Trainable': 985410
}
colors = ['#9b59b6', '#2ecc71']
bars = axes[1].bar(params_data.keys(), params_data.values(), color=colors)
axes[1].set_ylabel('Parameters', fontsize=12)
axes[1].set_title('Model Parameters', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', labelsize=10)

# Format y-axis
axes[1].set_ylim(0, 1.5e9)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
chart4_path = os.path.join(output_dir, 'chart4_statistics.png')
plt.savefig(chart4_path, dpi=150, bbox_inches='tight')
plt.close()

print("Charts generated successfully!")

# Create slides
print("Creating PPT slides...")

# Slide 1: Title
add_title_slide(prs,
    "ESM2 Embedding Integration",
    "UnifyImmun Project Modification Summary\nApril 2026")

# Slide 2: Background and Motivation
slide2_content = [
    "Original Embedding Method: nn.Embedding",
    ("Vocab size: 21 (20 amino acids + padding)", 1),
    ("Embedding dim: 64 (learned from scratch)", 1),
    ("No pretrained knowledge", 1),
    "New ESM2 Embedding Method:",
    ("Model: ESM2-650M (33 layers, 1280-dim)", 1),
    ("Pretrained on 250M protein sequences", 1),
    ("Contains rich biological knowledge", 1),
    "Key Benefits:",
    ("Capture evolutionary information", 1),
    ("Better protein sequence representation", 1),
    ("Improved binding prediction accuracy", 1),
]
add_content_slide(prs, "Background & Motivation", slide2_content, chart1_path)

# Slide 3: Implementation Architecture
slide3_content = [
    "ESM2TokenizerWrapper",
    ("Converts amino acid sequences to ESM2 tokens", 1),
    ("Adds <cls> and <eos> automatically", 1),
    "ESM2Embedding Class",
    ("Loads pretrained EsmModel (frozen)", 1),
    ("Projects 1280-dim to 64-dim", 1),
    "New Encoder Classes",
    ("Encoder_ESM2: for HLA/TCR (length 34)", 1),
    ("Encoder_padding_ESM2: for peptide (length 15)", 1),
    "Model Classes",
    ("Mymodel_HLA_ESM2, Mymodel_TCR_ESM2", 1),
]
add_content_slide(prs, "Implementation Architecture", slide3_content, chart2_path)

# Slide 4: Training Pipeline
slide4_content = [
    "New Training Scripts (6 files):",
    ("HLA_ESM2.py, TCR_ESM2.py (Phase 1)", 1),
    ("HLA_ESM2_2.py, TCR_ESM2_2.py (Phase 2)", 1),
    ("run_all_phases_esm2.py (pipeline)", 1),
    "Encoder Transfer Flow:",
    ("Phase 1: HLA→TCR (encoder_P transfer)", 1),
    ("Phase 2: TCR→HLA→TCR (iterative)", 1),
    "SwanLab Integration:",
    ("All metrics logged to SwanLab", 1),
    ("Real-time cloud monitoring", 1),
    "Estimated Training Time:",
    ("~150 hours per phase", 1),
    ("~600 hours total pipeline", 1),
]
add_content_slide(prs, "Training Pipeline & Files", slide3_content, chart3_path)

# Actually use chart3 for slide 4
prs.slides[3].shapes.add_picture(chart3_path, Inches(7), Inches(1.2), width=Inches(5.5))

# Save presentation
ppt_path = os.path.join(output_dir, 'ESM2_Embedding_Integration_Summary.pptx')
prs.save(ppt_path)

print(f"\nPPT saved to: {ppt_path}")
print("\nPPT Contents:")
print("  Slide 1: Title")
print("  Slide 2: Background & Motivation (with embedding comparison chart)")
print("  Slide 3: Implementation Architecture (with architecture diagram)")
print("  Slide 4: Training Pipeline (with training flow diagram)")