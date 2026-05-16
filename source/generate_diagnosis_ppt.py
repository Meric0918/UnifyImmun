"""
生成FGM训练问题排查PPT报告
基于实际排查过程：loss不变化 -> FGM BatchNorm问题 -> 解决方案
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# 创建PPT
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# 定义颜色
TITLE_COLOR = RGBColor(0, 51, 102)
SUBTITLE_COLOR = RGBColor(102, 102, 102)
CONTENT_COLOR = RGBColor(51, 51, 51)
HIGHLIGHT_COLOR = RGBColor(192, 0, 0)
SUCCESS_COLOR = RGBColor(0, 128, 0)
CODE_BG_COLOR = RGBColor(245, 245, 245)

def add_code_box(slide, code_text, left, top, width, height):
    """添加代码框"""
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    rect.fill.solid()
    rect.fill.fore_color.rgb = CODE_BG_COLOR
    rect.line.fill.background()

    code_box = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.1), width - Inches(0.2), height - Inches(0.2))
    tf = code_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = code_text
    p.font.size = Pt(11)
    p.font.color.rgb = CONTENT_COLOR
    p.font.name = "Courier New"

def add_bullet_text(textbox, items, font_size=16, color=None):
    """添加带项目符号的文本"""
    tf = textbox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color if color else CONTENT_COLOR
        p.space_before = Pt(8)

# ========== 第1页：问题描述与排查思路 ==========
slide1 = prs.slides.add_slide(prs.slide_layouts[6])

# 标题
title_box = slide1.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.8))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "第1页：问题描述与排查思路"
p.font.size = Pt(32)
p.font.bold = True
p.font.color.rgb = TITLE_COLOR

# 分隔线
line = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.1), Inches(12.333), Inches(0.02))
line.fill.solid()
line.fill.fore_color.rgb = TITLE_COLOR
line.line.fill.background()

# 问题现象
problem_title = slide1.shapes.add_textbox(Inches(0.5), Inches(1.4), Inches(6), Inches(0.4))
tf = problem_title.text_frame
p = tf.paragraphs[0]
p.text = "问题现象"
p.font.size = Pt(22)
p.font.bold = True
p.font.color.rgb = HIGHLIGHT_COLOR

problem_box = slide1.shapes.add_textbox(Inches(0.5), Inches(1.9), Inches(6), Inches(2))
add_bullet_text(problem_box, [
    "训练文件: HLA_ESM2.py",
    "现象: 第2个epoch后train_loss不再变化",
    "模型: ESM2(frozen) + Transformer + CrossAttention + MLP",
    "关键特征: 使用FGM对抗训练 + BatchNorm1d"
], color=CONTENT_COLOR)

# 排查思路
investigate_title = slide1.shapes.add_textbox(Inches(0.5), Inches(4.0), Inches(6), Inches(0.4))
tf = investigate_title.text_frame
p = tf.paragraphs[0]
p.text = "排查思路"
p.font.size = Pt(22)
p.font.bold = True
p.font.color.rgb = TITLE_COLOR

investigate_box = slide1.shapes.add_textbox(Inches(0.5), Inches(4.5), Inches(6), Inches(2.5))
add_bullet_text(investigate_box, [
    "1. 检查学习率设置 → lr=1e-3, 可能偏高",
    "2. 检查梯度流动 → 梯度正常,非NaN/Inf",
    "3. 检查FGM实现 → 发现可疑点!",
    "4. 检查BatchNorm状态 → 锁定问题根因"
], color=CONTENT_COLOR)

# 模型架构图（右侧）
arch_title = slide1.shapes.add_textbox(Inches(7), Inches(1.4), Inches(6), Inches(0.4))
tf = arch_title.text_frame
p = tf.paragraphs[0]
p.text = "模型架构概览"
p.font.size = Pt(22)
p.font.bold = True
p.font.color.rgb = TITLE_COLOR

arch_box = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7), Inches(1.9), Inches(5.8), Inches(5))
arch_box.fill.solid()
arch_box.fill.fore_color.rgb = CODE_BG_COLOR
arch_box.line.color.rgb = TITLE_COLOR

arch_text = slide1.shapes.add_textbox(Inches(7.2), Inches(2.1), Inches(5.5), Inches(4.8))
tf = arch_text.text_frame
tf.word_wrap = True
arch_items = [
    "ESM2 Embedding (frozen)",
    "    ↓ 1280 → 64 (projection可训练)",
    "Encoder_H + Encoder_P",
    "    ↓ Transformer Layer",
    "Cross Attention",
    "    ↓ peptide × HLA",
    "Projection (可训练)",
    "    Linear → ReLU → BatchNorm1d",
    "    → Linear → ReLU → Linear → 输出",
    "",
    "⚠️ BatchNorm1d在projection层中!",
    "    running_mean/var在forward时更新"
]
for i, item in enumerate(arch_items):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = item
    p.font.size = Pt(14)
    p.font.name = "Courier New"
    if "⚠️" in item:
        p.font.color.rgb = HIGHLIGHT_COLOR
        p.font.bold = True
    else:
        p.font.color.rgb = CONTENT_COLOR
    p.space_before = Pt(4)

# ========== 第2页：深度诊断与问题定位 ==========
slide2 = prs.slides.add_slide(prs.slide_layouts[6])

title_box = slide2.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.8))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "第2页：深度诊断与问题定位"
p.font.size = Pt(32)
p.font.bold = True
p.font.color.rgb = TITLE_COLOR

line = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.1), Inches(12.333), Inches(0.02))
line.fill.solid()
line.fill.fore_color.rgb = TITLE_COLOR
line.line.fill.background()

# FGM原始代码分析
fgm_title = slide2.shapes.add_textbox(Inches(0.5), Inches(1.4), Inches(6), Inches(0.4))
tf = fgm_title.text_frame
p = tf.paragraphs[0]
p.text = "原始FGM实现 (有问题)"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = HIGHLIGHT_COLOR

original_code = '''def attack(self, epsilon=1.0):
    # Step 1: 扰动projection参数
    for name, param in self.model.named_parameters():
        if param.requires_grad and "projection" in name:
            param.data.add_(r_at)  # 参数已被扰动!

    # Step 2: 备份BatchNorm running stats
    for name, module in self.model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            self.bn_backup[name] = {
                'running_mean': module.running_mean.clone(),
                ...
            }

    # 问题: 备份在扰动后执行!
    # running stats还未被污染,但下次forward会被污染'''

add_code_box(slide2, original_code, Inches(0.5), Inches(1.85), Inches(6), Inches(2.5))

# 问题执行流程
flow_title = slide2.shapes.add_textbox(Inches(0.5), Inches(4.5), Inches(6), Inches(0.4))
tf = flow_title.text_frame
p = tf.paragraphs[0]
p.text = "问题执行流程"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = HIGHLIGHT_COLOR

flow_box = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(4.95), Inches(6), Inches(2.3))
flow_box.fill.solid()
flow_box.fill.fore_color.rgb = RGBColor(255, 240, 240)
flow_box.line.color.rgb = HIGHLIGHT_COLOR

flow_text = slide2.shapes.add_textbox(Inches(0.7), Inches(5.15), Inches(5.6), Inches(2))
tf = flow_text.text_frame
tf.word_wrap = True
flow_items = [
    "1. 正常forward → backward → 参数梯度计算完成",
    "2. FGM attack → projection参数被扰动",
    "3. FGM forward (对抗样本) → BatchNorm用扰动参数计算",
    "   → running_mean/var被异常激活值污染!",
    "4. FGM restore → 只恢复参数权重",
    "   → running stats仍处于污染状态!",
    "5. 后续epoch → BatchNorm表现异常 → loss停滞"
]
for i, item in enumerate(flow_items):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = item
    p.font.size = Pt(14)
    if "!" in item or "污染" in item:
        p.font.color.rgb = HIGHLIGHT_COLOR
        p.font.bold = True
    else:
        p.font.color.rgb = CONTENT_COLOR
    p.space_before = Pt(6)

# 问题根因（右侧）
root_title = slide2.shapes.add_textbox(Inches(7), Inches(1.4), Inches(5.8), Inches(0.4))
tf = root_title.text_frame
p = tf.paragraphs[0]
p.text = "根本原因"
p.font.size = Pt(22)
p.font.bold = True
p.font.color.rgb = HIGHLIGHT_COLOR

root_box = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7), Inches(1.9), Inches(5.8), Inches(5.3))
root_box.fill.solid()
root_box.fill.fore_color.rgb = RGBColor(255, 240, 240)
root_box.line.color.rgb = HIGHLIGHT_COLOR

root_text = slide2.shapes.add_textbox(Inches(7.2), Inches(2.1), Inches(5.5), Inches(5))
tf = root_text.text_frame
tf.word_wrap = True
root_items = [
    "BatchNorm的双重属性:",
    "",
    "• 权重参数 (weight, bias)",
    "  → 可训练,会被FGM扰动和恢复",
    "",
    "• 运行统计量 (running_mean, running_var)",
    "  → 不是参数! 是forward时动态计算的状态",
    "  → FGM无法扰动它们 (它们没有grad)",
    "",
    "问题:",
    "FGM attack时projection参数被扰动",
    "→ 对抗forward时BatchNorm收到异常激活值",
    "→ running stats基于异常值更新",
    "→ restore只恢复参数,不恢复running stats",
    "→ 污染的running stats持续影响后续训练",
    "",
    "类比:",
    "就像洗衣服时把脏水倒进洗衣机,",
    "然后只把衣服拿出来,脏水还在里面!"
]
for i, item in enumerate(root_items):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = item
    p.font.size = Pt(15)
    if item.startswith("BatchNorm") or item.startswith("问题:") or item.startswith("类比:"):
        p.font.bold = True
        p.font.color.rgb = HIGHLIGHT_COLOR
    else:
        p.font.color.rgb = CONTENT_COLOR
    p.space_before = Pt(4)

# ========== 第3页：解决方案与验证 ==========
slide3 = prs.slides.add_slide(prs.slide_layouts[6])

title_box = slide3.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.8))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "第3页：解决方案与验证"
p.font.size = Pt(32)
p.font.bold = True
p.font.color.rgb = TITLE_COLOR

line = slide3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.1), Inches(12.333), Inches(0.02))
line.fill.solid()
line.fill.fore_color.rgb = TITLE_COLOR
line.line.fill.background()

# 解决方案代码
sol_title = slide3.shapes.add_textbox(Inches(0.5), Inches(1.4), Inches(7), Inches(0.4))
tf = sol_title.text_frame
p = tf.paragraphs[0]
p.text = "修复方案: 冻结BatchNorm track_running_stats"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = SUCCESS_COLOR

fixed_code = '''class FGM_ESM2:
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.bn_track_backup = {}  # 新增

    def attack(self, epsilon=1.0):
        # 关键修复: 先冻结BatchNorm的running stats更新
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self.bn_track_backup[name] = module.track_running_stats
                module.track_running_stats = False  # 冻结!

        # 然后扰动projection参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and "projection" in name:
                self.backup[name] = param.data.clone().detach()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # 恢复参数
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

        # 恢复BatchNorm track状态
        for name, module in self.model.named_modules():
            if name in self.bn_track_backup:
                module.track_running_stats = self.bn_track_backup[name]
        self.bn_track_backup = {}'''

add_code_box(slide3, fixed_code, Inches(0.5), Inches(1.85), Inches(7.2), Inches(3.8))

# 其他优化（右侧）
opt_title = slide3.shapes.add_textbox(Inches(8), Inches(1.4), Inches(4.8), Inches(0.4))
tf = opt_title.text_frame
p = tf.paragraphs[0]
p.text = "其他优化建议"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = TITLE_COLOR

opt_box = slide3.shapes.add_textbox(Inches(8), Inches(1.9), Inches(4.8), Inches(2))
tf = opt_box.text_frame
tf.word_wrap = True
opt_items = [
    "学习率调整:",
    "  修复前: lr=1e-3 (偏高)",
    "  修复后: lr=1e-4 (推荐)",
    "",
    "原因:",
    "  • 只训练projection层(小型MLP)",
    "  • ESM2 frozen,输出是稳定的embedding",
    "  • 太高的lr导致参数更新剧烈",
    "  • 配合FGM扰动,训练更不稳定"
]
for i, item in enumerate(opt_items):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = item
    p.font.size = Pt(14)
    if item.startswith("学习率") or item.startswith("原因"):
        p.font.bold = True
    else:
        p.font.color.rgb = CONTENT_COLOR
    p.space_before = Pt(4)

# 总结
summary_title = slide3.shapes.add_textbox(Inches(8), Inches(4.0), Inches(4.8), Inches(0.4))
tf = summary_title.text_frame
p = tf.paragraphs[0]
p.text = "排查总结"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = SUCCESS_COLOR

summary_box = slide3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8), Inches(4.5), Inches(4.8), Inches(2.8))
summary_box.fill.solid()
summary_box.fill.fore_color.rgb = RGBColor(240, 255, 240)
summary_box.line.color.rgb = SUCCESS_COLOR

summary_text = slide3.shapes.add_textbox(Inches(8.2), Inches(4.7), Inches(4.5), Inches(2.5))
tf = summary_text.text_frame
tf.word_wrap = True
summary_items = [
    "排查步骤:",
    "1. 问题描述确认 → loss不变化",
    "2. 代码审查 → 定位可疑模块",
    "3. 深度分析 → 理解FGM+BN交互",
    "4. 根因锁定 → BN running stats污染",
    "",
    "关键洞察:",
    "BatchNorm的running stats不是参数",
    "但在forward时会被更新",
    "FGM restore漏掉了它们!"
]
for i, item in enumerate(summary_items):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = item
    p.font.size = Pt(13)
    if item.startswith("排查") or item.startswith("关键"):
        p.font.bold = True
        p.font.color.rgb = SUCCESS_COLOR
    else:
        p.font.color.rgb = CONTENT_COLOR
    p.space_before = Pt(4)

# 保存
output_path = "/home/mclab/mjp/unifyimmun/docs/FGM_Training_Diagnosis.pptx"
prs.save(output_path)
print(f"PPT已保存: {output_path}")