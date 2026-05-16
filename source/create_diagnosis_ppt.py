"""
生成诊断排查PPT报告
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
TITLE_COLOR = RGBColor(0, 51, 102)  # 深蓝色
SUBTITLE_COLOR = RGBColor(102, 102, 102)  # 灰色
CONTENT_COLOR = RGBColor(51, 51, 51)  # 深灰色
HIGHLIGHT_COLOR = RGBColor(192, 0, 0)  # 红色（用于问题强调）
SUCCESS_COLOR = RGBColor(0, 128, 0)  # 绿色（用于解决方案）
CODE_BG_COLOR = RGBColor(245, 245, 245)  # 浅灰色背景

def add_title_slide(prs, title, subtitle):
    """添加标题页"""
    slide_layout = prs.slide_layouts[6]  # 空白布局
    slide = prs.slides.add_slide(slide_layout)

    # 添加标题
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(12.333), Inches(1))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = TITLE_COLOR
    p.alignment = PP_ALIGN.CENTER

    # 添加副标题
    subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.8), Inches(12.333), Inches(0.5))
    tf = subtitle_box.text_frame
    p = tf.paragraphs[0]
    p.text = subtitle
    p.font.size = Pt(24)
    p.font.color.rgb = SUBTITLE_COLOR
    p.alignment = PP_ALIGN.CENTER

    return slide

def add_content_slide(prs, title, content_sections):
    """添加内容页"""
    slide_layout = prs.slide_layouts[6]  # 空白布局
    slide = prs.slides.add_slide(slide_layout)

    # 添加标题
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.8))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = TITLE_COLOR

    # 添加分隔线
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.1), Inches(12.333), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = TITLE_COLOR
    line.line.fill.background()

    # 添加内容
    y_position = Inches(1.4)
    for section_title, section_content, section_color in content_sections:
        # 小标题
        section_box = slide.shapes.add_textbox(Inches(0.5), y_position, Inches(12.333), Inches(0.4))
        tf = section_box.text_frame
        p = tf.paragraphs[0]
        p.text = section_title
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = section_color
        y_position += Inches(0.45)

        # 内容
        content_box = slide.shapes.add_textbox(Inches(0.5), y_position, Inches(12.333), Inches(2))
        tf = content_box.text_frame
        tf.word_wrap = True

        for line_content in section_content:
            p = tf.add_paragraph()
            p.text = line_content
            p.font.size = Pt(16)
            p.font.color.rgb = CONTENT_COLOR
            p.level = 0
            p.space_before = Pt(6)

        y_position += Inches(len(section_content) * 0.35 + 0.3)

    return slide

def add_code_box(slide, code_text, left, top, width, height):
    """添加代码框"""
    # 添加背景矩形
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    rect.fill.solid()
    rect.fill.fore_color.rgb = CODE_BG_COLOR
    rect.line.fill.background()

    # 添加代码文本
    code_box = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.1), width - Inches(0.2), height - Inches(0.2))
    tf = code_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = code_text
    p.font.size = Pt(12)
    p.font.color.rgb = CONTENT_COLOR
    p.font.name = "Courier New"

# ========== 第1页：问题描述与初步诊断 ==========
slide1 = add_content_slide(prs, "第1页：问题描述与初步诊断", [
    ("问题现象", [
        "• 训练文件：/home/mclab/mjp/unifyimmun/source/HLA_ESM2.py",
        "• 问题：训练1个epoch后，train_loss不再变化，模型无法继续优化",
        "• 模型架构：ESM2 frozen + Transformer Encoder + Cross Attention + MLP分类层"
    ], TITLE_COLOR),

    ("初步排查：梯度诊断", [
        "• 创建诊断脚本 debug_gradient.py 检查梯度状态",
        "• 结果：所有可训练参数都有有效梯度，无NaN/Inf/零梯度",
        "• Loss变化：0.6699 → 0.6495（单次forward正常下降）",
        "• 结论：基础训练逻辑正确，问题不在梯度流动"
    ], TITLE_COLOR),

    ("可疑点锁定", [
        "• FGM对抗训练：epsilon=1.0 可能过大",
        "• BatchNorm1d：projection层中使用，可能受FGM影响",
        "• DataLoader：shuffle=False，数据顺序固定",
        "• 无学习率调度器：固定lr=1e-3"
    ], HIGHLIGHT_COLOR),
])

# ========== 第2页：深度诊断与问题定位 ==========
slide2_layout = prs.slide_layouts[6]
slide2 = prs.slides.add_slide(slide2_layout)

# 标题
title_box = slide2.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.8))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "第2页：深度诊断与问题定位"
p.font.size = Pt(32)
p.font.bold = True
p.font.color.rgb = TITLE_COLOR

# 分隔线
line = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.1), Inches(12.333), Inches(0.02))
line.fill.solid()
line.fill.fore_color.rgb = TITLE_COLOR
line.line.fill.background()

# FGM诊断结果
section1_box = slide2.shapes.add_textbox(Inches(0.5), Inches(1.4), Inches(6), Inches(0.4))
tf = section1_box.text_frame
p = tf.paragraphs[0]
p.text = "FGM攻击诊断 (debug_fgm.py)"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = TITLE_COLOR

content1_box = slide2.shapes.add_textbox(Inches(0.5), Inches(1.85), Inches(6), Inches(2.5))
tf = content1_box.text_frame
tf.word_wrap = True
contents1 = [
    "• 正常Forward Loss: 0.7206",
    "• FGM Attack后 Loss: 3.1108 ↑",
    "• Loss跳跃: +2.39（攻击扰动过大）",
    "",
    "关键发现：",
    "• projection层参数在attack后被大幅修改",
    "• FGM restore后参数恢复正常",
    "• 但Loss仍不稳定..."
]
for line in contents1:
    p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(16)
    p.font.color.rgb = CONTENT_COLOR if not line.startswith("关键") else HIGHLIGHT_COLOR
    p.space_before = Pt(6)

# BatchNorm诊断结果（右侧）
section2_box = slide2.shapes.add_textbox(Inches(6.5), Inches(1.4), Inches(6), Inches(0.4))
tf = section2_box.text_frame
p = tf.paragraphs[0]
p.text = "BatchNorm Running Stats诊断"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = HIGHLIGHT_COLOR

content2_box = slide2.shapes.add_textbox(Inches(6.5), Inches(1.85), Inches(6), Inches(2.5))
tf = content2_box.text_frame
tf.word_wrap = True
contents2 = [
    "• 初始 running_mean: 0.0",
    "• 第1次Forward后: 变化 0.13",
    "• FGM Attack Forward后: 变化 0.35 ↑↑",
    "",
    "⚠️ 问题根源发现：",
    "• BatchNorm的running_mean和running_var",
    "  在FGM attack期间被异常激活值污染",
    "• FGM restore只恢复了参数权重",
    "• running stats未被恢复，持续污染后续训练"
]
for line in contents2:
    p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(16)
    p.font.color.rgb = HIGHLIGHT_COLOR if "⚠️" in line or "问题根源" in line else CONTENT_COLOR
    p.space_before = Pt(6)

# 问题根因说明框
problem_box = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(4.5), Inches(12.333), Inches(2.5))
problem_box.fill.solid()
problem_box.fill.fore_color.rgb = RGBColor(255, 240, 240)
problem_box.line.color.rgb = HIGHLIGHT_COLOR

problem_text = slide2.shapes.add_textbox(Inches(0.7), Inches(4.7), Inches(12), Inches(2.3))
tf = problem_text.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "根本原因分析"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = HIGHLIGHT_COLOR

p = tf.add_paragraph()
p.text = ""
p.space_before = Pt(10)

p = tf.add_paragraph()
p.text = "FGM对抗训练机制：attack时对参数添加梯度方向扰动，restore时恢复原始参数"
p.font.size = Pt(14)
p.font.color.rgb = CONTENT_COLOR

p = tf.add_paragraph()
p.text = "但BatchNorm层的运行统计量(running_mean, running_var)不是参数，而是forward时动态更新的状态"
p.font.size = Pt(14)
p.font.color.rgb = CONTENT_COLOR

p = tf.add_paragraph()
p.text = "FGM attack产生的异常激活值导致running stats被错误更新，restore无法恢复它们"
p.font.size = Pt(14)
p.font.color.rgb = HIGHLIGHT_COLOR

p = tf.add_paragraph()
p.text = "被污染的running stats持续影响后续所有forward，导致训练不稳定、loss停滞"
p.font.size = Pt(14)
p.font.color.rgb = HIGHLIGHT_COLOR

# ========== 第3页：解决方案与验证 ==========
slide3_layout = prs.slide_layouts[6]
slide3 = prs.slides.add_slide(slide3_layout)

# 标题
title_box = slide3.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.8))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "第3页：解决方案与验证结果"
p.font.size = Pt(32)
p.font.bold = True
p.font.color.rgb = TITLE_COLOR

# 分隔线
line = slide3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.1), Inches(12.333), Inches(0.02))
line.fill.solid()
line.fill.fore_color.rgb = TITLE_COLOR
line.line.fill.background()

# 解决方案代码
solution_title = slide3.shapes.add_textbox(Inches(0.5), Inches(1.4), Inches(12.333), Inches(0.4))
tf = solution_title.text_frame
p = tf.paragraphs[0]
p.text = "解决方案：修改FGM类，备份并恢复BatchNorm Running Stats"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = SUCCESS_COLOR

# 代码框
code_text = '''class FGM_ESM2:
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.bn_backup = {}  # 新增：备份BatchNorm running stats

    def attack(self, epsilon=1.0):
        # 攻击projection层参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and "projection" in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

        # 新增：备份BatchNorm running stats
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self.bn_backup[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'num_batches_tracked': module.num_batches_tracked.clone()
                }

    def restore(self):
        # 恢复参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and "projection" in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

        # 新增：恢复BatchNorm running stats
        for name, module in self.model.named_modules():
            if name in self.bn_backup:
                module.running_mean = self.bn_backup[name]['running_mean']
                module.running_var = self.bn_backup[name]['running_var']
                module.num_batches_tracked = self.bn_backup[name]['num_batches_tracked']
        self.bn_backup = {}'''

add_code_box(slide3, code_text, Inches(0.5), Inches(1.9), Inches(7.5), Inches(3.8))

# 验证结果（右侧）
verify_title = slide3.shapes.add_textbox(Inches(8.2), Inches(1.9), Inches(4.5), Inches(0.4))
tf = verify_title.text_frame
p = tf.paragraphs[0]
p.text = "验证结果"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = SUCCESS_COLOR

verify_box = slide3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.2), Inches(2.35), Inches(4.5), Inches(2.5))
verify_box.fill.solid()
verify_box.fill.fore_color.rgb = RGBColor(240, 255, 240)
verify_box.line.color.rgb = SUCCESS_COLOR

verify_text = slide3.shapes.add_textbox(Inches(8.4), Inches(2.55), Inches(4.1), Inches(2.3))
tf = verify_text.text_frame
tf.word_wrap = True
verify_contents = [
    "修复前：",
    "• running_mean变化: 0.35 ❌",
    "• running_var变化: 0.09 ❌",
    "",
    "修复后：",
    "• running_mean变化: 0.00 ✓",
    "• running_var变化: 0.00 ✓",
    "",
    "Loss变化: 0.68 → 0.66 ✓",
    "训练正常进行"
]
for line in verify_contents:
    p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(14)
    if "❌" in line:
        p.font.color.rgb = HIGHLIGHT_COLOR
    elif "✓" in line:
        p.font.color.rgb = SUCCESS_COLOR
    else:
        p.font.color.rgb = CONTENT_COLOR
    p.space_before = Pt(6)

# 其他建议
other_title = slide3.shapes.add_textbox(Inches(8.2), Inches(5.0), Inches(4.5), Inches(0.4))
tf = other_title.text_frame
p = tf.paragraphs[0]
p.text = "备选方案"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = TITLE_COLOR

other_text = slide3.shapes.add_textbox(Inches(8.2), Inches(5.45), Inches(4.5), Inches(1.8))
tf = other_text.text_frame
tf.word_wrap = True
other_contents = [
    "1. 降低epsilon: 1.0 → 0.1~0.3",
    "2. 将BatchNorm换成LayerNorm",
    "3. 去掉FGM对抗训练",
    "4. 添加学习率调度器"
]
for line in other_contents:
    p = tf.add_paragraph()
    p.text = line
    p.font.size = Pt(14)
    p.font.color.rgb = CONTENT_COLOR
    p.space_before = Pt(8)

# 保存PPT
output_path = "/home/mclab/mjp/unifyimmun/docs/FGM_BatchNorm_Diagnosis.pptx"
prs.save(output_path)
print(f"PPT已保存至: {output_path}")