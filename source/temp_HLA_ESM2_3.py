"""
HLA_ESM2 Round 3
Loads encoder_P from TCR_ESM2_2/encoder_P_ESM2_2.pth
Saves encoder_P to HLA_ESM2_3/encoder_P_ESM2_3.pth
"""
import re

original = open("HLA_ESM2_2.py").read()

# Replace encoder loading path
modified = re.sub(
    r'"TCR_ESM2", "encoder_P_ESM2\.pth"',
    '"TCR_ESM2_2", "encoder_P_ESM2_2.pth"',
    original
)

# Replace encoder save filename
modified = re.sub(
    r'"encoder_P_ESM2_2\.pth"',
    '"encoder_P_ESM2_3.pth"',
    modified
)

# Replace save directory
modified = re.sub(
    r'"HLA_ESM2_2"',
    '"HLA_ESM2_3"',
    modified
)

# Replace experiment name
modified = re.sub(
    r'experiment_name="HLA_ESM2[^"]*"',
    'experiment_name="HLA_ESM2_3"',
    modified
)

exec(modified)
