# =========================
# STAGE LABEL DEFINITIONS
# =========================

# Stage 1 → normal vs abnormal(fullbody)
STAGE1_MAP = {
    "nm": 0,   # normal
    "fb": 1    # abnormal
}

# Stage 2 → region classification
# df has 4+ labels but we collapse to 2
STAGE2_MAP = {
    "a-r0": 0,
    "a-l0": 0,
    "a-r0.5": 0,
    "a-l0.5": 0,  # ARM → 0

    "l-r0.5": 1,
    "l-l0.5": 1   # LEG → 1
}

# Stage 3A → arm side
ARM_STAGE3_MAP = {
    "a-r0": 0,
    "a-r0.5": 0,  # right arm

    "a-l0": 1,
    "a-l0.5": 1   # left arm
}

# Stage 3B → leg side
LEG_STAGE3_MAP = {
    "l-r0.5": 0,  # right leg
    "l-l0.5": 1   # left leg
}
