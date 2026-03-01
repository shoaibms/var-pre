"""
Master Color Palette for VAR-PRE Publication Figures
=====================================================
Centralized color definitions for consistent cross-figure styling.
All figure scripts import from this module.

Color distribution:
    ~10-15%  dark greens       (deep, high-contrast anchors)
    ~50-60%  medium greens     (primary plot elements)
    ~20%     light yellow-green  (warm accents, highlights)
    ~10-20%  light blue-green    (cool accents, diversity)
"""

# =============================================================================
# DI COLORMAP STOPS  (Decoupling Index, diverging at DI = 1.0)
# =============================================================================
# Coupled (DI < 1): cool blue-green tones
# Neutral (DI = 1): light mint
# Anti-aligned (DI > 1): warm yellow-green tones

COUPLED_GREEN   = "#005824"   # dark green anchor
MID_BLUEGREEN   = "#008080"   # teal (blue-green mid-tone)
LIGHT_BLUEGREEN = "#00CED1"   # dark turquoise (light blue-green)
NEUTRAL_GREEN   = "#98FB98"   # pale green (visible midpoint)
LIGHT_YLGREEN   = "#00FF7F"   # spring green (light yellow-green)
MID_YLGREEN     = "#9ACD32"   # yellow-green
ANTI_YLGREEN    = "#94CB64"   # lime (anti-aligned end)

# =============================================================================
# FUNCTIONAL ROLE COLORS
# =============================================================================

# Text and UI
TEXT_PRIMARY     = "#1b4332"   # dark forest (titles, labels)
TEXT_SECONDARY   = "#2E8B57"   # dark sea green (spines, axes)
TEXT_TERTIARY    = "#5F9EA0"   # cadet blue (muted annotations)
SPINE_COLOR      = "#008080"   # teal (axis spines -- blue-green accent)
GRID_COLOR       = "#B2DFDB"   # light teal (grid lines -- visible, not faded)
BG_WHITE         = "#FFFFFF"

# Strategy comparison (Figures 1, 4)
STRATEGY_TOPVAR  = "#9ACD32"   # yellow-green (variance = risky)
STRATEGY_RANDOM  = "#B8B8B8"   # neutral grey
STRATEGY_TOPSHAP = "#008080"   # teal (importance = reliable, blue-green)
STRATEGY_ALL     = "#00CED1"   # dark turquoise (all features)

# Regime classification
REGIME_COUPLED      = "#008080"   # teal (blue-green family)
REGIME_MIXED        = "#3CB371"   # medium sea green
REGIME_ANTI_ALIGNED = "#9ACD32"   # yellow-green

# Mechanism roles (Figure 3)
SIGNAL_GREEN     = "#00BFFF"   # deep sky blue (signal -- blue-green accent)
VARIANCE_COLOR   = "#9ACD32"   # yellow-green
BETWEEN_COLOR    = "#00CED1"   # dark turquoise (between-class -- blue-green)
WITHIN_COLOR     = "#00FF7F"   # spring green (within-class -- yellow-green)

# Q4 hidden biomarker (Figure 5)
Q4_GLOW          = "#7FFF00"   # chartreuse glow accent
Q4_DEEP          = "#4ADE45"   # rich chartreuse-green
Q4_FILL          = "#2D6A1E"   # dark olive-green solid fill

# Accent colors (for diversity in multi-element plots)
TURQUOISE        = "#40E0D0"
DARK_TURQUOISE   = "#00CED1"
DEEP_SKY_BLUE    = "#00BFFF"
MED_TURQUOISE    = "#48D1CC"
SPRING_GREEN     = "#00FF7F"
TEAL             = "#008080"
STEEL_BLUE       = "#4682B4"
CORNFLOWER       = "#6495ED"
NATURE_GREEN     = "#00A087"

# Neutrals
GREY             = "#808080"
GREY_LIGHT       = "#B8B8B8"
GREY_LIGHTER     = "#D3D3D3"
GREY_PALE        = "#F0F0F0"
SLATE_GREY       = "#708090"
ANTHRACITE       = "#585656"

# =============================================================================
# DATASET IDENTITY  (consistent markers and labels across all figures)
# =============================================================================

DS_MARKERS  = {"mlomics": "o", "ibdmdb": "s", "ccle": "D", "tcga_gbm": "^"}
DS_SHORT    = {"mlomics": "MLO", "ibdmdb": "IBD", "ccle": "CCLE", "tcga_gbm": "GBM"}
DS_DISPLAY  = {"mlomics": "MLOmics", "ibdmdb": "IBDMDB", "ccle": "CCLE", "tcga_gbm": "TCGA-GBM"}

DS_LABEL_COLORS = {
    "mlomics":  "#3CB371",   # medium sea green
    "ibdmdb":   "#00CED1",   # dark turquoise (blue-green)
    "ccle":     "#00BFFF",   # deep sky blue (blue-green)
    "tcga_gbm": "#9ACD32",   # yellow-green
}

# =============================================================================
# STANDARD FONT SIZES  (all figures)
# =============================================================================

FONT = {
    "base":   10.5,
    "title":  12,
    "label":  11,
    "tick":   9.5,
    "legend": 9.5,
    "panel":  13,
    "annot":  8.5,
}

# =============================================================================
# PALETTES / COLORMAPS  (sequential gradients for heatmaps)
# =============================================================================

green_sequential = ["#E0F7FA", "#80CBC4", "#00CED1", "#008080", "#005824"]

sequential = ["#E0F7FA", "#40E0D0", "#00BFFF", "#4682B4"]

bugreen = [
    "#E0F7FA", "#B2DFDB", "#80CBC4", "#4DB6AC",
    "#00CED1", "#00897B", "#008080", "#005824",
]

ylgreen = [
    "#F9FBE7", "#F0F4C3", "#DCE775", "#CDDC39",
    "#9ACD32", "#7CB342", "#558B2F", "#33691E",
]

greens = [
    "#E8F5E9", "#C8E6C9", "#A5D6A7", "#81C784",
    "#66BB6A", "#4CAF50", "#388E3C", "#1B5E20",
]

diverging_bg = ["#00BFFF", "#00CED1", "#FFFFFF", "#00FF7F", "#9ACD32"]

neutral_ui = ["#F6F8FA", "#D0D7DE", "#6E7681"]

# =============================================================================
# VAD DIAGNOSTIC FRAMEWORK  (Figure 6)
# =============================================================================

vad_zones = {
    "safe":         "#008080",   # teal (blue-green)
    "safe_bg":      "#B2DFDB",   # light teal
    "uncertain":    "#9ACD32",   # yellow-green
    "uncertain_bg": "#F0F4C3",   # light yellow-green
    "risky":        "#f9e055",   # golden yellow
    "risky_bg":     "#FFF9C4",   # pale yellow
}

vad_models = {
    "xgb":       "#005824",   # deep green
    "xgb_light": "#00CED1",   # dark turquoise
    "rf":        "#3CB371",   # medium sea green
    "rf_light":  "#98FB98",   # pale green
}

# =============================================================================
# FLAT COLORS LIST  (legacy compatibility)
# =============================================================================

COLORS = [
    "#005824", "#008080", "#00CED1", "#98FB98", "#00FF7F",
    "#9ACD32", "#94CB64", "#2E8B57", "#3CB371", "#00BFFF",
    "#48D1CC", "#40E0D0", "#00A087", "#1b4332", "#4682B4",
    "#5F9EA0", "#6495ED", "#B2DFDB", "#80CBC4",
    "#808080", "#B8B8B8", "#D3D3D3", "#F0F0F0", "#708090",
    "#585656",
    "#7FFF00", "#4ADE45", "#2D6A1E",
    "#008080", "#9ACD32", "#f9e055", "#005824", "#3CB371", "#00CED1",
]
