"""One visual language for the app, the figures and the slides.

Colour has a job everywhere it appears:

* The three SVD factors keep fixed colours, so a reader can follow them from the
  shape diagram to every chart:  U (terms) teal,  Σ (strength) ochre,  Vᵀ (documents) violet.
* Categories (true topics) use a 4-colour set validated for colour-blind readers in
  scatter plots, always paired with a distinct marker shape.
* Signed values (loadings, correlations) use a blue ↔ red diverging scale with a grey zero.
* Non-negative magnitudes (entries of A) use a single ink-blue ramp.
"""

INK = "#18202B"
INK_SOFT = "#4D5766"
INK_FAINT = "#8892A0"
PAPER = "#FAFBFC"
PANEL = "#EEF2F5"
RULE = "#D5DCE3"

U_COLOR = "#0F7A80"      # term space
S_COLOR = "#B07A10"      # singular values
V_COLOR = "#6A4FC4"      # document space
A_COLOR = "#2B4C7E"      # the data matrix itself
PRIMARY = U_COLOR

CATEGORY_COLORS = ["#2a78d6", "#e34948", "#1baf7a", "#eda100"]
CATEGORY_SYMBOLS_PLOTLY = ["circle", "square", "diamond", "triangle-up"]
CATEGORY_MARKERS_MPL = ["o", "s", "D", "^"]
OTHER_COLOR = "#9AA3AE"

DIVERGING = ["#1c5cab", "#6da7ec", "#f0efec", "#ec8a85", "#b3261e"]  # negative -> positive
SEQUENTIAL = ["#f4f7fb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

FONT_BODY = "Lexend"
FONT_HEAD = "Literata"


def category_color(i: int) -> str:
    return CATEGORY_COLORS[i] if i < len(CATEGORY_COLORS) else OTHER_COLOR


def category_style(categories) -> dict:
    """Map each category to (colour, plotly symbol, matplotlib marker) in a fixed order."""
    out = {}
    for i, c in enumerate(categories):
        out[c] = (
            category_color(i),
            CATEGORY_SYMBOLS_PLOTLY[i % len(CATEGORY_SYMBOLS_PLOTLY)],
            CATEGORY_MARKERS_MPL[i % len(CATEGORY_MARKERS_MPL)],
        )
    return out
