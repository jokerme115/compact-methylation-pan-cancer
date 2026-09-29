"""Single source of truth for the submission-figure colour palette.

Every manuscript figure must take its colours from this module, so that one
quantity keeps one colour across the whole paper: the 500-CpG compact candidate
is always ``BLUE`` (#4C65A4) and the 1,000-CpG reference anchor is always
``ORANGE`` (#F3A683).

The seven steps below are the ordered cool-to-warm palette supplied for the
revision.  Panel-background tints are derived from those same steps through
``to_rgba`` instead of being additional hard-coded hex values, so the palette
stays closed under tinting.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, to_rgba

# --- ordered palette -------------------------------------------------------
P1 = "#4C65A4"  # deep indigo  -- primary series (500-CpG compact candidate)
P2 = "#8B8FCA"  # periwinkle   -- second cool series, cool pole of the diverging map
P3 = "#B7AED9"  # pale violet  -- light cool step
P4 = "#E8C1CC"  # dusty pink   -- light warm step
P5 = "#FFD7A8"  # apricot      -- highlight bands and pale warm fills
P6 = "#F3A683"  # soft orange  -- secondary series (1,000-CpG reference anchor)
P7 = "#B9786F"  # clay red     -- warm pole, accents and annotations

PALETTE = (P1, P2, P3, P4, P5, P6, P7)

# --- neutrals --------------------------------------------------------------
BLACK = "#202020"
DARK = "#202020"
GREY = "#6E7183"
LIGHT = "#E6E4ED"

# --- semantic slots --------------------------------------------------------
BLUE = P1
BLUE_LIGHT = P2
ORANGE = P6
ORANGE_LIGHT = P7
RED = P7
TEAL = P2
GOLD = P6
GREEN = P7
PURPLE = P3

# --- background tints, derived from the palette ----------------------------
BLUE_TINT = to_rgba(P1, 0.10)
ORANGE_TINT = to_rgba(P6, 0.16)
GREEN_TINT = to_rgba(P7, 0.14)
PALE_BLUE = to_rgba(P1, 0.07)
PALE_ORANGE = to_rgba(P5, 0.42)

# --- shared colour maps ----------------------------------------------------
# Sequential: white -> pale violet -> periwinkle -> deep indigo.  Used for the
# normalised confusion matrices and other one-signed magnitude fields.
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "palette_sequential", ["#FFFFFF", P3, P2, P1]
)
# Diverging: cool pole / white / warm pole, used for signed coefficients.
CMAP_DIV = LinearSegmentedColormap.from_list(
    "palette_diverging", [P1, "#FFFFFF", P7]
)
