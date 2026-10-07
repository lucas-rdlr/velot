#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Figure Generator (LaTeX + PNG)
#
# Default:
#   - Creates PNG only
#   - Deletes PDF and auxiliary files
#
# Optional:
#   --pdf    Keep PDF output
#
# Usage:
#   ./make_figure.sh
#   ./make_figure.sh figureS11
#   ./make_figure.sh figureS11 --pdf
# ============================================================================

OUTBASE="panel6"
KEEP_PDF=false

# ----------------------------------------------------------------------------
# Parse arguments
# ----------------------------------------------------------------------------

for arg in "$@"; do
    case "$arg" in
        --pdf)
            KEEP_PDF=true
            ;;
        *)
            OUTBASE="$arg"
            ;;
    esac
done

# ----------------------------------------------------------------------------
# Find required tools
# ----------------------------------------------------------------------------

TECTONIC="$(command -v tectonic || true)"
PDFLATEX="$(command -v pdflatex || true)"
PDFTOPPM="$(command -v pdftoppm || true)"
GS="$(command -v gs || true)"

if [[ -z "${TECTONIC}" && -z "${PDFLATEX}" ]]; then
    echo "[ERROR] Need tectonic or pdflatex in PATH."
    echo "Install:"
    echo "  sudo apt-get install texlive-latex-base"
    echo "or"
    echo "  conda install -c conda-forge tectonic"
    exit 1
fi

# ----------------------------------------------------------------------------
# Create LaTeX source
# ----------------------------------------------------------------------------

echo "===================================================================="
echo "Creating LaTeX source..."
echo "===================================================================="

cat > "${OUTBASE}.tex" <<'EOF'
\documentclass[border=5mm]{standalone}

\usepackage{silence}
\WarningsOff*

\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{multirow}
\usepackage{helvet}

\renewcommand{\familydefault}{\sfdefault}

\setlength{\parindent}{0pt}
\setlength{\tabcolsep}{0pt}
\setlength{\parskip}{0pt}

% Safe include for paths with underscores
\newcommand{\img}[2][]{\includegraphics[#1]{\detokenize{#2}}}

% Panel macro
% #1 = letter
% #2 = title
% #3 = image path
% #4 = includegraphics options
\newcommand{\Panel}[4]{%
  \begin{minipage}[t]{\linewidth}
    \centering
    \begin{tikzpicture}[baseline=(img.north)]

      \node[anchor=north west, inner sep=0] (img)
      {\img[#4]{#3}};

      % Letter
      \node[
        anchor=north west,
        text height=1.5ex,
        text depth=0ex,
        font=\bfseries\sffamily\Large,
        xshift=-1mm,
        yshift=5mm
      ] at (img.north west) {#1};

      % Title
      \node[
        anchor=north,
        text height=1.5ex,
        text depth=0ex,
        font=\sffamily\small,
        xshift=0mm,
        yshift=5mm
      ] at (img.north) {#2};

    \end{tikzpicture}
  \end{minipage}%
}

\begin{document}

\begin{minipage}{200mm}

% --------------------------------------------------------------------------
% Row 1
% --------------------------------------------------------------------------

\begin{minipage}[t]{0.33\linewidth}
\Panel{a}{VAMPFlow neural meta-states}{figure6_a.png}{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.33\linewidth}
\Panel{b}{VAMPFlow neural meta-states direction}{figure6_b.png}{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.33\linewidth}
\Panel{c}{Coarse meta-states transition}{figure6_c.png}{width=\linewidth}
\end{minipage}

\vspace{5mm}

% --------------------------------------------------------------------------
% Row 2
% --------------------------------------------------------------------------

\hfill\begin{minipage}[t]{0.33\linewidth}
\Panel{d}{Cell-type composition per meta-state}{figure6_d.png}{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.33\linewidth}
\Panel{e}{Alluvial: meta-states vs real cell-types}{figure6_e.png}{width=\linewidth}
\end{minipage}\hfill\null

\end{minipage}

\end{document}
EOF

echo "✓ LaTeX source created: ${OUTBASE}.tex"

# ----------------------------------------------------------------------------
# Compile PDF
# ----------------------------------------------------------------------------

echo ""
echo "===================================================================="
echo "Compiling PDF..."
echo "===================================================================="

if [[ -n "${TECTONIC}" ]]; then
    echo "Using tectonic..."
    "${TECTONIC}" -c minimal "${OUTBASE}.tex"
else
    echo "Using pdflatex..."
    "${PDFLATEX}" -interaction=nonstopmode -halt-on-error \
        "${OUTBASE}.tex" >/dev/null 2>&1
fi

echo "✓ PDF created: ${OUTBASE}.pdf"

# ----------------------------------------------------------------------------
# Convert to PNG
# ----------------------------------------------------------------------------

echo ""
echo "===================================================================="
echo "Converting to PNG..."
echo "===================================================================="

if [[ -n "${PDFTOPPM}" ]]; then

    echo "Using pdftoppm (300 DPI)..."

    pdftoppm \
        -png \
        -r 300 \
        -singlefile \
        "${OUTBASE}.pdf" \
        "${OUTBASE}"

    echo "✓ PNG created: ${OUTBASE}.png"

elif [[ -n "${GS}" ]]; then

    echo "Using ghostscript (300 DPI)..."

    gs \
        -dSAFER \
        -dBATCH \
        -dNOPAUSE \
        -sDEVICE=png16m \
        -r300 \
        -sOutputFile="${OUTBASE}.png" \
        "${OUTBASE}.pdf" \
        >/dev/null 2>&1

    echo "✓ PNG created: ${OUTBASE}.png"

else

    echo "⚠ No PDF rasterizer found."
    echo "Install:"
    echo "  sudo apt-get install poppler-utils"
    echo "or"
    echo "  sudo apt-get install ghostscript"

fi

# ----------------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------------

echo ""
echo "===================================================================="
echo "Cleaning auxiliary files..."
echo "===================================================================="

rm -f \
    "${OUTBASE}.aux" \
    "${OUTBASE}.log" \
    "${OUTBASE}.tex"

# Remove PDF unless requested
if [[ "${KEEP_PDF}" == false ]]; then
    rm -f "${OUTBASE}.pdf"
    echo "✓ PDF removed"
else
    echo "✓ PDF kept"
fi

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------

echo ""
echo "===================================================================="
echo "✓ SUCCESS"
echo "===================================================================="

if [[ "${KEEP_PDF}" == true ]]; then
    echo "  ${OUTBASE}.pdf"
fi

[[ -f "${OUTBASE}.png" ]] && echo "  ${OUTBASE}.png"

echo "===================================================================="
