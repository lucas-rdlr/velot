#!/usr/bin/env bash
set -euo pipefail

# Spatial Hubs Figure Generator - LaTeX/Bash Version
# Letters overlaid on images WITHOUT boxes

OUTBASE="${1:-panel4}"

# Find required tools
TECTONIC="$(command -v tectonic || true)"
PDFLATEX="$(command -v pdflatex || true)"
PDFTOPPM="$(command -v pdftoppm || true)"
GS="$(command -v gs || true)"

if [[ -z "${TECTONIC}" && -z "${PDFLATEX}" ]]; then
  echo "[ERROR] Need tectonic or pdflatex in PATH."
  echo "Install: sudo apt-get install texlive-latex-base"
  echo "    or: conda install -c conda-forge tectonic"
  exit 1
fi

echo "===================================================================="
echo "Creating LaTeX source..."
echo "===================================================================="

# Generate LaTeX document
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

% Panel macro: letter overlaid on image (NO BOX)
% #1 = letter
% #2 = title
% #3 = image path
% #4 = includegraphics options
\newcommand{\Panel}[4]{%
  \begin{minipage}[t]{\linewidth}
    \centering
    \begin{tikzpicture}
      \node[anchor=north west, inner sep=0] (img) {\img[#4]{#3}};
      
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

% Row 1
\begin{minipage}[t]{0.32\linewidth}
\Panel{a}{Cell clusters}{figure4_a.png}{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.33\linewidth}
\Panel{b}{Diffusion pseudotime}{figure4_b.png}{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.33\linewidth}
\Panel{c}{VelOT confidence}{figure4_c.png}{width=\linewidth}
\end{minipage}\hfill
\vspace{5mm}


% Row 2
\begin{minipage}[t]{0.33\linewidth}
\Panel{d}{Raw VelOT quiver}{figure4_e.png}{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.33\linewidth}
\Panel{e}{Smooth VelOT quiver}{figure4_f.png}{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.33\linewidth}
\Panel{f}{VelOT velocity stream}{figure4_d.png}{width=\linewidth}
\end{minipage}
\vspace{5mm}


% Row 3
\begin{minipage}[t]{0.33\linewidth}
\Panel{g}{VelOT training losses}{figure4_g.png}{width=\linewidth, height=67mm}
\end{minipage}\hfill
\begin{minipage}[t]{0.33\linewidth}
\Panel{h}{VelOT velocity metrics}{figure4_h.png}{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.33\linewidth}
\Panel{i}{Single trajectory}{figure4_i.png}{width=\linewidth}
\end{minipage}

\end{minipage}
\end{document}
EOF

echo "✓ LaTeX source created: ${OUTBASE}.tex"
echo ""
echo "===================================================================="
echo "Compiling PDF..."
echo "===================================================================="

# Compile
if [[ -n "${TECTONIC}" ]]; then
  echo "Using tectonic..."
  "${TECTONIC}" -c minimal "${OUTBASE}.tex"
else
  echo "Using pdflatex..."
  "${PDFLATEX}" -interaction=nonstopmode -halt-on-error "${OUTBASE}.tex" >/dev/null 2>&1
fi

echo "✓ PDF created: ${OUTBASE}.pdf"
echo ""
echo "===================================================================="
echo "Converting to PNG and JPG..."
echo "===================================================================="

# Rasterize
if [[ -n "${PDFTOPPM}" ]]; then
  echo "Using pdftoppm (300 DPI)..."
  pdftoppm -png -r 300 -singlefile "${OUTBASE}.pdf" "${OUTBASE}"
  pdftoppm -jpeg -r 300 -singlefile "${OUTBASE}.pdf" "${OUTBASE}"
  echo "✓ PNG: ${OUTBASE}.png"
  echo "✓ JPG: ${OUTBASE}.jpg"
elif [[ -n "${GS}" ]]; then
  echo "Using ghostscript (300 DPI)..."
  gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r300 \
     -sOutputFile="${OUTBASE}.png" "${OUTBASE}.pdf" 2>/dev/null
  gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=jpeg -r300 -dJPEGQ=95 \
     -sOutputFile="${OUTBASE}.jpg" "${OUTBASE}.pdf" 2>/dev/null
  echo "✓ PNG: ${OUTBASE}.png"
  echo "✓ JPG: ${OUTBASE}.jpg"
else
  echo "⚠ No pdftoppm or gs found - only PDF created"
  echo "Install: sudo apt-get install poppler-utils"
fi

echo ""
echo "===================================================================="
echo "✓ SUCCESS! Figure created with letters on images (no boxes)"
echo "===================================================================="
echo "Output files:"
echo "  ${OUTBASE}.pdf"
[[ -f "${OUTBASE}.png" ]] && echo "  ${OUTBASE}.png"
[[ -f "${OUTBASE}.jpg" ]] && echo "  ${OUTBASE}.jpg"
echo "===================================================================="

echo "===================================================================="
echo "Cleaning up auxiliary files..."
echo "===================================================================="

# Delete auxiliary files
rm -f "${OUTBASE}.aux" "${OUTBASE}.log" "${OUTBASE}.tex" "${OUTBASE}.jpg"

echo "✓ Cleanup complete. Remaining files:"
echo "  ${OUTBASE}.pdf"
[[ -f "${OUTBASE}.png" ]] && echo "  ${OUTBASE}.png"
