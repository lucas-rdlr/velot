#!/usr/bin/env bash
set -euo pipefail

# Spatial Hubs Figure Generator - LaTeX/Bash Version
# Letters overlaid on images WITHOUT boxes

OUTBASE="${1:-panel3}"

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
    % baseline=(img.north): without this, a tikzpicture's outer alignment reference
    % point defaults to the BOTTOM of its bounding box, so minipage[t] ends up matching
    % panels' bottoms instead of tops whenever their heights differ (e.g. panels with a
    % forced height= that doesn't match their natural aspect ratio, like e/f used to).
    \begin{tikzpicture}[baseline=(img.north)]
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
\begin{minipage}[t]{0.24\linewidth}
\Panel{a}{knn gradient stream}{figure3_h.png}{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.24\linewidth}
\Panel{b}{knn gradient+MLP stream}{figure3_j.png}{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.24\linewidth}
\Panel{c}{OT stream}{figure3_f.png}{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.24\linewidth}
\Panel{d}{True velocity stream}{figure3_d.png}{width=\linewidth}
\end{minipage}
\vspace{5mm}

% Row 3
\begin{minipage}[t]{0.32\linewidth}
\Panel{e}{knn gradient quiver}{figure3_g.png}{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.32\linewidth}
\Panel{f}{knn gradient+MLP quiver}{figure3_i.png}{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.32\linewidth}
\Panel{g}{OT quiver}{figure3_e.png}{width=\linewidth}
\end{minipage}
\vspace{5mm}

% Row 4
\begin{minipage}[t]{0.98\linewidth}
\Panel{h}{knn gradient velocity metrics}{figure3_l.png}{width=\linewidth}
\end{minipage}
\vspace{5mm}

% Row 5
\begin{minipage}[t]{0.98\linewidth}
\Panel{i}{OT velocity metrics}{figure3_k.png}{width=\linewidth}
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
rm -f "${OUTBASE}.aux" "${OUTBASE}.log" "${OUTBASE}.tex" "${OUTBASE}.jpg" "${OUTBASE}.pdf"

echo "✓ Cleanup complete. Remaining files:"
[[ -f "${OUTBASE}.png" ]] && echo "  ${OUTBASE}.png"