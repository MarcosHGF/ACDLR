# Paper Materials

This folder contains the article-facing materials:

- final LaTeX manuscript;
- figures selected for publication;
- tables exported from benchmark summaries;
- supplementary material;
- final protocol description.

The repository is currently organized as the implementation and reproducibility
artifact for a computer-vision paper-style project. The LaTeX manuscript is:

```text
paper/acdlr_artigo_cientifico.tex
```

Compile it from this folder with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error acdlr_artigo_cientifico.tex
```

All figures required by the manuscript are stored in:

```text
paper/figures/
```
