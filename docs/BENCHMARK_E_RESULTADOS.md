# Benchmark E Resultados

Este documento explica o benchmark atual: **ACDLR x Ellipse R-CNN** no mesmo
dataset visual anotado.

## Objetivo

Comparar:

- **ACDLR**: metodo principal, sem IA, apenas processamento classico;
- **Ellipse R-CNN**: detector neural visual pre-treinado para crateras, usando
  `wdoppenberg/crater-rcnn`.

Os dois metodos sao avaliados no mesmo split, com as mesmas labels e a mesma
funcao de matching.

## Dataset

```text
data/LU3M6TGT_yolo_format
```

As anotacoes YOLO sao convertidas para circulos:

```text
cx = x_center * image_width
cy = y_center * image_height
radius = (box_width_px + box_height_px) / 4
```

Ellipse R-CNN prediz elipses:

```text
[a, b, cx, cy, theta]
```

e o benchmark converte para circulo:

```text
radius = (a + b) / 2
```

## Metricas

| Metrica | Formula | Interpretacao |
|---|---|---|
| TP | deteccoes que casaram com anotacoes | acertos |
| FP | deteccoes sem anotacao correspondente | falsos positivos |
| FN | anotacoes nao detectadas | falsos negativos |
| Precision | TP / (TP + FP) | pureza das deteccoes |
| Recall | TP / (TP + FN) | cobertura das crateras anotadas |
| F1 | 2PR / (P + R) | equilibrio entre precision e recall |
| Center error ratio | erro de centro / raio anotado | erro espacial normalizado |
| Radius error ratio | erro de raio / raio anotado | erro de tamanho normalizado |

Matching:

```text
center_error <= 1.34 * gt_radius
radius_error <= 1.0 * gt_radius
```

## Rodar

Preparar peso:

```bash
python scripts/setup_ellipse_rcnn_pretrained.py
```

Comparacao:

```bash
python scripts/run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 3 --visual-count 2
```

Saidas:

```text
artifacts/acdlr_vs_ellipse_rcnn/comparison_report.md
artifacts/acdlr_vs_ellipse_rcnn/visual_comparison.png
artifacts/acdlr_vs_ellipse_rcnn/charts/acdlr_vs_ellipse_rcnn_metrics.png
```

## Estado Atual

O benchmark ACDLR x Ellipse R-CNN ja foi executado no smoke test local:

| Metodo | Imagens | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ACDLR | 3 | 0.6294 | 0.5114 | 0.5643 |
| Ellipse R-CNN | 3 | 0.3922 | 0.0568 | 0.0993 |

Saidas geradas:

```text
artifacts/acdlr_vs_ellipse_rcnn/comparison_report.md
artifacts/acdlr_vs_ellipse_rcnn/visual_comparison.png
artifacts/acdlr_vs_ellipse_rcnn/charts/acdlr_vs_ellipse_rcnn_metrics.png
```

## Interpretacao

Nesta amostra pequena, o ACDLR ficou superior em F1 porque recuperou muito mais
crateras anotadas. O Ellipse R-CNN teve recall muito baixo no dataset local,
mesmo mantendo bons erros de centro/raio nos poucos acertos. Isso sugere domain
shift e diferenca de escala/anotacao: o modelo pre-treinado detecta algumas
crateras com boa geometria, mas deixa muitas anotacoes do nosso dataset passar.

Qualquer outro modelo deve ser documentado como novo experimento separado, para
nao misturar protocolos de avaliacao.
