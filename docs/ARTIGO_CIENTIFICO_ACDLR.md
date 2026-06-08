# ACDLR: Metodo Classico Para Deteccao De Crateras Lunares E Comparacao Com Ellipse R-CNN

## Resumo

O **ACDLR**, *Automated Crater Detection and Landing Risk*, e um sistema de
visao computacional para detectar crateras lunares em imagens visuais e estimar
risco visual de pouso por regiao. O metodo principal nao usa IA, CNN,
aprendizado profundo ou treinamento. Ele usa tecnicas classicas: CLAHE,
suavizacao, bordas, filtros circulares multi-escala, validacao geometrica,
validacao fotometrica e deduplicacao.

Para comparacao neural justa, o repositorio usa **Ellipse R-CNN** com o peso
pre-treinado **`wdoppenberg/crater-rcnn`**. A escolha se encaixa no protocolo
porque Ellipse R-CNN roda em imagens visuais, prediz crateras como elipses e
pode ser avaliado no mesmo dataset YOLO do ACDLR.

## Pergunta Cientifica

> Um detector classico, explicavel e sem treinamento consegue competir com uma
> CNN visual pre-treinada para crateras quando ambos sao avaliados no mesmo
> dataset anotado?

## Metodo ACDLR

Pipeline:

```text
imagem visual
  -> tons de cinza
  -> CLAHE
  -> suavizacao
  -> bordas
  -> candidatos circulares multi-escala
  -> validacao geometrica/fotometrica
  -> refinamento local
  -> deduplicacao
  -> crateras (x, y, r)
```

O ACDLR e controlavel por parametros como `min_radius`, `max_radius`,
`canny_threshold` e `strictness`.

## Baseline CNN: Ellipse R-CNN

Referencias:

- Repositorio standalone do modelo: https://github.com/wdoppenberg/ellipse-rcnn
- Projeto completo/TRN relacionado: https://github.com/wdoppenberg/crater-detection
- Peso pre-treinado: https://huggingface.co/wdoppenberg/crater-rcnn

O projeto `wdoppenberg/crater-detection` contem o sistema completo de navegacao
lunar com deteccao e casamento de padroes de crateras. Para o benchmark deste
repositorio, usamos apenas o detector pre-treinado standalone: codigo
`wdoppenberg/ellipse-rcnn` e peso `wdoppenberg/crater-rcnn`.
- Paper base: https://arxiv.org/abs/2001.11584

Ellipse R-CNN prediz elipses:

```text
[a, b, cx, cy, theta]
```

Para comparar com o ACDLR, cada elipse vira um circulo:

```text
radius = (a + b) / 2
```

Assim, ACDLR e Ellipse R-CNN sao avaliados com a mesma estrutura `(x, y, r)`.

## Dataset E Metricas

Dataset:

```text
data/LU3M6TGT_yolo_format
```

Labels YOLO sao convertidas para circulos:

```text
cx = x_center * image_width
cy = y_center * image_height
radius = (box_width_px + box_height_px) / 4
```

Metricas:

- precision;
- recall;
- F1;
- true positives;
- false positives;
- false negatives;
- erro medio de centro;
- erro medio de raio.

Matching:

```text
center_error <= 1.34 * gt_radius
radius_error <= 1.0 * gt_radius
```

## Como Rodar

Preparar Ellipse R-CNN:

```bash
python scripts/setup_ellipse_rcnn_pretrained.py
```

Rodar comparacao:

```bash
python scripts/run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 5 --visual-count 3
```

Saidas:

```text
artifacts/acdlr_vs_ellipse_rcnn/comparison_report.md
artifacts/acdlr_vs_ellipse_rcnn/visual_comparison.png
artifacts/acdlr_vs_ellipse_rcnn/charts/acdlr_vs_ellipse_rcnn_metrics.png
```

## Estado Atual

O benchmark ACDLR x Ellipse R-CNN foi validado no dataset local em smoke test:

| Metodo | Imagens | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ACDLR | 3 | 0.6294 | 0.5114 | 0.5643 |
| Ellipse R-CNN | 3 | 0.3922 | 0.0568 | 0.0993 |

Neste smoke test, o ACDLR aparece superior em F1 porque apresenta recall muito
maior. O Ellipse R-CNN acerta poucas crateras, mas nos acertos tem erros medios
de centro e raio menores, indicando que o problema principal neste dataset e
cobertura/recall, nao necessariamente localizacao dos poucos positivos.


## Limitacoes

- Ellipse R-CNN pode ter domain shift, pois nao foi treinado exatamente neste
  dataset.
- YOLO boxes sao aproximadas como circulos.
- O smoke test pequeno nao substitui o `valid` completo.
- Para publicacao, congelar parametros antes da avaliacao final.
