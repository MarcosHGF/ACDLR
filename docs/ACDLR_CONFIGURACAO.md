# ACDLR: Funcionamento E Configuracao

Este documento explica como o ACDLR detecta crateras lunares sem usar IA, CNN
ou treinamento. A comparacao neural principal do repositorio agora e feita com
**Ellipse R-CNN**, separado do metodo ACDLR.

## Pipeline ACDLR

1. divide imagens grandes em tiles com sobreposicao;
2. realca contraste e bordas;
3. procura assinaturas circulares de crateras em multiplas escalas;
4. valida candidatos por criterios geometricos e fotometricos;
5. remove duplicatas;
6. mede crateras detectadas;
7. calcula risco visual de pouso por regiao;
8. compara deteccoes com anotacoes YOLO quando existe ground truth.

## Arquivos Principais

| Arquivo | Papel |
|---|---|
| `core/preprocessing.py` | Conversao para cinza, CLAHE, blur, sharpening e bordas |
| `core/detection.py` | Detector classico de crateras do ACDLR |
| `core/tiling.py` | Split em tiles e recomposicao para coordenadas globais |
| `core/measurement.py` | Conversao de circulos em medidas fisicas |
| `core/risk.py` | Score de risco e ponto de pouso |
| `core/evaluation.py` | Precision, recall, F1 e matching com ground truth |
| `app.py` | Interface Streamlit e parametros interativos |

## Pre-Processamento

| Parametro | Valor padrao | Efeito |
|---|---:|---|
| `clahe_clip` | `2.5` | Realce de contraste local |
| `blur_ksize` | `5` | Reducao de ruido fino |
| `canny_threshold` | `45` | Sensibilidade de borda |

## Detector

| Parametro | Valor padrao | Efeito |
|---|---:|---|
| `tile_size` | `1024` | Tamanho do tile |
| `overlap` | `96` | Sobreposicao entre tiles |
| `min_radius` | `4` | Menor cratera considerada |
| `max_radius` | `70` | Maior cratera considerada |
| `strictness` | `16` | Rigor dos validadores |

## Benchmark

Labels YOLO sao convertidas para circulos:

```text
cx = x_center * image_width
cy = y_center * image_height
radius = (box_width_px + box_height_px) / 4
```

Matching:

```text
center_error <= 1.34 * gt_radius
radius_error <= 1.0 * gt_radius
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

## Comparador IA Principal: Ellipse R-CNN

Ellipse R-CNN nao faz parte do ACDLR. Ele e usado apenas para comparar o metodo
classico contra uma CNN visual pre-treinada para crateras.

| Parametro | Valor |
|---|---|
| repositorio | `external/ellipse-rcnn` |
| modelo | `artifacts/ellipse_rcnn_pretrained/crater-rcnn` |
| peso esperado | `model.safetensors` |
| score threshold | `0.60` |
| max detections | `150` |

Comando:

```bash
python scripts/run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 25 --visual-count 8
```

Ellipse R-CNN prediz elipses `[a, b, cx, cy, theta]`. Para usar as mesmas
metricas do ACDLR, o benchmark converte:

```text
radius = (a + b) / 2
```

## Saidas

```text
artifacts/acdlr_yolo_benchmark/
artifacts/acdlr_vs_ellipse_rcnn/
```

O competidor neural ativo e Ellipse R-CNN, sempre executado fora do metodo
ACDLR e avaliado pelo mesmo protocolo de metricas.
