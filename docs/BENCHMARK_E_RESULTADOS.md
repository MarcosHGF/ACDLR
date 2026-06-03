# Benchmark E Resultados

Este documento explica como o benchmark funciona, quais metricas sao usadas e
quais resultados ja foram obtidos.

## Objetivo

Comparar:

- **ACDLR**: metodo principal, sem IA, apenas processamento classico;
- **CNN YOLOv11**: metodo concorrente, treinado com anotacoes YOLO.

Os dois metodos sao avaliados no mesmo dataset, no mesmo split e com as mesmas
anotacoes.

## Dataset De Benchmark

Dataset local:

```text
data/LU3M6TGT_yolo_format
```

Contagem atual:

| Split | Imagens |
|---|---:|
| train | 8756 |
| valid | 1545 |

As anotacoes sao caixas YOLO. Para comparar com o ACDLR, cada box e convertido
para uma cratera circular:

```text
cx = x_center * image_width
cy = y_center * image_height
radius = (box_width_px + box_height_px) / 4
```

## Matching

Uma deteccao e considerada correta quando:

```text
erro_de_centro <= center_tolerance * raio_anotado
erro_de_raio   <= radius_tolerance * raio_anotado
```

Na comparacao atual:

```text
center_tolerance = 1.34
radius_tolerance = 1.0
```

O matching e guloso: os pares possiveis sao ordenados pelo menor erro
normalizado, e cada deteccao/anotacao so pode ser usada uma vez.

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

## Rodar Benchmark ACDLR

```powershell
python scripts\benchmark_yolo_dataset.py ^
  --split valid ^
  --max-images 25 ^
  --out-dir artifacts\acdlr_yolo_benchmark ^
  --min-radius 4 ^
  --max-radius 70 ^
  --canny-threshold 45 ^
  --strictness 16 ^
  --center-tolerance 1.34 ^
  --radius-tolerance 1.0
```

## Rodar Benchmark CNN

```powershell
python scripts\benchmark_crater_cnn_yolo.py ^
  --weights artifacts\crater_cnn_yolo_train\moon_small\weights\best.pt ^
  --max-images 25 ^
  --conf 0.001 ^
  --iou 0.15 ^
  --max-det 150 ^
  --center-tolerance 1.34 ^
  --radius-tolerance 1.0
```

## Rodar Comparacao Completa

```powershell
python scripts\run_acdlr_vs_crater_cnn_comparison.py --max-images 5
```

Para gerar uma comparacao maior:

```powershell
python scripts\run_acdlr_vs_crater_cnn_comparison.py --max-images 25
```

## Resultado Atual

Ultimo teste executado:

| Campo | Valor |
|---|---|
| Dataset | `data/LU3M6TGT_yolo_format` |
| Split | `valid` |
| Imagens avaliadas | 3 |
| Tolerancia centro | `1.34 * raio` |
| Tolerancia raio | `1.0 * raio` |
| CNN | YOLOv11, 1 epoca, fracao pequena do treino |
| ACDLR | processamento classico, sem IA |

Resultado:

| Metodo | Deteccoes | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACDLR | 286 | 352 | 180 | 106 | 172 | 0.6294 | 0.5114 | 0.5643 |
| CNN YOLOv11 | 361 | 352 | 117 | 244 | 235 | 0.3241 | 0.3324 | 0.3282 |

Neste teste pequeno, o ACDLR teve melhor F1.

## Como Interpretar A Imagem Lado A Lado

Arquivo:

```text
artifacts/acdlr_vs_crater_cnn/visual_comparison.png
```

Cores:

- verde: true positive;
- vermelho: false positive;
- amarelo: false negative.

Leitura:

- muitos verdes indicam deteccoes corretas;
- muitos vermelhos indicam metodo agressivo demais;
- muitos amarelos indicam crateras perdidas;
- F1 alto indica melhor equilibrio entre perder menos crateras e inventar menos
  deteccoes.

## Por Que A CNN Ficou Fraca No Smoke Test

A CNN atual foi treinada rapidamente para validar o pipeline. Ela nao representa
o desempenho maximo de uma rede neural bem treinada.

Configuracao do smoke test:

```text
epochs = 1
fraction = 0.02
device = cpu
conf = 0.001
iou = 0.15
max_det = 150
```

Para uma comparacao mais forte, use:

```powershell
python scripts\run_acdlr_vs_crater_cnn_comparison.py ^
  --max-images 100 ^
  --force-cnn-train ^
  --cnn-train-epochs 30 ^
  --cnn-train-fraction 1.0
```

Se houver GPU:

```powershell
python scripts\run_acdlr_vs_crater_cnn_comparison.py ^
  --max-images 100 ^
  --force-cnn-train ^
  --cnn-train-epochs 50 ^
  --cnn-train-fraction 1.0 ^
  --cnn-device 0
```

## Arquivos Gerados

Comparacao:

```text
artifacts/acdlr_vs_crater_cnn/comparison_report.md
artifacts/acdlr_vs_crater_cnn/run_summary.json
artifacts/acdlr_vs_crater_cnn/visual_comparison.png
```

ACDLR:

```text
artifacts/acdlr_vs_crater_cnn/acdlr/acdlr_yolo_summary.json
artifacts/acdlr_vs_crater_cnn/acdlr/acdlr_yolo_benchmark.csv
artifacts/acdlr_vs_crater_cnn/acdlr/visuals/
```

CNN:

```text
artifacts/acdlr_vs_crater_cnn/crater_cnn_yolo/cnn_yolo_summary.json
artifacts/acdlr_vs_crater_cnn/crater_cnn_yolo/cnn_yolo_benchmark.csv
artifacts/acdlr_vs_crater_cnn/crater_cnn_yolo/visuals/
```

## Cuidados Na Apresentacao

- Nao afirmar que o smoke test prova superioridade final do ACDLR.
- Afirmar que, neste teste pequeno, o ACDLR superou a CNN subtreinada.
- Para conclusao forte, treinar a CNN por mais epocas e rodar o `valid`
  completo.
- Destacar que o ACDLR e explicavel e nao usa IA, enquanto a CNN depende de
  treinamento.
