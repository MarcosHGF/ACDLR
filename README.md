# ACDLR - Automated Crater Detection and Landing Risk

Aplicacao de visao computacional para detectar crateras lunares por
processamento classico de imagem e estimar risco visual de pouso por regiao.

![Imagem do projeto](./output.jpg)

## Resumo

O ACDLR detecta crateras em imagens da superficie lunar **sem IA no metodo
principal**. O pipeline usa OpenCV, filtros, bordas, realce de contraste,
validacao geometrica/fotometrica e analise por grade. Depois da deteccao, o
sistema mede crateras, calcula um score de risco por regiao e sugere uma area
mais favoravel para pouso dentro da imagem.

O comparador de IA principal do repositorio agora e **Ellipse R-CNN**
(`wdoppenberg/crater-rcnn`). Ele foi escolhido porque roda em imagens visuais
lunares, possui peso pre-treinado de crateras e produz elipses que podem ser
comparadas com as mesmas labels YOLO usadas pelo ACDLR.

A comparacao justa de acertos e ACDLR x Ellipse R-CNN no mesmo split visual
YOLO, com o baseline neural executado como modelo externo pre-treinado.

## O Que O Projeto Faz

1. Carrega uma imagem lunar ou um tile do dataset local.
2. Aplica pre-processamento: tons de cinza, CLAHE, suavizacao e realce.
3. Detecta crateras com processamento classico:
   - filtro casado multi-escala;
   - resposta por energia de aro;
   - candidatos por maximos locais;
   - validacao por contraste, borda, gradiente e suporte angular;
   - refinamento local por Hough;
   - deduplicacao de crateras sobrepostas.
4. Mede centro, raio, diametro e area das crateras.
5. Divide a imagem em grade e calcula risco visual por regiao.
6. Mostra imagens intermediarias, crateras detectadas, mapa de risco e melhor
   ponto sugerido.
7. Gera relatorios, graficos e visualizacao lado a lado ACDLR x Ellipse R-CNN.

## Estrutura Principal

```text
app.py                                            Interface Streamlit
core/detection.py                                 Detector classico ACDLR
core/preprocessing.py                             Pre-processamento
core/risk.py                                      Score de risco e ponto de pouso
core/evaluation.py                                Metricas de benchmark
scripts/benchmark_yolo_dataset.py                 Avalia ACDLR no dataset YOLO
scripts/setup_ellipse_rcnn_pretrained.py          Clona Ellipse R-CNN e baixa peso
scripts/benchmark_ellipse_rcnn_yolo_dataset.py    Avalia Ellipse R-CNN no dataset
scripts/run_acdlr_vs_ellipse_rcnn_comparison.py   Roda ACDLR x Ellipse R-CNN
docs/AI_BASELINE_ELLIPSE_RCNN.md                  Justificativa do baseline IA
docs/ACDLR_CONFIGURACAO.md                        Como o ACDLR funciona/configs
docs/ARTIGO_CIENTIFICO_ACDLR.md                   Texto tecnico em formato artigo
configs/                                          Configuracoes dos experimentos
paper/                                            Materiais reservados para artigo
reports/                                          Relatorios selecionados
```

Os scripts antigos de baselines anteriores foram removidos da estrutura ativa para
evitar comparacoes misturadas. O baseline neural do artigo agora e somente
Ellipse R-CNN, executado sem alterar o repositorio externo e avaliado no mesmo
dataset visual do ACDLR.

## Dataset

O dataset anotado usado no benchmark ACDLR fica em:

```text
data/LU3M6TGT_yolo_format
```

Estrutura esperada:

```text
data/LU3M6TGT_yolo_format/
  train/images
  train/labels
  valid/images
  valid/labels
```

As anotacoes seguem formato YOLO:

```text
classe x_centro y_centro largura altura
```

Para comparar com o ACDLR, cada box YOLO e convertido em uma cratera circular:

```text
cx = x_centro * largura_da_imagem
cy = y_centro * altura_da_imagem
raio = media(largura_box_px, altura_box_px) / 2
```

## Instalacao Global

Clone este repositorio:

```bash
git clone <URL_DO_REPOSITORIO_ACDLR> ACDLR
cd ACDLR
```

Crie o ambiente e instale dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Opcional com Conda:

```bash
conda env create -f environment.yml
conda activate acdlr
```

## Preparar O Baseline IA Visual

Comando recomendado:

```bash
python scripts/setup_ellipse_rcnn_pretrained.py
```

Esse comando:

1. clona `https://github.com/wdoppenberg/ellipse-rcnn` em `external/ellipse-rcnn`;
2. instala `ellipse-rcnn[hf]`;
3. baixa `wdoppenberg/crater-rcnn` para
   `artifacts/ellipse_rcnn_pretrained/crater-rcnn`;
4. gera um `manifest.json` com origem, caminhos e observacoes.

Se quiser apenas clonar/registrar o baseline sem baixar o arquivo grande:

```bash
python scripts/setup_ellipse_rcnn_pretrained.py --skip-download
```

Clone manual equivalente:

```bash
mkdir -p external
git clone https://github.com/wdoppenberg/ellipse-rcnn.git external/ellipse-rcnn
```

Referencias:

- Repositorio standalone do modelo: https://github.com/wdoppenberg/ellipse-rcnn
- Projeto completo/TRN relacionado: https://github.com/wdoppenberg/crater-detection
- Peso: https://huggingface.co/wdoppenberg/crater-rcnn
- Paper base: https://arxiv.org/abs/2001.11584

Observacao: `wdoppenberg/crater-detection` e o projeto completo de navegacao
lunar que usa Ellipse R-CNN e pattern matching. Para inferencia standalone do
detector, o proprio projeto aponta para `wdoppenberg/ellipse-rcnn`; por isso o
ACDLR usa `ellipse-rcnn` + o peso treinado `wdoppenberg/crater-rcnn`.

## Como Rodar A Interface

```bash
streamlit run app.py
```

Abra:

```text
http://localhost:8501
```

Na interface voce pode escolher um tile, enviar uma imagem, ajustar parametros
do ACDLR, ver resultados intermediarios e visualizar a comparacao ACDLR x
Ellipse R-CNN quando ela ja tiver sido gerada.

## Rodar Apenas O ACDLR

Teste pequeno:

```bash
python scripts/benchmark_yolo_dataset.py --split valid --max-images 25
```

Configuracao usada no benchmark compacto atual:

```bash
python scripts/benchmark_yolo_dataset.py \
  --dataset-dir data/LU3M6TGT_yolo_format \
  --split valid \
  --max-images 25 \
  --out-dir artifacts/acdlr_yolo_benchmark \
  --min-radius 4 \
  --max-radius 70 \
  --canny-threshold 45 \
  --strictness 16 \
  --center-tolerance 1.34 \
  --radius-tolerance 1.0
```

Saidas:

```text
artifacts/acdlr_yolo_benchmark/acdlr_yolo_report.md
artifacts/acdlr_yolo_benchmark/acdlr_yolo_summary.json
artifacts/acdlr_yolo_benchmark/acdlr_yolo_benchmark.csv
artifacts/acdlr_yolo_benchmark/visuals/
```

## Rodar A Comparacao ACDLR x Ellipse R-CNN

Comando principal do benchmark compacto:

```bash
python scripts/run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 25 --visual-count 8
```

Teste rapido para verificar instalacao:

```bash
python scripts/run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 3 --visual-count 2
```

Esse comando:

1. roda o ACDLR no dataset visual anotado local;
2. roda Ellipse R-CNN no mesmo split visual;
3. calcula precision, recall e F1;
4. gera grafico de metricas;
5. gera imagem lado a lado;
6. gera relatorio Markdown e JSON de resumo.

Saidas principais:

```text
artifacts/acdlr_vs_ellipse_rcnn/comparison_report.md
artifacts/acdlr_vs_ellipse_rcnn/visual_comparison.png
artifacts/acdlr_vs_ellipse_rcnn/charts/acdlr_vs_ellipse_rcnn_metrics.png
artifacts/acdlr_vs_ellipse_rcnn/run_summary.json
artifacts/acdlr_vs_ellipse_rcnn/acdlr/acdlr_yolo_summary.json
artifacts/acdlr_vs_ellipse_rcnn/ellipse_rcnn/ellipse_rcnn_yolo_summary.json
```

## Estado Do Benchmark Atual

O benchmark ACDLR x Ellipse R-CNN foi validado no dataset local com:

```bash
python scripts/run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 25 --visual-count 8
```

Resultado do experimento de 25 imagens:

| Metodo | Det. | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACDLR | 2610 | 2994 | 1399 | 1211 | 1595 | 0.536 | 0.467 | 0.499 |
| Ellipse R-CNN | 316 | 2994 | 178 | 138 | 2816 | 0.563 | 0.059 | 0.108 |

Interpretacao curta: o Ellipse R-CNN teve precision ligeiramente maior, mas
recall muito menor neste dataset local. Por isso o ACDLR ficou melhor em F1
neste protocolo zero-shot/pre-treinado. Isso nao significa que processamento
classico seja superior a CNNs em geral; significa que, neste conjunto visual e
sem fine-tuning local da CNN, o ACDLR recuperou mais crateras anotadas.

Artefatos gerados:

```text
artifacts/acdlr_vs_ellipse_rcnn/comparison_report.md
artifacts/acdlr_vs_ellipse_rcnn/visual_comparison.png
artifacts/acdlr_vs_ellipse_rcnn/charts/acdlr_vs_ellipse_rcnn_metrics.png
```

## Documentacao

- [Como rodar](docs/COMO_RODAR.md)
- [Apresentacao e fala pronta](docs/APRESENTACAO.md)
- [Baseline IA visual: Ellipse R-CNN](docs/AI_BASELINE_ELLIPSE_RCNN.md)
- [Configuracao do ACDLR](docs/ACDLR_CONFIGURACAO.md)
- [Artigo tecnico completo](docs/ARTIGO_CIENTIFICO_ACDLR.md)
- [Benchmark e resultados](docs/BENCHMARK_E_RESULTADOS.md)
- [Dataset card](docs/DATASET.md)
- [Protocolo experimental](docs/EXPERIMENT_PROTOCOL.md)
- [Reprodutibilidade](docs/REPRODUCIBILITY.md)
- [Estrutura dos arquivos](docs/ESTRUTURA.md)
- [Checklist de repositorio de artigo](docs/ARTICLE_REPOSITORY_CHECKLIST.md)

## Como Citar

O repositorio inclui `CITATION.cff`. Antes de publicar, substitua os autores
genericos pelos nomes finais do grupo e, se houver, adicione DOI ou URL final.

Referencia curta sugerida:

```text
ACDLR Team. ACDLR: Automated Crater Detection and Landing Risk. 2026.
Classical computer-vision pipeline for lunar crater detection and visual
landing-risk estimation, with an Ellipse R-CNN pretrained visual-CNN comparison.
```

## Licenca

Projeto academico e educacional.
