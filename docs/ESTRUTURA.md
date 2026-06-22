# Estrutura Dos Arquivos

Este documento descreve a organizacao atual do repositorio ACDLR como artefato
de um artigo de visao computacional. A estrutura ativa compara somente:

- **ACDLR**: metodo classico, sem IA, baseado em processamento de imagem.
- **Ellipse R-CNN**: baseline neural externo, pre-treinado para crateras e
  executado em modo vanilla/frozen.

## Raiz

```text
README.md
app.py
requirements.txt
requirements-dev.txt
requirements-ai.txt
environment.yml
pyproject.toml
CITATION.cff
benchmark_classical.py
tune_classical.py
prepare_default_dataset.py
output.jpg
```

| Arquivo | Funcao |
|---|---|
| `README.md` | Visao geral, instalacao, comandos e resultado atual |
| `app.py` | Interface Streamlit para deteccao, risco e comparacao visual |
| `requirements.txt` | Dependencias principais do ACDLR |
| `requirements-dev.txt` | Dependencias de testes |
| `requirements-ai.txt` | Dependencias opcionais do baseline Ellipse R-CNN |
| `environment.yml` | Ambiente Conda reproduzivel |
| `pyproject.toml` | Metadados Python e configuracoes de ferramentas |
| `CITATION.cff` | Metadados de citacao do software |
| `benchmark_classical.py` | Executor classico usado pelo benchmark do ACDLR |
| `tune_classical.py` | Busca de parametros do detector classico |
| `prepare_default_dataset.py` | Preparacao de tiles locais |
| `output.jpg` | Imagem de apresentacao |

## Core

```text
core/
  README.md
  preprocessing.py
  detection.py
  measurement.py
  risk.py
  visualization.py
  tiling.py
  evaluation.py
```

| Arquivo | Funcao |
|---|---|
| `preprocessing.py` | Tons de cinza, CLAHE, blur, realce e bordas |
| `detection.py` | Detector ACDLR com filtro multi-escala e validacao geometrica |
| `measurement.py` | Converte circulos detectados em medidas fisicas |
| `risk.py` | Calcula score de risco e ponto sugerido de pouso |
| `visualization.py` | Desenha crateras, grade de risco, legenda e resultado final |
| `tiling.py` | Divide imagens grandes em tiles e recompõe coordenadas |
| `evaluation.py` | Calcula precision, recall, F1 e erros normalizados |

## Scripts

```text
scripts/
  README.md
  benchmark_yolo_dataset.py
  setup_ellipse_rcnn_pretrained.py
  benchmark_ellipse_rcnn_yolo_dataset.py
  run_acdlr_vs_ellipse_rcnn_comparison.py
  run_toricelli_step_test.py
  validate_lroc_toricelli_dataset.py
```

| Script | Funcao |
|---|---|
| `benchmark_yolo_dataset.py` | Avalia o ACDLR no dataset YOLO anotado |
| `setup_ellipse_rcnn_pretrained.py` | Clona Ellipse R-CNN e baixa o peso `wdoppenberg/crater-rcnn` |
| `benchmark_ellipse_rcnn_yolo_dataset.py` | Avalia Ellipse R-CNN no mesmo dataset visual |
| `run_acdlr_vs_ellipse_rcnn_comparison.py` | Roda ACDLR e Ellipse R-CNN, gera graficos, visuais e relatorio |
| `run_toricelli_step_test.py` | Teste historico do fluxo Torricelli/LROC |
| `validate_lroc_toricelli_dataset.py` | Validacao de dataset local Torricelli/LROC |

## Configs

```text
configs/
  acdlr_default.yaml
  benchmark_smoke.yaml
  benchmark_valid25.yaml
```

| Arquivo | Funcao |
|---|---|
| `acdlr_default.yaml` | Parametros padrao do detector classico |
| `benchmark_smoke.yaml` | Protocolo pequeno para validacao rapida |
| `benchmark_valid25.yaml` | Protocolo para 25 imagens do split `valid` |

## Dataset

```text
data/LU3M6TGT_yolo_format/
  train/images
  train/labels
  valid/images
  valid/labels
  data.yaml
```

As imagens e labels ficam fora do versionamento por serem dados locais/grandes.
As anotacoes YOLO sao convertidas para crateras circulares para permitir a
mesma metrica entre ACDLR e Ellipse R-CNN.

## External

```text
external/ellipse-rcnn/
```

| Pasta | Funcao |
|---|---|
| `external/ellipse-rcnn` | Codigo vanilla do baseline neural externo |

O repositorio externo e clonado por `scripts/setup_ellipse_rcnn_pretrained.py`.
Ele nao deve ser alterado para manter a comparacao honesta.

## Artifacts

```text
artifacts/
  acdlr_yolo_benchmark/
  acdlr_vs_ellipse_rcnn/
  ellipse_rcnn_pretrained/
```

| Pasta | Funcao |
|---|---|
| `acdlr_yolo_benchmark` | Resultados do ACDLR puro no dataset local |
| `acdlr_vs_ellipse_rcnn` | Relatorio, grafico e visual ACDLR x Ellipse R-CNN |
| `ellipse_rcnn_pretrained` | Manifesto e pesos do modelo `wdoppenberg/crater-rcnn` |

Os arquivos em `artifacts/` sao gerados automaticamente e nao precisam ser
versionados.

## Docs

```text
docs/
  AI_BASELINE_ELLIPSE_RCNN.md
  ACDLR_CONFIGURACAO.md
  ARTIGO_CIENTIFICO_ACDLR.md
  COMO_RODAR.md
  METODOLOGIA.md
  BENCHMARK_E_RESULTADOS.md
  DATASET.md
  EXPERIMENT_PROTOCOL.md
  REPRODUCIBILITY.md
  ARTICLE_REPOSITORY_CHECKLIST.md
  ESTRUTURA.md
```

Use estes documentos para apresentacao, reproducao e explicacao tecnica do
projeto.

## Paper E Reports

```text
paper/
  README.md
reports/
  README.md
```

`paper/` guarda materiais para escrita do artigo. `reports/` aponta para os
relatorios gerados em `artifacts/`.
