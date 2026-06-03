# Estrutura Dos Arquivos

## Raiz

```text
README.md
app.py
requirements.txt
requirements-dev.txt
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
| `README.md` | Visao geral, comandos principais e resultado atual |
| `app.py` | Interface Streamlit |
| `requirements.txt` | Dependencias Python |
| `requirements-dev.txt` | Dependencias de desenvolvimento e testes |
| `environment.yml` | Ambiente Conda reproduzivel |
| `pyproject.toml` | Metadados Python e configuracoes de ferramentas |
| `CITATION.cff` | Metadados de citacao do software |
| `benchmark_classical.py` | Benchmark antigo para anotacoes CSV/JSON manuais |
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
| `detection.py` | Detector classico de crateras |
| `measurement.py` | Converte circulos em medidas fisicas |
| `risk.py` | Calcula score de risco e ponto de pouso |
| `visualization.py` | Desenha crateras, grade e resultado final |
| `tiling.py` | Divide imagem em tiles e recompõe coordenadas |
| `evaluation.py` | Calcula precision, recall, F1 e erros |

## Tests

```text
tests/
  README.md
  test_evaluation.py
```

| Arquivo | Funcao |
|---|---|
| `test_evaluation.py` | Testes de regressao das metricas de benchmark |

## Scripts

```text
scripts/
  README.md
  benchmark_yolo_dataset.py
  train_crater_cnn_yolo.py
  benchmark_crater_cnn_yolo.py
  run_acdlr_vs_crater_cnn_comparison.py
```

| Script | Funcao |
|---|---|
| `benchmark_yolo_dataset.py` | Avalia o ACDLR em dataset YOLO |
| `train_crater_cnn_yolo.py` | Treina baseline CNN YOLOv11 |
| `benchmark_crater_cnn_yolo.py` | Avalia CNN no mesmo dataset |
| `run_acdlr_vs_crater_cnn_comparison.py` | Roda comparacao completa ACDLR x CNN |

Scripts antigos relacionados a DeepMoon podem existir como referencia historica,
mas o comparador principal atual e o YOLOv11/CNN.

## Configs

```text
configs/
  acdlr_default.yaml
  benchmark_smoke.yaml
  benchmark_valid25.yaml
  cnn_baseline_smoke.yaml
```

| Arquivo | Funcao |
|---|---|
| `acdlr_default.yaml` | Parametros padrao do detector classico |
| `benchmark_smoke.yaml` | Protocolo do teste pequeno ACDLR x CNN |
| `benchmark_valid25.yaml` | Protocolo para 25 imagens do split valid |
| `cnn_baseline_smoke.yaml` | Parametros de treino/inferencia da CNN rapida |

## Dataset

```text
data/LU3M6TGT_yolo_format/
  train/images
  train/labels
  valid/images
  valid/labels
  data.yaml
```

As imagens e labels ficam ignoradas pelo Git por serem dados locais/grandes.

## External

```text
external/crater-identification/
```

Repositorio externo usado como referencia para o comparador CNN YOLOv11.

## Artifacts

```text
artifacts/
  acdlr_vs_crater_cnn/
  crater_cnn_yolo_train/
  ultralytics_config/
```

| Pasta | Funcao |
|---|---|
| `acdlr_vs_crater_cnn` | Relatorio, JSONs e imagem lado a lado |
| `crater_cnn_yolo_train` | Pesos treinados da CNN |
| `ultralytics_config` | Configuracao local do Ultralytics |

Os arquivos em `artifacts/` sao gerados automaticamente e nao precisam ser
versionados.

## Docs

```text
docs/
  ACDLR_CONFIGURACAO.md
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

`paper/` reserva espaco para o manuscrito e figuras finais. `reports/` serve
para relatorios estaveis selecionados para apresentacao ou publicacao. Saidas
brutas continuam em `artifacts/`.
