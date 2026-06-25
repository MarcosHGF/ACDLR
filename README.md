# ACDLR - Automated Crater Detection and Landing Risk

Aplicacao academica de visao computacional para detectar crateras lunares por
processamento classico de imagem, avaliar acertos em dataset anotado e comparar
o metodo ACDLR com uma CNN externa pre-treinada para crateras.

O metodo principal do projeto e o **ACDLR puro**: ele nao usa IA, CNN,
treinamento supervisionado ou modelos generativos. A CNN aparece apenas como
baseline externo de comparacao.

## Resumo

O ACDLR detecta crateras em imagens lunares usando OpenCV, realce de contraste,
bordas, filtros circulares multi-escala, validacao geometrica/fotometrica,
refinamento local e deduplicacao. A interface Streamlit permite:

- selecionar imagens do dataset anotado;
- rodar o ACDLR puro;
- ver etapas intermediarias do processamento;
- medir TP, FP, FN, precision, recall e F1 quando existe label YOLO;
- estudar o mesmo detector em outros datasets circulares;
- opcionalmente rodar Ellipse R-CNN e ver a comparacao lado a lado.

O baseline neural atual e **Ellipse R-CNN** com o peso
`wdoppenberg/crater-rcnn`, usando o repositorio aberto
`wdoppenberg/ellipse-rcnn`.

## Estrutura

```text
app.py                                            Interface Streamlit
core/preprocessing.py                             Pre-processamento
core/detection.py                                 Detector classico ACDLR
core/evaluation.py                                Matching e metricas
core/risk.py                                      Grade de risco visual
scripts/benchmark_yolo_dataset.py                 Benchmark ACDLR puro
scripts/setup_ellipse_rcnn_pretrained.py          Prepara Ellipse R-CNN
scripts/benchmark_ellipse_rcnn_yolo_dataset.py    Benchmark Ellipse R-CNN
scripts/run_acdlr_vs_ellipse_rcnn_comparison.py   Comparacao ACDLR x CNN
docs/COMO_RODAR.md                                Passo a passo completo
paper/                                            Artigo e figuras
data/                                             Datasets locais, nao versionados
external/                                         Repositorios externos clonados
artifacts/                                        Saidas de benchmark e pesos
```

## Requisitos

- Python 3.11 ou superior para o ACDLR.
- Python 3.12 ou superior recomendado para o baseline Ellipse R-CNN.
- Git instalado.
- Windows PowerShell, Linux shell ou macOS shell.
- Dataset YOLO de crateras em `data/LU3M6TGT_yolo_format`.
- Internet apenas para clonar repositorios externos e baixar pesos.

## Instalacao Rapida

```bash
git clone https://github.com/MarcosHGF/ACDLR.git ACDLR
cd ACDLR
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset De Crateras

O app e os benchmarks esperam este caminho:

```text
data/LU3M6TGT_yolo_format/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  data.yaml
```

O dataset oficial usado no projeto e:

```text
https://www.kaggle.com/datasets/riccardolagrassa/lu3m6tgt
```

Baixe e organize automaticamente com:

```bash
python scripts/setup_lu3m6tgt_dataset.py
```

Se o KaggleHub pedir autenticacao, configure sua conta Kaggle ou baixe o zip
manualmente pela pagina acima.

Esse script usa internamente:

```python
import kagglehub

path = kagglehub.dataset_download("riccardolagrassa/lu3m6tgt")
print("Path to dataset files:", path)
```

Se preferir baixar o `.zip` manualmente pelo Kaggle, extraia para que a pasta
final seja exatamente:

```text
data/LU3M6TGT_yolo_format
```

Conferencia rapida:

```bash
python -c "from pathlib import Path; p=Path('data/LU3M6TGT_yolo_format'); print((p/'valid/images').exists(), (p/'valid/labels').exists())"
```

As labels devem seguir YOLO:

```text
class x_center y_center width height
```

No benchmark, cada box YOLO e convertido para cratera circular:

```text
cx = x_center * image_width
cy = y_center * image_height
radius = (box_width_px + box_height_px) / 4
```

## Rodar A Aplicacao

```bash
streamlit run app.py
```

Abra:

```text
http://localhost:8501
```

Na aba **Crateras lunares**:

1. selecione `Dataset padrao`;
2. escolha um tile;
3. deixe `Rodar Ellipse R-CNN nesta imagem` desligado para ACDLR puro;
4. clique em `Run Analysis on selected dataset tile`;
5. veja a tabela **ACDLR puro nesta imagem** com TP, FP, FN, precision, recall e F1.

Para comparar com a CNN na mesma imagem, ligue `Rodar Ellipse R-CNN nesta imagem`
na sidebar. Se os pesos ainda nao existirem, siga a secao abaixo.

## Preparar Ellipse R-CNN

O comando recomendado clona o repositorio externo, instala o pacote e baixa o
peso pre-treinado:

```bash
python scripts/setup_ellipse_rcnn_pretrained.py
```

Ele cria:

```text
external/ellipse-rcnn/
artifacts/ellipse_rcnn_pretrained/crater-rcnn/
```

Repositorio externo usado:

```text
https://github.com/wdoppenberg/ellipse-rcnn
```

Peso pre-treinado usado:

```text
https://huggingface.co/wdoppenberg/crater-rcnn
```

Projeto completo relacionado:

```text
https://github.com/wdoppenberg/crater-detection
```

Se quiser preparar manualmente:

```bash
mkdir -p external
git clone https://github.com/wdoppenberg/ellipse-rcnn.git external/ellipse-rcnn
python -m pip install -r requirements-ai.txt
python -m pip install -e "external/ellipse-rcnn[hf]"
```

Se o download automatico do Hugging Face falhar, baixe manualmente
`model.safetensors` em:

```text
https://huggingface.co/wdoppenberg/crater-rcnn/tree/main
```

e coloque em:

```text
artifacts/ellipse_rcnn_pretrained/crater-rcnn/model.safetensors
```

## Rodar Benchmarks

ACDLR puro no dataset YOLO:

```bash
python scripts/benchmark_yolo_dataset.py --split valid --max-images 25
```

Comparacao ACDLR x Ellipse R-CNN:

```bash
python scripts/run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 25 --visual-count 8
```

Teste rapido de instalacao:

```bash
python scripts/run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 3 --visual-count 2
```

Saidas principais:

```text
artifacts/acdlr_yolo_benchmark/
artifacts/acdlr_vs_ellipse_rcnn/comparison_report.md
artifacts/acdlr_vs_ellipse_rcnn/visual_comparison.png
artifacts/acdlr_vs_ellipse_rcnn/charts/acdlr_vs_ellipse_rcnn_metrics.png
artifacts/acdlr_vs_ellipse_rcnn/run_summary.json
```

## Resultado Compacto Atual

Experimento com 25 imagens do split `valid`:

| Metodo | Det. | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACDLR | 2610 | 2994 | 1399 | 1211 | 1595 | 0.536 | 0.467 | 0.499 |
| Ellipse R-CNN | 316 | 2994 | 178 | 138 | 2816 | 0.563 | 0.059 | 0.108 |

Interpretacao: neste dataset local e sem fine-tuning da CNN, o ACDLR obteve
maior F1 por recuperar mais crateras anotadas. A CNN teve precision ligeiramente
maior, mas recall baixo. Isso nao prova superioridade geral de metodos classicos;
apenas descreve este protocolo zero-shot/pre-treinado.

## Solucao De Problemas

Se a porta `8501` estiver ocupada:

```bash
streamlit run app.py --server.port 8502
```

Se a tabela de acertos nao aparecer no app, verifique se a imagem selecionada
vem de:

```text
data/LU3M6TGT_yolo_format/valid/images
```

e se existe label com o mesmo nome em:

```text
data/LU3M6TGT_yolo_format/valid/labels
```

Se a CNN nao carregar, confirme:

```text
artifacts/ellipse_rcnn_pretrained/crater-rcnn/model.safetensors
external/ellipse-rcnn/
```

## Documentacao

- [Passo a passo completo](docs/COMO_RODAR.md)
- [Baseline IA visual](docs/AI_BASELINE_ELLIPSE_RCNN.md)
- [Configuracao do ACDLR](docs/ACDLR_CONFIGURACAO.md)
- [Benchmark e resultados](docs/BENCHMARK_E_RESULTADOS.md)
- [Dataset card](docs/DATASET.md)
- [Artigo LaTeX](paper/acdlr_artigo_cientifico.tex)

## Citacao

```text
ACDLR Team. ACDLR: Automated Crater Detection and Landing Risk. 2026.
Classical computer-vision pipeline for lunar crater detection and visual
landing-risk estimation, with an Ellipse R-CNN pretrained visual-CNN comparison.
```

## Licenca

Projeto academico e educacional.
