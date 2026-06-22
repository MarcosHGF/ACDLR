# Como Rodar O ACDLR Do Zero

Este guia usa a comparacao justa atual: **ACDLR x Ellipse R-CNN** no mesmo
dataset visual YOLO.

## 1. Clonar O Projeto

```bash
git clone <URL_DO_REPOSITORIO_ACDLR> ACDLR
cd ACDLR
```

## 2. Criar Ambiente

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Colocar O Dataset

Estrutura esperada:

```text
data/LU3M6TGT_yolo_format/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
```

As labels devem estar em formato YOLO:

```text
classe x_centro y_centro largura altura
```

## 4. Preparar Ellipse R-CNN

```bash
python scripts/setup_ellipse_rcnn_pretrained.py
```

Esse comando clona:

```text
https://github.com/wdoppenberg/ellipse-rcnn
```

e tenta baixar:

```text
https://huggingface.co/wdoppenberg/crater-rcnn
```

Se o download automatico falhar por `cas-bridge.xethub.hf.co`, baixe
manualmente `model.safetensors` no Hugging Face e coloque em:

```text
artifacts/ellipse_rcnn_pretrained/crater-rcnn/model.safetensors
```

## 5. Rodar ACDLR Puro

```bash
python scripts/benchmark_yolo_dataset.py --split valid --max-images 25
```

## 6. Rodar Comparacao Justa ACDLR x Ellipse R-CNN

Teste pequeno:

```bash
python scripts/run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 3 --visual-count 2
```

Teste maior:

```bash
python scripts/run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 25 --visual-count 8
```

Saidas:

```text
artifacts/acdlr_vs_ellipse_rcnn/comparison_report.md
artifacts/acdlr_vs_ellipse_rcnn/visual_comparison.png
artifacts/acdlr_vs_ellipse_rcnn/charts/acdlr_vs_ellipse_rcnn_metrics.png
artifacts/acdlr_vs_ellipse_rcnn/run_summary.json
```

## 7. Rodar Interface

```bash
streamlit run app.py
```

Abra:

```text
http://localhost:8501
```

## 8. Por Que Essa Comparacao Faz Sentido

- ACDLR e Ellipse R-CNN rodam nas mesmas imagens visuais.
- Ambos usam as mesmas labels YOLO convertidas para crateras circulares.
- ACDLR produz circulos diretamente.
- Ellipse R-CNN produz elipses; o benchmark converte para circulos com:

```text
radius = (a + b) / 2
```

Assim, os dois metodos ficam no mesmo protocolo: mesma imagem, mesma anotacao,
mesma conversao para circulos e mesmas metricas.
