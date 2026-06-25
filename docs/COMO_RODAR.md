# Como Rodar O ACDLR Do Zero

Este guia reproduz a aplicacao completa: ACDLR puro, dataset anotado, app
Streamlit e comparacao opcional com Ellipse R-CNN.

## 1. Clonar O Projeto

```bash
git clone https://github.com/MarcosHGF/ACDLR.git ACDLR
cd ACDLR
```

## 2. Criar Ambiente Python

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

### Conda Opcional

```bash
conda env create -f environment.yml
conda activate acdlr
python -m pip install -r requirements.txt
```

## 3. Colocar O Dataset De Crateras

O dataset anotado deve ficar em:

```text
data/LU3M6TGT_yolo_format/
```

Estrutura obrigatoria:

```text
data/LU3M6TGT_yolo_format/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  data.yaml
```

O dataset oficial usado no projeto esta no Kaggle:

```text
https://www.kaggle.com/datasets/riccardolagrassa/lu3m6tgt
```

Baixe e organize automaticamente:

```bash
python scripts/setup_lu3m6tgt_dataset.py
```

Se o KaggleHub pedir autenticacao, configure sua conta Kaggle ou baixe o zip
manualmente pela pagina acima.

Esse comando usa `kagglehub` com:

```python
import kagglehub

path = kagglehub.dataset_download("riccardolagrassa/lu3m6tgt")
print("Path to dataset files:", path)
```

Se estiver em `.zip`, extraia para dentro de `data/` mantendo o nome:

```text
data/LU3M6TGT_yolo_format
```

No Windows PowerShell, exemplo:

```powershell
Expand-Archive .\LU3M6TGT_yolo_format.zip -DestinationPath .\data -Force
```

Valide a estrutura:

```bash
python -c "from pathlib import Path; p=Path('data/LU3M6TGT_yolo_format'); print('images:', (p/'valid/images').exists()); print('labels:', (p/'valid/labels').exists())"
```

As labels devem seguir YOLO:

```text
class x_center y_center width height
```

## 4. Rodar O App Somente Com ACDLR

```bash
streamlit run app.py
```

Abra:

```text
http://localhost:8501
```

Na interface:

1. abra a aba `Crateras lunares`;
2. selecione `Dataset padrao`;
3. escolha uma imagem do dataset;
4. deixe `Rodar Ellipse R-CNN nesta imagem` desligado;
5. clique em `Run Analysis on selected dataset tile`;
6. veja os resultados do ACDLR puro no final da analise.

A tabela aparece na secao **Resultados no dataset anotado** e inclui:

```text
Deteccoes, GT, TP, FP, FN, Precision, Recall, F1, Center err/r, Radius err/r
```

Se a imagem for upload manual, a tabela de acertos so aparece se existir label
YOLO correspondente. Para imagens sem label, o app mostra apenas analise visual.

## 5. Preparar A CNN Externa

O projeto usa Ellipse R-CNN somente como baseline neural. O ACDLR continua sem
IA.

Comando recomendado:

```bash
python scripts/setup_ellipse_rcnn_pretrained.py
```

Esse script faz:

1. clona `https://github.com/wdoppenberg/ellipse-rcnn` em `external/ellipse-rcnn`;
2. instala o pacote `ellipse-rcnn[hf]`;
3. baixa `wdoppenberg/crater-rcnn` para `artifacts/ellipse_rcnn_pretrained/crater-rcnn`;
4. cria `manifest.json` com as origens usadas.

Instalacao manual equivalente:

```bash
mkdir -p external
git clone https://github.com/wdoppenberg/ellipse-rcnn.git external/ellipse-rcnn
python -m pip install -r requirements-ai.txt
python -m pip install -e "external/ellipse-rcnn[hf]"
```

Peso pre-treinado:

```text
https://huggingface.co/wdoppenberg/crater-rcnn
```

Se o download automatico falhar, baixe `model.safetensors` manualmente em:

```text
https://huggingface.co/wdoppenberg/crater-rcnn/tree/main
```

e coloque em:

```text
artifacts/ellipse_rcnn_pretrained/crater-rcnn/model.safetensors
```

## 6. Rodar O App Com Comparacao CNN

Depois de preparar a CNN:

```bash
streamlit run app.py
```

Na sidebar, ligue:

```text
Rodar Ellipse R-CNN nesta imagem
```

Ao rodar a analise em uma imagem do dataset, o app mostra:

- tabela do ACDLR puro;
- visual ACDLR;
- visual Ellipse R-CNN;
- tabela comparativa ACDLR x Ellipse R-CNN quando houver label YOLO.

## 7. Rodar Benchmark ACDLR Puro

```bash
python scripts/benchmark_yolo_dataset.py --split valid --max-images 25
```

Saidas:

```text
artifacts/acdlr_yolo_benchmark/acdlr_yolo_report.md
artifacts/acdlr_yolo_benchmark/acdlr_yolo_summary.json
artifacts/acdlr_yolo_benchmark/acdlr_yolo_benchmark.csv
artifacts/acdlr_yolo_benchmark/visuals/
```

## 8. Rodar Benchmark ACDLR x CNN

Teste compacto usado no artigo:

```bash
python scripts/run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 25 --visual-count 8
```

Teste rapido de instalacao:

```bash
python scripts/run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 3 --visual-count 2
```

Saidas:

```text
artifacts/acdlr_vs_ellipse_rcnn/comparison_report.md
artifacts/acdlr_vs_ellipse_rcnn/visual_comparison.png
artifacts/acdlr_vs_ellipse_rcnn/charts/acdlr_vs_ellipse_rcnn_metrics.png
artifacts/acdlr_vs_ellipse_rcnn/run_summary.json
artifacts/acdlr_vs_ellipse_rcnn/acdlr/acdlr_yolo_summary.json
artifacts/acdlr_vs_ellipse_rcnn/ellipse_rcnn/ellipse_rcnn_yolo_summary.json
```

## 9. Conferir Se Esta Tudo Certo

Compile os arquivos Python:

```bash
python -m py_compile app.py core/detection.py core/evaluation.py scripts/benchmark_yolo_dataset.py scripts/benchmark_ellipse_rcnn_yolo_dataset.py scripts/run_acdlr_vs_ellipse_rcnn_comparison.py
```

Rode os testes:

```bash
python -m pytest
```

Verifique o dataset:

```bash
python -c "from pathlib import Path; p=Path('data/LU3M6TGT_yolo_format'); print(len(list((p/'valid/images').glob('*'))), len(list((p/'valid/labels').glob('*.txt'))))"
```

## 10. Problemas Comuns

### Porta 8501 ocupada

```bash
streamlit run app.py --server.port 8502
```

### App nao mostra metricas

Use uma imagem do dataset em `valid/images` que tenha label com mesmo nome em
`valid/labels`.

### CNN nao carrega

Confirme se existem:

```text
external/ellipse-rcnn/
artifacts/ellipse_rcnn_pretrained/crater-rcnn/model.safetensors
```

Depois rode novamente:

```bash
python scripts/setup_ellipse_rcnn_pretrained.py
```

### Hugging Face falha no download

Baixe manualmente `model.safetensors` no navegador e coloque no caminho:

```text
artifacts/ellipse_rcnn_pretrained/crater-rcnn/model.safetensors
```

## 11. Comparacao Correta

A comparacao faz sentido porque:

- ACDLR e Ellipse R-CNN usam as mesmas imagens;
- ambos sao avaliados contra as mesmas labels YOLO;
- labels YOLO sao convertidas para circulos;
- Ellipse R-CNN prediz elipses e o benchmark converte para circulos com:

```text
radius = (a + b) / 2
```

Assim os dois metodos sao medidos no mesmo formato `(x, y, r)` e pelas mesmas
metricas.
