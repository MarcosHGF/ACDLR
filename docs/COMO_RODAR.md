# Como Rodar O ACDLR Do Zero

Este guia foi escrito para um repositorio publico/artigo. Ele nao depende do
caminho de uma maquina especifica. Os comandos funcionam a partir de uma pasta
qualquer onde voce queira instalar o projeto.

## 0. Pre-Requisitos

Instale antes:

- Git
- Python 3.11 ou superior
- pip
- Opcional: Conda
- Opcional: GPU/CUDA para treinar a CNN mais rapido

Verifique:

```bash
git --version
python --version
python -m pip --version
```

No Windows, use PowerShell. No Linux/macOS, use Bash ou equivalente.

## 1. Clonar Este Repositorio

Substitua `<URL_DO_REPOSITORIO_ACDLR>` pela URL real do repositório do projeto.

```bash
git clone <URL_DO_REPOSITORIO_ACDLR> ACDLR
cd ACDLR
```

Se voce recebeu o projeto como `.zip`, extraia e entre na pasta:

```bash
cd ACDLR
```

Todos os comandos abaixo assumem que voce esta dentro da pasta raiz do projeto.

## 2. Criar Ambiente Python

### Opcao A: venv

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

### Opcao B: Conda

```bash
conda env create -f environment.yml
conda activate acdlr
```

## 3. Clonar O Repositorio CNN Usado Na Comparacao

O ACDLR nao usa IA. A CNN e apenas o metodo concorrente usado no benchmark.
O repositorio externo usado como referencia YOLOv11/CNN e:

```text
https://github.com/sydney-machine-learning/crater-identification
```

Clone exatamente dentro de `external/crater-identification`:

```bash
mkdir -p external
git clone https://github.com/sydney-machine-learning/crater-identification.git external/crater-identification
```

No Windows PowerShell, se `mkdir -p` nao funcionar:

```powershell
New-Item -ItemType Directory -Force external
git clone https://github.com/sydney-machine-learning/crater-identification.git external\crater-identification
```

Verifique se o peso base YOLO existe:

Windows:

```powershell
Test-Path external\crater-identification\YOLOv11model\yolo11n.pt
```

Linux/macOS:

```bash
test -f external/crater-identification/YOLOv11model/yolo11n.pt && echo ok
```

Observacao importante: esse `yolo11n.pt` e usado como peso base. O script do
ACDLR treina/fine-tuna um `best.pt` local para a comparacao no dataset de
crateras.

## 4. Colocar O Dataset

O dataset anotado deve ficar em:

```text
data/LU3M6TGT_yolo_format
```

Estrutura obrigatoria:

```text
data/LU3M6TGT_yolo_format/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
```

Cada imagem deve ter um `.txt` de anotacao YOLO com o mesmo nome-base:

```text
valid/images/exemplo.png
valid/labels/exemplo.txt
```

Formato de cada linha no `.txt`:

```text
classe x_centro y_centro largura altura
```

Todos os valores de coordenada/tamanho devem estar normalizados entre 0 e 1.

Se voce quiser criar um `data.yaml` relativo ao repositorio, use:

```yaml
train: train/images
val: valid/images
nc: 1
names: ["crater"]
```

O dataset e ignorado pelo Git porque pode ser grande. Em um clone novo, voce
precisa baixar/copiar o dataset manualmente para essa pasta.

## 5. Rodar Testes De Sanidade

Instale dependencias de desenvolvimento:

```bash
python -m pip install -r requirements-dev.txt
```

Rode:

```bash
python -m pytest
```

Resultado esperado:

```text
2 passed
```

## 6. Rodar A Interface Streamlit

```bash
streamlit run app.py
```

Abra:

```text
http://localhost:8501
```

Na interface voce pode:

- escolher um tile do dataset local;
- enviar uma imagem propria;
- ajustar raio minimo/maximo, Canny, strictness, grade e escala;
- ver pre-processamento, crateras detectadas e mapa de risco;
- ver a comparacao ACDLR x CNN quando ela ja tiver sido gerada.

## 7. Rodar Apenas O Metodo ACDLR

Comando rapido:

```bash
python scripts/benchmark_yolo_dataset.py --split valid --max-images 5
```

Comando recomendado para subset pequeno:

Windows PowerShell:

```powershell
python scripts\benchmark_yolo_dataset.py ^
  --dataset-dir data\LU3M6TGT_yolo_format ^
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

Linux/macOS:

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

## 8. Rodar A Comparacao ACDLR x CNN YOLOv11

Este comando roda o benchmark completo:

```bash
python scripts/run_acdlr_vs_crater_cnn_comparison.py --max-images 5
```

Ele faz:

1. verifica pesos da CNN;
2. se nao existir `best.pt`, treina um baseline pequeno;
3. roda o ACDLR no mesmo dataset;
4. roda a CNN no mesmo dataset;
5. calcula precision, recall e F1;
6. gera relatorio e imagem lado a lado.

Saidas:

```text
artifacts/acdlr_vs_crater_cnn/comparison_report.md
artifacts/acdlr_vs_crater_cnn/visual_comparison.png
artifacts/acdlr_vs_crater_cnn/run_summary.json
artifacts/acdlr_vs_crater_cnn/acdlr/acdlr_yolo_summary.json
artifacts/acdlr_vs_crater_cnn/crater_cnn_yolo/cnn_yolo_summary.json
```

Para forcar novo treino pequeno da CNN:

```bash
python scripts/run_acdlr_vs_crater_cnn_comparison.py --max-images 5 --force-cnn-train
```

Para reusar pesos ja treinados:

```bash
python scripts/run_acdlr_vs_crater_cnn_comparison.py --max-images 5 --skip-cnn-train
```

Para avaliar mais imagens:

```bash
python scripts/run_acdlr_vs_crater_cnn_comparison.py --max-images 25
```

## 9. Treinar A CNN Separadamente

Smoke test rapido em CPU:

```bash
python scripts/train_crater_cnn_yolo.py --epochs 1 --fraction 0.02
```

Treino maior em CPU:

```bash
python scripts/train_crater_cnn_yolo.py --epochs 10 --fraction 0.20 --batch 8
```

Treino com GPU:

```bash
python scripts/train_crater_cnn_yolo.py --epochs 50 --fraction 1.0 --device 0
```

Pesos gerados:

```text
artifacts/crater_cnn_yolo_train/moon_small/weights/best.pt
artifacts/crater_cnn_yolo_train/moon_small/weights/last.pt
```

## 10. Avaliar A CNN Separadamente

```bash
python scripts/benchmark_crater_cnn_yolo.py \
  --weights artifacts/crater_cnn_yolo_train/moon_small/weights/best.pt \
  --max-images 25 \
  --conf 0.001 \
  --iou 0.15 \
  --max-det 150
```

No PowerShell, use `^` no lugar de `\` para quebrar linha:

```powershell
python scripts\benchmark_crater_cnn_yolo.py ^
  --weights artifacts\crater_cnn_yolo_train\moon_small\weights\best.pt ^
  --max-images 25 ^
  --conf 0.001 ^
  --iou 0.15 ^
  --max-det 150
```

## 11. Reproduzir O Smoke Test Do Artigo

Se `best.pt` ja existir:

```bash
python scripts/run_acdlr_vs_crater_cnn_comparison.py --max-images 3 --visual-count 3 --skip-cnn-train
```

Se `best.pt` nao existir:

```bash
python scripts/run_acdlr_vs_crater_cnn_comparison.py --max-images 3 --visual-count 3 --force-cnn-train
```

Resultado registrado no smoke test atual:

| Metodo | Precision | Recall | F1 |
|---|---:|---:|---:|
| ACDLR | 0.6294 | 0.5114 | 0.5643 |
| CNN YOLOv11 | 0.3241 | 0.3324 | 0.3282 |

## 12. Problemas Comuns

### O dataset nao foi encontrado

Confirme que existe:

```text
data/LU3M6TGT_yolo_format/valid/images
data/LU3M6TGT_yolo_format/valid/labels
```

### O repositorio CNN nao foi encontrado

Rode:

```bash
git clone https://github.com/sydney-machine-learning/crater-identification.git external/crater-identification
```

### A CNN nao detecta nada

O treino pode ter sido curto ou o limiar de confianca pode estar alto. No smoke
test foi usado:

```text
--conf 0.001 --iou 0.15 --max-det 150
```

### Streamlit nao abre

Tente definir outra porta:

```bash
streamlit run app.py --server.port 8502
```

### Ultralytics tenta escrever fora do projeto

Os scripts configuram automaticamente:

```text
artifacts/ultralytics_config
```

como pasta local de configuracao.
