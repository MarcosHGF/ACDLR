# ACDLR - Automated Crater Detection and Landing Risk

Aplicacao de visao computacional para detectar crateras lunares por processamento
classico de imagem e estimar uma pontuacao visual de risco de pouso por regiao.

![Imagem do projeto](./output.jpg)

## Resumo

O ACDLR detecta crateras em imagens da superficie lunar sem usar IA no metodo
principal. O pipeline usa OpenCV, filtros, bordas, realce de contraste,
validacao geometrica/fotometrica e analise por grade. Depois da deteccao, o
sistema mede as crateras, calcula um score de risco por regiao e sugere a area
mais favoravel para pouso dentro da imagem.

Tambem existe um benchmark comparativo contra uma CNN YOLOv11. Essa CNN e
apenas o metodo concorrente; ela nao faz parte do ACDLR. A comparacao roda nos
mesmos tiles anotados em YOLO e usa as mesmas metricas.

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

## Estrutura Principal

```text
app.py                                  Interface Streamlit
core/detection.py                       Detector classico ACDLR
core/preprocessing.py                   Pre-processamento
core/risk.py                            Score de risco e ponto de pouso
core/evaluation.py                      Metricas de benchmark
scripts/benchmark_yolo_dataset.py       Avalia ACDLR no dataset YOLO
scripts/train_crater_cnn_yolo.py        Treina baseline CNN YOLOv11
scripts/benchmark_crater_cnn_yolo.py    Avalia a CNN no mesmo dataset
scripts/run_acdlr_vs_crater_cnn_comparison.py
                                        Roda ACDLR x CNN e gera relatorio
configs/                                Configuracoes dos experimentos
docs/                                   Documentacao detalhada
paper/                                  Materiais reservados para artigo
reports/                                Relatorios selecionados para publicacao
CITATION.cff                            Metadados de citacao
environment.yml                         Ambiente Conda reproduzivel
pyproject.toml                          Metadados Python do projeto
requirements-dev.txt                    Dependencias de desenvolvimento/testes
tests/                                  Testes de regressao do artefato
```

## Dataset

O dataset anotado usado no benchmark fica em:

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

Estado atual do dataset local:

| Split | Imagens |
|---|---:|
| train | 8756 |
| valid | 1545 |

As anotacoes seguem formato YOLO:

```text
classe x_centro y_centro largura altura
```

Todos os valores de coordenada e tamanho sao normalizados entre 0 e 1. Para
comparar com o ACDLR, cada box YOLO e convertido em uma cratera circular:

```text
cx = x_centro * largura_da_imagem
cy = y_centro * altura_da_imagem
raio = media(largura_box_px, altura_box_px) / 2
```

## Instalacao

Clone o repositorio do ACDLR e entre na pasta:

```bash
git clone https://github.com/MarcosHGF/ACDLR.git ACDLR
cd ACDLR
```

Instale as dependencias:

```bash
python -m pip install -r requirements.txt
```

Dependencias principais:

- OpenCV
- NumPy
- Matplotlib
- Streamlit
- scikit-image
- Ultralytics, apenas para o benchmark CNN

Para desenvolvimento/testes:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Para usar a comparacao CNN, clone tambem o repositorio externo usado como
referencia:

```bash
mkdir -p external
git clone https://github.com/sydney-machine-learning/crater-identification.git external/crater-identification
```

Depois coloque o dataset YOLO em:

```text
data/LU3M6TGT_yolo_format
```

Guia completo: [Como rodar](docs/COMO_RODAR.md).
Detalhes do metodo e parametros: [ACDLR: funcionamento e configuracao](docs/ACDLR_CONFIGURACAO.md).

## Como Rodar A Interface

```bash
streamlit run app.py
```

Abra no navegador:

```text
http://localhost:8501
```

Na interface voce pode:

- escolher um tile do dataset local;
- enviar uma imagem manualmente;
- ajustar grade, escala e parametros de deteccao;
- ver pre-processamento, crateras detectadas e mapa de risco;
- visualizar a comparacao ACDLR x CNN YOLOv11 quando o benchmark ja tiver sido
  gerado.

## Como Rodar O Benchmark ACDLR

Teste pequeno no split `valid`:

```bash
python scripts/benchmark_yolo_dataset.py \
  --split valid \
  --max-images 25 \
  --out-dir artifacts/acdlr_yolo_benchmark
```

Configuracao usada na comparacao com CNN:

```bash
python scripts/benchmark_yolo_dataset.py \
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

## Como Rodar A Comparacao ACDLR x CNN

Comando principal:

```bash
python scripts/run_acdlr_vs_crater_cnn_comparison.py --max-images 5
```

Esse comando:

1. treina uma CNN YOLOv11 pequena se ainda nao existir peso treinado;
2. roda o ACDLR no mesmo split anotado;
3. roda a CNN no mesmo split anotado;
4. calcula precision, recall e F1 para os dois;
5. gera uma imagem lado a lado;
6. gera relatorio Markdown e JSONs de resumo.

Saidas principais:

```text
artifacts/acdlr_vs_crater_cnn/comparison_report.md
artifacts/acdlr_vs_crater_cnn/visual_comparison.png
artifacts/acdlr_vs_crater_cnn/acdlr/acdlr_yolo_summary.json
artifacts/acdlr_vs_crater_cnn/crater_cnn_yolo/cnn_yolo_summary.json
```

Para forcar novo treino pequeno da CNN antes de comparar:

```bash
python scripts/run_acdlr_vs_crater_cnn_comparison.py \
  --max-images 5 \
  --force-cnn-train
```

Para um teste maior:

```bash
python scripts/run_acdlr_vs_crater_cnn_comparison.py --max-images 25
```

## Resultados Obtidos No Teste Atual

O ultimo smoke test executado usou:

- dataset: `data/LU3M6TGT_yolo_format`;
- split: `valid`;
- imagens avaliadas: 3;
- tolerancia de matching: centro <= `1.34 * raio_anotado`;
- tolerancia de raio: erro <= `1.0 * raio_anotado`;
- CNN: YOLOv11 treinada por 1 epoca em uma fracao pequena do treino;
- ACDLR: sem treino, sem rede neural, apenas processamento classico.

| Metodo | Deteccoes | GT | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACDLR | 286 | 352 | 180 | 106 | 172 | 0.6294 | 0.5114 | 0.5643 |
| CNN YOLOv11 | 361 | 352 | 117 | 244 | 235 | 0.3241 | 0.3324 | 0.3282 |

Resultado do teste pequeno: **ACDLR venceu em F1**.

Visual gerado:

```text
artifacts/acdlr_vs_crater_cnn/visual_comparison.png
```

Na imagem:

- verde = true positive;
- vermelho = false positive;
- amarelo = false negative.

Esses resultados sao um teste de validacao do pipeline, nao uma conclusao
final. Para resultado de apresentacao forte, rode mais imagens e treine a CNN
por mais epocas.

## Documentacao Detalhada

- [Como rodar](docs/COMO_RODAR.md)
- [Metodologia do ACDLR](docs/METODOLOGIA.md)
- [Benchmark e resultados](docs/BENCHMARK_E_RESULTADOS.md)
- [Dataset card](docs/DATASET.md)
- [Protocolo experimental](docs/EXPERIMENT_PROTOCOL.md)
- [Reprodutibilidade](docs/REPRODUCIBILITY.md)
- [Estrutura dos arquivos](docs/ESTRUTURA.md)
- [Checklist de repositorio de artigo](docs/ARTICLE_REPOSITORY_CHECKLIST.md)

## Como Citar

O repositorio inclui `CITATION.cff`. Antes de publicar, substitua os autores
genericos pelos nomes finais do grupo e, se houver, adicione o DOI ou URL do
repositorio.

Referencia curta sugerida:

```text
ACDLR Team. ACDLR: Automated Crater Detection and Landing Risk. 2026.
Classical computer-vision pipeline for lunar crater detection and visual
landing-risk estimation.
```

## Estrutura De Repositorio De Artigo

O repositorio agora segue uma estrutura mais proxima de um artefato de artigo
de visao computacional:

- metodo implementado em `core/`;
- scripts reproduziveis em `scripts/`;
- metadados Python em `pyproject.toml`;
- testes em `tests/`;
- configs dos experimentos em `configs/`;
- documentacao tecnica em `docs/`;
- metadados de citacao em `CITATION.cff`;
- ambiente reproduzivel em `requirements.txt` e `environment.yml`;
- artefatos gerados em `artifacts/`;
- espaco reservado para material do paper em `paper/`;
- relatorios estaveis para apresentacao/publicacao em `reports/`.

## Observacoes Sobre A CNN De Comparacao

A CNN usada no benchmark foi estruturada a partir do repositorio aberto
`sydney-machine-learning/crater-identification`, associado ao artigo
**Deep learning framework for crater detection and identification on the Moon
and Mars**.

Essa escolha e pratica porque o repositorio usa YOLO/Ultralytics e se encaixa
no dataset local em formato YOLO. O DeepMoon continua sendo uma referencia
historica importante em deteccao de crateras por deep learning, mas ele usa DEMs
lunares e nao encaixa diretamente nos tiles visuais anotados deste projeto.

## Limitacoes

- O score de risco e didatico, nao uma metrica oficial de seguranca espacial.
- Sombras, iluminacao lateral e crateras degradadas podem gerar falsos positivos.
- O benchmark atual pequeno serve para validar o pipeline; nao substitui uma
  avaliacao completa no split `valid`.
- Se a CNN for treinada por poucas epocas, ela nao representa o teto real de
  desempenho de deep learning.
- O ACDLR deve permanecer sem IA no metodo principal; a CNN e apenas comparador.

## Licenca

Projeto academico e educacional.
