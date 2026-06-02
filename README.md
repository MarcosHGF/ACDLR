# ACDLR — Automated Crater Detection and Landing Risk

> Aplicação de visão computacional para detecção automática de crateras em imagens da superfície lunar e estimativa visual simplificada de risco de pouso.

![Imagem do projeto](./output.jpg)

---

## Sobre o projeto

O **ACDLR** é um projeto de visão computacional desenvolvido para detectar crateras em imagens da superfície lunar e gerar uma **análise visual simplificada de risco de pouso**.

A proposta do projeto não é reproduzir um sistema real de navegação espacial, mas sim construir uma ferramenta **didática, visual e interativa**, capaz de demonstrar como técnicas clássicas de processamento de imagens podem ser aplicadas à identificação automática de padrões e à geração de uma métrica simples de apoio à decisão.

O sistema recebe uma imagem lunar, realiza o pré-processamento, detecta crateras automaticamente, divide a imagem em regiões e calcula uma pontuação de risco baseada em características visuais observadas.

---

## Objetivo

O objetivo principal do projeto é:

- detectar crateras automaticamente em imagens lunares;
- analisar a distribuição espacial dessas crateras;
- estimar uma pontuação visual de risco por região;
- indicar a área mais favorável para pouso dentro da imagem analisada;
- apresentar os resultados de forma clara, interativa e visualmente forte.

---

## O que o sistema faz

O ACDLR executa as seguintes etapas:

1. **Recebe uma imagem da superfície lunar**
2. **Aplica pré-processamento**
   - conversão para tons de cinza;
   - suavização para redução de ruído;
   - realce de contraste, quando necessário;
3. **Detecta crateras automaticamente**
   - filtro casado multi-escala com assinatura visual de cratera;
   - geração de candidatos por máximos locais;
   - validação geométrica, fotométrica e por suporte de borda;
   - refinamento local por Hough apenas como ajuste final;
4. **Divide a imagem em regiões**
   - por exemplo, em uma grade 3x3;
5. **Calcula a pontuação de risco**
   - número de crateras;
   - tamanho médio estimado;
   - densidade de crateras por região;
6. **Exibe o resultado final**
   - imagem original;
   - crateras destacadas;
   - grade de análise;
   - pontuação de risco por região;
   - melhor região para pouso.

---

## Como funciona

O fluxo do projeto pode ser resumido da seguinte forma:

```text
Imagem lunar
   ↓
Pré-processamento
   ↓
Filtro casado multi-escala e validação geométrica/fotométrica
   ↓
Identificação das crateras
   ↓
Divisão da imagem em regiões
   ↓
Cálculo de risco por região
   ↓
Visualização final da análise
```

---

## Stack utilizada

**Linguagem principal**
- Python

**Bibliotecas**
- OpenCV
Responsável pela leitura das imagens, pré-processamento, filtros clássicos, detecção de bordas, refinamento geométrico e anotação visual dos resultados.
- NumPy
Utilizado para manipulação de arrays, operações matriciais e apoio aos cálculos da pontuação de risco.
- Matplotlib
Pode ser usado para visualização de etapas intermediárias, comparação entre resultados e apoio à apresentação.
- Streamlit
Utilizado para construir a interface interativa, permitindo upload da imagem, execução do processamento e exibição imediata dos resultados.

---

## Dataset / dados utilizados

O projeto utiliza **imagens públicas da superfície lunar**, escolhidas pela qualidade visual e pela presença clara de crateras.

Para a execução prática e simples do projeto, a ideia é trabalhar com um conjunto de imagens estáticas de teste, suficiente para demonstrar:

entrada de imagens lunares;
detecção automática de crateras;
análise por regiões;
classificação visual de risco de pouso.

---

## Dataset utilizado no projeto
LROC NAC ROI_TORICELILOA (https://data.lroc.im-ldi.com/lroc/view_rdr/NAC_ROI_TORICELILOA)
Escala aproximada: 1.10 m/px

Isso permite, se desejado, converter medidas em pixels para uma estimativa física simples:
```text
metros = pixels * 1.10
```

---

## Funcionalidades principais
- upload de imagem lunar;
- pré-processamento automático;
- detecção de crateras por visão computacional clássica;
- grade de análise por regiões;
- cálculo de risco por região;
- destaque visual da área mais favorável para pouso;
- interface amigável para demonstração.

---

## Critérios da pontuação de risco

A pontuação de risco é uma métrica simplificada e didática, baseada em informações visuais extraídas da imagem.

Exemplos de critérios:

- quantidade de crateras detectadas em cada região;
- tamanho médio aproximado das crateras;
- densidade de crateras por área;
- maior cratera detectada na região;
- cobertura estimada das crateras sobre a célula;
- distância livre sugerida para o ponto de pouso;
- distribuição espacial dos obstáculos.

O score atual usa componentes físicos limitados em uma escala de 0 a 100, em vez de normalizar sempre a imagem para que alguma região seja obrigatoriamente "100". Isso ajuda a comparar tiles diferentes de forma mais coerente.

Modelo atual:

| Componente | Peso máximo | Saturação didática |
|---|---:|---:|
| Densidade de crateras | 30 pontos | 120 crateras/km² |
| Diâmetro médio | 20 pontos | 80 m |
| Maior diâmetro | 35 pontos | 140 m |
| Cobertura por crateras | 15 pontos | 8% da célula |

Esses valores não representam critérios oficiais de pouso lunar. Eles são limites interpretáveis para uma aplicação acadêmica e podem ser recalibrados quando houver mais tiles anotados.

> Observação: esta pontuação não representa uma medida física real de segurança espacial. Ela foi proposta como uma métrica visual coerente com o objetivo acadêmico do projeto.

---

## Saída esperada

Ao final da análise, o sistema deve exibir:

- a imagem original;
- a imagem com crateras marcadas;
- a grade de regiões;
- a pontuação de cada região;
- a classificação final de risco;
- a região sugerida como mais favorável para pouso.

---

## Diferencial do projeto

O diferencial do ACDLR está em transformar um problema clássico de detecção de crateras em uma aplicação:

- mais visual;
- mais intuitiva;
- mais interativa;
- mais adequada para apresentação acadêmica;
- mais original dentro do contexto de uma disciplina introdutória de visão computacional.

Em vez de apenas detectar crateras, o projeto organiza essa detecção como uma ferramenta de apoio visual à decisão.

---

## Comparacao com CNN aberta

O ACDLR nao incorpora deep learning no pipeline. A comparacao neural principal
agora substitui DeepMoon pelo repositorio aberto **crater-identification**:
https://github.com/sydney-machine-learning/crater-identification.

Esse repositorio esta associado ao artigo **Deep learning framework for crater
detection and identification on the Moon and Mars**, publicado na Nature/npj
Space Exploration em 2026: https://www.nature.com/articles/s44453-026-00036-x.

A escolha encaixa melhor neste projeto porque o metodo CNN usa YOLO/Ultralytics
em imagens anotadas no formato YOLO, que e o mesmo formato do novo dataset
`data/LU3M6TGT_yolo_format`. DeepMoon continua sendo uma referencia historica
importante, mas usa DEMs lunares e uma pilha legada, entao nao e o melhor
competidor executavel para tiles visuais anotados.

| Criterio | ACDLR | CNN YOLOv11 |
|---|---|---|
| Entrada | Imagens/tiles visuais da superficie lunar | Imagens/tiles visuais anotados em YOLO |
| Metodo central | Visao computacional classica com CLAHE, filtro casado, bordas e validacao geometrica/fotometrica | Rede CNN YOLOv11 para deteccao de objetos |
| Treinamento | Nao exige treinamento; parametros sao explicitos | Exige treino ou pesos `best.pt` treinados |
| Saida | Circulos `(x, y, r)` | Boxes YOLO convertidos para circulos `(x, y, r)` |
| Metricas | Precisao, recall, F1, erro de centro e erro de raio | As mesmas metricas, no mesmo dataset |
| Papel no projeto | Metodo principal, explicavel e sem IA | Competidor neural para medir teto de desempenho |

O ponto principal da comparacao e: **ACDLR explicita os criterios visuais e
geometricos; a CNN aprende esses criterios a partir de anotacoes**. Por isso,
a comparacao executavel roda os dois metodos no mesmo split anotado, com a
mesma tolerancia de matching.

---

## Roteiro de evolucao sem incorporar IA no ACDLR

1. Criar um pequeno benchmark anotado manualmente com tiles do LROC NAC ROI_TORICELILOA.
2. Medir metricas comparaveis: precisao, recall, F1, erro medio de centro e erro medio de raio.
3. Ajustar o detector clássico usando esses resultados, não apenas inspeção visual.
4. Tornar o score de risco mais absoluto e comparável entre imagens.
5. Exibir comparacao direta entre ACDLR e a CNN YOLOv11 no mesmo dataset.
6. Documentar limitações com honestidade: iluminação, sombras, crateras degradadas e falsos positivos.

---

## Benchmark classico do ACDLR

O repositorio inclui um script para avaliar o pipeline classico contra anotacoes manuais usando metricas comparaveis:

```bash
python benchmark_classical.py --images-dir data/lroc_nac_roi_toriceliloa_tiles --annotations-dir data/annotations
```

O relatório gerado inclui:

- precisão;
- recall;
- F1;
- erro médio de centro em pixels;
- erro médio de centro normalizado pelo raio anotado;
- erro médio de raio em pixels;
- erro médio de raio normalizado pelo raio anotado.

Para cada imagem, crie um arquivo de anotação com o mesmo nome-base:

```text
data/lroc_nac_roi_toriceliloa_tiles/torricelli_y00000_x00000.png
data/annotations/torricelli_y00000_x00000.csv
```

Formato CSV aceito:

```csv
cx,cy,radius_px
512,438,22
610,701,14
```

Também é aceito JSON como lista de objetos ou objeto com chave `craters`.

---

## Comparacao executavel ACDLR x CNN YOLOv11

O comparador principal agora e o YOLOv11/CNN do repositorio aberto
`sydney-machine-learning/crater-identification`. O repo fica em
`external/crater-identification` e o ACDLR continua sem IA.

Para treinar um baseline CNN pequeno e comparar os dois metodos no mesmo
dataset anotado:

```bash
python scripts/run_acdlr_vs_crater_cnn_comparison.py --max-images 5
```

O script gera:

- `artifacts/acdlr_vs_crater_cnn/comparison_report.md`;
- `artifacts/acdlr_vs_crater_cnn/visual_comparison.png`;
- `artifacts/acdlr_vs_crater_cnn/acdlr/acdlr_yolo_summary.json`;
- `artifacts/acdlr_vs_crater_cnn/crater_cnn_yolo/cnn_yolo_summary.json`.

Se quiser treinar a CNN separadamente antes da comparacao:

```bash
python scripts/train_crater_cnn_yolo.py --epochs 1 --fraction 0.02
python scripts/benchmark_crater_cnn_yolo.py ^
  --weights artifacts/crater_cnn_yolo_train/moon_small/weights/best.pt ^
  --max-images 5 ^
  --conf 0.001 ^
  --iou 0.15 ^
  --max-det 150
```

O `yolo11n.pt` original do repo e usado como base YOLO. Para metricas honestas
de crateras, use `best.pt` ou `last.pt` treinado no dataset de crateras.

---

## Benchmark no novo dataset YOLO anotado

O dataset `data/LU3M6TGT_yolo_format` usa imagens em `images/` e anotacoes
YOLO em `labels/`. Para avaliar o ACDLR nesse formato com criterio mais rigido:

```bash
python scripts/benchmark_yolo_dataset.py ^
  --split valid ^
  --max-images 25 ^
  --out-dir artifacts/yolo_benchmark_final25
```

Configuracao recomendada com tolerancia permissiva usada no comparador:

```bash
python scripts/benchmark_yolo_dataset.py ^
  --split valid ^
  --max-images 25 ^
  --out-dir artifacts/yolo_benchmark_cnn_tolerance25 ^
  --min-radius 4 ^
  --max-radius 70 ^
  --canny-threshold 45 ^
  --strictness 16 ^
  --center-tolerance 1.34 ^
  --radius-tolerance 1.0
```

Saidas:

- `artifacts/yolo_benchmark_final25/acdlr_yolo_report.md`;
- `artifacts/yolo_benchmark_cnn_tolerance25/acdlr_yolo_report.md`;
- `artifacts/yolo_benchmark_cnn_tolerance25/acdlr_yolo_summary.json`;
- `artifacts/yolo_benchmark_cnn_tolerance25/visuals/`.

Para gerar a comparacao visual completa ACDLR x CNN:

```bash
python scripts/run_acdlr_vs_crater_cnn_comparison.py --max-images 5
```

O treino padrao acima e propositalmente pequeno, usado como smoke test. Para
uma CNN competitiva de verdade, aumente `--cnn-train-epochs`,
`--cnn-train-fraction` e avalie mais imagens.

---

## Ajuste do detector clássico

Depois de criar anotacoes manuais, a etapa seguinte e ajustar os parametros do detector com evidencia quantitativa. O script abaixo executa uma busca em grade e ordena os resultados por F1, mantendo o ACDLR como metodo classico sem treino neural.

```bash
python tune_classical.py --images-dir data/lroc_nac_roi_toriceliloa_tiles --annotations-dir data/annotations
```

Exemplo com faixas personalizadas:

```bash
python tune_classical.py ^
  --images-dir data/lroc_nac_roi_toriceliloa_tiles ^
  --annotations-dir data/annotations ^
  --min-radius-values 8,10,12 ^
  --max-radius-values 35,40,50 ^
  --canny-threshold-values 50,60,70 ^
  --strictness-values 28,34,40
```

O arquivo `tuning_results.csv` mostra os conjuntos de parâmetros ranqueados. A escolha recomendada é começar pelo maior F1 e, em caso de empate, preferir menor erro normalizado de centro e raio.

---

## Possível pipeline de implementação

```text
1. Ler imagem
2. Converter para grayscale
3. Aplicar blur / suavização
4. Melhorar contraste com CLAHE
5. Aplicar filtro casado multi-escala
6. Gerar candidatos a crateras
7. Validar centro, raio, contraste, aro e suporte de borda
8. Refinar localmente e remover duplicatas
9. Dividir imagem em grade
10. Calcular score de risco por região
11. Sugerir ponto de pouso por distância livre
12. Exibir resultado final
```

---

## Como executar
1. Clone o repositório
```bash
git clone https://github.com/MarcosHGF/acdlr.git
cd acdlr
```

2. Crie e ative o ambiente virtual
```bash
python -m venv venv
```
    Windows
```bash
venv\Scripts\activate
```
    Linux / macOS
```bash
source venv/bin/activate
```
3. Instale as dependências
```bash
pip install -r requirements.txt
```
4. Gerar tiles a partir de uma imagem grande
```bash
python prepare_default_dataset.py --input caminho/para/sua_imagem_grande.png
```

Para reproduzir a validacao feita com o dataset oficial LROC
NAC_ROI_TORICELILOA, use o produto 5M:

```bash
python prepare_default_dataset.py --input data/raw/NAC_ROI_TORICELILOA_E047S0284_5M.TIF --tile-size 1024 --overlap 64 --prefix torricelli_5m
python scripts/validate_lroc_toricelli_dataset.py --images-dir data/lroc_nac_roi_toriceliloa_tiles --out-dir artifacts/toricelli_validation --scale-m-per-px 5.0
```

O script salva um CSV descritivo, um relatorio Markdown e prints em:

```text
artifacts/toricelli_validation/
```

5. Execute a aplicação
```bash
streamlit run app.py
```

6. Opcionalmente, rode o benchmark com anotações manuais
```bash
python benchmark_classical.py --images-dir data/lroc_nac_roi_toriceliloa_tiles --annotations-dir data/annotations
```

7. Ajuste os parâmetros do detector a partir das métricas
```bash
python tune_classical.py --images-dir data/lroc_nac_roi_toriceliloa_tiles --annotations-dir data/annotations
```

---

## Exemplo de uso
- Abra a aplicação no navegador;
- Faça upload de uma imagem da superfície lunar;
- Execute o processamento;
- Visualize as crateras detectadas;
- Analise a pontuação de risco em cada região;
- Observe a região indicada como mais favorável para pouso.

---

## Tecnologias de referência

Este projeto foi inspirado e apoiado conceitualmente por referências como:

- Lunar Crater Detector
- YOLOv11/CNN para deteccao de crateras
- OpenCV Hough Circle Transform
- OpenCV Canny Edge Detection
- Moon Crater Database

---

## Limitações e ameaças à validade

O ACDLR e uma ferramenta academica de demonstracao. A comparacao com a CNN YOLOv11 ajuda a organizar a avaliacao, mas nao transforma o projeto em um sistema cientifico de navegacao real.

| Área | Limitação | Impacto | Mitigação |
|---|---|---|---|
| Comparacao com CNN | A CNN pode ser treinada no dataset anotado; ACDLR nao usa treino | Se treino e validacao forem misturados, o resultado fica otimista | Treinar no `train`, avaliar no `valid` e reportar parametros |
| Benchmark | O conjunto anotado deve ter split claro e exemplos suficientes | Um teste pequeno serve para validar pipeline, nao conclusao final | Aumentar `--max-images` e avaliar o split `valid` completo |
| Detecção | Sombras, iluminação lateral, crateras degradadas e textura do relevo podem confundir o detector | Pode haver falsos positivos e falsos negativos | Ajustar parâmetros com validação anotada e registrar casos de erro |
| Escala | Medidas físicas dependem do valor de metros por pixel | O score de risco muda se a escala estiver incorreta | Usar a escala correta do dataset antes de comparar tiles |
| Score de risco | A pontuação é didática e simplificada | Não certifica segurança real de pouso | Apresentar como apoio visual à decisão, não como métrica operacional |
| Validação | Ajustar e avaliar no mesmo conjunto pequeno pode gerar overfitting | Resultados podem parecer melhores do que a generalização real | Separar tiles anotados em validação e teste |
| Cobertura do terreno | O sistema considera crateras detectadas, mas não modela rochas, inclinação real, propriedades do regolito ou iluminação operacional | A região sugerida pode ignorar riscos físicos não visíveis no pipeline | Declarar escopo visual e não substituir análise geológica/engenharia |

---

## Licença

Este projeto é de caráter acadêmico e educacional.
