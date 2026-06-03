# ACDLR: Funcionamento E Configuracao

Este documento explica como o ACDLR esta configurado no repositorio e como o
metodo detecta crateras lunares sem usar IA, CNN ou treinamento. A CNN YOLOv11
aparece apenas como metodo concorrente no benchmark.

## Visao Geral

ACDLR significa **Automated Crater Detection and Landing Risk**. O metodo
principal e um pipeline de processamento classico de imagem:

1. divide imagens grandes em tiles com sobreposicao;
2. realca contraste e bordas;
3. procura assinaturas circulares de crateras em multiplas escalas;
4. valida candidatos por criterios geometricos e fotometricos;
5. remove duplicatas;
6. mede crateras detectadas;
7. calcula risco visual de pouso por regiao;
8. compara deteccoes com anotacoes YOLO quando existe ground truth.

Arquivos principais:

| Arquivo | Papel |
|---|---|
| `core/preprocessing.py` | Conversao para cinza, CLAHE, blur, sharpening e bordas |
| `core/detection.py` | Detector classico de crateras do ACDLR |
| `core/tiling.py` | Split em tiles e recomposicao para coordenadas globais |
| `core/measurement.py` | Conversao de circulos em medidas fisicas |
| `core/risk.py` | Score de risco e ponto de pouso |
| `core/evaluation.py` | Precision, recall, F1 e matching com ground truth |
| `core/visualization.py` | Overlays finais, grade de risco e legendas |
| `configs/acdlr_default.yaml` | Configuracao padrao documentada |
| `app.py` | Interface Streamlit e parametros interativos |

## Pipeline Do Detector

### 1. Tiling

Imagens grandes sao processadas por partes para preservar crateras pequenas e
evitar perda de detalhe por redimensionamento. Cada tile e analisado localmente
e depois suas deteccoes sao convertidas para a coordenada global da imagem.

Configuracao padrao:

| Parametro | Valor | Onde altera |
|---|---:|---|
| `tile_size` | `1024` px | `configs/acdlr_default.yaml` e sidebar do `app.py` |
| `overlap` | `96` px | `configs/acdlr_default.yaml` e sidebar do `app.py` |

Sobreposicao evita que crateras cortadas na borda de um tile sejam perdidas.
Depois, `tiling.deduplicate` remove repeticoes geradas entre tiles vizinhos.

### 2. Pre-processamento

O pre-processamento transforma a imagem em uma representacao mais favoravel para
bordas e crateras:

1. converte para tons de cinza;
2. aplica CLAHE para realce local de contraste;
3. usa blur para reduzir ruido fino;
4. gera imagem realcada para deteccao;
5. calcula bordas usadas na validacao.

Configuracao padrao:

| Parametro | Valor | Efeito |
|---|---:|---|
| `clahe_clip` | `2.5` | Maior valor aumenta contraste local, mas tambem pode amplificar ruido |
| `blur_ksize` | `5` | Maior valor suaviza textura e reduz falsos positivos pequenos |

### 3. Geracao De Candidatos

O ACDLR nao treina uma rede. Ele usa filtros circulares explicitos em varias
escalas. Para cada raio possivel, o detector procura uma assinatura visual de
cratera:

- centro mais escuro;
- aro circular com resposta forte;
- gradiente/borda distribuido ao redor do circulo.

Dois sinais geram candidatos:

| Sinal | Objetivo |
|---|---|
| filtro casado multi-escala | encontrar padrao centro/aro semelhante a cratera |
| energia de anel nas bordas | reforcar candidatos com contorno circular |

Os maximos locais com resposta alta viram candidatos. A exigencia minima muda
com o parametro `strictness`.

### 4. Validacao E Refinamento

Cada candidato passa por uma busca local de centro e raio. O detector calcula
um score baseado em:

- contraste entre centro, aro e exterior;
- magnitude de gradiente no aro;
- quantidade de borda no circulo;
- suporte angular do contorno;
- penalizacao para candidatos incompletos ou mal posicionados.

Depois disso, candidatos fracos sao rejeitados. Os sobreviventes passam por
deduplicacao, mantendo a cratera com melhor score quando duas deteccoes se
sobrepoem.

Configuracao padrao:

| Parametro | Valor | Efeito |
|---|---:|---|
| `min_radius` | `4` px | Menor cratera aceita |
| `max_radius` | `70` px | Maior cratera aceita |
| `canny_threshold` | `45` | Controla bordas usadas na validacao |
| `strictness` | `16` | Controla agressividade: maior reduz falsos positivos, mas pode perder crateras |

## Score De Risco De Pouso

Depois da deteccao, a imagem e dividida em uma grade. Cada celula recebe um
score de risco de `0` a `100`, calculado com pesos fixos e interpretaveis.

Configuracao padrao:

| Componente | Peso | Saturacao |
|---|---:|---:|
| densidade de crateras | `30` | `120` crateras/km2 |
| diametro medio | `20` | `80` m |
| maior diametro | `35` | `140` m |
| cobertura por crateras | `15` | `8%` |

A melhor regiao e a celula com menor risco. Dentro dela, o ponto de pouso e
sugerido por transformada de distancia: crateras sao expandidas por um fator de
seguranca, bordas da celula sao evitadas, e o ponto mais distante dos obstaculos
e escolhido.

Configuracao padrao:

| Parametro | Valor |
|---|---:|
| `grid_rows` | `3` |
| `grid_cols` | `3` |
| `default_scale_m_per_px` | `5.0` |
| `safety_factor` do ponto de pouso | `1.25` |
| `border_padding_px` | `12` |

## Comparacao Com Anotacoes YOLO

Quando existe ground truth, cada anotacao YOLO e convertida em circulo:

```text
cx = x_centro_normalizado * largura
cy = y_centro_normalizado * altura
raio = media(largura_box_px, altura_box_px) / 2
```

Uma deteccao conta como true positive quando:

| Criterio | Valor padrao |
|---|---:|
| erro do centro | `<= 1.34 * raio_anotado` |
| erro do raio | `<= 1.0 * raio_anotado` |

Esses mesmos criterios sao usados no ACDLR e na CNN, garantindo uma comparacao
justa no benchmark.

Metricas reportadas:

- precision;
- recall;
- F1;
- true positives;
- false positives;
- false negatives;
- erro medio de centro;
- erro medio de raio.

## Comparador CNN YOLOv11

A CNN nao faz parte do metodo ACDLR. Ela e usada para competir com o metodo
classico no mesmo dataset anotado.

Configuracao atual usada na interface para visualizar ACDLR x CNN:

| Parametro | Valor |
|---|---:|
| pesos | `artifacts/crater_cnn_yolo_train/moon_small/weights/best.pt` |
| `imgsz` | `416` |
| `conf` | `0.001` |
| `iou` | `0.15` |
| `max_det` | `150` |
| device | `cpu` |

O valor baixo de `conf` foi escolhido para o teste pequeno porque o modelo CNN
foi treinado rapidamente. Em uma comparacao final de artigo, a CNN deve ser
treinada por mais epocas e validada no split `valid` completo.

## Como Ajustar O ACDLR

| Objetivo | Ajustes recomendados |
|---|---|
| Reduzir falsos positivos | aumentar `strictness`, aumentar `canny_threshold`, aumentar `min_radius` |
| Recuperar crateras pequenas | reduzir `min_radius`, reduzir levemente `strictness`, reduzir `blur_ksize` |
| Melhorar crateras grandes | aumentar `max_radius`, aumentar `tile_size` se houver contexto suficiente |
| Reduzir duplicatas | aumentar sobreposicao com cuidado e manter deduplicacao ativa |
| Melhorar imagem com pouco contraste | aumentar `clahe_clip` gradualmente |
| Imagem com muito ruido | aumentar `blur_ksize` ou `strictness` |

Para comparacoes publicaveis, mantenha os parametros fixos antes de rodar o
benchmark final. Ajustar parametros olhando diretamente o split de validacao
pode inflar resultados.

## Configuracao Padrao Atual

Resumo de `configs/acdlr_default.yaml`:

```yaml
method: ACDLR
detector:
  tile_size: 1024
  overlap: 96
  clahe_clip: 2.5
  blur_ksize: 5
  min_radius: 4
  max_radius: 70
  canny_threshold: 45
  strictness: 16
risk:
  grid_rows: 3
  grid_cols: 3
  default_scale_m_per_px: 5.0
benchmark_matching:
  center_tolerance: 1.34
  radius_tolerance: 1.0
```

## Saidas Geradas

Na interface, o ACDLR mostra:

- imagem original;
- pre-processamento;
- crateras detectadas;
- mapa de risco;
- resultado final com grade e ponto de pouso;
- comparacao final ACDLR x CNN YOLOv11 quando pesos da CNN existem.

Nos benchmarks, as principais saidas ficam em:

```text
artifacts/acdlr_yolo_benchmark/
artifacts/acdlr_vs_crater_cnn/
```

Os overlays de benchmark agora usam uma faixa superior para legendas e metricas.
Assim, as legendas nao cobrem crateras nem atrapalham a leitura das imagens.
