# Metodologia Do ACDLR

O ACDLR e o metodo principal do projeto. Ele nao usa IA, nao usa CNN e nao faz
treinamento. Toda a deteccao vem de processamento classico de imagem.

## Fluxo Geral

```text
Imagem lunar
  -> tiling
  -> pre-processamento
  -> geracao de candidatos circulares
  -> validacao visual/geometrica
  -> refinamento local
  -> deduplicacao
  -> medicao das crateras
  -> analise de risco por grade
  -> visualizacao final
```

## 1. Entrada E Tiling

O sistema recebe uma imagem BGR lida pelo OpenCV. Para imagens grandes, ela e
dividida em tiles com sobreposicao. Isso melhora estabilidade e evita perder
crateras perto das bordas.

Parametros principais:

- `tile_size`: tamanho do tile;
- `overlap`: sobreposicao entre tiles.

No final, as deteccoes locais sao convertidas para coordenadas globais e
deduplicadas.

## 2. Pre-Processamento

O modulo `core/preprocessing.py` prepara a imagem:

- converte para tons de cinza;
- aplica CLAHE para realce local de contraste;
- suaviza ruido;
- gera imagem realcada;
- calcula pistas de borda.

O objetivo e aumentar a diferenca entre aro, sombra e interior das crateras sem
usar aprendizado de maquina.

## 3. Geracao De Candidatos

O detector em `core/detection.py` cria candidatos por escala. Para varios raios
possiveis, ele calcula respostas de:

- filtro casado com assinatura de cratera;
- energia anular sobre magnitude de gradiente;
- maximos locais nas respostas.

Cada candidato possui:

- centro aproximado `(x, y)`;
- raio aproximado `r`;
- intensidade de resposta.

## 4. Validacao De Crateras

Cada candidato passa por verificacoes locais. O detector analisa uma janela ao
redor do candidato e mede:

- contraste entre interior, aro e exterior;
- suporte de borda no aro;
- proeminencia de gradiente no aro;
- cobertura angular do aro;
- variacao por setores;
- sinais de regiao sem dados ou borda da imagem.

O parametro `strictness` controla a seletividade:

- menor strictness: mais deteccoes, mais risco de falsos positivos;
- maior strictness: menos deteccoes, maior exigencia de evidencia visual.

## 5. Refinamento Local

Depois de uma primeira pontuacao, o ACDLR faz pequenos ajustes no centro e no
raio. Em alguns casos usa Hough local apenas como refinamento, nao como detector
principal.

O resultado final de cada cratera e:

```text
x, y, raio
```

## 6. Deduplicacao

Como a mesma cratera pode aparecer em escalas ou tiles diferentes, o sistema
remove duplicatas por proximidade de centro e sobreposicao de raio. A deteccao
com melhor score e preservada.

## 7. Medicao

O modulo `core/measurement.py` transforma circulos detectados em medidas:

- raio em pixels;
- diametro em pixels;
- diametro em metros;
- area aproximada.

A conversao fisica depende da escala:

```text
metros = pixels * metros_por_pixel
```

## 8. Score De Risco

O modulo `core/risk.py` divide a imagem em uma grade, por exemplo 3 x 3. Para
cada celula calcula:

- quantidade de crateras;
- densidade por km2;
- diametro medio;
- maior diametro;
- cobertura percentual da celula por crateras.

O score vai de 0 a 100 e usa componentes fixos:

| Componente | Peso maximo | Saturacao |
|---|---:|---:|
| Densidade de crateras | 30 | 120 crateras/km2 |
| Diametro medio | 20 | 80 m |
| Maior diametro | 35 | 140 m |
| Cobertura por crateras | 15 | 8% da celula |

Classificacao:

| Score | Classe |
|---:|---|
| 0 a 32.99 | LOW |
| 33 a 65.99 | MEDIUM |
| 66 a 100 | HIGH |

A melhor regiao de pouso e a celula com menor score.

## 9. Sugestao De Ponto De Pouso

Depois de escolher a melhor celula, o sistema cria uma mascara de seguranca e
expande as crateras por um fator de seguranca. Em seguida usa distance transform
para encontrar o ponto mais distante das crateras dentro da regiao.

Esse ponto e didatico: ajuda na visualizacao, mas nao representa uma solucao
real de navegacao ou engenharia aeroespacial.

## 10. Por Que O Metodo E Explicavel

O ACDLR e explicavel porque cada decisao vem de criterios visuais:

- contraste;
- borda;
- circularidade;
- gradiente;
- escala;
- cobertura angular;
- densidade por regiao.

Isso permite ajustar e justificar o metodo sem depender de pesos aprendidos por
rede neural.
