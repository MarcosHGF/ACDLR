# AI Baseline Visual: Ellipse R-CNN

O comparador de IA justo para o dataset visual do ACDLR passa a ser
**Ellipse R-CNN** com o peso **`wdoppenberg/crater-rcnn`**.

## Por Que Este Baseline

| Criterio | Ellipse R-CNN |
|---|---|
| Entrada | imagens visuais lunares/camera-style |
| Saida | elipses de crateras `[a, b, cx, cy, theta]` |
| Peso pre-treinado | `wdoppenberg/crater-rcnn` no Hugging Face |
| Codigo aberto | `https://github.com/wdoppenberg/ellipse-rcnn` |
| Encaixe com ACDLR | elipses viram circulos com `radius=(a+b)/2` |
| Comparacao justa | mesmo split visual YOLO, mesmas labels, mesmas metricas |

Referencias:

- Repositorio: https://github.com/wdoppenberg/ellipse-rcnn
- Peso Hugging Face: https://huggingface.co/wdoppenberg/crater-rcnn
- Paper base do modelo: https://arxiv.org/abs/2001.11584

## Por Que Nao AINavi Como Padrao

AINavi/DetectorCraters e interessante porque fornece pesos MMDetection para
arquiteturas como VFNet, Cascade Mask R-CNN, FCOS e RetinaNet. O problema
pratico e que o pacote publicado no Zenodo tem cerca de **10.4 GB** e exige um
ambiente MMDetection/MMCV mais pesado e sensivel a versoes.

Para um repositorio de artigo que precisa rodar em teste pequeno e gerar
comparacao lado a lado, Ellipse R-CNN e mais adequado:

- peso menor;
- API direta em PyTorch;
- saida geometrica de crateras;
- roda em imagem visual, nao DEM;
- encaixa no nosso avaliador sem converter dominio.

AINavi pode ficar como baseline futuro para uma secao extra de comparacao
pesada.

## Como Rodar

Preparar repositorio, dependencias e peso:

```bash
python scripts/setup_ellipse_rcnn_pretrained.py
```

Rodar a comparacao justa:

```bash
python scripts/run_acdlr_vs_ellipse_rcnn_comparison.py --max-images 5 --visual-count 3
```

Saidas:

```text
artifacts/acdlr_vs_ellipse_rcnn/comparison_report.md
artifacts/acdlr_vs_ellipse_rcnn/visual_comparison.png
artifacts/acdlr_vs_ellipse_rcnn/charts/acdlr_vs_ellipse_rcnn_metrics.png
artifacts/acdlr_vs_ellipse_rcnn/acdlr/acdlr_yolo_summary.json
artifacts/acdlr_vs_ellipse_rcnn/ellipse_rcnn/ellipse_rcnn_yolo_summary.json
```

## Observacao Sobre Download

O Hugging Face hospeda `model.safetensors` via infraestrutura Xet. Se a rede
local nao resolver `cas-bridge.xethub.hf.co`, o download automatico pode falhar.
Nesse caso, baixe manualmente:

```text
https://huggingface.co/wdoppenberg/crater-rcnn/tree/main
```

e coloque:

```text
artifacts/ellipse_rcnn_pretrained/crater-rcnn/model.safetensors
```

## O Que Pode Ser Afirmado

Correto:

> ACDLR e Ellipse R-CNN foram avaliados no mesmo dataset visual anotado, usando
> a mesma conversao de labels e as mesmas metricas.

Incorreto:

> A comparacao usa uma CNN treinada localmente no nosso dataset.

Neste protocolo, Ellipse R-CNN e executado como baseline externo pre-treinado,
sem fine-tuning local.
