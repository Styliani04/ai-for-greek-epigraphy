# AI for Greek Epigraphy

An applied data science project exploring ancient Greek inscriptions through natural language processing, machine learning, and computer vision. The work uses the ICDAR 2026 Greek Squeezes dataset, which combines inscription images (squeezes) with text transcriptions in Latin proxy characters.

This is a university team project for the Applied Data Science course at Athens University of Economics and Business (2026).

## Project scope

The notebook develops an end-to-end exploratory pipeline for two complementary sources of evidence: inscription text and inscription images.

| Area | Methods used |
| --- | --- |
| Text preparation | Latin-proxy to Greek transliteration and greedy word segmentation of *scriptio continua* text |
| Text analysis | Vocabulary statistics, word-frequency analysis, and rule-assisted proper-name exploration |
| Text clustering | TF-IDF features, K-Means clustering, frequent-bigram analysis, and LDA topic modeling |
| Dating experiment | Aeneas / Predicting the Past model compared with a general-purpose LLM on selected inscriptions |
| Image preprocessing | Grayscale conversion, CLAHE contrast enhancement, unsharp masking, Sauvola thresholding, and slab masking |
| Image clustering | ViT embeddings, PCA, UMAP visualization, K-Means, and representative images closest to each cluster centroid |
| OCR exploration | TrOCR baseline and the official ICDAR Character Error Rate (CER) evaluation code |

## Repository contents

| File | Description |
| --- | --- |
| `solution.ipynb` | Main notebook containing the implemented analysis and documented discussion in Greek |
| `B1_preprocessing_pipeline.png` | Example image-preprocessing pipeline output |
| `clustering_analysis.png` | Visual analysis of text clusters |
| `KMeans_elbow_curve.png` | Elbow-method plot for text-cluster selection |
| `A5.png` | Supporting visual output for the text-analysis section |
| `ocr_results_backup.json` | Saved OCR-related output data |
| `prompts_3220127_3220215_3220152.md` | Coursework record of AI-assisted work and prompts |
| `team_assignment_greek_squeezes_2026 (1).pdf` | Assignment specification |

## Text-analysis workflow

### Transcription and segmentation

The annotations use Latin proxy characters. The notebook maps these characters to uppercase Greek according to the supplied proxy convention, retaining special symbols when the source is unreadable or ambiguous.

Because transcriptions are written without spaces, the project applies a greedy longest-match word-segmentation approach. A vocabulary is built from an ancient Greek papyri corpus, then the algorithm attempts the longest valid word at each position before falling back to a shorter candidate.

This approach is interpretable but has known limitations: damaged text, unknown names, unusual spellings, and incomplete words can prevent a correct vocabulary match.

### Clustering and topics

The text-clustering section represents inscriptions with TF-IDF vectors and applies K-Means with nine clusters. The choice is supported by an elbow curve and contextual knowledge of common inscription types. It also applies LDA for topic exploration and uses frequently occurring bigrams to identify repeated formulae.

The resulting clusters are exploratory groups, not verified historical categories. In particular, text length and missing-character markers can materially affect similarity.

## Image-analysis workflow

Inscription images are enhanced before visual analysis through contrast adjustment, sharpening, adaptive thresholding, and masking of the stone/squeeze region. The project extracts image embeddings with `google/vit-base-patch16-224`, reduces them using PCA and UMAP, and clusters the two-dimensional representation with K-Means.

Representative images are selected by distance to each cluster centroid. The notebook also contains an optional LLM-based step to propose concise labels for visual clusters based on the catalogue filenames; these suggestions are interpretive and should be reviewed by an epigraphist.

## OCR and evaluation

The notebook includes a TrOCR baseline for text recognition and the official ICDAR competition CER evaluation code. CER is calculated on Latin proxy characters, matching the competition evaluation format.

The published notebook should be described as an exploratory project. Some notebook sections retain assignment-template `TODO` markers or require the full dataset, models, and runtime setup to reproduce end-to-end results.

## Requirements

The notebook was designed for Google Colab and expects a CUDA-enabled runtime for the larger vision and language models where available. It also relies on files not included in this repository, including the Greek Squeezes annotations and images.

Core Python packages used by the notebook include:

```shell
pip install pandas numpy matplotlib seaborn scikit-learn datasets \
  opencv-python pillow torch torchvision transformers umap-learn \
  textdistance jax
```

Additional tools are needed for particular sections:

- `predictingthepast` / Aeneas for the dating experiment.
- Google Colab or an equivalent environment with access to the dataset archive.
- The ICDAR Greek Squeezes dataset and its annotation/image ZIP files.

Package versions are not pinned, so a fresh environment may need dependency adjustments.

## Dataset and sources

The project is based on the ICDAR 2026 Competition in Text Recognition on Greek Squeezes dataset. Refer to the assignment PDF and the official competition page for access conditions, data attribution, and licensing.

The notebook cites the following tools and sources:

- ICDAR 2026 Competition in Text Recognition on Greek Squeezes.
- Aeneas / Predicting the Past by Google DeepMind.
- Microsoft TrOCR.
- Vision Transformer (ViT).
- Classical Language Toolkit (CLTK).

## Setup outline

1. Obtain the annotation and image ZIP archives from the official dataset source.
2. Update the first notebook cells with your own local or Google Drive paths.
3. Run the data-pairing cells to match annotation IDs to image IDs.
4. Execute the required text- or image-analysis sections in order.
5. Download model weights only from their official sources, and record the versions used for reproducibility.

The notebook uses paths such as `/content/Annotations/`, `/content/Images1/`, and `/content/Images4/`; these are Google Colab paths and must be changed for a local environment.

## API-key safety

The original notebook contained a Gemini API key in a code cell used for cluster-topic naming. Do **not** commit real API keys to this repository.

Use an environment variable instead:

```python
import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
```

Before running the cell, set `GEMINI_API_KEY` in the notebook environment or secret manager. If the original key has ever been published publicly, rotate or revoke it in Google AI Studio / Google Cloud.

## Team and contribution

This repository documents a team coursework project. Contributions should be attributed precisely in the notebook or this README before publishing, rather than implying that the full implementation was produced by one person.

## Limitations

- Ancient Greek inscriptions may be fragmentary, damaged, or ambiguous.
- Word segmentation and name recognition are heuristic and depend on vocabulary coverage.
- Clusters and LLM-generated labels support exploration; they are not historical ground truth.
- OCR, dating, and image-clustering quality should be assessed using held-out data and domain-expert review.
- The dataset, model checkpoints, and compute environment are not bundled with this repository.
