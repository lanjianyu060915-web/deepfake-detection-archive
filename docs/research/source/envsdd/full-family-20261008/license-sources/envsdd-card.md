---
license: cc-by-4.0
task_categories:
  - audio-classification
language:
  - en
tags:
  - audio
  - deepfake-detection
  - environmental-sound
  - sound-event-detection
  - anti-spoofing
  - interspeech2025
  - esdd
size_categories:
  - 100K<n<1M
dataset_info:
  features:
    - name: audio
      dtype: audio
    - name: label
      dtype: string
    - name: attack_type
      dtype: string
    - name: generative_model
      dtype: string
    - name: dataset
      dtype: string
    - name: filename
      dtype: string
    - name: scene
      dtype: string
    - name: event
      dtype: string
    - name: caption
      dtype: string
  splits:
    - name: train
      num_bytes: 25899407707
      num_examples: 139055
    - name: validation
      num_bytes: 7397011712
      num_examples: 39710
    - name: test
      num_examples: 39768
    - name: remain
      num_examples: 107259
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: validation
        path: data/validation-*
      - split: test
        path: data/test-*
      - split: remain
        path: data/remain-*
---

# EnvSDD: Environmental Sound Deepfake Detection

<p align="center">
  <a href="https://arxiv.org/abs/2505.19203"><img src="https://img.shields.io/badge/arXiv-2505.19203-b31b1b.svg" alt="arXiv"/></a>
  <a href="https://github.com/apple-yinhan/EnvSDD"><img src="https://img.shields.io/badge/GitHub-Code-black?logo=github" alt="GitHub"/></a>
  <a href="https://envsdd.github.io/"><img src="https://img.shields.io/badge/Demo-Page-blue" alt="Demo Page"/></a>
  <a href="https://sites.google.com/view/esdd-challenge"><img src="https://img.shields.io/badge/ICASSP_2026-ESDD_Challenge-orange" alt="Challenge"/></a>
  <img src="https://img.shields.io/badge/Interspeech_2025-Accepted-brightgreen" alt="Interspeech 2025"/>
  <img src="https://img.shields.io/badge/License-CC_BY_4.0-green" alt="License"/>
</p>

> **Accepted at Interspeech 2025** 🎉 &nbsp;|&nbsp; The first [Environmental Sound Deepfake Detection (ESDD) Challenge](https://sites.google.com/view/esdd-challenge) is launching with **ICASSP 2026**!

## Overview

**EnvSDD** is the **first large-scale curated dataset** for Environmental Sound Deepfake Detection (ESDD). Audio generation models can now produce extremely realistic soundscapes — creating new risks for misinformation and public safety. While speech and singing deepfake detection have received widespread research attention, environmental sound deepfakes remain a largely open problem.

EnvSDD bridges this gap with:
- **45.25 hours** of real environmental audio and **316.74 hours** of AI-generated (deepfake) audio
- Both **monophonic** and **polyphonic** conditions
- **5 text-to-audio (TTA)** models and **2 audio-to-audio (ATA)** models
- **4 test conditions** spanning in-domain and challenging out-of-domain scenarios

All audio is at **16 kHz, 4 seconds per clip**.

![Dataset Pipeline](https://envsdd.github.io/static/images/overview_1.png)
*Pipeline for EnvSDD: real clips from 6 source datasets → captions via LLM → deepfakes via TTA and ATA models.*

---

## Quickstart

```python
from datasets import load_dataset

ds_train  = load_dataset("AustinXiao/EnvSDD", split="train")       # 139,055 examples
ds_val    = load_dataset("AustinXiao/EnvSDD", split="validation")  #  39,710 examples
ds_test   = load_dataset("AustinXiao/EnvSDD", split="test")        #  39,768 examples
ds_remain = load_dataset("AustinXiao/EnvSDD", split="remain")      # 107,259 examples

# Inspect a sample
sample = ds_train[0]
print(sample["label"])            # "real" or "fake"
print(sample["attack_type"])      # "TTA", "ATA", or "-" (for real clips)
print(sample["generative_model"]) # e.g. "AudioLDM", "TangoFlux", or "-"
print(sample["dataset"])          # source dataset name, e.g. "UrbanSound8K"
print(sample["caption"])          # text caption used for TTA generation
print(sample["scene"])            # acoustic scene label (polyphonic clips)
print(sample["event"])            # sound event label (monophonic clips)

# Play audio in a notebook
import IPython.display as ipd
audio = sample["audio"]
ipd.Audio(audio["array"], rate=audio["sampling_rate"])
```

---

## Splits

| Split | # Examples | Source | Zenodo |
|---|---|---|---|
| `train` | 139,055 | EnvSDD-Development | [10.5281/zenodo.15220951](https://zenodo.org/records/15220951) |
| `validation` | 39,710 | EnvSDD-Development | [10.5281/zenodo.15220951](https://zenodo.org/records/15220951) |
| `test` | 39,768 | EnvSDD-Test | [10.5281/zenodo.15241138](https://zenodo.org/records/15241138) |
| `remain` | 107,259 | EnvSDD-Remain | [10.5281/zenodo.15239720](https://zenodo.org/records/15239720) |

The **test** split covers 4 evaluation conditions (in-domain, unseen models, unseen datasets, and hardest out-of-domain). The **remain** split contains additional data not used in the original paper's evaluation.

---

## Data Fields

| Field | Type | Description |
|---|---|---|
| `audio` | `Audio` | 16 kHz WAV clip (4 seconds) |
| `label` | `string` | `"real"` or `"fake"` |
| `attack_type` | `string` | `"TTA"` (text-to-audio), `"ATA"` (audio-to-audio), or `"-"` for real clips |
| `generative_model` | `string` | Generation model name (e.g. `"AudioLDM"`, `"TangoFlux"`), or `"-"` for real |
| `dataset` | `string` | Source dataset the real clip was drawn from |
| `filename` | `string` | Original filename of the source audio clip |
| `scene` | `string` | Acoustic scene label (e.g. `"airport"`, `"office"`); polyphonic clips |
| `event` | `string` | Sound event label (e.g. `"siren"`, `"gun_shot"`); monophonic clips |
| `caption` | `string` | Text caption used as input to TTA models |

---

## Dataset Statistics

### Real Data Sources

| Dataset | Audio Type | # Real Clips |
|---|---|---|
| UrbanSound8K | Monophonic | 4,523 |
| DCASE 2023 Task7 Dev | Monophonic | 500 |
| TAU UAS 2019 Open Dev | Polyphonic | 31,700 |
| TUT SED 2016 | Polyphonic | 1,695 |
| TUT SED 2017 | Polyphonic | 1,806 |
| Clotho | Polyphonic | 500 |

### Generation Models

| Model | Deepfake Type |
|---|---|
| AudioLDM | TTA & ATA |
| AudioLDM 2 | TTA & ATA |
| AudioGen | TTA |
| TangoFlux | TTA |
| AudioLCM | TTA |

---

## Filtering Examples

```python
from datasets import load_dataset

ds = load_dataset("AustinXiao/EnvSDD")

# Real audio only
real = ds["train"].filter(lambda x: x["label"] == "real")

# TTA fakes only
tta = ds["train"].filter(lambda x: x["attack_type"] == "TTA")

# Specific model
tangoflux = ds["train"].filter(lambda x: x["generative_model"] == "TangoFlux")

# Specific source dataset
urbansound = ds["train"].filter(lambda x: x["dataset"] == "UrbanSound8K")
```

---

## Citation

If you use EnvSDD in your research, please cite:

```bibtex
@inproceedings{yin2025envsdd,
  title     = {{EnvSDD}: Benchmarking Environmental Sound Deepfake Detection},
  author    = {Yin, Han and Xiao, Yang and Das, Rohan Kumar and Bai, Jisheng and Liu, Haohe and Wang, Wenwu and Plumbley, Mark D},
  booktitle = {Proc. Interspeech 2025},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.19203}
}
```

---

## ESDD Challenge 2026

Based on EnvSDD, the **first Environmental Sound Deepfake Detection Challenge** is held with **ICASSP 2026**, focusing on robust detection that generalizes to unseen sound generators.

👉 [Challenge website & registration](https://sites.google.com/view/esdd-challenge)

---

## License

Released under **CC BY 4.0**. Please also respect the individual licenses of the original source datasets (UrbanSound8K, DCASE 2023, TAU UAS 2019, TUT SED 2016/2017, Clotho) when using this data.

---

## Authors

Han Yin¹², Yang Xiao³⁴, Rohan Kumar Das⁴, Jisheng Bai¹, Haohe Liu⁵, Wenwu Wang⁵, Mark D Plumbley⁵

¹Northwestern Polytechnical University &nbsp;·&nbsp; ²Alibaba Group &nbsp;·&nbsp; ³University of Melbourne &nbsp;·&nbsp; ⁴Fortemedia Singapore &nbsp;·&nbsp; ⁵University of Surrey (CVSSP)

Contact: `hanyin@kaist.ac.kr`, `yang.xiao.1@student.unimelb.edu.au`

---

## Links

| | |
|---|---|
| 📄 Paper | [arXiv 2505.19203](https://arxiv.org/abs/2505.19203) |
| 💻 Code | [github.com/apple-yinhan/EnvSDD](https://github.com/apple-yinhan/EnvSDD) |
| 🔊 Demo | [envsdd.github.io](https://envsdd.github.io/) |
| 🏆 ESDD Challenge | [sites.google.com/view/esdd-challenge](https://sites.google.com/view/esdd-challenge) |
