[![GitHub](https://img.shields.io/github/license/Marsrocky/Awesome-WiFi-CSI-Sensing?color=blue)](https://github.com/Marsrocky/Awesome-WiFi-CSI-Sensing/blob/main/LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/Marsrocky/Awesome-WiFi-CSI-Sensing/graphs/commit-activity)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)
[![DOI](https://zenodo.org/badge/511110383.svg)](https://zenodo.org/badge/latestdoi/511110383)
# Enhanced Wi-Fi Sensing: Leveraging Phase and Amplitude of CSI for Superior Accuracy
## Introduction
PA-CSI is the library for WiFi CSI HAR that leverages both amplitude and phase features from Wi-Fi signals, incorporating attention mechanisms across both temporal and channel dimensions, along with multi-scale convolutional neural networks (CNNs). It is implemented by PyTorch and Tensorflow. Our paper [*Enhanced Wi-Fi Sensing: Leveraging Phase and Amplitude of CSI for Superior Accuracy*](10.20944/preprints202412.2585.v1) that under review process. 

```
@article{yang2023benchmark,
  title={Enhanced Wi-Fi Sensing: Leveraging Phase and Amplitude of CSI for Superior Accuracy},
  author={Thai Duy Quy, Chih-Yang Lin, Timothy K. Shih},
  year={2025}
}
```

## Requirements

1. Install `pytorch` and `torchvision` (we use `pytorch==1.12.0` and `torchvision==0.13.0`).
2. `pip install -r requirements.txt`

**Note that the project runs perfectly in Linux OS (`Ubuntu`). If you plan to use `Windows` to run the codes, you need to modify the all the `/` to `\\` in the code regarding the dataset directory for the CSI data loading.**

## Run
### Download Processed Data
Please download and organize the [processed datasets](https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt?usp=sharing) in this structure:

