# Enhanced Wi-Fi Sensing: Leveraging Phase and Amplitude of CSI for Superior Accuracy
## Introduction
PA-CSI is the library for WiFi CSI HAR that leverages both amplitude and phase features from Wi-Fi signals, incorporating attention mechanisms across both temporal and channel dimensions, along with multi-scale convolutional neural networks (CNNs). It is implemented by PyTorch and Tensorflow. This is our paper [*Enhanced Human Activity Recognition Using Wi-Fi Sensing: Leveraging Phase and Amplitude with Attention Mechanisms*](10.3390/s25041038). 

```
@article{yang2023benchmark,
  title={Enhanced Wi-Fi Sensing: Leveraging Phase and Amplitude of CSI for Superior Accuracy},
  author={Thai Duy Quy, Chih-Yang Lin, Timothy K. Shih},
  year={2025}
}
```

## Requirements

1. Install `pytorch` and CUDA (we use `torch=2.5.1` and `cu124`).
2. Install `tensorflow` and CUDA (we use `tensorflow=2.10.0` and `cu124`).
3. `pip install -r requirements.txt`

**Note that the project runs perfectly in Windows OS (`Windows`). If you plan to use `Linux` or `Ubuntu` to run the codes, you need to modify some symbols in the code regarding the dataset directory for the CSI data loading.**

## Run
### Download Processed Data
Data original can be downloaded at: 
1. [MultiEnv](https://github.com/lcsig/Dataset-for-Wi-Fi-based-human-activity-recognition-in-LOS-and-NLOS-indoor-environments): This dataset was collected in three scenarios: line-of-sight (LOS) in both the office and hall, and non-line-of-sight (NLOS).
2. [StanWiFi](https://github.com/ermongroup/Wifi_Activity_Recognition): This dataset contains continuous CSI data for six activities without precise segmentation timestamps for each sample.

Try to run dataset.py to process the data from original dataset.

Data processed can be downloaded at: [PA-CSI dataset](https://drive.google.com/drive/folders/1fiJBDWDC3WKkD5pLYRwpLCWMXPVzCQ-i?usp=sharing)



