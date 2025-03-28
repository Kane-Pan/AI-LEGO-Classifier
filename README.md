# LEGO AI Classifier (Work in Progress)

This code was developed as part of my final project for PRAXIS II, an engineering design course for the Engineering Science program at the University of Toronto. My team was tasked with designing a solution to help members of ToroLUG, a local Toronto LEGO community, sort LEGO bricks more efficiently. My contribution to the project focused on building a computer vision-based LEGO brick classifier using deep learning. Trained on over **600,000 images** across **447 official LEGO brick classes**, the model achieves a **99.49% validation accuracy** and can be used in real-time to identify LEGO parts through a camera feed.

> **Note:** This is an active project. Some improvements and new features to be considered include creating an interactive GUI, adding colour recognition, improving LEGO detection, and code to send signals to an Arduino.

---

## Features

- Real-time LEGO brick classification via camera or image input
- Pretrained ResNet18 model using transfer learning
- PyTorch-based training pipeline with checkpointing
- Can Classify 447 different LEGO bricks

---

## Included Files

|File|Description|
|----------------------------|-------------------------------------------------------------------------|
| `train_lego_classifier.py`|PyTorch script to train the model on the LEGO image dataset.|
| `lego-live_classifier.py`|Script for real-time classification using connected camera or image input.|

---

## Dataset

This model was trained using the open-access dataset:

**[LEGO Bricks for Training Classification Network (v1.1)](https://mostwiedzy.pl/en/open-research-data/lego-bricks-for-training-classification-network,202309140842198941751-0)**  
By Gdańsk University of Technology  
Dataset DOI: [https://doi.org/10.34808/rcza-jy08](https://doi.org/10.34808/rcza-jy08)
- 447 official LEGO part numbers  
- More than 600,000 total images (90% renders, 10% real photos)  
- White background, randomized lighting and orientation

---

## Download Pretrained Model

[Download AI_Lego_Classifier_Model.pth](https://github.com/Kane-Pan/AI-Lego-Classifier/releases)  
*(Uploaded in the latest release as a `.pth` checkpoint)*

This model checkpoint includes:
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `last_epoch`
- `best_acc`
- `best_model_wts`
  
---
