# Multimodal Deception Detector

## I am exploring physio-linguistic processes to understand which are more likely to occur when people are lying compared to when they are telling the truth.

This repo contains my discovery process as I attempt to disect signals of deception in human behavior including multimodal approaches that integrate spoken language (sentiment and tone analysis), physiological features (e.g., heart rate, respiration rate, galvanic skin response, skin temperature) extracted from video as well as gestures and facial expressions.

## Local Development

```
bash
cd scripts
./pipenv_setup.sh
```

## Things to Explore

1. What is deception detection and why is it important?
2. What is meant by multimodal deception detection?
3. Which modalities can be used for deception detection?
4. What are the typical features that can be extracted from each modality to benefit the process of detecting deceit?
5. How can the multimodal features be integrated?
6. What are the advantages of using multimodal features compared to features from a single modality?
7. What is the difference between deception data collected in a lab and real-life deception data?
8. What are the advantages and limitations of processing multimodal lab-setting data and real-life deception data?
9. How can I generate my own deception detection dataset?

## Datasets

[Various software modules and data sets that are/were used in my research. They are made available under the terms of GNU General Public License. Both data and software are distributed without any warranty.](https://web.eecs.umich.edu/~mihalcea/downloads.html)

### I also looked into using:

[Bag of Lies](https://iab-rubric.org/index.php/bag-of-lies)
Bag-of-Lies is a multi-modal dataset consisting of video, audio and eye gaze from 35 unique subjects collected using a carefully designed experiment for the task of automated deception detection (binary classification into truth/lie). It has a total of 325 manually annotated recordings consisting of 162 lies and 163 truths. Along with it, EEG (13 channels) data is also available for 22 unique subjects.

For the experiment, each subject was shown 6-10 select images and were asked to describe them deceptively or otherwise based on their choice. Video and Audio were captured using a standard phone camera and microphone, gaze data was collected using GazePoint GP3 Eye tracking system and EEG Data was captured using EPOC+ headset.

1. V. Gupta, M. Agarwal, M. Arora, T. Chakraborty, R. Singh, M. Vatsa, "Bag-of-Lies: A Multimodal Dataset for Deception Detection," in IEEE Conference on Computer Vision and Pattern Recognition Workshop on Challenges and Opportunities for Privacy and Security, 2019. Available: [Bag of Lies](https://iab-rubric.org/index.php/bag-of-lies)

[DOLOS Dataset](https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning/tree/main)

## Academic Papers

- [Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient Crossmodal Learning](https://arxiv.org/pdf/2303.12745.pdf)
- [State of Research in ML and Deception Detection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9910662/)
- [Multimodal Deception Detection Using Real-Life Trial Data](https://ieeexplore.ieee.org/document/9165161)
- [Audio Deep Fake Detection](https://huggingface.co/blog/Andyrasika/deepfake-detect)
- [Explainable Verbal Deception Detection using Transformers](https://arxiv.org/abs/2210.03080)
- [Building a Better Lie Detector with BERT: The Difference Between Truth and Lies](https://ucinlp.github.io/files/papers/bertdecept-ijcnn20.pdf) -[](https://www.mdpi.com/2076-3417/12/12/5878) -[Detecting Deceptive Utterances Using Deep Pre-Trained Neural Networks](https://www.mdpi.com/2076-3417/12/12/5878)
- [Automated Deception Detection from Videos: Using End-to-End Learning Based High-Level Features and Classification Approaches](https://arxiv.org/abs/2307.06625)

## Single-Modal Models

### VIDEO

- [VideoMAE](https://huggingface.co/docs/transformers/main/en/model_doc/videomae)

### AUDIO

## Crossmodal Models

- [VILBERT](https://paperswithcode.com/method/vilbert)
- [CLIP](https://huggingface.co/docs/transformers/en/model_doc/clip)

## Resources

- [DOLOS GitHub](https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning)
- [liarliar.ai](https://liarliar.ai/)
- [Truthsayer](https://github.com/everythingishacked/Truthsayer)
- [Webcam Pulse Detector](https://github.com/thearn/webcam-pulse-detector)
- [facial emotion recog](https://github.com/victor369basu/facial-emotion-recognition)

## Tutorials

- [Video Classification Hugging Face](https://huggingface.co/docs/transformers/tasks/video_classification#visualize-the-preprocessed-video-for-better-debugging)

## Deployment

## Plan

1. Define Benchmark Model Architecture

   - V1: use VideoMAE to understand visual data.
   - V2: establish benchmark via models that can handle multimodal inputs, such as CLIP and ViLBERT.

2. Preprocess Data

   - Audio: convert speech to text using an Automatic Speech Recognition (ASR) system / extract audio features directly
   - Video: extract frames or features that are relevant to deception detection.

3. Choose a Pre-trained Model

4. Fine-Tuning for Deception Detection

5. Implement Multimodal Fusion
   Integrate the features learned from the audio/text and visual models using early fusion, late fusion, or hybrid methods (determine how the model combines the information from different modalities to make a prediction).

6. Train the Model

7. Evaluate the Model
   Use appropriate metrics for deception detection, such as accuracy, precision, recall, and F1 score.

8. Iterative Improvement
9. Deploy to SageMaker
10. Build / Deploy a frontend
