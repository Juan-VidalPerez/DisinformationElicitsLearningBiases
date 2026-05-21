# Disinformation Elicits Learning Biases: Code and Data Repository

## Overview
This repository contains the behavioral data and MATLAB computational modeling pipeline for the paper **"Disinformation elicits learning biases"**. 

You can read the reviewed preprint here: [https://elifesciences.org/reviewed-preprints/106073#s2](https://elifesciences.org/reviewed-preprints/106073#s2)

The study investigates how feedback reliability and disinformation shape reward learning. To maintain clear organization between the different experimental designs used in the paper, the repository is split into two main directories.

## Repository Structure

### 1. `DiscoveryStudy/`
This folder contains the codebase and datasets for the initial discovery (pilot) study.
* **Design:** Strictly blocked trials (1 bandit pair per block).
* **Agents:** Features 4 agents with objective credibilities of 0.5, 0.7, 0.85, and 1.0.
* **Contents:** The pipeline used to generate the SI section 1 figures (Figures S2, S3, S4).

### 2. `MainStudy/`
This folder contains the codebase and datasets for the primary experiment.
* **Design:** Interleaved trials (3 concurrent bandit pairs intermixed per block).
* **Agents:** Features 3 agents with objective credibilities of 0.5, 0.75, and 1.0.
* **Contents:** The pipeline used to generate the main manuscript figures (Figures 3, 4, 5, 6).

---
*For specific instructions on how to reproduce the figures and statistical tests for each study, please refer to the dedicated `README.md` file located inside each respective subfolder.*# DisinformationElicitsLearningBiases
