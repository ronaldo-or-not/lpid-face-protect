# Data & Ethics

## Purpose
LPID is a **privacy-protection (defensive) tool**. Its purpose is to make it hard for a user's publicly posted face photos to be scraped without consent and used to train face-recognition models. It is not meant for identifying or tracking other people, or for poisoning anyone else's data.

## What is not included
- **No face datasets.** The datasets used in the paper's experiments (CASIA-WebFace, IMDB-Face, Places365, etc.) follow their respective distribution licenses and are not redistributed here. Obtain them from their original sources.
- **No model checkpoints or raw experiment logs.** (See "Reproducibility scope" below.)

## Reproducibility scope (important)
This repository provides the **protection tool (`protect.py`) and a description of the method**. It does not include the full paper benchmark pipeline (surrogate training → attacker from-scratch training → evaluation) or the raw results. The numbers reported in the paper were produced in the author's training environment (Colab/GPU); Table 1 and Figure 4 are single runs, and Table 3 is averaged over three attacker-training runs. All superiority claims are limited to **"among the compared methods."**

## Responsible use
- Use only on photos of yourself, or of people who have given explicit consent.
- Use outputs as lossless PNG (re-saving as JPEG weakens the protection — a known limitation).
