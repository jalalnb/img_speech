EEG-Based Speech Decoding: Spoken and Imagined Persian Speech
This repository contains the official code and resources for the M.Sc. thesis, "Analysis of Spatio-Temporal Patterns of EEG Data for Spoken and Imagined Speech on a Set of Semantically Related Persian Words". The project focuses on decoding spoken and imagined speech from non-invasive EEG signals, presenting a novel dataset and a deep learning framework for both classification and speech synthesis.

Abstract
Brain-Computer Interfaces (BCIs) that decode imagined speech from brain signals offer new hope for restoring communication for individuals with severe speech disorders. This field, however, faces challenges like the low signal-to-noise ratio in EEG and a lack of rich, public datasets, especially for languages like Persian. This project introduces two key contributions:

The Persian Imagined Speech Dataset (PISD): A new corpus of EEG signals from 20 participants imagining and speaking nine distinct Persian words. The vocabulary is designed to cover complex linguistic relationships (antonymy, synonymy, homophony), providing a challenging benchmark.

Spatio-Spectral-Temporal Former (S2T-Former): An integrated deep learning framework for both classifying EEG signals and synthesizing speech from them. The architecture uses a CNN backbone for local feature extraction and a Transformer encoder for long-range temporal modeling. For speech synthesis, the encoder is paired with a conditional diffusion model-based decoder.

The proposed model achieves high accuracy in subject-dependent classification tasks and demonstrates superior performance in speech synthesis compared to GAN-based baselines.

Key Features
Novel Persian Dataset: Introduction of the Persian Imagined Speech Dataset (PISD), a new resource for EEG-based speech research.

Hybrid Deep Learning Model: The S2T-Former, a hybrid CNN-Transformer architecture designed to capture complex patterns in EEG data.

Dual-Task Framework: Capable of performing both multi-class classification (identifying imagined words) and generative synthesis (reconstructing audio from EEG).

High-Fidelity Speech Synthesis: Utilizes a conditional diffusion model to generate high-quality speech, outperforming traditional generative approaches.

Self-Supervised Learning: Employs a masked signal modeling strategy for pre-training, enabling efficient learning from limited labeled data.

Project Structure
The repository is organized as follows:

.
├── data/
│   ├── raw/          # Raw PISD dataset files
│   └── processed/    # Preprocessed data for training
├── notebooks/
│   └── eda.ipynb     # Jupyter notebook for exploratory data analysis
<!-- ├── results/
│   ├── checkpoints/  # Saved model weights
│   ├── logs/         # Training logs
│   └── outputs/      # Generated audio, figures, etc. -->
<!-- ├── src/
│   ├── models/
│   │   ├── s2t_former.py      # S2T-Former model definition
│   │   └── diffusion_decoder.py # Diffusion model for synthesis
│   ├── preprocessing/
│   │   └── feature_extraction.py # DWT and other feature engineering
│   ├── train.py          # Main training script
│   ├── evaluate.py       # Evaluation script
│   └── utils.py          # Utility functions -->
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

Getting Started
Prerequisites
Python 3.9 or higher

PyTorch

MNE-Python for EEG data processing

Scikit-learn, NumPy, Pandas

Installation
Clone the repository:

git clone [https://github.com/your-username/eeg-persian-speech.git](https://github.com/your-username/eeg-persian-speech.git)
cd eeg-persian-speech

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Usage
1. Data Preprocessing
First, run the preprocessing script to prepare the raw PISD data.

python src/preprocessing/feature_extraction.py --data_path ./data/raw --output_path ./data/processed

2. Training
Train the model for either the classification or synthesis task.

Classification:

python src/train.py --task classification --data_path ./data/processed --model_save_path ./results/checkpoints --epochs 100 --batch_size 16

Speech Synthesis:

python src/train.py --task synthesis --data_path ./data/processed --model_save_path ./results/checkpoints --epochs 200 --batch_size 8

3. Evaluation
Evaluate a trained model on the test set.

python src/evaluate.py --task classification --model_path ./results/checkpoints/s2t_former_best.pth --data_path ./data/processed

<!-- Citation
If you use this work, please cite the original thesis:

@mastersthesis{Nematbakhsh2025,
  author       = {Nematbakhsh, Mohammad Jalal},
  title        = {Analysis of Spatio-Temporal Patterns of EEG Data for Spoken and Imagined Speech on a Set of Semantically Related Persian Words},
  school       = {Sharif University of Technology},
  year         = {2025},
  month        = {September},
  supervisor   = {Sameti, Hossein and Aghajan, Hamid Karbalayi}
} -->

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
This research was conducted at the Department of Computer Engineering, Sharif University of Technology.

Special thanks to supervisors Dr. Hossein Sameti and Dr. Hamid Karbalayi Aghajan for their guidance.