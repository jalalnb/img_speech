# EEG-Based Speech Decoding: Spoken and Imagined Persian Speech

This repository contains the official code and resources for the M.Sc. thesis,  
**"Analysis of Spatio-Temporal Patterns of EEG Data for Spoken and Imagined Speech on a Set of Semantically Related Persian Words"**.

The project focuses on decoding spoken and imagined speech from non-invasive EEG signals, presenting a novel dataset and a deep learning framework for both classification and speech synthesis.

---

## Abstract

Brain-Computer Interfaces (BCIs) that decode imagined speech from brain signals offer new hope for restoring communication for individuals with severe speech disorders. This field, however, faces challenges like the low signal-to-noise ratio in EEG and a lack of rich, public datasets, especially for languages like Persian. This project introduces two key contributions:

- **The Persian Imagined Speech Dataset (PISD):**  
  A new corpus of EEG signals from 20 participants imagining and speaking nine distinct Persian words. The vocabulary is designed to cover complex linguistic relationships (antonymy, synonymy, homophony), providing a challenging benchmark.

- **Spatio-Spectral-Temporal Former (S2T-Former):**  
  An integrated deep learning framework for both classifying EEG signals and synthesizing speech from them. The architecture uses a CNN backbone for local feature extraction and a Transformer encoder for long-range temporal modeling. For speech synthesis, the encoder is paired with a conditional diffusion model-based decoder.

The proposed model achieves high accuracy in subject-dependent classification tasks and demonstrates superior performance in speech synthesis compared to GAN-based baselines.

---

## Key Features

- **Novel Persian Dataset:** Introduction of the Persian Imagined Speech Dataset (PISD), a new resource for EEG-based speech research.

- **Hybrid Deep Learning Model:** The S2T-Former, a hybrid CNN-Transformer architecture designed to capture complex patterns in EEG data.

- **Dual-Task Framework:** Capable of performing both multi-class classification (identifying imagined words) and generative synthesis (reconstructing audio from EEG).

- **High-Fidelity Speech Synthesis:** Utilizes a conditional diffusion model to generate high-quality speech, outperforming traditional generative approaches.

- **Self-Supervised Learning:** Employs a masked signal modeling strategy for pre-training, enabling efficient learning from limited labeled data.

---

## Project Structure


---

## Getting Started

### Prerequisites

- Python 3.9 or higher  
- PyTorch  
- MNE-Python for EEG data processing  
- Scikit-learn, NumPy, Pandas  

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/eeg-persian-speech.git
cd eeg-persian-speech


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Winsdows use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt
```


License

This project is licensed under the MIT License - see the LICENSE
 file for details.


Acknowledgments

This research was conducted at the Department of Computer Engineering, Sharif University of Technology.

Special thanks to supervisors Dr. Hossein Sameti and Dr. Hamid Karbalayi Aghajan for their guidance.