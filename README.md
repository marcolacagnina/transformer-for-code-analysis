# PyTorch Transformer for code complexity classification

This project is a PyTorch Transformer classifier, with the aim to classify the time complexity of a Python code. 
In particular, the model classifies within 7 classes:
- O(1)
- O(logn)
- O(n)
- O(n^2)
- O(n*m)
- O(n+m)
- O(n*logn)

## Dataset

The model is trained on `final_dataset.json`, a pre-processed dataset included in the `data/` directory.

The original data was sourced from the [**BigOBench**](https://huggingface.co/datasets/facebook/BigOBench) dataset and has been cleaned, normalized, and filtered to prepare it for the complexity classification task. 
The pre-processing steps are not included in this repository to maintain focus on the modeling and training pipeline.

## Prerequisites

To run this project, you will need the following installed on your system:

* **Python 3.9+**
* **pip** (Python package installer)
* **Git LFS** (Large File Storage) to handle the dataset file.

An hardware accelerator is highly recommended for training the model in a reasonable amount of time:
* NVIDIA GPU with **CUDA** support.
* The scripts will fall back to **CPU** if no GPU is available, but training will be significantly slower.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/marcolacagnina/transformer-for-code-analysis.git
    cd transformer-for-code-analysis/
    ```

2.  **Download the dataset:**
    After cloning, you need to pull the LFS files to download the actual dataset:
    ```bash
    git lfs pull
    ```

3.  **Create a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the model, run the main training script:
```bash
python train.py
```

## Output
Running the training script via `python train.py` will generate the following files inside the `outputs/` directory:

* **`best_model.pt`**: This file contains the state dictionary of the best performing model, saved based on the highest validation accuracy achieved during training. You can load this file to make predictions or continue training.

* **`training_progress.png`**: This is an image file that visualizes the training process, plotting the training loss and validation accuracy for each epoch.

### Training Performance

![Training Progress](outputs/training_progress.png)

### Citation

```bibtex
@misc{chambon2025bigobenchllmsgenerate,
      title={BigO(Bench) -- Can LLMs Generate Code with Controlled Time and Space Complexity?}, 
      author={Pierre Chambon and Baptiste Roziere and Benoit Sagot and Gabriel Synnaeve},
      year={2025},
      eprint={2503.15242},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.15242}, 
}
