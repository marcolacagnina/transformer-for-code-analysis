# PyTorch Transformer for code complexity classification

This project is a PyTorch Transformer classifier, with the aim to classify the time complexity of a Python code. 
In particular, the model classify within 7 classes:
- O(1)
- O(logn)
- O(n)
- O(n^2)
- O(n*m)
- O(n+m)
- O(n*logn)

## Dataset

This project utilizes the [**BigOBench**](https://huggingface.co/datasets/facebook/BigOBench) dataset, available on Hugging Face.

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
