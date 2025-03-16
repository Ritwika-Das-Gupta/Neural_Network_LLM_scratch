# NanoGPT

NanoGPT is a lightweight, decoder-only GPT-style transformer model implemented in PyTorch for character-level language modeling using the Tiny Shakespeare dataset. It features multi-head self-attention, feed-forward layers, layer normalization, and positional embeddings to generate coherent text based on learned patterns.

## Model Architecture
The model consists of the following key components:
- **Embedding Layer**: Converts characters into dense vector representations.
- **Multi-Head Self-Attention**: Enables contextual token interactions within the input sequence.
- **Feed-Forward Layers**: Applies transformations to token representations for improved feature extraction.
- **Layer Normalization**: Stabilizes training and accelerates convergence.
- **Output Layer**: Predicts the next token in the sequence.

### Model Parameters
The model has a total of approximately **10 million** trainable parameters, making it efficient for lightweight training while maintaining strong text generation capabilities.

![Model Diagram](nanogpt_diagram.png) *(A conceptual illustration of the NanoGPT architecture.)*

## Features
- **Decoder-only Transformer**: Optimized for autoregressive text generation.
- **Character-level Language Modeling**: Trained on Tiny Shakespeare dataset.
- **Token and Positional Embeddings**: Helps retain word order and contextual understanding.
- **Efficient PyTorch Implementation**: Enables easy experimentation and modification.
- **Training Stability**: Layer normalization improves convergence speed and robustness.

## Installation
Ensure you have Python and PyTorch installed. You can install PyTorch via:
```sh
pip install torch
```

## Usage
### Training
The training logic is embedded within. To train the model, simply run:
```sh
python gpt.py
```
The script will read `input.txt`, train the model, and generate text.

### Generating Text
After training, you can generate text using:
```sh
python gpt.py
```
Generated output will be saved in `output.txt`.

## Hyperparameters
| Parameter        | Value |
|-----------------|-------|
| Batch Size      | 64    |
| Block Size      | 256   |
| Max Iterations  | 5000  |
| Learning Rate   | 3e-4  |
| Embedding Dim   | 384   |
| Attention Heads | 6     |
| Layers         | 6     |
| Dropout        | 0.2   |

## Notes
- The model is trained on Tiny Shakespeare text.
- Training progress is printed every 500 iterations.
- The trained model can generate new Shakespeare-like text.

## Future Improvements
- Save and load trained models.
- Implement `train.py` and `generate.py` for modularity.
- Optimize performance with mixed precision training.
