# Transformer From Scratch – Linux Command Generation

A Transformer Encoder-Decoder implemented with PyTorch to translate natural language instructions into Linux shell commands.

This project is a hands-on exploration of the Transformer architecture introduced in the paper **"Attention Is All You Need" (Vaswani et al., 2017)**.

The architecture was implemented by following **Umar Jamil's "Coding a Transformer from Scratch" tutorial**, then adapted to a custom sequence-to-sequence task: generating Linux commands from natural language.

Unlike approaches based on fine-tuning pretrained language models, this project implements and trains the Transformer architecture from scratch, while using a pretrained GPT-2 tokenizer from Hugging Face.

The main objective is to gain a deeper understanding of how Transformers work internally, from attention mechanisms to model training and evaluation.

---

## 1. Project Overview

The goal is to build a sequence-to-sequence model capable of translating natural language instructions into Linux commands.

### Example

**Input:**

```text
Display the current working directory
```

**Expected output:**

```bash
pwd
```

Another example:

**Input:**

```text
Create a directory named projects
```

**Expected output:**

```bash
mkdir projects
```

The model uses an Encoder-Decoder architecture:

1. The encoder processes the natural language instruction.
2. The decoder generates the corresponding Linux command token by token.
3. Cross-attention enables the decoder to attend to the encoder representations.

The project focuses on implementing, training, and evaluating this architecture rather than relying on an existing pretrained Transformer.

---

## 2. Learning Objectives

The main objectives of this project are:

- Understand the internal architecture of Transformers.
- Implement attention mechanisms using PyTorch.
- Understand the differences between self-attention and cross-attention.
- Work with embeddings and positional encoding.
- Implement causal and padding masks.
- Build custom PyTorch datasets and dataloaders.
- Understand the training process of sequence-to-sequence models.
- Investigate overfitting and generalization.
- Gain practical experience with Hugging Face tokenizers.

The project also serves as an opportunity to improve my understanding of PyTorch and deep learning engineering.

---

## 3. Transformer Architecture

The model follows the original Encoder-Decoder Transformer architecture introduced in:

**Attention Is All You Need – Vaswani et al. (2017)**

The implementation is based on Umar Jamil's Transformer walkthrough.

Rather than using `torch.nn.Transformer`, the main architectural components are implemented directly using PyTorch.

### 3.1 Input Embeddings

Input tokens are converted into dense vector representations using an embedding layer.

The embeddings are scaled by the square root of the model dimension:

\[
X = \text{Embedding}(tokens) \times \sqrt{d_{model}}
\]

This produces the initial representations used by the Transformer.

### 3.2 Positional Encoding

Since Transformers do not process tokens sequentially like recurrent neural networks, positional information must be introduced explicitly.

The implementation uses sinusoidal positional encoding.

For a position `pos` and dimension `i`:

\[
PE(pos, 2i) =
\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

\[
PE(pos, 2i+1) =
\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

These encodings are added to token embeddings to incorporate positional information.

### 3.3 Scaled Dot-Product Attention

The attention mechanism computes relationships between queries, keys, and values.

The attention operation is defined as:

\[
Attention(Q,K,V) =
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Where:

- `Q`: Query matrix.
- `K`: Key matrix.
- `V`: Value matrix.
- `d_k`: Dimension of the key vectors.

The scaling factor helps prevent excessively large attention scores, improving the stability of the softmax operation.

### 3.4 Multi-Head Attention

Instead of performing attention only once, the model uses multiple attention heads.

Each head learns different representations of the input sequence.

\[
MultiHead(Q,K,V) =
Concat(head_1,\ldots,head_h)W^O
\]

Each attention head is computed independently before concatenation.

### 3.5 Encoder

The encoder consists of multiple identical layers.

Each encoder layer contains:

1. Multi-Head Self-Attention.
2. Residual connections and Layer Normalization.
3. Position-wise Feed-Forward Network.
4. Residual connections and Layer Normalization.

The encoder produces contextual representations of the input instruction.

### 3.6 Decoder

The decoder generates the output sequence autoregressively.

Each decoder layer contains:

1. Masked Multi-Head Self-Attention.
2. Encoder-Decoder Cross-Attention.
3. Position-wise Feed-Forward Network.

Residual connections and layer normalization are also applied.

A causal mask prevents the decoder from accessing future tokens during training.

### 3.7 Projection Layer

The decoder output is projected onto the tokenizer vocabulary.

The resulting logits are used to predict the next token.

---

## 4. Technologies

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| PyTorch | Transformer implementation and training |
| Hugging Face Transformers | GPT-2 tokenizer |
| Pandas | Dataset preprocessing |
| Scikit-learn | Dataset splitting |
| NumPy | Numerical operations |
| Docker | Planned containerization |

---

## 5. Dataset

The dataset contains pairs of natural language instructions and their corresponding Linux commands.

Each example consists of:

- `input`: Natural language instruction.
- `output`: Expected Linux command.

### Example

```json
{
    "input": "List all files in the current directory",
    "output": "ls -a"
}
```

### 5.1 Data Preprocessing

The preprocessing pipeline includes:

1. Loading the dataset using Pandas.
2. Removing missing values.
3. Removing duplicate entries.
4. Splitting the dataset into training, validation, and test sets.
5. Tokenizing input instructions and target commands.
6. Preparing padded sequences.
7. Creating attention masks.

The dataset is divided into:

```text
data/
├── linuxcommands.json
└── processed/
    ├── train.json
    ├── val.json
    └── test.json
```

### 5.2 Tokenization

The project uses the pretrained GPT-2 tokenizer provided by Hugging Face.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

Additional special tokens are introduced to support sequence-to-sequence learning:

```python
special_tokens = {
    "bos_token": "<|bos|>",
    "pad_token": "<|pad|>"
}

tokenizer.add_special_tokens(special_tokens)
```

The tokenizer converts natural language instructions and Linux commands into sequences of token IDs.

The Transformer embedding layers are trained from scratch.

### 5.3 PyTorch Dataset

A custom PyTorch dataset handles the preparation of model inputs and target sequences.

The implementation uses:

- `torch.utils.data.Dataset`
- `torch.utils.data.DataLoader`
- A custom `collate_fn`

The dataloader handles batching and sequence padding.

---

## 6. Model Training

The model is trained using supervised learning.

Each training example contains an input instruction and its expected Linux command.

### 6.1 Training Pipeline

The training process follows these steps:

1. Tokenize the input instruction.
2. Tokenize the expected Linux command.
3. Prepare decoder inputs and prediction targets.
4. Pass the sequences through the Transformer.
5. Compute the cross-entropy loss.
6. Perform backpropagation.
7. Update the model parameters.
8. Repeat for multiple epochs.

### 6.2 Teacher Forcing

During training, the decoder receives the previous ground-truth tokens rather than its own predictions.

For example:

**Expected output:**

```bash
mkdir projects
```

Conceptually, the decoder is trained to predict:

```text
Decoder input:  <BOS> mkdir projects
Target:         mkdir projects <EOS>
```

This training strategy is known as **teacher forcing**.

### 6.3 Optimization

The training pipeline uses:

- Cross-Entropy Loss.
- AdamW optimizer.
- Gradient clipping.
- Mini-batch training.

Gradient clipping is used to limit excessively large gradients during training.

### 6.4 Training Configuration

The final hyperparameters will be documented after the experiments.

| Hyperparameter | Value |
|----------------|-------|
| Architecture | Transformer Encoder-Decoder |
| Model dimension | TBD |
| Encoder layers | TBD |
| Decoder layers | TBD |
| Attention heads | TBD |
| Feed-forward dimension | TBD |
| Dropout | TBD |
| Batch size | TBD |
| Learning rate | TBD |
| Optimizer | AdamW |
| Loss function | Cross-Entropy Loss |
| Epochs | TBD |
| Tokenizer | GPT-2 |

---

## 7. Experiments

The training process is divided into two main experiments.

### 7.1 Experiment 1: Overfitting a Small Dataset

The first experiment focuses on a subset of 10 training examples.

The objective is to verify that the Transformer can memorize a small dataset.

This is an important sanity check when implementing a neural network architecture from scratch.

A successful experiment should demonstrate that:

- The training loss decreases significantly.
- The model can reproduce the training targets.
- The forward and backward passes work correctly.
- The optimization process is functioning as expected.

#### Results

To be completed after training.

| Metric | Result |
|--------|--------|
| Number of training examples | 10 |
| Training loss | TBD |
| Training exact match accuracy | TBD |

### 7.2 Experiment 2: Generalization

The second experiment evaluates whether the model can generate commands for natural language instructions that were not included in the training data.

The model is trained on a larger dataset and evaluated on held-out examples.

The main objectives are to:

- Evaluate generalization to unseen instructions.
- Compare training and validation losses.
- Detect potential overfitting.
- Analyze incorrect predictions.
- Investigate the effect of training configuration.

#### Results

To be completed after evaluation.

| Metric | Result |
|--------|--------|
| Training loss | TBD |
| Validation loss | TBD |
| Test exact match accuracy | TBD |

---

## 8. Evaluation

The model is evaluated using several metrics.

### 8.1 Training Loss

Training loss measures how well the model fits the training examples.

A decreasing training loss indicates that the optimization process is reducing prediction errors.

### 8.2 Validation Loss

Validation loss evaluates the model on data not used for parameter updates.

Monitoring both training and validation losses helps identify overfitting.

### 8.3 Exact Match Accuracy

Exact Match Accuracy measures the percentage of generated commands that exactly match their reference commands.

\[
ExactMatch =
\frac{\text{Correct Predictions}}{\text{Total Predictions}}
\times 100
\]

For example, if the model generates 80 correct commands out of 100 test examples:

\[
ExactMatch = 80\%
\]

This metric is strict because multiple Linux commands may sometimes achieve the same result.

### 8.4 Prediction Examples

Actual predictions will be added after the evaluation experiments.

| Instruction | Expected Command | Generated Command |
|-------------|------------------|-------------------|
| Display current directory | `pwd` | TBD |
| List all files | `ls -a` | TBD |
| Create a directory named test | `mkdir test` | TBD |
| Remove a file named example.txt | `rm example.txt` | TBD |

---

## 9. Project Structure

The project is organized into separate modules for data processing, model architecture, masking, and training.

```text
.
├── data/
│   ├── linuxcommands.json
│   └── processed/
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── src/
│   ├── model.py
│   ├── linux_dataset.py
│   └── mask.py
│
├── README.md
└── ...
```

### Main Modules

**`src/model.py`**

Contains the Transformer architecture and its components:

- Input embeddings.
- Positional encoding.
- Multi-head attention.
- Encoder and decoder blocks.
- Feed-forward networks.
- Projection layer.

**`src/linux_dataset.py`**

Handles dataset preparation for PyTorch training.

**`src/mask.py`**

Contains functions used to create attention masks.

Additional modules for training, evaluation, and inference will be documented as the project is completed.

---

## 10. Challenges and Lessons Learned

Implementing a Transformer from scratch provided practical experience with several deep learning concepts.

### Understanding Attention

One of the main learning objectives was understanding how attention mechanisms operate on query, key, and value representations.

Implementing scaled dot-product attention helped clarify the role of matrix multiplication, attention scores, and softmax normalization.

### Working with Tensor Dimensions

The implementation required understanding how tensor shapes change throughout the Transformer.

This was particularly important for:

- Splitting embeddings into attention heads.
- Computing attention scores.
- Concatenating attention outputs.
- Handling batch and sequence dimensions.

### Attention Masking

Another important aspect was understanding how masks influence attention.

The project uses masking to prevent attention to padding tokens and future decoder tokens.

### Training Neural Networks with PyTorch

Developing the training pipeline provided practical experience with:

- Custom datasets and dataloaders.
- Loss computation.
- Gradient calculation.
- Optimizer updates.
- Gradient clipping.

### Generalization

An important objective is understanding the difference between memorizing training examples and learning representations that generalize to unseen data.

The final experiments will investigate these behaviors.

---

## 11. Limitations

This project is primarily intended for educational and experimental purposes.

The Transformer weights are initialized from scratch, without pretrained language model parameters.

Consequently:

- The model must learn representations from the available training dataset.
- Generalization may be limited by the size and diversity of the dataset.
- Training can be computationally expensive.
- Generated Linux commands may be incorrect or syntactically invalid.
- Exact Match Accuracy does not capture all forms of semantic equivalence.

The model is not intended to replace production-ready natural language-to-shell systems.

**Security note:** Generated Linux commands should always be reviewed before execution. This project does not automatically execute generated commands.

---

## 12. Future Improvements

- [ ] Verify that the Transformer can overfit 10 examples.
- [ ] Train the model on the full dataset.
- [ ] Monitor training and validation losses.
- [ ] Evaluate generalization on unseen instructions.
- [ ] Implement and evaluate autoregressive generation.
- [ ] Analyze incorrect predictions.
- [ ] Experiment with different hyperparameters.
- [ ] Add training curves and evaluation results.
- [ ] Document reproducible training and inference commands.
- [ ] Containerize the application with Docker.

---

## 13. References & Acknowledgments

### Transformer Architecture

**Vaswani et al. (2017)**

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

The original research paper introducing the Transformer architecture.

### Umar Jamil – Coding a Transformer from Scratch

The Transformer implementation in this project was developed by following Umar Jamil's excellent walkthrough:

- [YouTube Tutorial – Coding a Transformer from Scratch](https://www.youtube.com/watch?v=ISNdQcPhsts)
- [Original GitHub Repository – pytorch-transformer](https://github.com/hkproj/pytorch-transformer)

The tutorial explains how to implement the Transformer architecture using PyTorch.

**Project adaptation:**

While the architecture implementation is based on the tutorial, this project applies it to a different task: generating Linux shell commands from natural language instructions.

The dataset preparation, tokenization integration, training experiments, and evaluation are developed for this specific use case.

### Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

---

## 14. Author

**Hamza Allali**

Engineering Student – Grenoble INP – Ensimag  
Master of Artificial Intelligence

Interested in Machine Learning, Deep Learning, NLP, and AI Engineering.