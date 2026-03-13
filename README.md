# Latent-Sentiment-Analyzer: Transformer-based Market Sentiment Manifold

[![NLP](https://img.shields.io/badge/NLP-Transformer-orange.svg)](Abstract.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced research framework that utilizes **Transformer-based architectures (BERT/RoBERTa)** to map unstructured social and news data into a **Latent Sentiment Manifold**. This mapping enables the prediction of market volatility and trend shifts by identifying hidden linguistic patterns.

## 🔬 Core Methodology
- **Deep NLP Feature Extraction**: Uses pre-trained language models to extract semantic representations of market sentiment.
- **Latent Projection**: Implements a non-linear projection from high-dimensional text embeddings to a lower-dimensional latent space.
- **Volatility Prediction**: Maps latent vectors to predicted market outcomes, focusing on volatility impact and regime changes.

## 🛠 Project Structure
- `src/model.py`: Transformer-based sentiment model and projection manifold logic.
- `data/`: Sample datasets for fine-tuning and sentiment mapping.
- `models/`: Pre-trained checkpoints and custom projection weights.

## 🚀 Quick Start
```bash
python src/model.py
```

---

**Lead Researcher:** Muhammad Zainurrahman  
**Framework:** PyTorch | Transformers (HuggingFace) | NumPy
