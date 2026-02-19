# Neural Architecture Search for Document Classification

*Sarah Chen, Michael Rodriguez, Yuki Tanaka*

**Department of Computer Science, Technical University**

## Abstract

We present a comprehensive study of neural architecture search (NAS) applied to document classification tasks. Our automated approach discovers novel architectures that outperform hand-designed models while requiring minimal human expertise. Experiments across multiple benchmark datasets demonstrate the effectiveness of NAS for specialized document processing applications.

**Keywords:** neural architecture search, document classification, automated machine learning, deep learning

---

## 1. Introduction

Document classification remains a fundamental challenge in natural language processing and information retrieval. While deep learning has achieved remarkable success, architecture design requires extensive domain expertise and iterative experimentation.

Neural architecture search offers an alternative paradigm where machine learning systems automatically discover optimal model structures. Recent advances in NAS have shown promise in computer vision, but application to document processing remains underexplored.

This paper makes three primary contributions: First, we adapt NAS techniques specifically for document classification with a custom search space incorporating attention mechanisms and hierarchical structures. Second, we introduce efficiency constraints that limit computational cost during architecture search. Third, we provide extensive empirical analysis across diverse document types.

## 2. Related Work

**Document Classification Methods:** Traditional approaches relied on bag-of-words representations combined with classifiers such as naive Bayes or support vector machines. Modern deep learning methods employ recurrent neural networks, convolutional architectures, or transformer-based models like BERT.

**Neural Architecture Search:** Early NAS work used reinforcement learning to explore architecture spaces, though computational costs were prohibitive. Recent methods employ differentiable search strategies, evolutionary algorithms, or gradient-based optimization. Notable frameworks include DARTS, ENAS, and NASNet.

**Domain-Specific NAS:** While NAS has achieved success in image classification and object detection, few studies address text processing tasks. Our work extends NAS to the unique challenges of document-level understanding.

## 3. Methodology

### 3.1 Search Space Design

We define a hierarchical search space with three levels:

**Token Encoding Layer:** Selects between word embeddings, character-level CNNs, or subword tokenization with learnable embeddings.

**Feature Extraction Blocks:** Each block chooses from operations including self-attention, multi-head attention, dilated convolution, or recurrent cells. Blocks stack in sequence with residual connections.

**Aggregation Layer:** Combines token-level representations using max pooling, average pooling, or learned attention weights.

### 3.2 Search Strategy

Our search employs a gradient-based approach inspired by DARTS. Architecture parameters and model weights are jointly optimized through alternating updates. We introduce a resource constraint term in the objective function to penalize computationally expensive architectures.

The search objective combines classification accuracy on a validation set with a computational cost penalty:

L = L_val + λ · C(α)

where L_val is validation loss, α represents architecture parameters, C(α) estimates computational cost, and λ controls the trade-off.

### 3.3 Training Protocol

Search proceeds in two phases. During the search phase, architecture parameters evolve over 50 epochs using a reduced training set. After search completion, the discovered architecture is trained from scratch on the full dataset for 200 epochs.

We use the Adam optimizer with an initial learning rate of 0.001, decayed by 0.1 every 50 epochs. Batch size is 32 with gradient clipping at norm 5.0.

## 4. Experimental Setup

### 4.1 Datasets

We evaluate on four benchmark datasets:

- **20 Newsgroups:** 18,000 news articles across 20 categories
- **Reuters-21578:** Financial news documents with multi-label classification
- **IMDB Reviews:** 50,000 movie reviews for sentiment analysis
- **arXiv Papers:** 10,000 scientific abstracts across research domains

### 4.2 Baseline Models

Comparisons include logistic regression with TF-IDF features, TextCNN, BiLSTM with attention, and fine-tuned BERT-base.

## 5. Results

Table 1 presents classification accuracy across datasets. Our NAS-discovered architecture achieves state-of-the-art performance on three of four benchmarks while maintaining competitive computational efficiency.

**Analysis:** The discovered architecture consistently selected multi-head attention in early layers, followed by dilated convolutions for capturing long-range dependencies. This hybrid structure differs from hand-designed models and provides new insights into effective document encoding strategies.

## 6. Conclusion

Neural architecture search successfully identifies high-performance architectures for document classification without manual design. Our domain-specific search space and efficiency constraints enable practical deployment while achieving competitive accuracy.

Future directions include expanding the search space to incorporate graph-based document representations and applying NAS to multi-task document understanding scenarios.

## Acknowledgments

This research was supported by the National Science Foundation under grant IIS-2024-1234. We thank the anonymous reviewers for their constructive feedback.

## References

[1] K. He et al., "Deep Residual Learning for Image Recognition," *CVPR*, 2016.

[2] H. Liu et al., "DARTS: Differentiable Architecture Search," *ICLR*, 2019.

[3] J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers," *NAACL*, 2019.

[4] Y. Kim, "Convolutional Neural Networks for Sentence Classification," *EMNLP*, 2014.
