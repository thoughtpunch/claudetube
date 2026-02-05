# Neural Networks & Deep Learning: A Visual Guide

*Synthesized from 3Blue1Brown's Neural Networks series*

This guide distills the core concepts from Grant Sanderson's acclaimed video series on neural networks and deep learning. The series progresses from basic neural network structure through modern transformers and diffusion models.

---

## Part 1: The Foundation

### What is a Neural Network?

A neural network is a mathematical function that can learn to recognize patterns. At its core, a neuron is simply **a thing that holds a number** between 0 and 1, called its **activation**.

**The Structure:**
- **Input layer**: Neurons that receive raw data (e.g., 784 neurons for a 28x28 pixel image)
- **Hidden layers**: Intermediate layers that progressively extract features
- **Output layer**: Produces the final result (e.g., 10 neurons for digit recognition)

**The Key Insight**: Activations in one layer determine activations in the next layer through:
1. **Weighted sums** of all activations from the previous layer
2. Adding a **bias** term
3. Passing through an **activation function** (sigmoid, ReLU)

```
activation = σ(w₁a₁ + w₂a₂ + ... + wₙaₙ + bias)
```

**Matrix Notation**: The entire transition between layers can be expressed as:
```
a' = σ(W·a + b)
```
Where W is the weight matrix, a is the activation vector, and b is the bias vector.

### Why Layers?

The hope is that each layer learns increasingly abstract representations:
- **Layer 2**: Might detect edges
- **Layer 3**: Might combine edges into patterns (loops, lines)
- **Output**: Combines patterns to recognize complete objects

This hierarchical abstraction applies beyond images to speech (sounds → syllables → words → phrases) and other domains.

---

## Part 2: Learning via Gradient Descent

### The Cost Function

Learning is framed as an optimization problem. We need to measure "how bad" our network is performing:

```
Cost = average of (predicted - actual)² across all training examples
```

The cost function takes all ~13,000 weights and biases as inputs and outputs a single number measuring overall performance.

### Gradient Descent

**The Core Algorithm**:
1. Compute the **gradient** of the cost function (which direction increases cost most rapidly)
2. Take a small step in the **negative gradient** direction
3. Repeat

**Key Insight**: Each component of the gradient tells you:
- **Sign**: Should this weight increase or decrease?
- **Magnitude**: How sensitive is the cost to changes in this weight?

The gradient encodes the relative importance of each parameter - which changes give you the "most bang for your buck."

### Why Smooth Activations Matter

Neural networks use continuous activations (not binary on/off) specifically so the cost function is smooth and differentiable. This enables gradient-based optimization.

---

## Part 3: Backpropagation

### The Intuition

Backpropagation computes how each weight and bias should change to reduce the cost. For a single training example:

1. **Start at the output**: Compare what we got vs. what we wanted
2. **Propagate desires backward**: Each neuron "wants" its activation to change a certain way
3. **Translate to weights**: These desired activation changes inform which weights matter most

**The "Neurons that fire together, wire together" insight**: The biggest increases to weights happen between neurons that are both highly active. If a neuron fires while seeing a "2", it gets more strongly connected to neurons that fire when "thinking about" a 2.

### Three Ways to Increase a Neuron's Activation

1. **Increase the bias**
2. **Increase the weights** (especially those connected to bright/active neurons)
3. **Change the activations in the previous layer**

The third option is what makes this "backpropagation" - we recursively apply the same process moving backward through the network.

### Stochastic Gradient Descent

Computing gradients over all training examples is slow. Instead:
1. Shuffle training data into **mini-batches** (~100 examples)
2. Compute gradient approximation using just the mini-batch
3. Take a step, move to next batch

This is "stochastic" because mini-batches give noisy but useful gradient estimates.

---

## Part 4: The Calculus of Backpropagation

### Chain Rule Applied

For a simple network, to find how sensitive cost C is to weight w:

```
∂C/∂w = ∂z/∂w · ∂a/∂z · ∂C/∂a
```

Where:
- **∂z/∂w** = activation from previous layer (stronger neurons matter more)
- **∂a/∂z** = derivative of activation function
- **∂C/∂a** = 2(a - y), proportional to the error

**The Key Takeaway**: This chain rule expression gives us a complete recipe for computing gradients efficiently, which is what enables neural networks to learn.

---

## Part 5: Large Language Models

### The Core Mechanism

An LLM is a sophisticated function that **predicts what word comes next** given input text. It assigns probabilities to all possible next words.

**Chatbots work by**:
1. Setting up a prompt describing user-AI interaction
2. Repeatedly predicting and sampling the next word
3. Appending each prediction and continuing

### The Scale

- **Parameters**: GPT-3 has 175 billion tunable parameters
- **Training data**: Reading the GPT-3 training data at human speed would take 2,600+ years
- **Compute**: Training requires over 100 million years of human-equivalent computation

### Pre-training vs. Fine-tuning

1. **Pre-training**: Learn to predict next words from massive internet text
2. **RLHF (Reinforcement Learning with Human Feedback)**: Workers flag problematic outputs, corrections adjust parameters toward preferred responses

---

## Part 6: Transformers

### The Architecture Overview

Transformers process text through repeated blocks of:
1. **Attention** - vectors "talk" to each other, sharing information
2. **MLP (Multi-Layer Perceptron)** - individual processing with no cross-talk
3. Normalization steps in between

### Embeddings: Words as Vectors

Each word becomes a high-dimensional vector (~12,288 dimensions in GPT-3). Critically, **directions in this space carry semantic meaning**:

- `woman - man ≈ queen - king` (gender direction)
- `Italy - Germany + Hitler ≈ Mussolini` (nationality + historical role)
- `Germany - Japan + sushi ≈ bratwurst` (cultural association)

**Dot products measure alignment**: Similar meanings → vectors point in similar directions → large dot product.

### The Full Flow

1. **Tokenization**: Break text into ~50,000 possible tokens
2. **Embedding**: Look up each token's vector in the embedding matrix
3. **Attention + MLP blocks**: Repeated 96 times in GPT-3
4. **Unembedding**: Final vector → probability distribution over next tokens
5. **Softmax**: Normalize to valid probabilities

### Context Size

The network processes a fixed number of tokens at once (2,048 for GPT-3). This limits how much text it can "see" when making predictions.

---

## Part 7: The Attention Mechanism

### The Core Problem

The word "mole" means different things in:
- "American shrew **mole**" (animal)
- "One **mole** of carbon dioxide" (chemistry)
- "Take a biopsy of the **mole**" (medical)

After initial embedding, all "mole" tokens get the same vector. Attention allows context to refine this.

### Query, Key, Value

For each token, compute three vectors:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I have to offer?"
- **Value (V)**: "What information would I pass along?"

**The Attention Pattern**:
1. Compute dot products between all query-key pairs
2. Apply softmax to normalize (creates the "attention pattern")
3. Use these weights to take a weighted sum of value vectors
4. Add the result to the original embedding

### Multi-Headed Attention

GPT-3 uses **96 attention heads** running in parallel, each with its own Q, K, V matrices. This allows the model to learn many different types of contextual relationships simultaneously.

### Masking

During training, we mask future tokens - later words can't influence earlier ones. This prevents "cheating" by seeing the answer when predicting.

---

## Part 8: MLPs and Fact Storage

### Where Facts Live

Research suggests that factual knowledge (e.g., "Michael Jordan plays basketball") lives primarily in the **MLP blocks**, not the attention layers.

### MLP Structure

Each MLP block:
1. **Up-projection**: Multiply by large matrix (12,288 → ~50,000 dimensions)
2. **ReLU**: Clip negative values to zero
3. **Down-projection**: Multiply by another matrix back to embedding size
4. **Add to original**: Output = input + MLP(input)

### The AND Gate Insight

The ReLU creates a clean yes/no signal. If a row of the up-projection matrix encodes "first name Michael + last name Jordan", the neuron activates (becomes positive) only when BOTH conditions are met.

Then the corresponding column of the down-projection can add the "basketball" direction to the embedding.

### Superposition: The Key to Scale

Nearly perpendicular vectors can store far more than n features in n dimensions. The **Johnson-Lindenstrauss lemma** shows this grows exponentially with dimension.

This may explain why models scale so well - a space with 10x dimensions can store far more than 10x features.

---

## Part 9: Diffusion Models (Image/Video Generation)

### The Core Idea

Diffusion models learn to **reverse the process of adding noise**:
1. Start with real images
2. Progressively add noise until pure randomness
3. Train a model to predict and remove the noise

### Generation Process

1. Start with pure random noise
2. Ask the model: "What noise was added?"
3. Subtract predicted noise, add small random perturbation
4. Repeat 50-100 times
5. Result: realistic image

### Why Random Noise During Generation?

Without random noise at each step, all generated images converge to the **mean/average** of the training data, producing blurry results.

The model actually learns to predict the mean of a distribution. Adding noise samples from that distribution, enabling diverse outputs.

### CLIP: Bridging Images and Text

**CLIP** (Contrastive Language-Image Pre-training) trains paired image and text encoders so that:
- Matching image-caption pairs → similar vectors
- Non-matching pairs → dissimilar vectors

This creates a shared embedding space where:
```
(image of hat) - (image of no hat) ≈ "hat"
```

### Classifier-Free Guidance

The key to prompt adherence:
1. Train model with AND without text conditioning
2. Compute: **guided = conditioned + α(conditioned - unconditioned)**
3. This amplifies the direction toward the prompted content

Without guidance, you get "a shadow in a desert." With guidance, you get the tree you asked for.

---

## The Parameter Count (GPT-3)

| Component | Parameters |
|-----------|-----------|
| Embedding matrix | 617 million |
| Unembedding matrix | 617 million |
| Attention heads (96 layers × 96 heads) | ~58 billion |
| MLP blocks (96 layers) | ~116 billion |
| **Total** | **~175 billion** |

About 2/3 of all parameters are in the MLPs.

---

## Key Takeaways

1. **Neural networks are functions**: They map inputs to outputs through matrix multiplications and simple nonlinearities

2. **Learning = optimization**: Find parameters that minimize a cost function via gradient descent

3. **Backpropagation**: Chain rule applied recursively to efficiently compute gradients

4. **Transformers scale**: The attention mechanism enables parallel processing and massive models

5. **Directions encode meaning**: High-dimensional embedding spaces can represent rich semantic relationships

6. **Superposition enables capacity**: Nearly-orthogonal directions allow storing exponentially more features than dimensions

7. **Diffusion reverses noise**: Image generation works by learning to "denoise" random inputs

8. **Guidance steers generation**: Amplifying the difference between conditioned and unconditioned outputs enables precise control

---

## Further Learning

Resources recommended in the series:
- Michael Nielsen's free online book on neural networks
- Chris Olah's blog posts
- Distill.pub articles
- Andrej Karpathy's content
- Welch Labs (for the diffusion chapter)

---

*This document synthesizes ~3.5 hours of video content from the 3Blue1Brown Neural Networks playlist. For visual intuitions and animations, the original videos are invaluable.*
