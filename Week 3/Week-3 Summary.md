# Week 3

### Introduction

In an era where Artificial Intelligence models are becoming increasingly powerful and pervasive, the ability to control, adapt, and even "forget" information is as critical as the capacity to learn. This week, we delve into the cutting edge of responsible AI, exploring the burgeoning field of machine unlearning—a crucial mechanism for addressing privacy concerns, mitigating biases, and ensuring compliance with evolving legal frameworks like GDPR. As models grow in complexity and data consumption, the prohibitive cost of retraining necessitates innovative solutions to selectively remove undesirable information.

Beyond the ethical imperative of unlearning, we also lay the groundwork for understanding how these intelligent systems are built from the ground up. This includes a comprehensive introduction to the machine learning pipeline, from raw data to actionable insights, and a deep dive into the fundamental concepts of neural networks. We will then transition to practical implementation using PyTorch, a leading deep learning framework, covering everything from tensor operations and model architecture to training workflows and classification tasks. By bridging the gap between theoretical concepts of responsible AI and the practicalities of deep learning development, this week equips you with the knowledge to not only understand the challenges of modern AI but also to build and deploy intelligent systems ethically and effectively.

### Core Concepts

| Term | Definition | Significance |
| :--- | :--- | :--- |
| **`Machine Unlearning`** | The process of removing the influence of specific training data from a trained machine learning model. | Essential for data privacy, regulatory compliance, and correcting model biases or errors without costly full retraining. |
| **`Approximate Unlearning`** | A method where a model is modified to behave as if specific data was never learned, aiming for distributional closeness rather than exact erasure. | Offers a practical and often more efficient alternative to exact unlearning, addressing its computational drawbacks. |
| **`Representation Engineering (RE)`** | An approach to control a language model's output behavior by directly modifying its internal hidden representations. | Enables post-training behavioral control, offering flexibility and efficiency in aligning LLMs with desired traits like honesty or harmlessness. |
| **`Overfitting`** | A model that learns the training data too well, including noise and specific patterns, leading to poor performance on new, unseen data. | Results in high accuracy on training data but low accuracy on test data, indicating a lack of generalization. |
| **`Bias-Variance Trade-off`** | The inherent conflict in model design where reducing bias (underfitting) often increases variance (overfitting), and vice-versa. | Finding the optimal balance is crucial for building models that generalize effectively to new data. |
| **`Backpropagation`** | An algorithm that efficiently computes the gradients of the loss function with respect to all the weights in the network, working backward from the output layer. | It's the cornerstone of neural network training, allowing the network to "learn" by attributing error to specific weights. |
| **`Learning Rate`** | A hyperparameter that controls the step size taken during each weight update in gradient descent. | Determines how aggressively or cautiously the model adjusts its weights, significantly impacting convergence speed and stability. |
| **`tensor`** | A multi-dimensional array, similar to NumPy arrays, that is the fundamental data structure in PyTorch. | Tensors are the universal data representation in PyTorch, used for all inputs, outputs, and model parameters in neural networks. |
| **`nn.Module`** | The base class for all neural network modules in PyTorch, providing a structured way to define network layers and forward passes. | Essential for building custom neural network architectures, inheriting core functionalities like parameter management. |
| **`Non-linearity (Activation Function)`** | A mathematical function applied to the output of each layer, introducing non-linear transformations into the network. | Allows neural networks to learn and model complex, non-linear relationships and decision boundaries in data. |

### Detailed Analysis

#### 1. The Imperative of Machine Unlearning
The week begins by establishing the critical need for **Unlearning** in modern AI. With models growing exponentially in parameters and training data, the ability to remove specific, undesirable information without costly full retraining becomes paramount. This is driven by:
* **Legal and Ethical Motivations**: The "Right to be Forgotten" (GDPR) mandates the erasure of personal data, posing a significant challenge for models that have integrated this data.
* **Data Quality and Safety**: Models can learn private data, stale knowledge, copyrighted materials, toxic content, dangerous capabilities, and misinformation from vast internet datasets.
* **Approaches**:
* **Access Revocation**: Focuses on removing a model's ability to utilize specific information, often requiring periodic retraining.
* **Model Correction/Editing**: Post-training modifications to alter behavior, remove biases, or update knowledge.
* **Forms of Unlearning**:
* **Exact Unlearning**: Aims for a model distributionally identical to one never trained on the forgotten data. **Sharding (SISA Training)** is a technique where data is divided into isolated shards, allowing only relevant constituent models to be retrained, significantly reducing cost.
* **Approximate Unlearning**: A more practical approach when exact unlearning is too costly or the data to be forgotten is vague.

#### 2. Nuances of Approximate Unlearning
The **Approximate unlearning** lesson delves deeper into the practicalities of making models "forget."
* **Limitations of Exact Unlearning**: Exact methods, especially sharding, can suffer from exponential performance degradation as the number of unlearning requests increases.
* **Defining Approximate Unlearning**: The goal is for the unlearned model to be "distributionally close" to a gold standard model (trained from scratch without the forget set), making it indistinguishable to an adversary.
* **Differential Privacy (DP)**: Provides a theoretical framework for achieving distributional closeness by introducing controlled noise, offering statistical guarantees against inference attacks.
* **Membership Inference Attacks**: A key evaluation metric; successful unlearning should make it impossible for an adversary to determine if a specific data point was part of the training set.
* **Empirical Unlearning**:
* **Known Example Space**: When the "forget set" is precisely defined, methods like fine-tuning on the retain set or targeted gradient steps can be used.
* **Unknown Example Space**: For abstract concepts (e.g., "unlearn Harry Potter"), methods like "just asking" LLMs to pretend not to know or using `counterfactuals` (flipping labels of forget examples) are employed.

#### 3. Evaluating Unlearning and Graph Unlearning
The **Evaluation of Unlearning and Graph Unlearning Part 1** and **Evaluation of Unlearning and Graph Unlearning Part 2** lessons address the critical challenge of measuring unlearning effectiveness.
* **Evaluation Challenges**: Historically, a lack of standardized benchmarks made evaluation difficult. The field has moved towards synthetic manipulations to create measurable unlearning goals.
* **Interclass Confusion**: A metric where synthetic confusion is introduced between classes, and unlearning is evaluated by how effectively this confusion is removed.
* **Corrective Machine Unlearning**: Aims to improve model accuracy on an affected domain by removing identified "wrong" data, even if only a representative subset is known.
* **LLM Benchmarks**:
* **TOFU (Task of Fictitious Unlearning for LLMs)**: Evaluates unlearning of synthetic author profiles, assessing knowledge retention.
* **WMDP (Weapons of Mass Destruction Proxy)**: Focuses on unlearning dangerous knowledge (biosecurity, cybersecurity), crucial for AI safety.
* **Hardness of Unlearning**: Varies from easy (infrequent text) to hardest (fundamental, implicitly embedded knowledge).
* **Representation Engineering**: Offers a higher-level approach to model control and unlearning by manipulating internal representation vectors, influencing traits like honesty or morality.
* **Graph Unlearning**: Extends unlearning to graph-structured data (social, healthcare, financial networks).
* **Foundations**: Graph Convolutional Networks (GCNs) and Message Passing Neural Networks (MPNNs) learn node representations by aggregating neighbor information.
* **Types of Operations**:
* **Node Feature Unlearning**: Removes specific attributes of a node.
* **Node Unlearning**: Removes an entire node and its connections.
* **Edge Unlearning**: Removes a specific connection between two nodes.
* **Methods**: Various algorithms like Graph Eraser, Guide, Projector, Graph Editor, GNN Delete, and Megu address these operations, each with strengths and limitations.
* **Challenges**: Preventing "over-forgetting" (inadvertently removing too much), supporting "continued learning" (incremental updates post-unlearning), and ensuring real-world applicability.

#### 4. Hands-on with Representation Engineering
The **Representation Engineering - Hands on** lesson provides a practical deep dive into controlling LLM behavior.
* **From Interpretability to Causal Control**: RE shifts from understanding *what* a model knows to actively shaping its internal "thoughts" by modifying hidden representations.
* **Illustrative Examples**: RE can control:
* **Honesty/Truthfulness**: Make models truthful even when prompted to lie.
* **Morality/Power**: Shift ethical stances or induce power-seeking behavior.
* **Emotional Tone**: Inject anger, fear, or happiness into responses.
* **Harmlessness/Jailbreak Prevention**: Restore safety mechanisms against adversarial prompts.
* **Fairness/Bias Reduction**: Mitigate stereotypical biases.
* **Fact Editing/Knowledge Unlearning**: Alter factual knowledge or remove concepts.
* **Memorization Control**: Prevent completion of memorized phrases.
* **Mechanism: Linear Artificial Tomography (LAT)**:
1. **Design Stimulus Prompts**: Create contrasting prompts (e.g., happy vs. not happy).
2. **Collect Internal Activations**: Extract hidden representations from a specific transformer layer.
3. **Find the Steering Vector**: Calculate the difference between average representations of contrasting concepts (e.g., using the `Mean Difference Method`).
* **Hands-on Demonstration**:
* Setup: Clone `repi` library, load model (e.g., Mistral).
* Configuration: Define `rep_token`, `hidden_layers`, `direction_method`, `coefficient`, `sign`.
* Extraction:
```python
from repi.pipeline import RepiPipelineRegistry
# ... model and tokenizer loading ...
repi_pipeline = RepiPipelineRegistry.get_pipeline(
model=model,
tokenizer=tokenizer,
rep_token=rep_token,
hidden_layers=hidden_layers,
direction_method=direction_method,
batch_size=batch_size
)
repi_pipeline.train(train_data, train_labels)
```
* Application: Generate text with and without steering.
* **Evaluation**: Assesses correlation (alignment), manipulation (steering effectiveness), and termination (concept removal).

### Introduction
The **Introduction to ML Part 1** and **Introduction to ML Part 2** lessons provide a foundational understanding of machine learning.
* **ML Pipeline**:
* **Data**: Task-specific annotated data (supervised) or large-scale unannotated data (unsupervised).
* **Pre-processing**: Cleaning and transforming raw data (e.g., tokenization, stemming, lemmatization for text; resizing for images).
* **Data Representation**: Converting processed data into numerical formats.
* **Bag of Words (BOW)**: Simple word counts, ignores order.
* **TF-IDF**: Weights words by frequency and rarity, improving discriminative power.
* **Word Vectors**: Dense, finite-dimensional representations capturing semantic meaning based on context (e.g., Word2Vec).
* **ML Algorithm Categories**:
* **Supervised Learning**: Learns from labeled data (X, Y) for tasks like:
* **Classification**: Predicting categorical labels (binary, multi-class).
* **Regression**: Predicting real-valued numbers.
* **Unsupervised Learning**: Finds patterns in unlabeled data (X) for tasks like:
* **Clustering**: Grouping similar data points.
* **Dimensionality Reduction**: Reducing features while retaining information (e.g., PCA).
* **Evaluation Metrics**:
* **Accuracy**: (Correct Predictions / Total), can be misleading for imbalanced datasets.
* **Confusion Matrix**: Visualizes TP, TN, FP, FN.
* **Precision**: TP / (TP + FP), important when FP cost is high.
* **Recall (Sensitivity)**: TP / (TP + FN), important when FN cost is high.
* **F1 Score**: Harmonic mean of precision and recall, balances both.
* **Model Fit**:
* **Underfitting**: Model too simple, fails to capture patterns (high bias).
* **Overfitting**: Model learns training data too well, including noise, performs poorly on new data (high variance).
* **Bias-Variance Trade-off**: Balancing model complexity for optimal generalization.
* **Data Splitting**:
* **Train-Test Split**: Divides data for training and unbiased evaluation.
* **Train-Validation-Test Split**: Adds a validation set for hyperparameter tuning.
* **K-Cross Validation**: Robust evaluation by averaging performance across multiple train-validation splits.
* **Sentiment Classification Pipeline Example**:
* Data Loading, Pre-processing (lowercasing, tokenization, stop word removal).
* Splitting:
```python
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```
* Vectorization (`TfidfVectorizer`):
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
```
* Model Training (`LogisticRegression`):
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
```
* Evaluation (`classification_report`):
```python
from sklearn.metrics import classification_report
print(classification_report(y_val, y_val_pred))
```
* Feature Importance: Analyzing model weights to understand influential words.

#### 6. Basics of Neural Networks & PyTorch
The **Basics of Neural Networks & PyTorch Part 1** and **Basics of Neural Networks & PyTorch Part 2** lessons introduce the core mechanics and tools for deep learning.
* **NN Training Workflow**:
* Initialization -> Forward Pass (prediction) -> Loss Calculation -> Weight Adjustment.
* **Gradients and Backpropagation**:
* `Gradient`: Quantifies each weight's contribution to the loss.
* `Partial Differentiation` and `Chain Rule`: Used to compute gradients.
* `Backpropagation`: Algorithm to efficiently compute all gradients from output to input.
* **Gradient Descent**: Iterative optimization to minimize loss by adjusting weights opposite to the gradient.
* `W_new = W_old - eta * (dL/dW)`
* `Learning Rate (eta)`: Controls step size; crucial hyperparameter.
* **Non-linearities**: `Activation Functions` (e.g., ReLU) are essential for NNs to learn complex, non-linear decision boundaries.
* **PyTorch Tensors**: Fundamental multi-dimensional arrays.
* Creation: `torch.tensor()`, `torch.rand()`, `torch.zeros()`, `torch.arange()`.
* Attributes: `dtype` (e.g., `torch.float32`, `torch.float16`), `device` (CPU/CUDA), `requires_grad` (for autograd).
* Operations: Element-wise (`+`, `*`), `torch.matmul()` (`@`).
* Manipulation: `reshape()`, indexing.
* NumPy Interoperability: `torch.from_numpy()`, `tensor.numpy()`.
* Reproducibility: `torch.manual_seed()`.
```python
import torch
# Example tensor creation
scalar = torch.tensor(7)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])
# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor_on_device = torch.tensor([1, 2, 3], device=device)
# Matrix multiplication
mat1 = torch.tensor([[1, 2], [3, 4]])
mat2 = torch.tensor([[5, 6], [7, 8]])
print(mat1 @ mat2)
```

#### 7. PyTorch Basic Workflow and Classification
The **PyTorch - Basic Workflow** and **PyTorch - Classification** lessons bring these concepts together into practical PyTorch development.
* **Data Preparation**: Create dummy data, convert to `torch.Tensor`, split into train/test sets.
```python
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, noise=0.03, random_state=42)
X_tensor = torch.from_numpy(X).type(torch.float32)
y_tensor = torch.from_numpy(y).type(torch.long)
```
* **Model Definition**: Inherit from `nn.Module` (for custom models) or use `nn.Sequential` (for simple stacks).
```python
import torch.nn as nn
class CircleModelV2(nn.Module):
def __init__(self):
super().__init__()
self.layer_1 = nn.Linear(in_features=2, out_features=10)
self.layer_2 = nn.Linear(in_features=10, out_features=10)
self.layer_3 = nn.Linear(in_features=10, out_features=1)
self.relu = nn.ReLU()
def forward(self, x):
return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
model_2 = CircleModelV2().to(device)
```
* **Loss Functions and Optimizers**:
* Binary Classification: `nn.BCEWithLogitsLoss()` (combines sigmoid and BCE for stability).
* Multi-class Classification: `nn.CrossEntropyLoss()`.
* Optimizer: `torch.optim.SGD()`, `torch.optim.Adam()`.
* **Training Loop**:
1. `model.train()`
2. Forward pass
3. Calculate loss
4. `optimizer.zero_grad()`
5. `loss.backward()` (backpropagation)
6. `optimizer.step()` (weight update)
* **Evaluation Loop**:
1. `model.eval()`
2. `torch.inference_mode()` (disable gradients)
3. Forward pass on test data
4. Calculate test loss/metrics
* **Non-linearities in Classification**: Demonstrated that linear models fail on non-linearly separable data (`make_circles`), while `ReLU` enables learning complex decision boundaries.
* **Multi-class Classification**: Uses `make_blobs` data, output layer `out_features` matches number of classes, and `nn.CrossEntropyLoss()`.
* **Saving/Loading Models**: `torch.save(model.state_dict(), path)`, `model.load_state_dict(torch.load(path))`.

### Key Takeaways
- [ ] Machine unlearning is essential for ethical AI, privacy compliance (GDPR), and mitigating harmful model behaviors without full retraining.
- [ ] Exact unlearning (e.g., SISA) provides strong guarantees but can be computationally expensive, leading to the rise of approximate unlearning.
- [ ] Approximate unlearning aims for "distributional closeness" to a model never trained on forgotten data, often leveraging Differential Privacy and evaluated via Membership Inference Attacks.
- [ ] Representation Engineering offers a powerful post-training method to control LLM behavior by manipulating internal "steering vectors," influencing traits like honesty, emotion, or bias.
- [ ] The machine learning pipeline involves data acquisition, pre-processing, data representation (e.g., word vectors), and rigorous evaluation.
- [ ] Understanding overfitting, underfitting, and the bias-variance trade-off is crucial for building models that generalize well to unseen data.
- [ ] Neural networks learn by iteratively minimizing a loss function through forward passes, backpropagation (gradient computation), and gradient descent (weight updates).
- [ ] Non-linear activation functions are fundamental for neural networks to model complex, non-linear relationships in data.
- [ ] PyTorch tensors are the core data structure, and managing their `dtype`, `device`, and `requires_grad` is vital for efficient deep learning.
- [ ] The PyTorch workflow involves defining models with `nn.Module`, selecting appropriate loss functions and optimizers, and executing iterative training and evaluation loops.

### Conclusion

This week's exploration has traversed the critical landscape of modern AI, from the ethical imperative of machine unlearning to the foundational mechanics of neural networks and their practical implementation in PyTorch. We've seen how the ability to selectively "forget" information is not merely a technical challenge but a societal necessity, driven by privacy regulations and the need to combat misinformation and bias. Techniques like Representation Engineering offer a glimpse into a future where AI's internal "thoughts" can be steered, allowing for more aligned and safer intelligent systems.

Simultaneously, we've built a robust understanding of the machine learning pipeline, demystifying how raw data transforms into actionable insights. The deep dive into neural network principles—gradients, backpropagation, and the crucial role of non-linearities—provides the theoretical bedrock for building complex models. Finally, the hands-on PyTorch sessions have equipped you with the practical skills to define, train, and evaluate classification models, demonstrating how these theoretical concepts translate into powerful, real-world applications. By mastering these intertwined domains, you are now better prepared to contribute to the development of AI that is not only intelligent and capable but also responsible, adaptable, and aligned with human values.