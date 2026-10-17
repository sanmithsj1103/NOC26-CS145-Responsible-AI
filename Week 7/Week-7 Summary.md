# Week 7

### Introduction

In an increasingly data-driven world, the deployment of artificial intelligence and machine learning systems profoundly impacts critical aspects of our lives, from financial decisions to judicial outcomes. While these technologies promise unprecedented efficiency and innovation, their responsible application hinges on addressing fundamental ethical considerations: privacy and fairness. This week's lessons delve into advanced concepts in privacy-preserving data analysis, moving beyond strict guarantees to more pragmatic approaches, and then pivot to the equally crucial challenge of ensuring algorithmic fairness. Understanding these principles is not merely an academic exercise; it is essential for building trustworthy AI systems that serve all individuals equitably, preventing discrimination, and maintaining public confidence in automated decision-making.

Our exploration begins by refining our understanding of differential privacy, introducing a more flexible framework that allows for greater data utility while still providing strong privacy guarantees. We then examine a powerful mechanism designed for privately selecting optimal choices from a set of options. Finally, we confront the pervasive issue of algorithmic bias, exploring how machine learning models can inadvertently perpetuate societal inequalities and introducing methods to mitigate these harms. This week equips us with the conceptual tools to design and evaluate AI systems that are not only powerful but also ethical and just.

### Core Concepts

| Term | Definition | Significance |
| :--- | :--- | :--- |
| **`Epsilon-Delta Differential Privacy`** | A relaxed privacy guarantee where a mechanism is `(ε, δ)`-differentially private, allowing a small probability `δ` of privacy failure in rare events. | Enables higher utility by permitting less noise, especially for complex data releases, while maintaining strong privacy for most cases. |
| **`Delta (δ)`** | A small, acceptable probability of privacy failure in `(ε, δ)`-differential privacy, representing a controlled "wiggle room" in the strict `ε` guarantee. | Crucial for balancing privacy and utility, allowing mechanisms to add less noise and preserve more data accuracy. |
| **`Gaussian Mechanism`** | A privacy-preserving technique that adds noise sampled from a Gaussian (normal) distribution to a function's output. | The primary method for achieving `(ε, δ)`-differential privacy, particularly beneficial for multi-dimensional outputs due to its noise scaling properties. |
| **`L2 Sensitivity`** | The maximum change in the L2 (Euclidean) norm of a function's output when a single individual's data is altered in the input dataset. | Used by the Gaussian mechanism to determine the appropriate scale of noise, offering better utility for vector-valued functions than L1 sensitivity. |
| **`Exponential Mechanism (EM)`** | An algorithm that selects an output from a finite set with a probability proportional to an exponential function of its utility score and the privacy parameter. | Ensures differential privacy when selecting an optimal item from a set of choices, treating equally good options with equal probability. |
| **`Fairness`** | The principle that machine learning algorithms should not produce outcomes systematically disadvantageous to certain groups or individuals. | Essential for ethical AI deployment, preventing discrimination, and fostering public trust in automated decision-making systems. |
| **`Protected Attribute`** | A characteristic of an individual (e.g., race, gender, age) that is legally or ethically protected from discrimination. | Central to defining and measuring fairness, as algorithms must ensure equitable treatment regardless of these attributes. |
| **`Statistical Parity`** | A fairness notion requiring that the probability of a favorable outcome is the same across different protected groups. | A common metric for assessing group-level fairness, aiming for equal representation in positive outcomes. |
| **`Equality of Opportunity`** | A fairness notion where, among individuals who truly deserve a positive outcome, the probability of receiving that outcome is the same across different protected groups. | Focuses on ensuring that deserving individuals from all groups have an equal chance, addressing bias in true positive rates. |
| **`Constrained Optimization`** | An optimization technique where an objective function is minimized or maximized subject to specific conditions or constraints. | A key algorithmic approach to achieving fairness by incorporating fairness criteria directly into the model training process. |

### Detailed Analysis

#### 1. Approximate Differential Privacy: A Pragmatic Approach to Data Release

The first lesson, **Approximate Differential Privacy**, introduces a crucial evolution from the strict `ε`-differential privacy (ε-DP) to `(ε, δ)`-differential privacy (ε, δ)-DP. While ε-DP guarantees that the probability ratio of any output for neighboring datasets is bounded by `e^ε`, `(ε, δ)`-DP adds a small term `δ` to this inequality: `P(M(X) ∈ S) ≤ e^ε * P(M(X') ∈ S) + δ`. This `δ` represents a minuscule, acceptable probability of privacy failure in rare events, typically set to a very small value like `10^-5` or `10^-9`. The primary motivation for this relaxation is to enable mechanisms that can add significantly less noise, thereby preserving more utility in the released data.

The workhorse for achieving `(ε, δ)`-DP is the **Gaussian Mechanism**. Unlike the Laplace mechanism, which adds noise from a Laplacian distribution and is typically used for `ε`-DP (or `(ε, 0)`-DP), the Gaussian mechanism adds noise sampled from a Gaussian (normal) distribution. A key distinction arises when dealing with multi-dimensional outputs (e.g., releasing a vector of `d` values). The Gaussian mechanism utilizes `L2 Sensitivity`, defined as the maximum L2 norm (Euclidean distance) between function outputs for neighboring datasets. For `d` outputs, the noise added by the Gaussian mechanism scales with `sqrt(d)`, whereas the Laplace mechanism's noise scales linearly with `d`. This `sqrt(d)` scaling, combined with a logarithmic factor `log(1/δ)`, means the Gaussian mechanism adds quadratically less noise for large `d`, leading to substantial utility gains.

**Approximate Differential Privacy** also maintains important properties. `Post-processing Immunity` ensures that any arbitrary function applied to the output of an `(ε, δ)`-differentially private mechanism does not diminish its privacy guarantees. Furthermore, `Composition Theorems` describe how privacy guarantees accumulate when multiple mechanisms are applied. While `Basic Composition` shows a linear degradation (`kε, kδ`) for `k` mechanisms, `Advanced Composition` for Gaussian mechanisms offers a much tighter bound, with the `ε` term scaling with `sqrt(k)`. This `sqrt(k)` scaling is particularly vital for iterative algorithms, such as those in machine learning, allowing for many more steps while maintaining a reasonable privacy budget.

#### 2. The Exponential Mechanism: Private Selection of Optimal Choices

The second lesson, **Exponential Mechanism**, addresses the challenge of privately selecting an optimal output from a finite set of choices based on sensitive data. It is motivated by problems like a seller setting a price for a digital good, where individual buyer valuations are private, but the chosen price is public. A naive approach of simply adding noise to the optimal price can lead to suboptimal outcomes or treat equally profitable options with unequal probabilities.

The **Exponential Mechanism** formalizes this problem by defining:
* `Input Data (x)`: The private information (e.g., buyer valuations).
* `Output Space (H)`: The finite set of possible choices (e.g., possible prices).
* `Scoring Function (S(x, h))`: Quantifies the utility of choosing an output `h` given `x` (e.g., revenue).
* `Sensitivity (ΔS)`: The maximum change in the scoring function for any output when a single individual's data changes.

The mechanism selects an output `h` from `H` with a probability proportional to `exp(ε * S(x, h) / (2 * ΔS))`. This means outputs with higher scores are more likely to be chosen, but all options have a non-zero probability. A key advantage is that if two different outputs yield the same utility score, they will be chosen with the same probability, addressing the fairness issue of simple noise addition. The **Exponential Mechanism** provides `ε-Differential Privacy` for the input data and a `Utility Guarantee`, ensuring that the chosen output's score is highly likely to be close to the optimal possible score, balancing privacy with practical utility. This mechanism is broadly applicable whenever a selection needs to be made from a discrete set based on private data.

#### 3. Fairness in Machine Learning: Mitigating Algorithmic Bias

The final lesson, **Fairness in Machine Learning**, shifts focus to the critical ethical dimension of algorithmic bias. It highlights that simply maximizing accuracy is insufficient for responsible AI, as algorithms can perpetuate and amplify societal biases present in their training data. Several real-world examples underscore this:
* Google's "Ugliest Language" incident, where search results reflected offensive stereotypes.
* The COMPAS recidivism prediction system, which showed racial bias by disproportionately labeling African-American defendants as high-risk.
* Amazon's AI recruiting tool, which exhibited gender bias by penalizing resumes with words associated with women.
* Latanya Sweeney's research, demonstrating that online ads correlated racially identifying names with arrest records.
* Bias in `Word Embeddings`, where semantic relationships encoded societal stereotypes (e.g., "man" closer to "officer," "woman" to "nurse").
* Even large language models (LLMs) acknowledge their potential to produce biased content.

Defining fairness is complex, but practical approaches often ground it in legal definitions. Algorithmic notions of fairness include:
* `Statistical Parity`: Requires the probability of a positive outcome to be the same across different `Protected Attributes` groups.
* `Equality of Opportunity`: Focuses on ensuring that among individuals who truly deserve a positive outcome, the classifier's positive prediction rate is the same across protected groups.

Fairness concerns extend to both unsupervised and supervised learning. For unsupervised learning, `Principal Component Analysis (PCA)` can exhibit bias, such as higher reconstruction errors for female faces. `Fair PCA` addresses this by minimizing the maximum *excess loss* across different populations, ensuring that no single group suffers disproportionately. For supervised learning, algorithms like `Logistic Regression` can be made fair through `Constrained Optimization`. This involves modifying the standard optimization problem to include constraints that enforce fairness, such as limiting the correlation between the model's predictions and the protected attribute.
```
Minimize_w Loss(w, Data)
Subject to: | Σ_i (s_i - s̄) * (w^T x_i) | ≤ τ
```
Here, `s_i` is the protected attribute, `s̄` is its average, `w^T x_i` is the model's raw output, and `τ` is a small tolerance. This approach restricts the parameter search space, balancing accuracy with non-discrimination. Fairness considerations also apply to sequential learning problems, emphasizing the broad applicability of these principles across various machine learning paradigms.

### Key Takeaways
- [ ] Epsilon-delta differential privacy offers a practical balance between privacy and utility by allowing a small, controlled probability `δ` of privacy failure.
- [ ] The Gaussian mechanism is the primary tool for `(ε, δ)`-DP, using `L2 Sensitivity` and adding noise that scales more favorably (`sqrt(d)`) for multi-dimensional outputs than the Laplace mechanism.
- [ ] `Post-processing Immunity` and `Advanced Composition` (with `sqrt(k)` scaling for `ε`) are crucial properties of approximate differential privacy, especially for iterative algorithms.
- [ ] The Exponential Mechanism privately selects an optimal item from a finite set of choices, ensuring `ε`-differential privacy and a `Utility Guarantee` by probabilistically favoring higher-scoring options.
- [ ] Simple noise addition to an optimal output is often insufficient for privacy-preserving selection, as it can lead to suboptimal outcomes or unequal probabilities for equally good options.
- [ ] Machine learning algorithms can perpetuate and amplify societal biases present in training data, leading to unfair and discriminatory outcomes in real-world applications.
- [ ] Fairness in ML requires explicit consideration beyond maximizing accuracy, with examples like COMPAS and Amazon's recruiting tool demonstrating tangible harm.
- [ ] Algorithmic fairness can be defined using notions like `Statistical Parity` (equal positive outcome rates across groups) and `Equality of Opportunity` (equal true positive rates across groups).
- [ ] `Fair PCA` and `Fair Logistic Regression` illustrate how fairness can be integrated into unsupervised and supervised learning through `Constrained Optimization` to mitigate bias.
- [ ] Achieving fairness often involves a trade-off with raw accuracy, necessitating careful balancing of ethical and performance objectives.

### Conclusion

This week's lessons have illuminated critical pathways toward building more responsible and ethical AI systems. By exploring **Approximate Differential Privacy** and the **Exponential Mechanism**, we've gained a deeper understanding of how to design privacy-preserving algorithms that are not only robust but also practical, offering a crucial balance between stringent privacy guarantees and the need for data utility. These advanced privacy techniques are indispensable for enabling data analysis and model training on sensitive information without compromising individual confidentiality.

Simultaneously, our dive into **Fairness in Machine Learning** has underscored the profound societal implications of algorithmic bias. The numerous real-world examples serve as a stark reminder that unchecked AI can exacerbate existing inequalities, making the pursuit of fairness an ethical imperative. By understanding various definitions of fairness and implementing techniques like `Constrained Optimization` in algorithms such as `Fair PCA` and `Fair Logistic Regression`, we can actively work to mitigate bias and ensure that AI systems treat all individuals equitably. Together, these concepts of advanced privacy and algorithmic fairness form foundational pillars for the development of trustworthy, impactful, and ethically sound artificial intelligence that truly benefits society.