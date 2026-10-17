# Week 11

### Introduction

In today's rapidly evolving technological landscape, artificial intelligence, particularly large language models (LLMs), stands at the forefront of innovation, demonstrating unprecedented capabilities in understanding and generating human-like text. However, the journey from raw potential to reliable application is complex, requiring sophisticated techniques to adapt these general-purpose models for specific tasks, robust frameworks to govern their ethical deployment, and rigorous methods to ensure their trustworthiness. This week, we delve into the critical processes that shape AI's utility and safety: **Finetuning and Jailbreaking: Hands-on** explores how LLMs are specialized for diverse applications and the vulnerabilities that can arise; **AI Governance** examines the essential policies, laws, and ethical guidelines needed to manage AI responsibly; and **Research Overview: SaGE- Quantifying moral consistency in LLMs** introduces a novel approach to measure and enhance the reliability of these powerful systems. Together, these topics provide a comprehensive understanding of the technical and societal challenges in building AI that is not only intelligent but also aligned with human values and expectations.

### Core Concepts

| Term | Definition | Significance |
| :--- | :--- | :--- |
| **`Fine-tuning`** | The process of adjusting the parameters of a pre-trained language model to better suit a specific downstream task using a task-specific dataset. | Adapts a general-purpose model to perform specialized functions, improving performance on targeted applications. |
| **`PEFT (Parameter Efficient Fine-tuning)`** | Techniques that modify only a small subset of a pre-trained model's parameters, or introduce new, smaller parameters, to adapt it to a new task. | Reduces computational cost and memory usage compared to full fine-tuning, mitigating catastrophic forgetting. |
| **`Jailbreaking`** | The act of circumventing a language model's safety alignments or ethical guidelines to elicit undesirable, harmful, or biased responses. | Highlights vulnerabilities in AI safety mechanisms and the ongoing need for robust alignment techniques. |
| **`RLHF (Reinforcement Learning from Human Feedback)`** | A training paradigm that uses human preferences to fine-tune models, rewarding helpful and safe responses while penalizing undesirable ones. | A key method for aligning LLMs with human values, making them more helpful and less harmful. |
| **`AI Governance`** | The overarching framework of policies, laws, and ethical guidelines designed to manage the development, deployment, and use of AI systems. | Ensures AI systems are developed and used responsibly, preventing harm and promoting beneficial outcomes across society. |
| **`EU AI Act`** | A landmark legislative proposal by the European Union to regulate AI systems based on their potential risk level. | Establishes a pioneering, risk-based legal framework for AI, setting a global precedent for comprehensive AI regulation. |
| **`Semantic Consistency`** | The ability of a language model to produce semantically equivalent answers when presented with semantically equivalent questions or contexts. | Its absence indicates unreliable and unpredictable LLM behavior, especially in sensitive domains, undermining trust. |
| **`Rule of Thumb (ROT)`** | A fundamental judgment or guiding principle about right or wrong behavior, extracted from a question-answer pair. | Serves as a normalized, abstract representation of an LLM's moral stance, enabling consistent comparison across responses. |
| **`SaGE Score`** | An entropy-based metric, scaled between 0 and 1, to quantify the consistency of an LLM's responses in free text generation. | Provides a single, interpretable number where higher scores indicate greater consistency and lower scores indicate more randomness or contradiction. |

### Detailed Analysis

#### 1. Fine-tuning and Mitigating Risks in LLM Deployment

The first lesson, **Finetuning and Jailbreaking: Hands-on**, explores how large language models are adapted for specific tasks and the inherent risks involved.

* **Transformer Architecture and Fine-tuning Fundamentals**:
* The Transformer architecture, with its encoder, decoder, multi-head attention, and feed-forward networks, forms the backbone of modern LLMs, enabling parallel processing and complex language understanding.
* Fine-tuning adapts pre-trained models to "downstream tasks" by adjusting parameters, leveraging the broad language knowledge acquired during pre-training.
* Three main types of fine-tuning exist:
* **Head-level fine-tuning**: Updates only the final output layer, being the least computationally expensive and often the first approach.
* **Full model fine-tuning**: Updates all weights, which is computationally intensive and risks "catastrophic forgetting" of general language capabilities.
* **Layer-wise specific fine-tuning**: A middle ground, updating selected layers for domain-specific data without the full expense or risk of full fine-tuning.
* A practical example involves fine-tuning a GPT-2 model for Twitter sentiment analysis, including steps like dependency installation, data loading, tokenizer loading, model initialization, metrics definition, hyperparameter definition, fine-tuning execution, evaluation, and loss monitoring.

* **Parameter Efficient Fine-tuning (PEFT)**:
* PEFT techniques address the computational cost and time of full fine-tuning.
* **Adapters**: Small neural network components inserted into existing layers, trained while original model weights remain fixed. They introduce new information pathways efficiently.
* **LoRa (Low-Rank Adaptation)**: A PEFT technique that introduces trainable low-rank matrices to adjust pre-trained weight matrices, significantly reducing the number of trainable parameters and improving efficiency.

* **Instruction Fine-tuning and Jailbreaking**:
* **Instruction fine-tuning**: A specialized PEFT approach that trains LLMs to follow user instructions more accurately, making them suitable for interactive tools.
* This process often involves **quantization**, reducing weight precision (e.g., float32 to 8-bit integer) to deploy large models on resource-limited hardware like T4 GPUs, though it can cause some performance degradation.
* Using templates like the "Absolute Obedient Agent" (AOA) during instruction fine-tuning can lead to **jailbreaking**. If a system prompt instructs absolute obedience, a model might generate harmful content in response to a malicious user query, overriding its safety alignments. This highlights an "alignment problem" where helpfulness can compromise safety.

* **Safety Alignment and RLHF**:
* To counter jailbreaking, **safety alignment** techniques are crucial.
* **Reinforcement Learning from Human Feedback (RLHF)** is a key method, using human preferences to fine-tune models, rewarding helpful and safe responses.
* **PPO (Proximal Policy Optimization)** is an algorithm used in RLHF to optimize a model's policy. It compares the active model's output distribution with a reference model (e.g., a classifier) and assigns rewards based on divergence, iteratively teaching the model to generate helpful and safe outputs.

#### 2. Establishing Frameworks for AI Governance

The second lesson, **AI Governance**, shifts focus to the broader policy, legal, and ethical considerations surrounding AI.

* **Evolving Approaches to AI Safety and Control**:
* AI safety draws parallels with traditional computer security, using:
* **Training Users**: Educating users about AI capabilities and limitations, such as OpenAI's FAQs on data usage.
* **Interface-Level Warnings**: Direct prompts within AI systems, like "Do not share sensitive information," to guide user behavior.
* **Silently Eliminating Threats (Governance & Regulation)**: Systemic solutions that operate in the background, like email spam filters, to prevent harmful AI applications or ensure safe development practices.

* **Foundational Principles Guiding AI Governance**:
* **Right to Erasure (Right to be Forgotten)**: A legal principle granting individuals the right to have their personal data deleted, influencing data management in AI systems.
* **US White House Executive Order on AI**: Outlines objectives for AI development, including safety, privacy, equity, and innovation, with directives like disclosure for foundational models.
* **OECD AI Principles**: A comprehensive international framework for trustworthy AI, emphasizing:
* Inclusive Growth, Sustainable Development, and Well-being.
* Human Rights and Democratic Values, including non-discrimination, equality, and human agency.
* Transparency and Explainability.
* Robustness, Security, and Safety.
* Accountability for AI actors.
* Investment in AI Research and Development.
* Fostering an Inclusive AI-Enabling Systems.
* Shaping Interoperable Governance and Policy Environment.
* Building Human Capacity.
* International Cooperation.

* **The EU AI Act: A Pioneering Regulatory Framework**:
* A landmark legislation adopting a risk-based approach to regulate AI systems within the EU.
* **Risk-Based Classification**:
* **Unacceptable Risk AI Systems**: Banned due to severe threats (e.g., cognitive behavioral manipulation, social scoring, real-time remote biometric identification).
* **High Risk AI Systems**: Subject to strict requirements due to potential impact on safety and fundamental rights (e.g., critical infrastructure, education, employment, law enforcement). These must be registered in an EU database.
* **Transparency Requirements**: Mandates disclosure for AI-generated content, prevention of illegal content, and summaries of copyrighted training data.
* **Stakeholders**: Defines roles and responsibilities for Providers, Deployers, Manufacturers, Importers, Distributors, and Representatives.

* **Global Efforts in Establishing AI Safety Institutes**:
* Nations are establishing AI Safety Institutes (e.g., UK, US) to conduct research, monitor AI development, evaluate risks, and inform policy.
* International cooperation, as seen at the Seoul summit, highlights a global consensus on coordinated efforts.
* India is also actively engaged in AI governance discussions, with plans for an AI Safety Institute.

* **Unresolved Questions in AI Governance**:
* Challenges include ensuring cross-sector and cross-geographical applicability of policies.
* Debates persist on whether governance should target the AI system itself or its specific applications and human users.

#### 3. Quantifying Moral Consistency in LLMs with SaGE

The final lesson, **Research Overview: SaGE- Quantifying moral consistency in LLMs**, addresses the critical issue of inconsistency in LLMs, particularly in moral reasoning.

* **The Pervasive Problem of LLM Inconsistency**:
* LLMs frequently provide contradictory answers to semantically equivalent questions phrased differently, posing a significant safety risk and eroding user trust.
* This lack of **semantic consistency** means LLMs often fail to make consistent decisions in semantically equivalent contexts.

* **The SaGE Pipeline: A Framework for Quantifying Consistency**:
* **Paraphrase Generation**: Generates multiple semantically equivalent versions of moral questions using models like Vicuna, ensuring quality with a **ParaScore** metric (semantic similarity and lexical divergence).
* **LLM Response Generation**: The target LLM generates responses for each paraphrased question.
* **Rule of Thumb (ROT) Generation**: A separate Vicuna model extracts the underlying moral principles (ROTs) from question-answer pairs, abstracting fundamental judgments.
* **Semantic Graph Construction**: ROTs are converted into semantic embeddings using a fine-tuned SBIRT-DeBIRTA model. These embeddings form nodes in a semantic graph, where cosine distance represents similarity, visually representing the distribution of moral judgments.
* **SaGE Score Calculation**: An entropy-based metric, the **SaGE score**, quantifies consistency (0-1). Low entropy (tightly clustered ROTs) indicates high consistency, while high entropy (scattered ROTs) indicates inconsistency.

* **Validating the SaGE Metric**:
* Human annotation experiments showed that the SaGE score correlated significantly better with human judgments of consistency than traditional lexical or semantic embedding metrics.
* Using extracted ROTs for semantic graph construction proved more effective than using raw LLM answers.

* **Key Findings on LLM Consistency**:
* **Pervasive Inconsistency**: Current state-of-the-art LLMs (GPT-3.5, GPT-4, Llama-2, Mistral, Falcon) exhibit surprisingly low consistency scores, even Llama-2 achieved only 0.575.
* **Consistency is Intrinsic**: Consistency is largely independent of extrinsic factors like the `temperature` parameter, suggesting it's an intrinsic property of the model's architecture and training.
* **No Correlation Between Accuracy and Consistency**: High accuracy on benchmarks like Truthful QA or HeLaSwag does not guarantee consistent behavior, challenging the sole reliance on accuracy metrics for trustworthiness.
* **Improving Consistency with Rules**: Providing LLMs with explicit "rules of thumb" within the instruction prompt significantly increased their consistency, suggesting strategies like Retrieval-Augmented Generation (RAG) could be beneficial.

### Key Takeaways
- [ ] Fine-tuning adapts pre-trained LLMs to specific tasks, with head-level fine-tuning being the most efficient initial approach.
- [ ] Parameter Efficient Fine-tuning (PEFT) techniques like Adapters and LoRa significantly reduce the computational cost and memory footprint of fine-tuning.
- [ ] Instruction fine-tuning can lead to "jailbreaking" if models are trained to prioritize obedience over safety alignments, highlighting the alignment problem.
- [ ] Reinforcement Learning from Human Feedback (RLHF), often using PPO, is crucial for aligning LLMs with human values to ensure helpful and safe behavior.
- [ ] AI governance encompasses policies, laws, and ethical guidelines, moving beyond technical safety to ensure responsible AI development and deployment.
- [ ] The EU AI Act is a pioneering risk-based regulatory framework, categorizing AI systems by their potential for harm and setting global precedents.
- [ ] LLMs frequently exhibit semantic inconsistency, providing contradictory answers to semantically equivalent questions, which poses a significant safety risk.
- [ ] The SaGE framework provides a novel, entropy-based metric to quantify LLM consistency, demonstrating that current LLMs are surprisingly inconsistent.
- [ ] LLM consistency is an intrinsic property, largely independent of factors like `temperature`, and does not correlate with accuracy on standard benchmarks.
- [ ] Providing LLMs with explicit rules or guiding principles can significantly improve their consistency, suggesting avenues for future development.

### Conclusion

This week's exploration underscores the multifaceted nature of deploying AI responsibly. From the technical intricacies of fine-tuning models for specialized tasks to the critical need for robust governance frameworks and the fundamental challenge of ensuring semantic consistency, the journey towards trustworthy AI is continuous and collaborative. The ability to efficiently adapt LLMs through techniques like PEFT and quantization unlocks immense potential across industries, yet it simultaneously demands vigilance against vulnerabilities like jailbreaking. The emergence of comprehensive regulatory efforts like the EU AI Act and the establishment of global AI Safety Institutes signify a collective commitment to steering AI development towards beneficial outcomes. Furthermore, research like the SaGE framework provides essential tools to quantify and address the often-overlooked issue of LLM inconsistency, pushing us beyond mere accuracy metrics to a deeper understanding of reliability. Ultimately, the future of AI hinges on our ability to integrate advanced technical capabilities with strong ethical principles and effective governance, ensuring that these powerful technologies serve humanity's best interests with predictability, safety, and trust.