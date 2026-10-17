# Week 4

### Introduction

In an increasingly AI-driven world, where algorithms influence everything from job applications to medical diagnoses, understanding and addressing bias is not just an academic exercise—it's a societal imperative. Imagine an AI system designed to recommend careers, subtly steering women towards nursing and men towards dentistry, or a translation tool automatically assigning gender to a neutral phrase based on stereotypical professions. These aren't far-fetched scenarios; they are real-world manifestations of bias embedded within artificial intelligence, particularly in Generative AI models like large language models (LLMs) and vision models. This week's module delves into the subtle yet pervasive ways biases can manifest in AI outputs, from image generation to text translation, and even in chatbot responses.

The implications of unchecked bias are profound and far-reaching. A student relying on an AI for homework might unknowingly perpetuate societal prejudices, or a professional using a translation tool could find meanings subtly altered by gender stereotypes. This module aims to equip you with the critical tools to recognize these issues, understand their origins, and appreciate the ongoing research dedicated to mitigating them. We will examine real-world examples, discuss various categories of bias, and consider how to define and address bias when it leads to negative impacts, ensuring that AI serves all of humanity fairly and equitably.

### Core Concepts

| Term | Definition | Significance |
| :--- | :--- | :--- |
| **`Bias (ML)`** | A systematic favoritism or discrimination towards certain groups or outcomes within a machine learning system, often leading to unfair or inaccurate results. | Crucial for understanding why models might produce unfair or stereotypical results, impacting their ethical deployment and societal trust. |
| **`Guardrails`** | Mechanisms implemented in AI systems to prevent the generation of harmful, offensive, or policy-violating content. | Essential for safety and ethical AI deployment, though their implementation can sometimes mask underlying biases or be bypassed. |
| **`RLHF`** | Reinforcement Learning from Human Feedback, a technique where human preferences are used to fine-tune AI models. | A key method for aligning AI behavior with human values, but also a potential pathway for human biases to enter models. |
| **`Negative Impact`** | The adverse consequences or harm caused by biased AI outputs, affecting individuals or groups. | The primary criterion for determining when bias becomes a "problem" that requires intervention and mitigation. |
| **`Data Representativeness`** | The extent to which a dataset accurately reflects the diversity and characteristics of the real-world population or phenomenon it aims to model. | Lack of representativeness is a primary source of bias, causing models to perform poorly or unfairly on underrepresented groups. |
| **`Annotator Beliefs`** | The personal opinions, cultural backgrounds, and inherent biases of individuals responsible for labeling or categorizing data. | Human biases in labeling directly transfer to the training data, influencing the model's learned patterns and decisions. |
| **`CrowS-Pairs`** | A benchmark dataset and methodology for measuring model bias by comparing model probabilities on minimally different sentence pairs. | A key tool for quantifying the degree of bias in a model, enabling before-and-after comparisons for de-biasing efforts. |
| **`Auto-Debias`** | A framework designed to mitigate bias in models, often by minimizing the divergence between probability distributions for contrasting concepts. | A primary method used to reduce specific biases, such as gender bias, by making model predictions independent of target concept words. |
| **`Positionality`** | The social and political context that shapes an individual's or a system's perspective, influencing data collection, annotation, and model design. | Explains why AI models often reflect the biases of their creators and the dominant societal groups represented in their training data. |
| **`Feedback Loop`** | A cyclical process where user interactions and behaviors with a deployed AI system inform and influence subsequent data collection and model updates. | Can inadvertently reinforce existing biases if user behavior itself is biased or if the feedback mechanism is not carefully designed. |

### Detailed Analysis

#### 1. Initial Demonstrations and Categorization of Bias (from **Bias - I**)
The exploration of AI bias begins with compelling interactive examples that reveal how popular AI platforms inadvertently perpetuate societal stereotypes. In **Bias - I**, demonstrations with image generation (Meta AI via WhatsApp) showed that prompts like "Generate an image of a doctor" typically resulted in male doctors, while "Generate an image of a nurse" yielded female nurses. Similarly, "Generate a person driving a car" often produced a male driver in a Western setting. These examples immediately highlighted biases related to gender, geographical location, and age.

Text generation examples further illustrated this. When ChatGPT was prompted with "Bob gave Mike 30mg. What was given? Answer in one word," the response varied based on the perceived cultural origin of the names. "Bob and Mike" often resulted in "substance," while "Ram and Lakshman" frequently led to "drugs," suggesting a bias in how the AI associated names with different connotations. Another test showed gender bias in career recommendations: "Jack" choosing between nursing and dentistry often resulted in "dentistry," while "Jane" might receive "nursing." Machine translation (Google Translate, English to Hindi) also exhibited gender bias, automatically assigning masculine (`mera`) for "doctor" and feminine (`meri`) for "nurse" to the gender-neutral "my friend."

Formally, bias is defined in several ways: psychologically as a systematic deviation from rationality, statistically as a systematic error in data, and in computer science/machine learning as systematic favoritism or discrimination towards certain groups or outcomes. For practical purposes, bias is considered a "problem" only when it leads to a **Negative Impact**. Researchers categorize bias in Natural Language Generation (NLG) tasks into dimensions such as Gender, Profession, Race, Religion, Sexuality, Age, Nationality, Disability, Physical Appearance, and Socioeconomic Status. The **Krauss Pairs** dataset is introduced as a valuable tool for detecting bias, consisting of minimally distant sentence pairs that highlight specific bias categories (e.g., changing "African-American" to "white" in a stereotypical sentence).

#### 2. Demographic, Geographic, and Ethical Dimensions of Bias (from **Bias - II**)
Building on the initial demonstrations, **Bias - II** delves into how demographic and geographic factors influence bias perception and manifestation. The lecture emphasizes that what is considered biased can vary significantly across cultures; for instance, a statement about race might carry different weight in the U.S. versus India. This concept is tied to **Positionality**, where the social and political context of data annotators and researchers influences the biases embedded in AI models.

The **Perspective API**, a tool for detecting toxic content, serves as a prime example of demographic and geographic bias. It showed performance mismatches, where toxicity scores for phrases might differ for users from the U.S. versus India, suggesting the model was trained on datasets predominantly reflecting Western norms. AI models also exhibit subtle biases with gender stereotypes. While a prompt like "men are stronger than women" might receive a nuanced response, flipping it to "women are weaker than men" often triggers **Guardrailing** mechanisms, where the model explicitly identifies the statement as a harmful stereotype and refuses to propagate it. This highlights the model's sensitivity to phrasing and the implementation of safety protocols.

Real-world incidents underscore the pervasive nature of AI bias. The St. George's Hospital Medical School case showed historical human bias in admissions. Microsoft's Tay Chatbot quickly learned and amplified racist and misogynistic remarks from user interactions. The Allen Institute's Delphi, an ethical judgment AI, produced alarming biases, deeming "Secure the existence of our people and a future for white children" as "good" and "Black man walks into your store" as "bad." Predictive policing algorithms have shown racial bias, and vision models like MidJourney exhibit ageism, sexism, racial bias, classism, and urbanism bias in generated images. Regulatory responses, such as the European AI Act and New York City laws concerning AI in hiring, are emerging to address these issues. The "Coded Bias" documentary is recommended for its exploration of algorithmic bias in facial recognition.

#### 3. The AI/ML Pipeline as a Source of Bias (from **Source of Bias**)
The lesson **Source of Bias** meticulously breaks down how bias can infiltrate every stage of an AI/ML system's lifecycle. It's not a singular event but a pervasive risk.
* **Data Collection**: Bias can be introduced if the data is not truly **Data Representativeness** of all demographics or if the questions posed during collection are framed in a biased way.
* **Data Labeling/Annotation**: Human **Annotator Beliefs**, cultural contexts, and unconscious biases are encoded into the "ground truth" that the model learns from. For example, annotators from different regions might label images or texts differently.
* **Model Training**: If the input data is already biased, the model will inevitably learn and reflect these biases. Using inappropriate metrics, such as relying solely on accuracy on **Unbalanced Data**, can mask underlying biases, leading to poor performance for minority classes.
* **Model Deployment in Production**: Users might engage in **Jailbreaking** to manipulate the model, revealing latent biases or vulnerabilities.
* **User Perception**: Users might perceive an AI's output as biased, even if not statistically evident, influencing trust and adoption. Children exposed to biased AI outputs might internalize those biases.
* **Feedback Loop**: Continuous interaction between users and the AI can reinforce existing biases if user behavior itself is biased or if the feedback mechanism is not carefully managed.

Visual examples from vision models further illustrate this. Prompting for "an Indian person" often generates stereotypical images (e.g., a sadhu), failing to capture the vast diversity. Similarly, "a house in India" might yield only rural dwellings, ignoring modern urban architecture. These examples highlight the lack of data representativeness and the impact of cultural biases in training data.

#### 4. Hands-on Approach to Bias Detection and Mitigation (from **Bias-Handson**)
The **Bias-Handson** lesson provides a practical, four-step demonstration of how bias can be introduced, measured, and mitigated in LLMs. The focus is on **Gender Bias**, defined as a stereotypical inclination towards one gender when neutrality is expected.

The process begins with **1. Hacking a Dataset to Create Gender Bias**. The "We Know Gender Schemas" dataset, originally used for **Coreference Resolution**, is manipulated. Professions are extracted and manually associated with stereotypical genders (e.g., "secretary" with female, "plumber" with male). Sentences are then filtered to include only those where a pronoun refers to an occupation, and that occupation's gender aligns with the ground truth stereotype. This creates a dataset of 120 sentences exclusively reinforcing gender stereotypes.
```python
# Illustrative logic for filtering sentences (simplified)
new_gendered_bias_data = []
for index, row in original_dataframe.iterrows():
occupation = get_occupation_from_sentence_id(row['sentence_id'])
pronoun_refers_to_occupation = (row['pronoun_ref_id'] == 0) # 0 for occupation, 1 for participant
if pronoun_refers_to_occupation and ground_truth_gender[occupation] == get_gender_from_pronoun(row['pronoun']):
new_gendered_bias_data.append(row['sentence'])
```
Next, **2. Training a Biased Model**. A sentence transformer model (all-mini-LM) is fine-tuned using this biased dataset with **Contrastive Tension Loss**. This loss function encourages similar embeddings for positive examples (sentences from the biased dataset) and distant embeddings for negative examples (randomly sampled dissimilar sentences). After 10 epochs, a "biased model" is saved.

**3. Measuring Model Bias with CrowS-Pairs** follows. The original unbiased model yields a **CrowS-Pairs** metric score of 55.5. The newly trained biased model scores 59.28, indicating successful induction of gender bias. CrowS-Pairs uses **Pseudo Log Likelihood Scoring** to compare model probabilities on minimally different sentence pairs.

Finally, **4. De-biasing the Model with Auto-Debias**. The **Auto-Debias** framework is used to mitigate the induced bias. It constructs contrasting sentence pairs using "Target Concept Words" (e.g., "he" vs. "she") and minimizes the **GS Divergence Loss** between the probability distributions for masked words in these pairs. This encourages the model to assign equal probabilities regardless of gender. After one epoch of de-biasing, the GS divergence loss significantly decreases. The de-biased model is then re-evaluated with CrowS-Pairs, yielding a score of 47, demonstrating successful bias reduction.

The lesson also critiques bias benchmarks, mentioning **Microsoft Salmon Work** which found quality issues in crowdsourced datasets. **Co-Bias Work** from IIIT Hyderabad emphasizes the importance of context in assessing bias reliability. The discussion concludes by noting that bias extends beyond language models to text-to-image generation, perpetuating visual stereotypes (e.g., Mexican people with sombreros, Indian people with tilaks).

### Key Takeaways
- [ ] AI models, especially Generative AI, can inadvertently perpetuate societal biases and stereotypes in their outputs across various tasks.
- [ ] Bias can be introduced at every stage of the AI/ML pipeline, from data collection and labeling to model training, deployment, and user interaction.
- [ ] The "positionality" of data annotators and researchers significantly influences the biases embedded in AI training datasets.
- [ ] AI systems employ "guardrails" and RLHF to prevent harmful content, but these mechanisms can sometimes mask underlying biases or be bypassed by subtle prompt changes.
- [ ] For practical purposes, bias is considered a "problem" primarily when it leads to a negative impact on individuals or groups.
- [ ] Datasets like "Krauss Pairs" and benchmarks like "CrowS-Pairs" are crucial tools for systematically identifying, categorizing, and measuring specific types of bias.
- [ ] Frameworks like "Auto-Debias" can be used to mitigate induced biases, demonstrating a quantifiable reduction in bias scores.
- [ ] Real-world incidents and research highlight that AI bias manifests across modalities (text, image, legal) and can have significant societal consequences.
- [ ] Existing bias benchmark datasets, particularly crowdsourced ones, often have quality issues, emphasizing the need for more robust and context-aware evaluation metrics.
- [ ] Critical evaluation of AI outputs, especially those generalizing about people or cultures, is essential for responsible AI engagement.

### Conclusion

The comprehensive exploration of bias in AI this week reveals a critical challenge in the development and deployment of intelligent systems. From the subtle nuances of language interpretation across cultures to the overt discrimination observed in legal and visual AI, it is clear that algorithms are not neutral arbiters but rather reflections of the data they consume and the human biases embedded within. The examples discussed, ranging from the **Perspective API**'s geographic disparities to the alarming ethical judgments of Delphi and the representational biases in MidJourney, underscore the urgent need for vigilance and proactive measures.

Understanding these biases is the first step towards building more equitable and responsible AI systems. As users and developers, recognizing when an AI output is influenced by bias, especially when it leads to **Negative Impact**, is paramount. This foundational knowledge will be crucial as we delve deeper into metrics, datasets, and cutting-edge research aimed at detecting, measuring, and ultimately mitigating bias in AI. The emergence of regulations in areas like AI hiring signifies a growing societal recognition of these issues. Ultimately, the goal is to build AI systems that are not only intelligent but also fair, transparent, and accountable, ensuring that technological progress benefits all of humanity without perpetuating or amplifying existing inequalities. Your ability to identify and question these sources of bias is a foundational skill for contributing to a future where AI is a force for good.