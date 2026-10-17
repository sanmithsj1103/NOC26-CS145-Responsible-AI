# Week 5

### Introduction

The rapid advancement of artificial intelligence, particularly in large language models (LLMs) and vision-language models (VLMs), has brought unprecedented capabilities but also significant ethical challenges. As these models learn from vast datasets reflecting real-world human communication and imagery, they inevitably absorb and can perpetuate societal biases, stereotypes, and prejudices. This week, we delve into the critical and multifaceted problem of AI bias, exploring how it manifests, how it can be systematically measured, and the innovative techniques researchers are developing to mitigate its harmful effects.

Understanding and addressing bias is not merely an academic exercise; it has profound real-world implications. Biased AI systems can lead to unfair outcomes in critical applications, from misrepresenting diverse populations in generated content to perpetuating discrimination in hiring or legal contexts. This week's lessons provide a comprehensive overview, moving from theoretical frameworks and benchmark datasets to practical, hands-on methods for detecting bias in textual data, and finally, examining the unique challenges posed by bias in multimodal visual systems. The goal is to equip students and practitioners with the knowledge to critically evaluate AI systems and contribute to the development of more equitable, reliable, and trustworthy AI.

### Core Concepts

| Term | Definition | Significance |
| :--- | :--- | :--- |
| **`Stereotype Score (SS)`** | A metric measuring the percentage of instances where a model prefers a stereotypical association over an anti-stereotypical one. | An ideal SS of 50% indicates neutrality, meaning the model shows no preference for either stereotypical or anti-stereotypical associations. |
| **`Pseudo Log Likelihood`** | A method to estimate the probability of a sentence by iteratively masking and predicting each token, then summing their log likelihoods. | Used in benchmarks like CrowdSpare to quantify how strongly a model associates with a given sentence, revealing bias. |
| **`Guardrails`** | Programmable controls or rules implemented in LLMs to prevent them from generating undesirable or biased content. | Essential for ensuring responsible AI behavior, though they can sometimes misinterpret user intent due to lack of context. |
| **`Co-Bias Metric`** | Context-Oriented Bias Indicator and Assessment Score, a metric derived from Context Variance, designed to measure the contextual reliability of a statement. | Provides a standardized, bounded score (0-1) to assess how reliably bias can be measured in a statement, with higher scores indicating better contextual reliability. |
| **`Gender-Bleached Input`** | A technique where human subjects in images are replaced with gender-neutral representations, such as robots, to isolate and study bias without human gender cues. | Helps researchers determine if bias stems from visual gender depiction or from the model's learned associations with professions or actions. |
| **`Pointwise Mutual Information (PMI)`** | A statistical measure that quantifies the association between two events or words, indicating how much more frequently they co-occur than would be expected by chance. | A powerful tool for uncovering implicit word associations and stereotypes within a corpus, ideal for detecting subtle biases. |
| **`Identity Labels`** | Specific terms or categories (e.g., "men," "women," "doctor," "nurse") used to represent different social groups or roles within a dataset. | These labels are the focal points of a bias audit, allowing researchers to investigate how different groups are represented and associated with other words. |

### Detailed Analysis

#### 1. Measuring and Mitigating Textual Bias: Benchmarks and Automated Approaches

The journey to combat bias in AI begins with robust measurement. In **Bias - III**, we explored several methodologies for quantifying and addressing bias in language models.
* **StereoSet** is a benchmark dataset designed to quantify stereotypical bias using Context Association Tests (CATs).
* It evaluates models based on three scores:
* **Language Modeling Score (LMS)**: Measures the model's ability to distinguish meaningful associations from meaningless ones (ideal 100%).
* **Stereotype Score (SS)**: Quantifies preference for stereotypical over anti-stereotypical associations (ideal 50% for neutrality).
* **Idealized CAT (ICAT)**: Combines LMS and SS, with an ideal score of 100 for a proficient and unbiased model.
* **CrowdSpare** is another challenge dataset focusing on social biases in masked language models, particularly for protected demographic groups.
* It uses pairs of minimally different sentences, one more stereotyping and one less so.
* Bias is quantified by comparing the **conditional pseudo log likelihood** of these sentence pairs, where higher likelihood for the stereotyping sentence indicates stronger bias.
* **Auto-de-bias** presents an automated method for bias mitigation.
* **Stage 1 (Maximizing Disagreement)**: Automatically identifies "discriminative prompts" that elicit the highest disagreement in masked token predictions across demographic groups (e.g., "The ____ was a good manager" showing different predictions for "he" vs. "she").
* **Stage 2 (Minimizing Disagreement)**: Fine-tunes the model using an "equalizing loss" to minimize the disagreement, making prediction distributions similar regardless of demographic words.
* Bias can also manifest in complex linguistic tasks like **Coreference Resolution**.
* Studies show gender bias where systems correctly resolve coreferences for male/neutral pronouns (e.g., "his," "their") with gender-neutral professions like "surgeon," but often fail or misinterpret "her," implicitly associating professions with male gender.

#### 2. Advanced Benchmarking, Contextual Nuance, and Architectural Solutions

Building on foundational measurement, **Bias - IV** delved into more sophisticated benchmarks, the critical role of context, and architectural innovations for de-biasing.
* **Benchmarking Co-reference and Generation Bias**:
* The **Win-Bias Dataset** evaluates gender bias in co-reference resolution using gender-balanced pro- and anti-stereotypical examples (e.g., "physician" with "he" vs. "she").
* The **BOLD (Bias in Open-ended Language Generation Dataset)**, with over 23,000 prompts across five domains (profession, gender, race, religion, political ideology), benchmarks biases in generative AI.
* **Reddit Bias** targets conversational language models, providing a unique dataset for studying bias in dialogue systems.
* **Addressing Performance Degradation and Benchmark Pitfalls**:
* De-biasing efforts can sometimes degrade model performance ("catastrophic forgetting"). Research explores fine-tuning less than 1% of parameters to reduce bias without compromising task performance.
* Critiques of benchmarks highlight issues like unclear articulation of measured bias, problematic target groups (e.g., "Norwegian salmon"), and ambiguous expectations for model behavior.
* **The Crucial Role of Context and Guardrails**:
* LLMs use "guardrails" to prevent harmful outputs, but these can misinterpret user intent due to a lack of context (e.g., blocking "I don't like women" without knowing the user's full context "because I'm gay").
* **Nemo Guardrails** is an open-source toolkit allowing customizable guardrails to control LLM output, emphasizing the need for context-aware responses.
* **Quantifying Contextual Bias with Co-Bias**:
* The **Co-Bias (Context-Oriented Bias Indicator and Assessment Score)** metric systematically evaluates the impact of context.
* It involves identifying context points, generating context-added versions of statements, calculating a `tau` score (log-likelihood of a statement), and then assessing **Context Variance (`Cv(X)`)**.
* `Cv(X)` measures the percentage variance in `tau` scores of context-added versions from the population mean, indicating contextual reliability. Co-Bias transforms `Cv(X)` to a 0-1 scale, where higher scores mean better reliability.
* **Multi-Adapter Architectures for Inclusive Language Models**:
* **Counterfactual Data Augmentation (CDA)**, while useful, is labor-intensive. **IDEB** (Inclusive Data for Bias Evaluation) automates CDA using Wikidata and generative models, though LLM-generated counterfactuals can sometimes introduce new biases.
* **MAFIA (Multiple Adapter Fused Inclusive Language Models)** uses multiple de-biasing adapters, specialized for different bias types (gender, race, religion), fused within the model architecture. This approach aims to overcome performance degradation and address the complex interplay of biases, showing lower bias scores on benchmarks like STS-B.

#### 3. Bias in Vision Language Models (VLMs)

**Bias in VLM's** highlighted how biases manifest in multimodal AI systems that process both visual and linguistic information.
* **Manifestations of Bias**:
* **Geographical and Cultural Bias**: VLMs often produce stereotypical images. For "an Indian person," models might consistently generate images resembling Sadhus, failing to capture diversity. Similarly, "house in India" might show an idealized or stereotypical dwelling, not representative of typical housing.
* **Professional Bias**: Models exhibit bias when identifying professions, especially with limited context. An image of a robot "baking bread" might be predicted as "female," while "welding a metal structure" as "male," even with gender-neutral inputs.
* **Measuring Bias in Image-to-Text Models**:
* Researchers use "informed prompts" (e.g., "predict the gender of the person doing [action]") versus "blind prompts" (e.g., "predict the gender of a person") to isolate contextual impact.
* **Gender-bleached input** is a key technique where human subjects are replaced with gender-neutral robots to study bias related to professions and actions without confounding human gender cues.
* The process involves:
* Selecting professions from official statistics.
* Using GPT-4 to generate 2-3 gender-neutral, simple-to-sketch actions for each profession.
* Validating prompts by asking GPT-4 to identify the profession from the action.
* **Bias in Any-to-Any Models**:
* VLMs exhibit varying biases across different social attributes (gender, race, age) and model types (e.g., GPT-4 Vision, DALL-E).
* Proprietary models tend to be more "neutral" than open-source ones.
* Bias can be direction-dependent (e.g., CODI's racial bias differs between image-to-text and any-to-image).
* Bias often increases in cross-modal settings compared to single-modality tasks.
* **Open Challenges and Cultural Relevance**:
* Measuring and de-biasing VLMs are complex.
* A critical challenge is ensuring **cultural relevance**, as models often lack understanding of diverse cultural contexts, leading to potentially offensive or unrepresentative outputs.

#### 4. Hands-on Bias Detection: Auditing Textual Datasets (Part 1)

In **Bias Handon Part 1**, we began a practical exploration of auditing NLP datasets for social biases, focusing on the SNLI (Stanford Natural Language Inference) corpus.
* **Natural Language Inference (NLI)**: This fundamental NLP task determines the semantic relationship (entailment, contradiction, neutral) between a "premise" and a "hypothesis."
* **SNLI Corpus**: A large-scale, crowdsourced dataset of human-written English sentence pairs.
* Its two-stage crowdsourcing process (premises from Flickr 30k, hypotheses generated by crowd workers) is a prime source for unconscious bias.
* **Measuring Bias with Pointwise Mutual Information (PMI)**:
* PMI quantifies the association between two words, indicating how much more frequently they co-occur than expected by chance. High PMI suggests strong, non-random association, potentially revealing stereotypes.
* The formula for PMI between words *i* and *j* is:
```
PMI(i, j) = log2 ( P(i, j) / (P(i) * P(j)) )
```
* **Preprocessing steps** are crucial for accurate PMI: lowercasing, stop word removal, tokenization, and frequency thresholding.
* The goal is to calculate PMI scores for "identity labels" (e.g., "men," "women") against other words in the corpus to uncover stereotypical associations.

#### 5. Hands-on Bias Detection: Practical Application (Part 2)

**Bias Handon Part 2** provided a detailed walkthrough of implementing the bias detection methodology using PMI.
* **Data Loading and Initial Preparation**:
* The SNLI 1.0 train dataset is loaded.
* A predefined list of "identity labels" (gender, ethnicity, etc.) is used as reference.
* **Deduplication** is performed to ensure accurate statistical calculations.
* **Text Preprocessing**:
* **Stop Word Removal**: Common words are removed using NLTK's stop words list.
```python
import nltk
from nltk.corpus import stopwords
import string

def remove_stopwords_and_punctuation(sentence):
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
tokens = sentence.lower().split() # Lowercase and split into tokens
filtered_tokens = [
word for word in tokens
if word not in stop_words and all(char not in punctuations for char in word)
]
return " ".join(filtered_tokens)
```
* **Punctuation Removal** and **Lowercasing** standardize the text.
* **Tokenization** breaks sentences into individual words.
* **Computing Unigram and Bigram Probabilities**:
* **Unigram Probabilities**: Frequencies of unique words are counted. Words below a frequency threshold (e.g., 15) are discarded.
```python
from collections import Counter

def compute_unigram_frequencies(corpus_tokens):
all_words = [word for sentence_tokens in corpus_tokens for word in sentence_tokens]
unigram_counts = Counter(all_words)
# Filter out low-frequency words
filtered_unigram_counts = {word: count for word, count in unigram_counts.items() if count >= 15}
return filtered_unigram_counts
```
* **Bigram Probabilities**: Frequencies of two-word sequences are counted, treating SNLI's premise and hypothesis sentences independently. Bigram pairs below a frequency threshold (e.g., 10) are discarded.
```python
def compute_bigram_frequencies(corpus_tokens):
bigram_counts = Counter()
for sentence_tokens in corpus_tokens:
for i in range(len(sentence_tokens) - 1):
bigram_counts[(sentence_tokens[i], sentence_tokens[i+1])] += 1
# Filter out low-frequency bigrams
filtered_bigram_counts = {bigram: count for bigram, count in bigram_counts.items() if count >= 10}
return filtered_bigram_counts
```
* **Calculating PMI**: The PMI function computes the score between two words using their unigram and bigram probabilities.
```python
import math

def calculate_pmi(word1, word2, unigram_freq, bigram_freq, total_unigrams, total_bigrams):
freq_w1 = unigram_freq.get(word1, 0)
freq_w2 = unigram_freq.get(word2, 0)
freq_w1_w2 = bigram_freq.get((word1, word2), 0)

if freq_w1 == 0 or freq_w2 == 0 or freq_w1_w2 == 0:
return 0 # Or a very low negative number to indicate no co-occurrence

p_w1 = freq_w1 / total_unigrams
p_w2 = freq_w2 / total_unigrams
p_w1_w2 = freq_w1_w2 / total_bigrams

if p_w1 == 0 or p_w2 == 0: # Avoid division by zero
return 0

pmi = math.log2(p_w1_w2 / (p_w1 * p_w2))
return pmi
```
* **Detecting Bias through PMI Analysis**:
* By calculating PMI between identity labels (e.g., "women," "men") and all other unigrams, and then sorting, clear stereotypes emerge.
* Examples: "gossiping" and "bikinis" associated with "women"; "refueling" and "arguing" with "men."
* Unique associations (e.g., "cheerleading" for "girls," "wrestle" for "boys") further highlight stereotypical representations.
* Analysis also extends to LGBT identity terms, showing associations like "pride" and "rights." This demonstrates how human-generated data perpetuates societal biases, emphasizing the need for careful crowdsourcing instructions and proactive mitigation.

### Key Takeaways
- [ ] Bias in language and vision models is a significant problem stemming from real-world data and requires systematic measurement and mitigation.
- [ ] Benchmarks like StereoSet and CrowdSpare quantify textual bias using metrics like Stereotype Score (SS) and pseudo log likelihood.
- [ ] Automated de-biasing methods like Auto-de-bias identify and correct biases by maximizing and then minimizing disagreement in model predictions.
- [ ] The absence of sufficient context is a major contributor to AI bias, and guardrails, while necessary, can misinterpret user intent.
- [ ] The Co-Bias metric quantifies the contextual reliability of statements, helping to assess how much a statement's interpretation changes with added context.
- [ ] Vision Language Models (VLMs) exhibit geographical, cultural, gender, and professional biases, often revealed through "gender-bleached inputs" and "informed" vs. "blind" prompts.
- [ ] Pointwise Mutual Information (PMI) is a powerful statistical tool for uncovering implicit word associations and stereotypes in textual datasets.
- [ ] Text preprocessing (deduplication, stop word removal, lowercasing, tokenization) is crucial for accurate bias analysis using PMI.
- [ ] Human-generated and crowdsourced datasets inherently contain and perpetuate societal biases, necessitating continuous auditing and improved data collection guidelines.
- [ ] De-biasing efforts must strive to reduce bias without degrading the model's overall performance, and multi-adapter architectures like MAFIA offer promising solutions.

### Conclusion

This week's exploration of bias in AI underscores a fundamental truth: the intelligence we build is a reflection of the data we feed it. From the subtle linguistic associations uncovered by PMI in textual datasets like SNLI to the overt stereotypical imagery generated by VLMs, the pervasive nature of bias demands our continuous attention. The methodologies discussed, ranging from sophisticated benchmarks like StereoSet and CrowdSpare to practical hands-on techniques for data auditing, provide essential tools for identifying and quantifying these biases.

However, identification is merely the first step. The development of automated de-biasing methods like Auto-de-bias and architectural innovations such as MAFIA represent significant strides towards mitigating these issues. Crucially, the emphasis on context, highlighted by the Co-Bias metric and the challenges faced by guardrails, points to a future where AI systems must not only be powerful but also deeply nuanced in their understanding of human communication and culture. As AI becomes increasingly integrated into education, healthcare, and daily life, ensuring its fairness, inclusivity, and cultural relevance is paramount. The ongoing research and practical applications discussed this week are vital contributions to building AI that serves all of humanity equitably and responsibly.