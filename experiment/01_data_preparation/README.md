# Dataset Preparation

To obtain the datasets to i) train and validate FocusNet and ii) create WordChecker, run all the notebooks and scripts in this folder. Follow the order of the numbered subfolders.

Below, we provide a brief description of the datasets and the preparation process.

## Training FocusNet

To train FocusNet, we prepared datasets from CommonGen and WordNet.

### CommonGen

[CommonGen](https://arxiv.org/abs/1911.03705) is a open-source dataset originally released for generative commonsense reasoning.
It contains a set of rules (i.e., concepts that must be included in the generated sentence) and a sample sentence that respect those rules (i.e., contains all the concepts).
In this case, we simply augment the dataset. The intuition is that if a sentence is valid for a set of concepts, it is also valid for any subset of those concepts.
For instance, if the sentence "The cat sits on the mat" is valid for the concepts {cat, mat, sits}, then it is also valid for {cat, mat} or {cat, sits}.

**N.b.** We only use the training and validation set of CommonGen at this stage.

### WordNet

WordNet is a lexical database for the English language coming from the ntlk library.
Each word comes with a definition and a hypernym (word with a broad meaning constituting a category, e.g., "animal" is a hypernym of "dog").
We use CommonGen unique set of concepts as rules, and we use WordNet to extract the definitions and hypernyms of those concepts.
The intuition is that, while CommonGen provides examples of sentences that contain a set of concepts, WordNet provides definitions that describe the meaning of those concepts.
Each word in CommonGen is therefore mapped to its definition and its hypernym's definition from WordNet.

**N.b.** We use 90% of the unique concepts from WordNet for training/validation, and the remaining 10% for testing.

### CommonNet

CommonNet is a dataset that combines CommonGen and WordNet. It simply concatenates the information from both datasets, producing an output dataframe with the following columns:

- concepts: set of concepts (i.e., rules) associated with the sentence
- target: a sentence matching the concepts. This means i) containing the concepts and ii) being the definition of the concepts according to WordNet.

Len of the matching concepts varies from 1 to 5 (avg: 2, std: 1). There are 273,916 rows in total.

### Dataset Split

The dataset is split into three partitions: training, validation, and test. We will use these partitions to train our FocusNet model and evaluate its performance.

The idea here is:

- to obtain, from the original dataset, a training and testing partition with disjoint rules (i.e., no concepts in common between the two partitions)
- To divide the training set with a K-Fold split, to obtain different training and validation folds.

This is done to robustly assess the model performance and pick the best hyperparameters. In short:

- Each partition (e.g., train, validation, test) contains rules that are NOT present in the other partitions
- For test, we only use samples from CommonGen. This means, we only assess the model ability to recognize whether a sentence contains a set of concepts or not.
- We made sure that the division is fair in terms of labelled samples per rule: each partition contains rare (i.e., few labelled examples given the rule), common, and frequent rules

### Ablation Studies

- Single-Rule Training: to assess the model ability to learn the coupling between words and definitions, we created a training set where each sample contains only one rule (i.e., one concept). This is done by filtering the CommonNet training set to keep only samples with one concept. See folder `03_single_rules_train` for more details.

- Common-Gen only training: to verify the importance of coupling words and definitions, we created a training set using only CommonGen samples. See folder `02b_CommonGen_only` for more details.

## Validating FocusNet: can we couple words and definitions?

To validate FocusNet, use the remaining 10 % of the unique concepts from WordNet to create two datasets: one with specific definitions and one with generic definitions.
On both cases, we give the model a concept and 100 possible definitions (one correct and 99 incorrect) and we ask the model to select the correct definition.

Notice that, while in the previous test set (one from CommonGen) we only assessed the model ability to recognize whether a sentence contains a set of concepts or not, here we assess the model ability to match concepts with their definitions.

## Inference: WordChecker - LLM's robustness to increasing number of rules

To create WordChecker, and assess the robustness of LLMs to an increasing number of rules, we started from the test set of CommonGen - which we haven't used so far.

For each row in CommonGen's test set, we first decided whether the sample would contain the exact matching rule or not.
If yes, we kept the original set of concepts as the exact matching rule and added more random concepts to increase the number of rules.
If not, we created a new set of concepts by randomly sampling concepts that are not in the original set.
We created different scenarios with an increasing number of rules: from 5 to 1000 rules.

For a fair evaluation, we made sure that each scenario contains the same concepts as the previous one, plus some new ones. For instance, the 5-rules scenario is a subset of the 10-rules scenario, which is a subset of the 20-rules scenario, and so on.
