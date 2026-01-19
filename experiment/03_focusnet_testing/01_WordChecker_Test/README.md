# Word Checker Test

This folder assesses whether the previously trained models can identify whether a sentence contains a forbidden English word or not.

Remember that containing a word does not mean exact match; for example, the word "run" is considered contained in the sentence "He is running fast."

**Important**: to run these tests, ensure that you have already trained the models by executing the scripts in the `02_focusnet_tuning` folder.

Remember that, as we design WordChecker as a task to identify forbidden words, a higher recall is more important than a higher precision (i.e., we prefer to have more false positives than false negatives).

## Input and Output

In this test, we give the model a sentence and **a forbidden word**. The model must output whether the sentence contains the forbidden word or not.
Particularly, the model will output:

- _IsCompliant_: a boolean indicating whether the sentence contains the forbidden word.
- _Score_: a float between 0 and 1 indicating the model's confidence in its prediction.

## How to run the tests

The folder contains scripts to run 5 different models:

- Default FocusNet: contrastive loss, trained on WordNet and CommonGen datasets, using attention, with a Random Forest classifier.
- KNN FocusNet: contrastive loss, trained on WordNet and CommonGen datasets, using attention, with a KNN classifier.
- No WordNet FocusNet: contrastive loss, trained only on CommonGen dataset, using attention, with a Random Forest classifier.
- No Attention FocusNet: contrastive loss, trained on WordNet and CommonGen datasets, without using attention, with a Random Forest classifier.
- No Contrastive FocusNet: trained on WordNet and CommonGen datasets, using attention, with a Random Forest classifier.
- NLP Baseline: a simple NLP-based baseline that checks for the presence of the forbidden word using lemmatization.

To run the tests for each model, execute the corresponding script. For instance, to run the Default FocusNet test, use the following command:

```bash
01_default_focusnet.sh
```
