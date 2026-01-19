# LLM Rule Constrainer 📖

## Introduction

Large Language Models (LLMs) are typically evaluated on instruction following under explicitly specified and task-relevant constraints. In real-world applications, however, models are often exposed to _large collections of generic constraints_, only a small subset of which is actually relevant to the task at hand. Correct behavior therefore requires not only satisfying constraints, but **identifying which constraints are binding**.

This repository accompanies the paper _“Large-Scale Constraint Generation: Can LLMs Parse Hundreds of Constraints?”_, which introduces **Large-Scale Constraint Generation (LSCG)** - a setting in which an LLM must autonomously determine the relevant constraints from a large pool of candidates and then perform the task while respecting only those constraints.

To study LSCG, we introduce two tasks:

- **Word Checker**, a controlled classification task that isolates constraint identification by requiring detection of violations from long lists of forbidden words.
- **Language Moderator**, a constrained generation task that requires rewriting a sentence to preserve its meaning while avoiding all applicable forbidden words.

Our experiments show that LLM performance degrades sharply as the number of candidate constraints increases, and that scaling models or applying standard test-time prompting strategies is insufficient to address this issue. To mitigate this failure mode, we propose **FoCusNet (Focused Constraints Net)**, a lightweight auxiliary model that filters large constraint pools to a small set of likely relevant candidates prior to LLM inference, substantially improving robustness under large-scale constraints.

## How to

First, install the project environment via:

```bash
conda create -n rule_constrainer python=3.10
```

Now, with the `base` environment on (run `conda deactivate` otherwise), run:

```bash
chmod +x set_PYTHONPATH.sh
./set_PYTHONPATH.sh
```

This script will make sure that your python interpreter can succesfully decode the structure of the project and succesfully import the python modules.

Eventually, activate the environment and install the dependencies:

```bash
conda activate rule_constrainer
pip install -r requirements.txt
```

You are now ready to run the experiments!

## Repo Structure

The repository is structured as follows:

- `core/`: Contains the main codebase for implementing LSCG tasks, models, and evaluation metrics.
- `experiments/`: Scripts for running experiments, including data preparation, model training, and evaluation.

To reproduce the experiments from the paper, please refer to the `experiments/` directory for detailed instructions and scripts. A numerical identifier for each subfolder indicates the order in which the experiments must be conducted. Follow the README files within each subdirectory for specific guidance on running different experiments.
