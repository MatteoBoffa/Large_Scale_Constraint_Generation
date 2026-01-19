# WordNet Semantic Grounding

This directory contains the code and resources for implementing WordNet-based semantic grounding in our experiments. Semantic grounding is the process of linking words to their meanings in a structured knowledge base, such as WordNet.

We assess both generic and specific groundings. Particularly, in the generic grunding we focus on the hypernym relationships in WordNet to establish broader category connections. On the other hand, in the specific grounding we directly link a word to their definitions in WordNet, providing a more precise semantic context.

In both cases, the model:

i) Encodes the definition (generic or specific)
ii) Encodes a pool of 100 words, only 1 being the correct grounding
iii) Computes similarity scores between the definition and each word in the pool

We compute MRI metrics to evaluate the model's ability to correctly identify the grounded word from the pool based on the computed similarity scores.

## How to run

To run the WordNet Semantic Grounding experiments, refer to the scripts located in `01_generic_definitions' and `02_specific_definitions`directories. Each directory contains the necessary code to perform the respective grounding tasks. For instance, to run FocusNet with generic definitions, navigate to the `01_generic_definitions`directory and execute`01_default_focusnet'.
