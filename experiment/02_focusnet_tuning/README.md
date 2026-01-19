# HOW TO RUN

In the folder `scripts`, we provide several bash scripts to train the default version of FocusNet and its ablations.

Particularly:

- `scripts/01_default_focusnet.sh`: trains the default FocusNet model (attention, contrastive loss, training on CommonGen and WordNet, Random Forest as classifier).
- `scripts/02_ablation_knn_head.sh`: trains an ablation version of FocusNet that uses a KNN head instead of a Random Forest classifier.
- `scripts/03_ablation_no_wordnet.sh`: trains an ablation version of FocusNet that does not use WordNet data during training (i.e., less semantic knowledge).
- `scripts/04_ablation_no_attention.sh`: trains an ablation version of FocusNet that does not use attention (i.e., it focuses on one rule at a time).
- `scripts/05_ablation_no_contrastive_loss.sh`: trains an ablation version of FocusNet that does not use contrastive loss during training (i.e., frozen encoder, classifier trained on top of fixed embeddings).
