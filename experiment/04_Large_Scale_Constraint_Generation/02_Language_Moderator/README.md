# Language Moderator

This directory contains the code to run the Language Moderator task. In Language Moderator, the model is given a sentence and an increasingly large set of candidate words. The model must generate a moderated version of the sentence by replacing any inappropriate words from the candidate set. If the sentence is already appropriate, the model should return it unchanged.

## How to run

In order to deploy the LLM for inference, refer to the subfolders under `./00_launch_<model>/`, where `<model>' is the chosen LLM. For instance, `./00_launch_deepseek8B/` contains the code to launch the DeepSeek 8B model.

Inside, you'll find `.sbatch' files that can be submitted to a SLURM scheduler to start the model server. The scheduler uses *slang* to manage the deployment of the model, and then continues to run the inference code. Remember that you can give a name to the job by modifying the `--job-name`parameter in the`.sbatch`files or by passing the`--job-name`argument when submitting the job with`sbatch`. For instance:

```bash
sbatch --job-name=deepseek8B_simple_prompt 00_launch_deepseek8B/01_launch_OneShot.sbatch
```
