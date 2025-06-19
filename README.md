# Supplementary Material of "TestSmells-LLMs_EmpiricalStudy_ReplicationPackage" paper

# Overview

This repository contains the resources and results from a study that investigates the impact of refactoring performed by Large Language Models on test smells, which include GPT-4, Gemini 1.5 Pro and LLaMA 3 70B models. The study involved evaluating various prompt templates for detecting and refactoring test smells, analyzing the outcomes, assessing the impact on code coverage metrics, and documenting the entire process.

## Repository Structure

### Prompt Templates

The `prompt templates` folder includes templates used in projects Python and Java:
- **10 templates** for **test smell detection**
- **10 templates** for **test smell refactoring**
>All text files. The prompt versions were numbered from version 0 to 4, followed by the abbreviation of the prompt techniques used

These templates were used to guide the analysis and refactoring processes.

### Dataset

The `dataset` folder contains the URLs of the repositories used in this study, which served as the basis for evaluating the effectiveness of the refactoring techniques. It includes:
- An `.xlsx` file with structured data.

### Results

The `results` folder contains the outcomes of the refactoring process. Each project was refactored with respect to a specific type of test smell, and the refactoring was performed using all LLMs. The folder includes:
- **Changes in code coverage metrics**: Analysis of how refactoring impacted code coverage for[_Python_](./results/coverage/python/final%20aggregated%20results/) and [_Java_](./results/coverage/java/final%20aggregated%20results/).
- **Mitigated and introduced test smells**: Documentation of test smells that were resolved or introduced during the refactoring process [_(results/final refactoring results)_](./results/final%20refactoring%20results/).
- **The projects refactored by each LLM**: Outcomes from refactoring performed by each LLM, enabling a comparison of their effectiveness. For [Python](./results/coverage/python/) and [Java](./results/coverage/java/).
>The *final* results are those outcomes achieved with the best prompt template.

### Source code

The 'src' folder contains the scripts used to perform the study. 
Some external resources we used to perform the study require individual installation and configuration. For this reason, if you intend to use the code provided, you will need to adapt it to work with your environment specific configurations
