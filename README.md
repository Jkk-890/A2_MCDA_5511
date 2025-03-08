# A2_MCDA_5511
Assignment 2 for MCDA 5511

To get the code to run you must

1. Select the ".venv (Python 3.12.9)" kernal
2. Run "uv sync" 

When you need to add a new package to the venv you can write "uv add <package_name>", 
this will automatically update the "pyproject.toml" file to include the dependency.

# README - Part 6: Automating RAG Evaluation Using LLM as a Judge

## Overview
In Part 6 of the assignment, we automate the evaluation of our Retrieval-Augmented Generation (RAG) system by leveraging an LLM as a judge. The goal is to assess the quality of the generated responses by comparing them against the ground truth (GT) answers. We utilize **Llama 3.1 (8B)** to conduct this automated evaluation.

## Methodology
1. **Dataset**: The dataset consists of Question-Answer (Q-A) pairs with retrieved documents, cosine similarity scores, generated responses (GR), and ground truth responses (GT).
2. **Evaluation Prompt**: The LLM is given a structured prompt with the question, retrieved top matches with similarity scores, the generated response, and the ground truth.
3. **Evaluation Criteria**:
   - The LLM assesses whether the generated response sufficiently matches the ground truth.
   - If the similarity between the generated response and the ground truth meets a predefined threshold (`th`), the output is **1** (correct).
   - Otherwise, the output is **0** (incorrect).
4. **Implementation**:
   - The function `evaluate_llama()` formulates the prompt and queries the LLM.
   - The function `run_llm_as_judge(th)` applies this evaluation for all Q-A pairs and saves the results.

## Implementation Details
- **Model Used**: Llama 3.1 (8B)
- **Evaluation Logic**:
  - The LLM strictly adheres to binary output (1 or 0) without explanation.
  - The threshold (`th`) determines the required similarity between GR and GT.
- **Challenges Encountered**:
  - The LLM tends to be **strict** in its evaluation due to out-of-context interpretations.
  - Small variations in numbers or phrasing resulted in **false negatives** (e.g., rounding differences in revenue figures).

## Observations & Adjustments
1. **Example Adjustments for Strictness**:
   - **Revenue Calculation**:
     - GT: "Avatar" (2009) grossed **$2.8 billion** → Adjusted to **2.8 billion**
     - GR: **2.9 billion** → Initially marked incorrect due to precision mismatch.
   - **Release Order Comparison**:
     - GT: "Titanic (1997) was released before The Matrix (1999)." → Adjusted to **Titanic** for consistency.
   - **Oldest Movie Revenue**:
     - GT: "The Birth of a Nation" (1915) grossed **50–100 million.**
     - GR: **96.77 million** → Adjusted for better alignment.
2. **LLM Behavior**:
   - The LLM was **overly strict** with small variations in phrasing or numerical representation.
   - Contextual inconsistencies were flagged even when the meaning was equivalent.

## Results
- The automated evaluation helped scale the testing process.
- However, strict adherence to exact wording caused some false negatives.
- Further improvements could include a **semantic similarity score** instead of binary evaluation.

## Next Steps
- Implement a **soft matching metric** (e.g., cosine similarity on response embeddings) for nuanced grading.
- Experiment with **prompt engineering** to make Llama 3.1 more lenient in handling minor phrasing differences.
- Compare results with human evaluation to refine automated assessment.

## File Output
- The updated Q-A dataset with LLM evaluation is saved as **`qa_pairs_results.csv`**.

