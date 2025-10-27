# Disentangling Faithfulness Hallucination in Retrieval Augmented Generation: A Systematic Benchmark and Analysis
After the retrieval and generation phases of the presented RAG architecture, we fulfilled an evaluation on the system results.
- First on the retrieval phase, we compare the performance of different retrieval strategies (sparse, dense, and hybrid) using MAP@k and NDCG@k at different cut-off values, where k = 3, 5, 10.
- Then, to make sure that the difference between these strategies are statistically significant, we applied statistical significance tests and then correction for multiple hypothesis testing.
- After that, we fulfilled an overall evaluation on the generated responses using a LLM-as-a-judge approach (with a closed LLM) and this approach has been further verified by human annotators.
