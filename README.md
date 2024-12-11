# RO-FIGS: Random Oblique Fast Interpretable Greedy-tree Sum method

Official code for the paper RO-FIGS: Efficient and Expressive Tree-Based Ensembles for Tabular Data accepted at IEEE CITREx Symposium 2025

**Paper abstract:**  Tree-based models are often robust to uninformative features and can accurately capture non-smooth, complex decision boundaries. Consequently, they often outperform neural network-based models on tabular datasets at a significantly lower computational cost. Nevertheless, the capability of traditional tree-based ensembles to express complex relationships efficiently is limited by using a single feature to make splits. To improve the efficiency and expressiveness of tree-based methods, we propose Random Oblique Fast Interpretable Greedy-Tree Sums (RO-FIGS). RO-FIGS builds on Fast Interpretable Greedy-Tree Sums, and extends it by learning trees with oblique or multivariate splits, where each split consists of a linear combination learnt from random subsets of features. This helps uncover interactions between features and improves performance. The proposed method is suitable for tabular datasets with both numerical and categorical features. We evaluate RO-FIGS on 22 real-world tabular datasets, demonstrating superior performance and much smaller models over other tree- and neural network-based methods. Additionally, we analyse their splits to reveal valuable insights into feature interactions, enriching the information learnt from SHAP summary plots, and thereby demonstrating the enhanced interpretability of RO-FIGS models. The proposed method is well-suited for applications, where balance between accuracy and interpretability is essential.

## Citation

Please cite our work as:
