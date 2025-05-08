# RO-FIGS: Random Oblique Fast Interpretable Greedy-tree Sum method

Official code for the paper [RO-FIGS: Efficient and Expressive Tree-Based Ensembles for Tabular Data](https://doi.org/10.1109/CITREx64975.2025.10974939) accepted at [IEEE CITREx Symposium 2025](https://ieee-ssci.org/?ui=trustworthy-explainable-and-responsible-ci)


**Paper abstract:**  Tree-based models are often robust to uninformative features and can accurately capture non-smooth, complex decision boundaries. Consequently, they often outperform neural network-based models on tabular datasets at a significantly lower computational cost. Nevertheless, the capability of traditional tree-based ensembles to express complex relationships efficiently is limited by using a single feature to make splits. To improve the efficiency and expressiveness of tree-based methods, we propose Random Oblique Fast Interpretable Greedy-Tree Sums (RO-FIGS). RO-FIGS builds on Fast Interpretable Greedy-Tree Sums, and extends it by learning trees with oblique or multivariate splits, where each split consists of a linear combination learnt from random subsets of features. This helps uncover interactions between features and improves performance. The proposed method is suitable for tabular datasets with both numerical and categorical features. We evaluate RO-FIGS on 22 real-world tabular datasets, demonstrating superior performance and much smaller models over other tree- and neural network-based methods. Additionally, we analyse their splits to reveal valuable insights into feature interactions, enriching the information learnt from SHAP summary plots, and thereby demonstrating the enhanced interpretability of O-FIGS models. The proposed method is well-suited for applications, where balance between accuracy and interpretability is essential.


**Supplementary material:** Supplementary material that accompanies the paper has been added to the `Supplementary_material.pdf`. This includes additional experiments, data, and analysis referenced in the paper.


## Citation
For attribution in academic contexts, please cite this work as:
```
@inproceedings{matjasec2025,
  author    = {Matjašec, Urška and Simidjievski, Nikola and Jamnik, Mateja},
  booktitle = {2025 IEEE Symposium on Trustworthy, Explainable and Responsible Computational Intelligence (CITREx)}, 
  title     = {{RO-FIGS}: Efficient and Expressive Tree-Based Ensembles for Tabular Data}, 
  year      = {2025},
  pages     = {1-7},
  doi       = {10.1109/CITREx64975.2025.10974939}
}
```


## Installation

To install the required dependencies, run:
```
pip install -r requirements.txt
```


## Usage

See two tutorials in `examples` folder for detailed examples and usage.

If you are using your own dataset, note that categorical features need to be transformed into numerical features, either by one-hot encoding or any other suitable method.

