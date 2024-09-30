# 3 Sequence-to-Sequence (Seq2Seq) Expansion Policy

As stated in **Section 2**, for **neural network-guided one-step retrosynthesis**, there are two primary methodologies used to **define the disconnection rules** and so **model the reverse reaction**: **<sup>1</sup>**
1. **Template-Based Methods**
2. **SMILES-Based (Template-Free) Methods**

This project aims to implement a SMILES-based (template-free) retrosynthetic method, and incorporate it into a **customised instance of AiZynthFinder**. The neural network architecture used is a **sequence-to-sequence (seq2seq)** model based on the work by *Pande et al.* **<sup>2</sup>**

## 3.1 Limitations of Template-Based Retrosynthetic Methods

A template-based retrosynthetic method is based on a predefined set of rules, and thus **inherit the limitations of these rules**:
1. The primary limitation of such methods is that they are **fundamentally dependent on the rules on which the neural network is trained**, and thus these approaches have **issues with making accurate predictions outside of this rule-based knowledge base**. **<sup>2</sup>**
2. There is also an **inherent trade-off** between defining **general rules**, which can **introduce noise** and **reduce the accuracy or reliability of a model’s predictions**, and defining **very specific rules**, which **restrict the model’s predictions to a narrow set of reactants and products**. **<sup>2</sup>**
3. Additionally, the reaction rules are **inadequate representations of the underlying chemistry** as they focus on **local reaction center molecular environments only**. **<sup>2</sup>**

## 3.2 Sequence-to-Sequence Model Overview



## References
**[1]** Saigiridharan, L. et al. (2024) ‘AiZynthFinder 4.0: Developments based on learnings from 3 years of industrial application’, Journal of Cheminformatics, 16(1). <br><br>
**[2]** Liu, B. et al. (2017) ‘Retrosynthetic reaction prediction using neural sequence-to-sequence models’, ACS Central Science, 3(10), pp. 1103–1113. <br><br>
