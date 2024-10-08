# 4. Retrosynthesis Sequence-to-Sequence Model

## 4.1 *Liu et al.* Sequence-to-Sequence Model

The sequence-to-sequence (Seq2Seq) model implementation in this project was based on the model developed by *Liu at al.* **<sup>1</sup>** This model processes target molecules in **molecular-input line-entry system (SMILES)** notation and outputs the prediced molecular precursors in the same notation.

### 4.1.1 Data Preparation

*Liu at al.* used a data set of **50,000 atom-mapped reactions** that were filtered form an open source patent database to represent typical medicinal chemistry reaction types. **<sup>2</sup>** These 50,000 reactions were **classified into 10 broad reaction types** **<sup>3</sup>** (**Table 1.**), preprocessed to **eliminate all reagents** and leave only reactants & products, and then **canonicalised**. Additionally, any reactions with multiple products were split into **multiple single product reactions**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/b4b11d31-9b6b-4527-ae15-ccd9b3abf921", alt="liu-et-al-reaction-types"/>
    <p>
      <b>Table 1</b> <b>Liu at al.</b> training data set reaction class distribution. <b><sup>1</sup></b>
    </p>
  </div>
<br>

Finally, the data set was split into training, validation and test data sets in a ratio of **8:1:1**.

### 4.1.2 Model Architecture

*Liu at al.* adapted the **open source Seq2Seq library** from *Brtiz et al.*. **<sup>4</sup>** This aim of this open source Seq2Seq library is to allow researchers to explore **novel architectures** with **minimal code changes**, and **define experimental parameters in a reproducible manner**.

In their large-scale analysis of **Neural Machine Translation (NMT) architecture hyperparameters**, *Brtiz et al.* found the following: **<sup>4</sup>**
1. 


## References
**[1]** Liu, B. et al. (2017) ‘Retrosynthetic reaction prediction using neural sequence-to-sequence models’, ACS Central Science, 3(10), pp. 1103–1113. <br><br>
**[2]** Lowe, D. M. (2012) ‘Extraction of Chemical Structures and Reactions from the Literature’; University of Cambridge. <br><br>
**[3]** Schneider, N., Stiefl, N. and Landrum, G.A. (2016) ‘What’s what: The (nearly) definitive guide to reaction role assignment’, Journal of Chemical Information and Modeling, 56(12), pp. 2336–2346. <br><br>
**[4]** Britz, D. et al. (2017) ‘Massive exploration of neural machine translation architectures’, Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. <br><br>
