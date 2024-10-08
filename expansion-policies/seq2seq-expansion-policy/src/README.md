# 4. Retrosynthesis Sequence-to-Sequence Model

## 4.1 *Liu et al.* Sequence-to-Sequence Model

The sequence-to-sequence (Seq2Seq) model implementation in this project was based on the model developed by *Liu at al.* **<sup>1</sup>** This model processes target molecules in **molecular-input line-entry system (SMILES)** notation and outputs the prediced molecular precursors in the same notation.

### 4.1.1 Data Preparation

*Liu at al.* used a data set of **50,000 atom-mapped reactions** that were filtered form an open source patent database to represent typical medicinal chemistry reaction types. **<sup>2</sup>** These 50,000 reactions were **classified into 10 broad reaction types** **<sup>3</sup>** (**Table 1.**), preprocessed to **eliminate all reagents** and leave only reactants & products, and then **canonicalised**. Additionally, any reactions with multiple products were split into **multiple single product reactions**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/b4b11d31-9b6b-4527-ae15-ccd9b3abf921", alt="liu-et-al-reaction-types"/>
    <p>
      <b>Table 1</b> <i>Liu at al.</i> training data set reaction class distribution. <b><sup>1</sup></b>
    </p>
  </div>
<br>

Finally, the data set was split into training, validation and test data sets in a ratio of **8:1:1**.

### 4.1.2 Model Architecture

*Liu at al.* adapted the **open source seq2seq library** from *Brtiz et al.*. **<sup>4</sup>** This aim of this open source Seq2Seq library is to allow researchers to explore **novel architectures** with **minimal code changes**, and **define experimental parameters in a reproducible manner**.

In their seminal paper, *Brtiz et al.* conducted a large-scale analysis of **Neural Machine Translation (NMT) architecture hyperparameters**. This provides insights into the **optimisation of NMT models** (such as seq2seq models), and establishing the extent to which model performance metrics are influenced by **random initialisation** and **hyperparameter variation**, helping to **distinguish statisitcally significant results** from **random noise**.

### Embedding Dimensionality

Using a valiation data set (**newtest2013**) *Brtiz et al.* evaluated the effect of **varying embedding dimensionality** on model performance (**Table 2**).

<div align="center">
  <table>
    <tr>
      <th>Embedding Dimensionality</th>
      <th>BLEU Score (newstest2013)</th>
      <th>Model Parameters</th>
    </tr>
    <tr>
      <td>128</td>
      <td>21.50 ± 0.16 (21.66)</td>
      <td>36.13M</td>
    </tr>
    <tr>
      <td>256</td>
      <td>21.73 ± 0.09 (21.85)</td>
      <td>46.20M</td>
    </tr>
    <tr>
      <td>512</td>
      <td>21.78 ± 0.05 (21.83)</td>
      <td>66.32M</td>
    </tr>
    <tr>
      <td>1024</td>
      <td>21.36 ± 0.27 (21.67)</td>
      <td>106.58M</td>
    </tr>
    <tr>
      <td>2048</td>
      <td>21.86 ± 0.17 (22.08)</td>
      <td>187.09M</td>
    </tr>
  </table>
  <p>
    <b>Table 2</b> <i>Brtiz et al.</i> newtest2013 BLEU score trends relative to embedding dimensionality. <b><sup>4</sup></b>
  </p>
</div>

For **small embedding sizes (128 & 256)**, the analysis shows a **surprisingly robust performance** even for the **relatively small number of parameters** afforded by these small embedding sizes, with their BLEU scores being only **-0.36** and **-0.12** less than the highest performing embedding dimension (**2048**). Additionally, it was found that models with these lower embeddings **converge almost twice as quickly** compared to larger embeddings.

For **medium sized embeddings (512 & 1024)**, there is a continuing **upward trend in BLEU scores**, however with signs of **dimenishing returns**. Additionally, there is an **increase in variability** illustrated by the **increase standard deviation (SD)** of 1024 dimensions.

For **large embedding sizes (2048)**, although this **achieves the highest BLEU score**, the score is only **marginally better than 512 dimensions** but with a **substantial increase in model parameters**. This drastic increase in parameters would raise concerns about **computational efficiency/overhead** and **potential overfitting**.


## References
**[1]** Liu, B. et al. (2017) ‘Retrosynthetic reaction prediction using neural sequence-to-sequence models’, ACS Central Science, 3(10), pp. 1103–1113. <br><br>
**[2]** Lowe, D. M. (2012) ‘Extraction of Chemical Structures and Reactions from the Literature’; University of Cambridge. <br><br>
**[3]** Schneider, N., Stiefl, N. and Landrum, G.A. (2016) ‘What’s what: The (nearly) definitive guide to reaction role assignment’, Journal of Chemical Information and Modeling, 56(12), pp. 2336–2346. <br><br>
**[4]** Britz, D. et al. (2017) ‘Massive exploration of neural machine translation architectures’, Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. <br><br>
