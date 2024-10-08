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

### 4.1.2 *Britz et al.* Open Source Seq2Seq Libary

*Liu at al.* adapted the **open source seq2seq library** from *Britz et al.*. **<sup>4</sup>** This aim of this open source Seq2Seq library is to allow researchers to explore **novel architectures** with **minimal code changes**, and **define experimental parameters in a reproducible manner**.

In their seminal paper, *Britz et al.* conducted a large-scale analysis of **Neural Machine Translation (NMT) architecture hyperparameters**. This provides insights into the **optimisation of NMT models** (such as seq2seq models), and establishing the extent to which model performance metrics are influenced by **random initialisation** and **hyperparameter variation**, helping to **distinguish statisitcally significant results** from **random noise**.

### Embedding Dimensionality

Using a valiation data set (**newtest2013**) *Britz et al.* evaluated the effect of **varying embedding dimensionality** on model performance (**Table 2**).

<div style="display: flex;" align="center">
  <table border="1" cellspacing="0" cellpadding="5">
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
    <b>Table 2</b> <i>Britz et al.</i> newtest2013 BLEU score trends relative to embedding dimensionality. <b><sup>4</sup></b> <i>N.B. The values in parentheses represent the maximum observed BLEU score within the given uncertainty range.</i>
  </p>
</div>

**Small Embedding Sizes (128 & 256)**
* Models employing smaller embedding dimensions of **128** and **256** demonstrate surprisingly robust performance despite their relatively modest number of parameters. Specifically, the BLEU scores for these embeddings are only **0.36** and **0.12** lower than the highest performing embedding dimension of **2048** respectively.
* Additionally, it was found that models with these lower embeddings **converge almost twice as quickly** compared to larger embeddings.
* While not included in **Table 2**, *Britz et al.* also found that the **gradient updates and their norms** remained **approximately constant across these embedding sizes**. This indicates that the **optimisation process is similarly effective regardless of embedding dimensionality**.
* Finally, *Britz et al.* **did not observe any signs of overfitting** with these smaller embeddings and observed **consistent training log perplexity**.

**Medium Embedding Sizes (512 & 1024)**
* Transitioning to medium embedding dimensions, specifically **512** and **1024**, *Britz et al.* observed a **continued upward trend** in BLEU scores, albeit with **diminishing returns**.
* The 512-dimensional embeddings achieve a BLEU score of **21.78**, showing **improvement over smaller sizes**.
* However, at 1024 dimensions, the BLEU score **slightly decreases to 21.36**, accompanied by an **increased standard deviation of 0.27**. This **rise in variability** suggests that while **some models benefit from larger embeddings, others may not**, leading to **less consistent performance**.
* Additionally, the **substantial increase in model parameters** at 1024 dimensions (**106.58M**) raises concerns about **computational efficiency** and **potential overfitting**.
* Like with small embedding sizes, *Britz et al.* found that the **gradient updates and their norms remained constant**, implying that the **optimisation process does not inherently favour larger embeddings**. This potentially **limits the effective utilisation of the additional parameters**.

**Large Embedding Sizes (2048)**
* At the largest embedding dimension of **2048**, the model achieves the **highest BLEU score of 21.86**, marginally outperforming the 512-dimensional embeddings by **0.08 points**.
* Despite this improvement, the **drastic increase in model parameters to 187.09M** introduces **significant computational overhead**, raising practical concerns regarding **training time, memory consumption**, and **scalability**.
* Moreover, the **minimal gain in BLEU score** suggests that the model **may not be making efficient use of the extra parameters**, as indicated by the **consistent training log perplexity across all embedding sizes**. As a result, the model is **not fully leveraging the capacity offered by such large embeddings**.

### Encoder and Decoder Recurrent Neural Network (RNN) Cell Variant

To evaluate the effect of encoder and decoder RNN cell variant on model performance, *Britz et al.* **compared three cell variants**:
1. **Long Short-Term Memory (LSTM) Cells**
2. **Gated Recurrent Unit (GNU) Cells**
3. **A Vanilla RNN Cell in the Decoder Only**

The LSTM and GNU cells are what are known as **gated cells**:
* **Gated cells** are **specialised types of RNN units** that **incorporate gates** - mechanisms designed to **regulate the flow of information**.
* These gates control **how information is retained, forgotten, or updated** as the network **processes sequential data**.
* The primary purpose of gated cells is to **address and mitigate common issues** encountered in **traditional RNNs**, such as the **vanishing gradient problem**, which **hampers the ability to learn long-term dependencies in data**.

Using **vanillar RNN cells**, deep networks **cannot efficiently propagate information and gradients though mulitple layers and time steps**, hence the need for gated cells. However, *Britz et al.* hypothesised that, with an **attention-based model**, the decoder should be able to make decisions almost **exclusively based on the current input and the attention context**, and so the **gating mechanism in the decoder is not strictly necessary**. 

To test this hyporthesis they added a **vanilla RNN cell in the decoder only** in their study of RNN cell variants.

<div style="display: flex;" align="center">
  <table border="1" cellspacing="0" cellpadding="5">
    <thead>
      <tr>
        <th><strong>Cell Variant</strong></th>
        <th><strong>BLEU Score (newstest2013)</strong></th>
        <th><strong>Model Parameters</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>LSTM</strong></td>
        <td>22.22 ± 0.08 (22.33)</td>
        <td>68.95M</td>
      </tr>
      <tr>
        <td><strong>GRU</strong></td>
        <td>21.78 ± 0.05 (21.83)</td>
        <td>66.32M</td>
      </tr>
      <tr>
        <td><strong>Vanilla-Dec</strong></td>
        <td>15.38 ± 0.28 (15.73)</td>
        <td>63.18M</td>
      </tr>
    </tbody>
  </table>
  <p>
    <b>Table 3</b> <i>Britz et al.</i> newtest2013 BLEU score trends relative to RNN cell variant. <b><sup>4</sup></b> <i>N.B. The values in parentheses represent the maximum observed BLEU score within the given uncertainty range.</i>
  </p>
</div>

In their experiments, *Britz et al.* found that **LSTM cells consistently outperformed GRU cells** (**Table 3**). As the **computational bottleneck** of their architecture was the **softmax operation**, they **did not observe large differences in training speed** between LSTM and GRU cells. **<sup>4</sup>**

Additionally, the vanilla decoder **performed significantly worse than both the gated variants**, disproving their hypothesis. This could indicate:
1. That the decoder **indeed passes information in its own state throughout multiple time steps** instead of **relying solely on the attention mechanism and current input**.
2. That the gating mechanism is **necessary to mask out irrelevant parts of the inputs**.

### Encoder and Decoder Depth

### 4.1.3 *Liu at al.* Model Architecture

## References
**[1]** Liu, B. et al. (2017) ‘Retrosynthetic reaction prediction using neural sequence-to-sequence models’, ACS Central Science, 3(10), pp. 1103–1113. <br><br>
**[2]** Lowe, D. M. (2012) ‘Extraction of Chemical Structures and Reactions from the Literature’; University of Cambridge. <br><br>
**[3]** Schneider, N., Stiefl, N. and Landrum, G.A. (2016) ‘What’s what: The (nearly) definitive guide to reaction role assignment’, Journal of Chemical Information and Modeling, 56(12), pp. 2336–2346. <br><br>
**[4]** Britz, D. et al. (2017) ‘Massive exploration of neural machine translation architectures’, Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. <br><br>
