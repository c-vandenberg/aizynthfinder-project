# 4. Retrosynthesis Sequence-to-Sequence Model

## 4.1 *Britz et al.* Analysis of Neural Machine Translation Architecture Hyperparameters

*Liu et al.* derived their seq2seq model architecture from the large-scale analysis of **Neural Machine Translation (NMT) architecture hyperparameters** by *Britz et al.*.

In their seminal paper, *Britz et al.* provide insights into the **optimisation of NMT models** (such as seq2seq models), and establish the extent to which model performance metrics are influenced by **random initialisation** and **hyperparameter variation**, helping to **distinguish statisitcally significant results** from **random noise**.

### i. Embedding Dimensionality

Using a valiation data set (**newtest2013**) *Britz et al.* evaluated the effect of **varying embedding dimensionality** on model performance (**Table 2**). *N.B* The values in parentheses represent the maximum observed BLEU score within the given uncertainty range.

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
    <b>Table 2</b> <i>Britz et al.</i> BLEU score trends on variation of embedding dimensionality. <b><sup>4</sup></b>
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

### ii. Encoder and Decoder Recurrent Neural Network (RNN) Cell Variant

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
    <b>Table 3</b> <i>Britz et al.</i> BLEU score trends on variation of RNN cell variant. <b><sup>4</sup></b>
  </p>
</div>

In their experiments, *Britz et al.* found that **LSTM cells consistently outperformed GRU cells** (**Table 3**). As the **computational bottleneck** of their architecture was the **softmax operation**, they **did not observe large differences in training speed** between LSTM and GRU cells. **<sup>4</sup>**

Additionally, the vanilla decoder **performed significantly worse than both the gated variants**, disproving their hypothesis. This could indicate:
1. That the decoder **indeed passes information in its own state throughout multiple time steps** instead of **relying solely on the attention mechanism and current input**.
2. That the gating mechanism is **necessary to mask out irrelevant parts of the inputs**.

### iii. Encoder and Decoder Depth

*Britz et al.* generally expected **deeper networks to converge to better solutions than shallower ones**. However, the **importance of network depth is unclear**, and so they explored the effect of both encoder and decoder depth **up to 8 layers**. 

Additionally, for deeper networks, they experimented with **two variants of residual connections** (**Res** and **ResD**) to **encourage gradient flow**.

<div style="display: flex;" align="center">
  <table border="1" cellspacing="0" cellpadding="5">
    <thead>
      <tr>
        <th><strong>Depth</strong></th>
        <th><strong>BLEU Score (newstest2013)</strong></th>
        <th><strong>Model Parameters</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Enc-2</strong></td>
        <td>21.78 ± 0.05 (21.83)</td>
        <td>66.32M</td>
      </tr>
      <tr>
        <td><strong>Enc-4</strong></td>
        <td>21.85 ± 0.32 (22.23)</td>
        <td>69.47M</td>
      </tr>
      <tr>
        <td><strong>Enc-8</strong></td>
        <td>21.32 ± 0.14 (21.51)</td>
        <td>75.77M</td>
      </tr>
      <tr>
        <td><strong>Enc-8-Res</strong></td>
        <td>19.23 ± 1.96 (21.97)</td>
        <td>75.77M</td>
      </tr>
      <tr>
        <td><strong>Enc-8-ResD</strong></td>
        <td>17.30 ± 2.64 (21.03)</td>
        <td>75.77M</td>
      </tr>
      <tr>
        <td><strong>Dec-1</strong></td>
        <td>21.76 ± 0.12 (21.93)</td>
        <td>64.75M</td>
      </tr>
      <tr>
        <td><strong>Dec-2</strong></td>
        <td>21.78 ± 0.05 (21.83)</td>
        <td>66.32M</td>
      </tr>
      <tr>
        <td><strong>Dec-4</strong></td>
        <td>22.37 ± 0.10 (22.51)</td>
        <td>69.47M</td>
      </tr>
      <tr>
        <td><strong>Dec-4-Res</strong></td>
        <td>17.48 ± 0.25 (17.82)</td>
        <td>68.69M</td>
      </tr>
      <tr>
        <td><strong>Dec-4-ResD</strong></td>
        <td>21.10 ± 0.24 (21.43)</td>
        <td>68.69M</td>
      </tr>
      <tr>
        <td><strong>Dec-8</strong></td>
        <td>01.42 ± 0.23 (1.66)</td>
        <td>75.77M</td>
      </tr>
      <tr>
        <td><strong>Dec-8-Res</strong></td>
        <td>16.99 ± 0.42 (17.47)</td>
        <td>75.77M</td>
      </tr>
      <tr>
        <td><strong>Dec-8-ResD</strong></td>
        <td>20.97 ± 0.34 (21.42)</td>
        <td>75.77M</td>
      </tr>
    </tbody>
  </table>
  <p>
    <b>Table 4</b> <i>Britz et al.</i> BLEU score trends on variation of encoder and decoder depth, and type of residual connection. <b><sup>4</sup></b>
  </p>
</div>

**Encoder Depth**
* The **Enc-2** model achieved a BLEU score of **21.78 ± 0.05**, while increasing the depth to **Enc-4** slightly improved the BLEU score to **21.85 ± 0.32**.
* However, further increasing the encoder depth to **Enc-8** resulted in a **decrease in BLEU score to 21.32 ± 0.14**, despite a **rise in model parameters from 66.32M to 75.77M**.
* This decline was further pronounced when **residual connections were introduced**, with **residually connected deeper encoders significantly more likely to diverge during training**.
* Therefore, contrary to their hypothesis, **deeper encoder networks did not consistently outperform their shallower counterparts**. In fact, deeper encoders **with residual connections** often **exhibited reduced performance** and **increased training instability**, as evidenced by their **larger SD**.

**Decoder Depth**
* Models with deeper decoder outperformed shallower ones by a small margin, with a **general increase in BLEU score from Dec-1 to Dec-4** (**21.76 ± 0.12** to **22.37 ± 0.10**).
* However, without residual connections, *Britz et al.* found it **impossible to train decoders with 8 or more layers** as evidenced by the BLEU score of **Dec-8** (**01.42 ± 0.23**)
* Additionally, for residual connections, **dense residual connections consistently outperformed regular residual connections** and **converged much faster** in terms of step count.

Contrary to their initial hypothesis, **deeper encoders and decoders did not consistently outperform their shallower counterparts**. These findings suggest that, while network depth is a **critical factor in model performance**, its **benefits are not linear** and are **highly contingent on the architectural strategies employed**, such as the **type and implementation of residual connections**. 

Additionally, the **lack of clear performance improvements with incresed depth**, coupled with **training instabilities in deeper configurations**, indicates a need for **more robust optimisation techniques** and **architectural innovations** to **fully harness the potential** of **deep sequential models** in NMT.

### iv. Unidirectional vs. Bidirectional Encoder

For **encoder directionality**, *Britz et al.* cited literature sources where **bidirectional encoders**, **<sup>6</sup>** **unidirectional encoders**, **<sup>7</sup>** and a **mix of both** **<sup>8</sup>** were employed.

Bidirectional encoders are able to create representations that **take into account both past and future inputs**, whereas unidirectional encoders can **only take past inputs into account**. However, the benefit of unidirectional encoders is that they can be **easily parallelized on GPUs**, allowing them to run faster than bidirectional encoders. **<sup>4</sup>**

A well as investigating encoder directionality, *Britz et al.* also investigated **source input reversal**. Reversing source inputs is a **commonly used technique** that allows the encoder to **create richer representations for earlier words**. Given that **errors can on the decoder side can easily cascade**, the **correctness of early words has a disproportionate impact**. **<sup>4</sup>**

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
        <td><strong>Bidi-2</strong></td>
        <td>21.78 ± 0.05 (21.83)</td>
        <td>66.32M</td>
      </tr>
      <tr>
        <td><strong>Uni-1</strong></td>
        <td>20.54 ± 0.16 (20.73)</td>
        <td>63.44M</td>
      </tr>
      <tr>
        <td><strong>Uni-1R</strong></td>
        <td>21.16 ± 0.35 (21.64)</td>
        <td>63.44M</td>
      </tr>
      <tr>
        <td><strong>Uni-2</strong></td>
        <td>20.98 ± 0.10 (21.07)</td>
        <td>65.01M</td>
      </tr>
      <tr>
        <td><strong>Uni-2R</strong></td>
        <td>21.76 ± 0.21 (21.93)</td>
        <td>65.01M</td>
      </tr>
      <tr>
        <td><strong>Uni-4</strong></td>
        <td>21.47 ± 0.22 (21.70)</td>
        <td>68.16M</td>
      </tr>
      <tr>
        <td><strong>Uni-4R</strong></td>
        <td>21.32 ± 0.42 (21.89)</td>
        <td>68.16M</td>
      </tr>
    </tbody>
  </table>
  <p>
    <b>Table 5</b> <i>Britz et al.</i> BLEU score trends on variation of encoder directionality, and source input reversal. <b><sup>4</sup></b>
  </p>
</div>

The investigation shows that, in genereal, **bidirectional encoders marginally outperform unidirectional encoders**, and the **introduction of reversed source inputs significantly boosts the performance of unidirectional encoders**. However, even with reversed inputs, **shallower bidirectional encoders remain competitive**, suggesting that bidirectionality provides **inherent advantages in capturing contextual information**.

Noteably, their results **do not include a bidirectional 2-layer encoder with reversed source input**, nor a **bidirectional 4-layer encoder with and without reversed source input**. This will be an **avenue for investigation in this project.**

### v. Attention Mechanism

*Britz et al.* compared the performance of **additive and multiplicative attention variants** across **varying attention dimensionalities** (**Table 6**).
1. **Additive Attention**:
   * The **additive mechanism**, introduced by *Bahdanau et al.*, **<sup>6</sup>** involves **combining the encoder states and decoder states** through a **feedforward network** before computing the attention scores.
2. **Multiplicative Attention**:
   * The **multiplicative mechanism**, introduced by *Luond et al.*, **<sup>7</sup>** **computes attention scores using a dot product** between the **transformed encoder and decoder states**, making it less computationally expensive.

They also experimented with using **no attention mechanism** by:
1. **Initializing the decoder state with the last encoder state (None-State)**.
2. **Concatenating the last decoder state to each decoder input (None-Input)**.

<div style="display: flex;" align="center">
  <table border="1" cellspacing="0" cellpadding="5">
    <thead>
      <tr>
        <th><strong>Attention Type</strong></th>
        <th><strong>BLEU Score (newstest2013)</strong></th>
        <th><strong>Model Parameters</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Mul-128</strong></td>
        <td>22.03 ± 0.08 (22.14)</td>
        <td>65.73M</td>
      </tr>
      <tr>
        <td><strong>Mul-256</strong></td>
        <td>22.33 ± 0.28 (22.64)</td>
        <td>65.93M</td>
      </tr>
      <tr>
        <td><strong>Mul-512</strong></td>
        <td>21.78 ± 0.05 (21.83)</td>
        <td>66.32M</td>
      </tr>
      <tr>
        <td><strong>Mul-1024</strong></td>
        <td>18.22 ± 0.03 (18.26)</td>
        <td>67.11M</td>
      </tr>
      <tr>
        <td><strong>Add-128</strong></td>
        <td>22.23 ± 0.11 (22.38)</td>
        <td>65.73M</td>
      </tr>
      <tr>
        <td><strong>Add-256</strong></td>
        <td>22.33 ± 0.04 (22.39)</td>
        <td>65.93M</td>
      </tr>
      <tr>
        <td><strong>Add-512</strong></td>
        <td>22.47 ± 0.27 (22.79)</td>
        <td>66.33M</td>
      </tr>
      <tr>
        <td><strong>Add-1024</strong></td>
        <td>22.10 ± 0.18 (22.36)</td>
        <td>67.11M</td>
      </tr>
      <tr>
        <td><strong>None-State</strong></td>
        <td>9.98 ± 0.28 (10.25)</td>
        <td>64.23M</td>
      </tr>
      <tr>
        <td><strong>None-Input</strong></td>
        <td>11.57 ± 0.30 (11.85)</td>
        <td>64.49M</td>
      </tr>
    </tbody>
  </table>
  <b>Table 6</b> <i>Britz et al.</i> BLEU score trends on variation of attention mechanism variants, and attention dimensionality. <b><sup>4</sup></b>
</div>

The investigation showed that **additive attention slightly but consistently outperformed multiplicative attention**, with attention dimensionality **having minimal effect on additive attention performance**. Additionally, the **abysmal performance of non-attention models** further emphasizes the **necessity of incorporating attention mechanisms in NMT models**.

### vi. Beam Search Strategies

**Beam Search** is a commonly used technique aimed at **identifying the most probable target sequences** by **exploring multiple translations through tree search**. In their study, *Britz et al.* evaluated the impact of **varying beam widths**, ranging from **1 (greedy search) to 100**, and the **incorporation of length normalisation penalities** of **0.5 and 1.0** on BLEU scores (**Table 7**).

<div style="display: flex;" align="center">
  <table border="1" cellspacing="0" cellpadding="5">
    <thead>
      <tr>
        <th><strong>Beam Width</strong></th>
        <th><strong>BLEU Score (newstest2013)</strong></th>
        <th><strong>Model Parameters</strong></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>B1</strong></td>
        <td>20.66 ± 0.31 (21.08)</td>
        <td>66.32M</td>
      </tr>
      <tr>
        <td><strong>B3</strong></td>
        <td>21.55 ± 0.26 (21.94)</td>
        <td>66.32M</td>
      </tr>
      <tr>
        <td><strong>B5</strong></td>
        <td>21.60 ± 0.28 (22.03)</td>
        <td>66.32M</td>
      </tr>
      <tr>
        <td><strong>B10</strong></td>
        <td>21.57 ± 0.26 (21.91)</td>
        <td>66.32M</td>
      </tr>
      <tr>
        <td><strong>B25</strong></td>
        <td>21.47 ± 0.30 (21.77)</td>
        <td>66.32M</td>
      </tr>
      <tr>
        <td><strong>B100</strong></td>
        <td>21.10 ± 0.31 (21.39)</td>
        <td>66.32M</td>
      </tr>
      <tr>
        <td><strong>B10-LP-0.5</strong></td>
        <td>21.71 ± 0.25 (22.04)</td>
        <td>66.32M</td>
      </tr>
      <tr>
        <td><strong>B10-LP-1.0</strong></td>
        <td>21.80 ± 0.25 (22.16)</td>
        <td>66.32M</td>
      </tr>
    </tbody>
  </table>
  <b>Table 7</b> <i>Britz et al.</i> BLEU score trends on variation of beam width, and aaddition of length penalities (LP). <b><sup>4</sup></b>
</div>

The investigation shows that the **optimal beam width** appears to reside around **5 to 10**, where **significant improvements** in BLEU score are observed **without incurring the diminishing returns associated with larger beams**. Additionally, the **introduction of length penalities enhances performance within this beam width range**.

## 4.2 *Liu et al.* Sequence-to-Sequence Model

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

From their analysis, *Britz et al.* released an **open source, TensorFlow-based package** specifically designed to implement **reproducible state of the art sequence-to-sequence models**. This aim of this open source seq2seq library is to allow researchers to explore **novel architectures** with **minimal code changes**, and **define experimental parameters in a reproducible manner**. **<sup>4</sup>*

*Liu et al.* adapted this open source library in the design of their characterwise seq2seq model. The encoder-decoder architecture consists of **bidrectional LSTM cells for the encoder** and **unidirectional LSTM cells for the decoder**. Additionally, they utilise a an **additive attention mechanism**. The key hyperparameters are shown in **Table 8**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/999ae54c-1d80-4f0a-8411-cb5d9391766e", alt="liu-et-al-model-hyperparameters"/>
    <p>
      <b>Table 8</b> Key hyperparameters of the seq2seq model by <i>Liu at al.</i> <b><sup>1</sup></b>
    </p>
  </div>
<br>

For model training, they **reverse

## 4.2 Project Sequence-to-Sequence Model

## 4.3 References
**[1]** Liu, B. et al. (2017) ‘Retrosynthetic reaction prediction using neural sequence-to-sequence models’, ACS Central Science, 3(10), pp. 1103–1113. <br><br>
**[2]** Lowe, D. M. (2012) ‘Extraction of Chemical Structures and Reactions from the Literature’; University of Cambridge. <br><br>
**[3]** Schneider, N., Stiefl, N. and Landrum, G.A. (2016) ‘What’s what: The (nearly) definitive guide to reaction role assignment’, Journal of Chemical Information and Modeling, 56(12), pp. 2336–2346. <br><br>
**[4]** Britz, D. et al. (2017) ‘Massive exploration of neural machine translation architectures’, Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. <br><br>
**[5]** He, K. et al. (2016) ‘Deep residual learning for image recognition’, 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). <br><br>
**[6]** Bahdanau, D. et al. (2015) ‘Neural machine translation by jointly learning to align and translate’, Proceedings of the 2015 International Conference on Learning Representations (ICLR). <br><br>
**[7]** Luong, M. et al. (2016) ‘Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models’, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics. <br><br>
**[8]** Wu, Y. et al. (2016) ‘Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation’. <br><br>
