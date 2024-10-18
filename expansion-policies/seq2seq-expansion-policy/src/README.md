# 4. Retrosynthesis Sequence-to-Sequence Model

## 4.1 *Britz et al.* Analysis of Neural Machine Translation Architecture Hyperparameters

*Liu et al.* derived their seq2seq model architecture from the large-scale analysis of **Neural Machine Translation (NMT) architecture hyperparameters** by *Britz et al.*.

In their seminal paper, *Britz et al.* provide insights into the **optimisation of NMT models** (such as seq2seq models), and establish the extent to which model performance metrics are influenced by **random initialisation** and **hyperparameter variation**, helping to **distinguish statisitcally significant results** from **random noise**.

### 4.1.1 Embedding Dimensionality

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

### 4.1.2 Encoder and Decoder Recurrent Neural Network (RNN) Cell Variant

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

### 4.1.3 Encoder and Decoder Depth

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

### 4.1.4 Unidirectional vs. Bidirectional Encoder

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

### 4.1.5 Attention Mechanism

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

### 4.1.6 Beam Search Strategies

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

*Liu at al.* used a data set of **50,000 atom-mapped reactions** that were filtered form an open source patent database to represent typical medicinal chemistry reaction types. **<sup>2</sup>** These 50,000 reactions were **classified into 10 broad reaction types** **<sup>3</sup>** (**Table 8.**), preprocessed to **eliminate all reagents** and leave only reactants & products, and then **canonicalised**. Additionally, any reactions with multiple products were split into **multiple single product reactions**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/b4b11d31-9b6b-4527-ae15-ccd9b3abf921", alt="liu-et-al-reaction-types"/>
    <p>
      <b>Table 8</b> <i>Liu at al.</i> training data set reaction class distribution. <b><sup>1</sup></b>
    </p>
  </div>
<br>

Finally, the data set was split into training, validation and test data sets in a ratio of **8:1:1**.

### i. Training
For model training, the training data **has reaction atom-mapping removed**, and each reaction example is **split into a source sequence** and a **target sequence**.
  * The source sequence is the **product SMILES sequence split into characters**, with a **reaction type token prepended to the sequence**.
  * Additionally, the source sequence is **reversed before being fed into the encoder**.
  * The target sequence is the **reactants SMILES split into characters**.

The seq2seq model is **evaluated on the validation data set every 4000 training steps**, and the model training is stopped **once the evaluation log perplexity starts to increase**.

### ii. Testing
The trained seq2seq model is **evaluated on the test data set with atom-mapping removed**. Each target molecule SMILES in the test data set is **split into characters**, a **reaction type token prepended to the sequence**, and is **reversed** before being fed into the trained model encoder.

Additionally, a **beam search procedure is used for model inference**: **<sup>1</sup>**
1. For each target molecule source sequence input, the **top N candidate output sequences** ranked by **overall sequence log probability at each time step during decoding are retained**,  where *N is the width of the beam**.
2. The decoding is stopped **once the lengths of the candidate sequences reach the maximum decode length of 140 characters**
3. The candidate sequences that contain an **end of sequence character are considered to be complete**. This was on average about **97% of all beam search predicted candidate sequences**.
4. These complete candidate sequences represent the **reactant sets predicted by the seq2seq model** for a particular target molecule, and they are **ranked by the overall sequence log probabilities**. The overall sequence log probability for a candidate sequence consists of the **log probabilities of the individual characters** in that candidate sequence.

### 4.1.2 Model Architecture

From their analysis, *Britz et al.* released an **open source, TensorFlow-based package** specifically designed to implement **reproducible state of the art sequence-to-sequence models**. This aim of this open source seq2seq library is to allow researchers to explore **novel architectures** with **minimal code changes**, and **define experimental parameters in a reproducible manner**. **<sup>4</sup>*

*Liu et al.* adapted this open source library in the design of their characterwise seq2seq model. The encoder-decoder architecture consists of **bidrectional LSTM cells for the encoder** and **unidirectional LSTM cells for the decoder**. Additionally, they utilise a an **additive attention mechanism**. The key hyperparameters are shown in **Table 9**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/999ae54c-1d80-4f0a-8411-cb5d9391766e", alt="liu-et-al-model-hyperparameters"/>
    <p>
      <b>Table 9</b> Key hyperparameters of the seq2seq model by <i>Liu at al.</i> <b><sup>1</sup></b>
    </p>
  </div>
<br>

## 4.2 Project Sequence-to-Sequence Model

### 4.2.1 Data Preparation

The training, validation, and testing data for developing the seq2seq model in this project were derived from the *Liu et al.* model codebase. **<sup>9</sup>**

These data sets had already been processed as per the process described in **4.1.1**, and split into:
1. **Train sources** (products)
2. **Train targets** (reactants)
3. **Validation sources** (products)
4. **Validation targets** (reactants)
5. **Test sources** (products)
6. **Test targets** (reactants)

As the ultimate goal of this project is to **incorporate this model into AiZynthFinder**, the **prepended reaction type token** in the source sequences **was removed** to leave just the split target molecule SMILES sequence. Additionally, the **spaces between the characters** were removed to give **raw canonical SMILES**.

Additionally, the sources and target datasets were **combined** so that they could be **split before each training run**. This would allow us to **control the split ratio** during the model development process.

### 4.2.2 Model Architecture

As this project is to be an introduction to seq2seq models, the model architecture was **not based on the open source library** provided by *Britz et al.*. Instead, a **custom model** was implemented based on the architecture described by *Liu et al.*, to act as a **baseline** for future model iterations.

### 4.1.3 Model Optimisation

### i. Deterministic Training Environment

**Determinism** when using machine learning frameworks is to have **exact reproducibility from run to run**, with a model's training run **yielding the same weights**, and a model's inference run **yielding the same prediction**. **<sup>10</sup>**

In the context of optimizing model performance, this is useful as it **reduces noise/random fluctuations in data** between training runs, ensuring any improvement or reduction in performance is solely the result of the hyperparameter change, change in model architecture etc.

Following the **NVIDIA documentation for Clara**, **<sup>10</sup>** the following steps were taken to ensure **deterministic training** in the [training environment set up](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/trainers/environment.py).
* Set environment variable for **Python's built-in has seed**.
* Set seeds for the **pseudo-random number generators** used in the model for reproducible random number generation.
* Enabling **deterministic operations** in TensorFlow.

Additionally, the environment set up gives the optional measure of **disabling GPU** and **limiting TensorFlow to single-threaded execution**. This is because modern GPUs and CPUs are designed to execute computations **in parallel across many cores**. This parallelism is typically managed **asynchronously**, meaning that the order of operations or the availability of computing resources can vary slightly from one run to another. 

It is this asynchronous parallelism that can introduce random noise, and hence, non-deterministic behaviour.

Setting up a custom deterministic training environment was used as an introduction to determinism in machine learning. Future models will use the [machine learning reproducibility framework package](https://github.com/NVIDIA/framework-reproducibility/tree/master) developed by NVIDIA.

### ii. Data Tokenization and Preprocessing Optimisation (DeepChem Tokenizer and TensorFlow TextVectorisation)

Despite promising training, validation and test accuracy (~68%) and loss (~0.10) for a full training run of an early model version, BLEU score remained very low (~2%). Additionally, once the seq2seq model was integrated into AiZynthFinder, analysis of the retrosynthesis predictions showed that they were converging on SMILES strings containing **all carbons** (either `C` or `c`).

Debugging of tokenizer showed that space characters between the individual chemical characters were also being tokenized. This explains the relatively **high token-level accuracy**, but very **low sequence-level accuracy** (BLEU score). This also may explain why the model was **overfitting to the most frequent tokens (i.e. `C` and `c`)**.

In an attempt to resolve this issue, various new analytics and debugging tactics were employed for a **more granular analysis** of model performance. This included:
* Logging tokenizer **word index mappings** and **token frequency distribution** for both **tokenized products** and **tokenized reactants**.
* **Verifying tokenization process** by manually tokenizing and detokenizing known canonical SMILES strings, and logging random SMILES strings from all data sets throughout the training process.
* Adding **more validation metrics**, particularly **string and chemical validity metrics**.

The initial tokenizer **fitted the tokenized SMILES list** on a `tensorflow.keras.preprocessing.text.Tokenizer` instance for **character-level tokenization**. This had the advantage of being **simple to implement**, and didn't introduce much **computational overhead** in the model training/inference runs.

However, not only was this resulting in spaec characters being tokenized, but it would also result in **loss of chemical semantic meaning** as there was no mechanism in place to account for **multi-character tokens** such as `Cl`, `Br` etc. As a result, these would be split into `C`, `l` and `B` `r` etc. 

Moreover, research into the **TensorFlow Keras documentation** found that the `tensorflow.keras.preprocessing` module was **deprecated**.

Therefore, an alternative strategy was employed whereby `deepchem.feat.smiles_tokenizer.BasicSmilesTokenizer` would be used to **generate the list of tokenized SMILES strings** while **preserving chemical information**, and this list would then be **adapted onto a `tensorflow.keras.layers.TextVectorization` layer instance**. This **`TextVectorization` layer** is a **more modern TensorFlow integration**, allowing for **better integration with the model graph**.

Analysis using the metrics described above showed that this new approach was vastly superior, with an **improvement of BLEU score to ~17%** even with **throttled hyperparameters**.

### iii. Loss Function and Callbacks Optimistion

### iv. Encoder Optimisation (Residual Connections)

Intial baseline model encoder architecture consisted of **2 bidirectional LSTM layers**, with hyperparameters matching those outlined by *Liu et al.* **<sup>1</sup>** (**Table 8**). However the **attention, encoder and decoder embedding dimensions**, as well as the **units** were all decreased first to **256**, then to **128** for efficient hardware usage while testing subsequent model versions.

The first siginificant encoder change implemented during the optimisation process was to **test 4 bidirectional LSTM layers**, as this was **missing in the analysis** by *Britz et al.*. This resulted in **marginal improvement**, but a **significant increase in computation**.

The second significant encoder change was the implementation of **residual connections**. 
* Residual connections are **direct pathways** that allow the **output of one layer to be added to the output of a deeper layer in the network**.
* Instead of data flowing **strictly through a sequence of layers**, residual connections provide **shortcuts that bypasss one or more layers**.

The benefits of residual connections include:
* **Mitigating the Vanishing/Exploding Gradient Problem**: Residual connections help this by **providing alternative pathways** for gradients to **flow backward through the network**, ensuring that gradients **remain sufficiently large** (mitigating vanishing gradients), while being **stable** (mitigating exploding gradients).
* **Enabling Identity Mappings**: Residual connections **apply identity mappings**, making it easier for **layers to learn identity functions** if necessary. This flexibility allows the network to **adaptively utilize or bypass certain layers**, enchancing its capacity to **model complex data**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/9082fa4e-0eb2-402b-a494-a29740efd7d4", alt="residual-connection"/>
    <p>
      <b>Fig 1</b> Residual connection in a FNN <b><sup>11</sup></b>
    </p>
  </div>
<br>

### vi. Decoder Optimisation (Residual Connections, Layer Normalisation)

Initial baseline model decoder architecture consisited of **4 unidirectional LSTM layers** with hyperparameters matching those outlined by *Liu et al.* **<sup>1</sup>** (**Table 8**). However, **decoder embedding dimension** and **units** were decreased first to **256**, then to **128** for efficient hardware usage while testing subsequent model versions.

The first significant change was the **adddition of residual connections were added to the decoder** (**Fig 1**). This resulted in an **improvement in both accuracy and loss** for training, validation and testing. This was at odds to what was reported by *Britz et al.* (**Table 4** and **Table 5**). This need for residual connections between layers is likley due to the increased semantic complexity of SMILES strings.

The second significant change was to incorporate **layer normalisation** into the decoder.
* **Normalisation** works by **mapping all the values of a feature** to be in the **range [0,1]**.
* Normalisation techniques are employed in neural networks to:
  * **Stabilise training**: By **standardising inputs to layers**, they help to **maintain consistent activation scales**.
  * **Accelerate Convergence**: This enables the use of **higher learning rates** without the **risk of divergence**.
  * **Improve generalisation**: By acting as a form of **regularisation**, reducing overfitting.
  * **Mitigate Internal Coveriate Shift**: By **reducing the change in the distribution of network activations** during training.

The first normalisation technique to consider is **batch normalisation**. In batch normalisation, the **inputs in each batch are scaled** so that they have a **mean of 0 (zero mean)** and a **standard deviation of 1 (unit standard deviation)**. Batch normalisation is applied **between the hidden layers of the encoder and/or decoder**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/6fdc7bd1-1f0f-450b-938e-83a2df51fb68", alt="batch-normalisation-overview"/>
    <p>
      <b>Fig 2</b> Section of a neural network with a Batch Normalisation Layer <b><sup>12</sup></b>
    </p>
  </div>
<br>

To get the output of any hidden layer `h` within a neural network, we pass the inputs through a **non-linear activation function**. To **normalise the neurons (activation) in a given layer (`k-1`)**, we can **force the pre-activations** to have a **mean of 0** and a **standard deviation of 1**. In batch normalisation this is achieved by **subtracting the mean from each of the input features across the mini-batch** and **dividing by the standard deviation**.

Following the output of the **layer `k-1`**, we can add a **layer that performs this batch normalisation operation** across the **mini-batch** so that the **pre-activations at layer `k` are unit Gaussians** (**Fig 2**.

As a high-level example, we can consider a mini-batch with **3 input samples**, with each **input vector** being **four features long**. Once the **mean and standard deviation** is computed for **each feature in the batch dimension**, we can **subtract the mean** and **divide by the standard deviation** (**Fig 3**). 

In reality, forcing all pre-activations to have a **zero mean** and **unit standard deviation** can be **too restrictive**, so batch normalisation **introduces additional parameters**, but this is beyond the scope of this project.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/08e5dda1-8a59-474f-8793-b287424579b2", alt="how-batch-normlisation-works"/>
    <p>
      <b>Fig 3</b> How batch normalisation works <b><sup>12</sup></b>
    </p>
  </div>
<br>

**Layer normalisation** is a normalisation technique introduced to address some of the limitations of **batch normalisation**. In layer normalisation, **all neurons in a particular layer** effectively have the **same distribution across all features for a given input**.
* For example, if each input has **`d` features, it is a **d-dimensional vector**. If there are **`B` elements** in a batch, the normalisation is done **along the length of the d-dimensional vector** and **not across the batch of size `B`**.

Normalising **across all features of each input removes the dependence on batches/batch statistics**. This makes layer normalisation **well suited for sequence models** such as seq2seq models, RNNs and transformers.

*Fig 4** illustrates the same example as earlier, but with **layer normalisation instead of batch normalisation**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/71187197-02ad-463a-934a-f15abd887344", alt="how-layer-normalisation-works"/>
    <p>
      <b>Fig 4</b> How layer normalisation works <b><sup>12</sup></b>
    </p>
  </div>
<br>

### vii. Attention Mechanism Optimisation

Initial baseline model used an **additive (Bahdanau) attention mechanism** in line with the mechanism used by *Liu et al.* **<sup>1</sup>**, with the **same dimension** (**Table 8**). However, **attention dimension** and **units** were decreased first to **256**, then to **128** for efficient hardware usage while testing subsequent model versions.

## 4.3 References
**[1]** Liu, B. et al. (2017) ‘Retrosynthetic reaction prediction using neural sequence-to-sequence models’, ACS Central Science, 3(10), pp. 1103–1113. <br><br>
**[2]** Lowe, D. M. (2012) ‘Extraction of Chemical Structures and Reactions from the Literature’; University of Cambridge. <br><br>
**[3]** Schneider, N., Stiefl, N. and Landrum, G.A. (2016) ‘What’s what: The (nearly) definitive guide to reaction role assignment’, Journal of Chemical Information and Modeling, 56(12), pp. 2336–2346. <br><br>
**[4]** Britz, D. et al. (2017) ‘Massive exploration of neural machine translation architectures’, Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. <br><br>
**[5]** He, K. et al. (2016) ‘Deep residual learning for image recognition’, 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). <br><br>
**[6]** Bahdanau, D. et al. (2015) ‘Neural machine translation by jointly learning to align and translate’, Proceedings of the 2015 International Conference on Learning Representations (ICLR). <br><br>
**[7]** Luong, M. et al. (2016) ‘Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models’, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics. <br><br>
**[8]** Wu, Y. et al. (2016) ‘Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation’. <br><br>
**[9]** Pandegroup (2017) ‘Pandegroup/reaction_prediction_seq2seq’, GitHub. Available at: https://github.com/pandegroup/reaction_prediction_seq2seq/tree/master (Accessed: 09 October 2024). <br><br>
**[10]** Determinism (2023) NVIDIA Docs. Available at: https://docs.nvidia.com/clara/clara-train-archive/3.1/nvmidl/additional_features/determinism.html (Accessed: 17 October 2024). <br><br>
**[11]** Wong, W. (2021) What is residual connection?, Medium. Available at: https://towardsdatascience.com/what-is-residual-connection-efb07cab0d55 (Accessed: 18 October 2024). <br><br>

