# 3 Sequence-to-Sequence (Seq2Seq) Expansion Policy

As stated in **Section 2**, for **neural network-guided one-step retrosynthesis**, there are two primary methodologies used to **define the disconnection rules** and so **model the reverse reaction**: **<sup>1</sup>**
1. **Template-Based Methods**
2. **SMILES-Based (Template-Free) Methods**

This project aims to implement a SMILES-based (template-free) retrosynthetic method, and incorporate it into a **customised instance of AiZynthFinder**. The neural network architecture used is a **sequence-to-sequence (Seq2Seq)** model based on the work by *Pande et al.* **<sup>2</sup>**

## 3.1 Limitations of Template-Based Retrosynthetic Methods

A template-based retrosynthetic method is based on a predefined set of rules, and thus **inherit the limitations of these rules**:
1. The primary limitation of such methods is that they are **fundamentally dependent on the rules on which the neural network is trained**, and thus these approaches have **issues with making accurate predictions outside of this rule-based knowledge base**. **<sup>2</sup>**
   
2. There is also an **inherent trade-off** between defining **general rules**, which can **introduce noise** and **reduce the accuracy or reliability of a model’s predictions**, and defining **very specific rules**, which **restrict the model’s predictions to a narrow set of reactants and products**. **<sup>2</sup>**
   
3. Additionally, the reaction rules are **inadequate representations of the underlying chemistry** as they focus on **local reaction center molecular environments only**. **<sup>2</sup>**
   
4. Finally, a rules-based system **does not fully account for stereochemistry**. **<sup>2</sup>**

## 3.2 Alternative SMILES-Based Retrosynthetic Method

A deep learning approach that **avoids a template-based/rule-based approach** could **avoid the above limitations**.

*Pande et al.* appraoched the problem as a **sequence-to-sequence prediction task**, **mapping a text-based linear notation of the reactants to that of the product, or vice versa**. **<sup>2</sup>** 

In their paper, *Pande et al.* reference the work of **Nam and Kim**, where a **neural Seq2Seq model** was employed for **forward reaction prediction**, using the **SMILES representation** of reactants as input to predict the SMILES of the product. **<sup>3</sup>** *Pande et al.* aimed to extend this approach to **retrosynthetic (backward) reaction prediction**.

## 3.2 Seq2Seq Model

A seq2seq model is a type ofg neural network architecture desgined **convert sequences from one domain** (e.g. sentences in English) to **sequences in another domain** (e.g. thew same sentences translated to French). 

Seq2seq models are especially useful for tasks where the **input and output are sequences of varying length**, which traditional neural networks struggle to handle. As such, they are widely used in the field of **natural langauge processing (NLP)**, such as machine translation, text summarisation and conversational modeling.

## 3.3 Architecture of Seq2Seq Models

At its core, a Seq2Seq model consists of two main components: an **encoder** and a **decoder**. These components **work in tandem** to **process input sequences** and **generate corresponding output sequences**.

Both the encoder and decoder are neural networks, specifically a type of **recurrent neural network (RNN)** called **Long Short-Term Memory (LSTM)** models (or sometimes **Gated Recurrent Unit (GRU)** models)

### 3.3.1 Encoder

* **Function**: The encoder reads the input sequence and **compresses it into a fixed-size vector**, called **internal state vectors** or **context vectors**. In the case of LSTM models, these are called the **hidden state vector** (`state_h`) and **cell state vector** (`state_c`).
* **Structure**: Often implemented using RNNs like LSTM or GRU. These can a a **single RNN layer**, or a **stack of several RNN layers**. However other architectures like **Transformers** can serve as encoders.
* **Operation**: It **reads the input sequence token by token** and **updates its hidden state accordingly**, thus **capturing the information from the entire input**. The **outputs of the encoder are discarded** and only the **internal states/context vectors are preserved**

The context vectors aims to **encapsulate the information for all input elements** in order to **help the decoder make accurate predictions**.

Using the example of an LSTM RNN, an LSTM unit maintains **two types of internal states**:
1. **Hidden State ($$h_t$$)**: This represents the **output of the LSTM at time step $$t$$**.
2. **Cell State ($$c_t$$)**: This **carries long-term dependencies** and acts as a **memory** for the LSTM.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/2bef59f1-672e-41e6-a211-dd25c8a8b412", alt="seq2seq-model-encoder"/>
    <p>
      <b>Fig 1</b> LSTM encoder architecture in a Seq2Seq model <b><sup>4</sup></b>
    </p>
  </div>
<br>

In **Fig 1**, we can see that the **LSTM encoder reads the input data, one token after the other**. Thus, if the input is a sequence of **length $$t$$**, we say that the LSTM encoder reads it in **$$t$$ time steps**:
1. **$$X_i$$**: The input sequence at **time step $$i$$**
   
2. **$$h_i$$ and $$c_i$$**: The LSTM encoder **maintains two states at each time step**. This is the **hidden state $$h$$** and the **cell state $$c$$**. Combined, these are the **internal state of the LSTM encoder** at **time step $$i$$**

3. **$$Y_ii$$**: This is the **output sequence at time stel $$i$$**. $$Y_ii$$ is actually a **probability distribution over the entire vocabulary** generated by using a **softmax activation**. Thus, each $$Y_ii$$ is a **vector of size `vocab_size`**, representing a probability distribution


### 3.3.2 Decoder

* **Function**: The decoder then **generates the output sequence** from the **context vector** provided by the encoder.
* **Structure**: Like the encoder, it is often implemented using RNNs like LSTM or GRU. These can a a **single RNN layer**, or a **stack of several RNN layers**
* **Operation**: It **produces the output sequence one token at a time**, using the **context vector** and its **own previously generated output tokens** to **predict the next token in the sequence**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/93afac63-7dc2-42d6-a307-6c1e3cf71c04", alt="seq2seq-lstm-encoder-decoder-internal-state-transfer"/>
    <p>
      <b>Fig 2</b> Encoder final internal state vectors are used as the initial state of the decoder <b><sup>5</sup></b>
    </p>
  </div>
<br>

An LSTM decoder uses the **final internal state vectors of the LSTM encoder** as its **initial state** (**Fig 2**). Using these **initial states**,  the LSTM decoder **starts generating the output sequence, one token at a time**, and the **internal state vectors of the output token** are used to **predict future output tokens**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/4a366c6d-b7ff-4876-a73c-1cdb53e35b24", alt="seq2seq-model-decoder"/>
    <p>
      <b>Fig 3</b> LSTM decoder architecture in a Seq2Seq model <b><sup>4</sup></b>
    </p>
  </div>
<br>

In **Fig 3**, we can see that at the beginning of its prediction: the decoder is fed the **final internal state vectors of the encoder**, and a **start token**
1. **Initialisation**
   * **Initial States**: The **final hidden state ($$h_0$$)** and the **final cell state ($$c_0$$)** from the LSTM encoder are used to **initialise the LSTM decoder's hidden and cell states**.

$$h_0^{dec} = h_{enc}$$

$$c_0^{dec} = c_{enc}$$

2. **Generating the First Token**
   * **Input to Decoder**: A **special start-of-sequence token** (e.g. **`<START>`**) is **fed into the decoder**.
   * **Processing**: The LSTM decoder **processes this token**, along with the **initial states** to **produce the first output token** and **update its internal states**.
     
$$h_1^{dec},\ c_1^{dec} = \text{LSTM}(\langle \text{START} \rangle,\ h_0^{dec},\ c_0^{dec})$$

   * **Output**: The **first token $$Y_1$$** (e.g. "Hello") is generated.

3. **Generating Subsequent Tokens**
   * **Input**:
      * **During training** - A technique called **teacher forcing** is used whereby the **ground truth token** (i.e. the **correct token**) from the **previous time step** is passed to the LSTM decoder for the next token generation.
      * **During Inference** - The **decoders own predicted token** from the previous time step is used.
   * **Processing**: The LSTM decoder **processes this input**, along with the **current hidden and cell states** to generate the **next token** and **update its internal states**

4. **Iterative Processing**
   * Step 3 **continues iteratively**:
      * **Internal State Propagation**: At each time step, the **current hidden and cell states** ($$ $$) **encapsulate all the information processed up to that point**.
      * **Prediction dependency**: The prediction of the next token depends not only on the **current input token**, but also on the **cumulative context captured in the internal states**.
    
$$h_t^{dec},\ c_t^{dec} = \text{LSTM}(x_{t-1},\ h_{t-1}^{dec},\ c_{t-1}^{dec})$$

where:
   * $$x_{t-1}$$ = The **input token at time $$t - 1$$** (either **ground truth token** if **teacher forcing**, or **previous prediction**).
   * $$h_t^{dec},\ c_t^{dec}$$ = The **previous hidden and cell states**


Within the decoder, there is **another key component** of a Seq2Seq model we must talk about; The **Attention Mechanism**.

### 3.3.3 Attention Mechanism

The **attention mechanism** is a **pivotal enhancement** to Seq2Seq models, significantly improving their performance/accuracy. Introduced to address the limitations of traditinoal Seq2Seq architectures, the attention mechanism allows the decoder to **dynamically focus on the most relevant parts of the input sequence at each time step**.

One of the **main limitations** of basic Seq2Seq models is that relying on the **compression of all input information into a single context vector can be problematic**, especially for **long sequences**, as it **may not capture all the necessary information effectively.**

The attention mechanism was introduced to **mitigate the limitations of the fixed-size context vector** by allowing the decoder to **access different parts of the input sequence dynamically during each time step of the output generation**. Therefore, instead of **summarising the entire input into a single vector**, attention enables the model to **create a context vector tailored to each output token**.

The **key aspects of the attention mechanism** include: **<sup>4</sup>** 
1. **Dynamic Weighting**
   * Instead of relying on a **fixed-length context vector** to encode the entire input sequence, attention mechanisms **assign different weights** to **different parts of the input sequence** based on their **relevance to the current step of the output sequence**.
   * This dynamic weighting enables the model's decoder to focus more on **relevant information** and **ignore irrelevant parts**.
     
2. **Soft Alignment**
   * Attention mechanisms create a **soft alignment** between the **input and output sequences** by computing a **distribution of attention weights over the input sequence**.
   * This allows the model's decoder to **consider multiple input elements simultaneously** at each time step, unlike **hard alignment methods** that force the model to **choose only one input element at each time step**.
     
3. **Scalability**
   * Attention mechanisms are **scalable to sequences of varying lengths**.
   * This means they can **adapt to longer input sequences without significantly increasing computational complexity**, unlike fixed-length context vectors, which may **struggle with long sequences**.
     
4. **Interpretable Representations**:
   * **Attention weights** represent the **model's decision-making process**.
   * By visualising these weights, researchers and practioners can **gain insight into which parts of the input sequence are most relevant** for generating specific parts of the output sequence
  
At a high-level, the **attention mechanism's role within the decoder** is as follows:
1. **Encoder Processing**
   * The encoder prcoesses the input sequence and **produces a sequence of hidden states** $$\\{h_1,h_2,...,h_n\\}$$, where each $$h_i$$ corresponds to a the **hidden state of an input token**

2. **Decoding with Attention**
   * For **each decoding time step $$t$$**:
     1. **Compute Alignment Scores:**
        * For each hidden state $$h_i$$, **calculate a score $$e_{t,i}$$** that indicates its **relevance to the current decoding step**. Common methods for this include:
        * **Example 1 - Dot-Product:** $$e_{t,i} = h_t^{dec} . h_i^{enc}$$
        * **Example 2 - Additive (Bahdanau) Attention:** $$e_{t,i} = v^T tanh(W_1h_t^{dec} + W_2h_i^{enc})$$
          
     2. **Generate Attention Weights**:
        * Apply a **softmax function** to the alignment scores to **obtain attention weights $$\alpha_{t,i}$$**:
          
          $$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{n} \exp(e_{t,j})}$$
        
        * These weights indicate the **importance of each encoder hidden state** for the **current decoding step**
          
     3. **Compute the Context Vector**:
        * Create a context vector **$$c_t$$** as a **weighted sum of the encoder hidden states**:
    
          $$c_t = \sum_{i=1}^{n} \alpha_{t,i}h_i^{enc}$$
          
        * This vector **encapsulates the most relevant information from the input sequence** for generating the current output token.

     4. **Generate the Output Token**:
        * The decoder then uses the context vector $$c_t$$, along with its **previous hidden state** and the **previous output token**, to generate the current output token:
          
          $$h_t^{dec} = \text{LSTM}(y_{t-1},h_{t-1}^{dec},c_t$$

          $$\hat{y}_t = \text{Softmax}(Wh_t^{dec} + Vc_t)$$
       
        * Here **$$y_{t-1}$$** is the **previous token**, **$$\hat{y}_t$$** is the **predicted token** at **step $$t$$**, and **$$W$$**, **$$V$$** are **weighted matrices**.
       
   3. **Iterative Process**:
      * Step 2 is **repeated iteratively** for **each token in the output sequence**, allowing the decoder to **focus on different parts of the input as needed**.
       


## References
**[1]** Saigiridharan, L. et al. (2024) ‘AiZynthFinder 4.0: Developments based on learnings from 3 years of industrial application’, Journal of Cheminformatics, 16(1). <br><br>
**[2]** Liu, B. et al. (2017) ‘Retrosynthetic reaction prediction using neural sequence-to-sequence models’, ACS Central Science, 3(10), pp. 1103–1113. <br><br>
**[3]** Nam, J., Kim, J. (2016) ‘Linking the Neural Machine Translation and the Prediction of Organic Chemistry Reactions’. 1612.09529. <br><br>
**[4]** Introduction to seq2seq models (2024) Analytics Vidhya. Available at: https://www.analyticsvidhya.com/blog/2020/08/a-simple-introduction-to-sequence-to-sequence-models/ (Accessed: 30 September 2024). <br><br>
**[5]** Chollet, F. (no date) The keras blog, The Keras Blog ATOM. Available at: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html (Accessed: 30 September 2024). <br><br>
