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
* **Function**: The encoder reads the input sequence and **compresses it into a fixed-size vector**, called **interal state vectors** or **context vectors**. In the case of LSTM models, these are called the **hidden state vector** (`state_h`) and **cell state vector** (`state_c`).
* **Structure**: Often implemented using RNNs like LSTM or GRU, though other architectures like **Transformers** can serve as encoders.
* **Operation**: It **reads the input sequence token by token** and **updates its hidden state accordingly**, thus **capturing the information from the entire input**. The **outputs of the encoder are discarded** and only the **internal states/context vectors are preserved**

The context vector aims to **encapsulate the information for all input elements** in order to **help the decoder make accurate predictions**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/2bef59f1-672e-41e6-a211-dd25c8a8b412", alt="seq2seq-model-encoder"/>
    <p>
      <b>Fig 1</b> Seq2Seq model LSTM encoder architecture <b><sup>4</sup></b>
    </p>
  </div>
<br>

In **Fig 1**, we can see that the **LSTM encoder reads the input data, one token after the other**. Thus, if the input is a sequence of **length $$t$$**, we say that the LSTM encoder reads it in **$$t$$ time steps**:
1. **$$X_i$$**: The input sequence at **time step $$i$$**
   
2. **$$h_i$$ and $$c_i$$**: The LSTM encoder **maintains two states at each time step**. This is the **hidden state $$h$$** and the **cell state $$c$$**. Combined, these are the **internal state of the LSTM encoder** at **time step $$i$$**

3. **$$Y_ii$$**: This is the **output sequence at time stel $$i$$**. $$Y_ii$$ is actually a **probability distribution over the entire vocabulary** generated by using a **softmax activation**. Thus, each $$Y_ii$$ is a **vector of size `vocab_size`**, representing a probability distribution


### 3.3.2 Decoder

The encoder takes the input sequence and **compresses it into a fixed-size vector**, often referred to as the **context vector** or the **thought vector**. This vector contains the **most important information** from the input sequence and serves as the **initial state for the decoder**.
2. **Decoder**: The decoder then generates the output sequence by **predicting one token at a time**.


## References
**[1]** Saigiridharan, L. et al. (2024) ‘AiZynthFinder 4.0: Developments based on learnings from 3 years of industrial application’, Journal of Cheminformatics, 16(1). <br><br>
**[2]** Liu, B. et al. (2017) ‘Retrosynthetic reaction prediction using neural sequence-to-sequence models’, ACS Central Science, 3(10), pp. 1103–1113. <br><br>
**[3]** Nam, J., Kim, J. (2016) ‘Linking the Neural Machine Translation and the Prediction of Organic Chemistry Reactions’. 1612.09529. <br><br>
**[4]** Introduction to seq2seq models (2024) Analytics Vidhya. Available at: https://www.analyticsvidhya.com/blog/2020/08/a-simple-introduction-to-sequence-to-sequence-models/ (Accessed: 30 September 2024). 
