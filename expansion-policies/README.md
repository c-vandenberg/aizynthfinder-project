# 2. AiZynthFinder's Expansion Policy Neural Network

For **neural network-guided one-step retrosynthesis**, there are two primary methodologies used to **define the disconnection rules** and so **model the reverse reaction**: **<sup>1</sup>**
1. **Template-Based Methods**
2. **SMILES-Based (Template-Free) Methods**

AiZynthFinder primarily uses a **template-based method** during its **expansion policy** (*c.f.* **Section 1.3**).

In template-based retrosynthetic methods, a set of **predefined molecular transformations** are applied to the target molecule. These template rules can either be **bespoke rules written by in-house synthetic chemists**, or obtained by **data mining reaction databases**. **<sup>2</sup>**

The neural network is trained on these template rules, and the trained model is fed a **molecular representation of the target molecule** (in AiZynthFinder's case, an **ECFP4 bit string**) as input. The trained model then **predicts the most appropriate template to use**, giving a **set of reaction precursors**. This process is then **recursively employed** to **construct a retrosynthetic tree**. **<sup>1</sup>** **<sup>3</sup>**

## 2.1 What is AiZynthFinder's Expansion Policy Neural Network?

As its standard template-based expantion policy, AiZynthFinder employs a type of ***feedforward neural network** called a **Muti-Layer Perceptron**. **<sup>4</sup>** This network is designed to predict the applicability of various reaction templates to a given target molecule during retrosynthetic planning.

The architecture effectively **maps molecular representations to reaction probabilities**, generating a **ranked list of reaction templates** representing the most feasbile sets of reactions.

## 2.2 Neural Networks Overview

**Neural networks** are machine learning models inspired by the structure and function of **biological neural networks** in animal brains. They consist of **layers of nodes (articial neurons)** that process input data to produce an output.

Each node/neuron can be thought of as a **linear regression model**, that involves **computing a weighted sum of inputs, plus a bias**. As such, each node/neuron consists of:
1. **Input Data**: The data the node receives.
   
2. **Weights**: Numerical parameters that determine the **strength and direction** of the **connection between neurons in adjacent layers**. Each input is **assigned a weight** which helps to **determine the correlation of each input to the output**.
   * **Positive Weights** indicate a **positive correlation between the input and the output**. With positive weights, as the **input increases**, the **neurons activation tends to increase**.
   * **Negative Weights** indicate a **negative correlation between the input and the output**. With negative weights, as the **input increases**, the **neurons activation tends to decrease**.
     
3. **Biases**: These **shift the activation threshold**, enabling the **neuron to activate even when all input signals are zero**. This allows the model to better fit training data by allowing neurons to **activate in a broader range of scenarios**.
   
4. **Output data**: The output value passed to the next node/neuron in the adjacent layer if it **exceeds the activation threshold**.
   * Once **all inputs are multipied by their respective weights and summed**, this value is **passed through an activation function**, which determines the output.
   * If this output **exceeds the given activation threshold**, the node/neuron **fires (or activates)**, and the **output data is passed to a node/neuron in the next layer** in the network.
   * As a result, the **output of one node/neuron** becomes the **input of the next node/neuron**.
  
Mathematically, this can be represented as:

1. **Input Data**: The node/neuron receives inputs from the previous layer (or the input data itself if it is the input layer).
   
2. **Weighted Sum**: Each input is **multiplied by its corresponding weight**, and all these products are **summed together along with the bias**.
   * This is why the bias can **shift the activation threshold**, because if it is a large enough value, it can cause the output to exceed the threshold even if all inputs are zero.

$$z =  \sum_{i=1}^{n} w_ix_i + b$$

where:
  * $$x_i$$ = Input from the $$i^{th}$$ neuron of the previous layer (or $$i^{th}$$ feature if input layer).
  * $$w_ix_i$$ = Weight associated with the $$i^{th}$$ input.
  * $$b$$ = The bias term.
  * $$z$$ = The **weighted sum + bias term**. This is also known as the neuron's **pre-activation value**.

3. **Activation Function**: The **weighted sum/pre-activation value** $$z$$ is then **passed through an activation function** $$\sigma(z)$$ to produce the **node/neuron's output**.
   * This activation function introduces **non-linear properties** to the neural network, allowing the model to **learn more complex patterns**. Without the activation function, the neural network would only be able to learn **linear patterns**.

$$ a = \sigma(z)$$

where:
  * $$a$$ = The **activated output of the neuron**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/f4c954c2-e0f7-43f4-a1a7-82d2699696e9", alt="simple-neural-network"/>
    <p>
      <b>Fig 1</b> Simple neural network architecture. <b><sup>5</sup></b>
    </p>
  </div>
<br>

The architecture of a simple neural network is shown in **Fig 1** and consists of:

1. **Input Layer**
   * **Function**: Nodes/neurons in the input layer receive inputs/input data and pass them onto the next layer (first hidden layer). Depending on the neural network architecture, the nodes/neurons in the input layer may also calculate weighted sums and pass them through activation functions.
   * **Structure**: The number of nodes/neurons in the input layer is determined by the number of dimensions/features of the input data.

2. **Hidden Layers**
   * **Function**: These layers are **not exposed directly to raw input or output data** and can be considered as the **computational engines** of the neural network. Each hidden layer's nodes/neurons take the outputs from **nodes in the previous layer** as input, compute a **weighted sum of these inputs** and apply an **activation function**. This output is then **pass the result to the next layer**. 
   * **Structure**: One or more layers with neurons applying activation functions to weighted sums of inputs.
  
3. **Output Layer**
   * **Function**: Produces the final prediction or outoput for the given inputs.
   * **Structure**: Number of nodes/neurons in the output layer depends on the desired output format (e.g., classes for classification).
  
**N.B.** Although **deep learning and neural networks are often used interchangeably**, it is worth noting that the **"deep"** in deep learning simply refers to the **depth of the layers** in a neural network. Generally, a neural network that consists of **more than three layers** (i.e. an input layer, one hidden layer, and an output layer) can be considered a **deep learning neural network** (**Fig 2**).

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/04f5f505-b0c1-44af-8f63-23142b4e8d21", alt="deep-learning-neural-network"/>
    <p>
      <b>Fig 2</b> Deep learning neural network schematic. <b><sup>6</sup></b>
    </p>
  </div>
<br>
  
## 2.3 Feedforward Neural Networks (FNNs)

**Feedforward Neural Networks (FNNs)** are one of the **simplest type** of artificial neural networks. In FNNs, **data moves in only on direction - forward -** from the input nodes, to the hidden nodes (if any), and to the output nodes. There are **no cycles or loops** in the network.

The first type of neural network developed was called **single-layer perceptrons**. These consisted of only an **input layer** and an **output layer** and could only recognise/predict **linear patterns** between the input and output data, as there were no hidden layers (and so no associated activation functions) to **introduce non-linearity**.

FNNs were the first type of artificial neural network invented and are simpler than their counterparts like **recurrent neural networks** and **convolutional neural networks**.

Modern FNNs are called **multilayer perceptrons (MLPs)** and consist of **one or more hidden layers** as well as the input and output layer. As a result, they are able to recognise/predict **non-linear patterns** between the input and output data, due to **nonlinear activation functions** present within the hidden layers.

The training of MLP FNNs involves two main phases:
1. **Feedforward Phase**:
   * In this phase, the input data is fed into the network, and it **propagates forward through the network.
   * At each hidden layer, the **weighted sum of the inputs** from the previous layer is calculated and **passed through an activation function**, introducing non-linearity into the model.
   * This process continues **until the output layer is reached**, and a **prediction is made**.
2. **Backpropagation Phase**:
   * Once a prediction is made, the **error** (the **difference between the predicted output** and the **actual output**) is calculated using a **loss function** (also known as a **cost function**).
   * This error is then **propagated back through the network**, and the **weights are adjusted to minimize the error**.
   * The process of adjusting the weights is typically done using a **gradient descent optimization algorithm**.

This is an **iterative process** where the training dataset is **passed through the network multiple times**, and each time the **weights are updated to reduce the error in prediction**. This process is known as **gradient descent**, and it continues until the model reaches a **point of convergence (i.e. where the loss funtion is at a minimum)**, or another **stop criterion is reached** (**Fig 3**).

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/16d294fc-c27e-442f-986d-a8d2619c7473", alt="loss-function-gradient-descent"/>
    <p>
      <b>Fig 3</b> Gradient descent of error in prediction, calculated by a loss function. <b><sup>6</sup></b>
    </p>
  </div>
<br>

## 2.4 Recurrent Neural Networks (RNNs)

**Recurrent Neural Networks (RNNs)** are a special type of artificial neural network adapted to work for **time series data** (i.e. **data that involves sequences**). The general format of this sequential data is $$x(t) = x(1), . . . , x(\tau)$$, with a **time step index $$t$$** ranging from **$$1 to \tau$$**.

RNNs are trained to process and convert **sequential data input into a specific sequential data output**.
   * Sequential data is data such as **words, sentences or time-series data** where **sequential components interrelate based on complex semantic and syntax rules**.

**Natural Language Processing (NLP)** is an example of a problem that involves sequential inputs. In an NLP problem, if you want to **predict the next word in a sentence**, it is important to **know the words before it**.

RNNs are called **recurrent** because they **perform the same task for every element of a sequence**, with the **output being dependent on the computations of the previous elements**. To put it another way, RNNs have a **'memory'** which **captures information about what has been calculated so far**.

### 2.4.1 Recurrent Neural Network Architecture

**Computational graphs** are a way to **formalise the structure and set of given computations**, such as those involved in **mapping inputs and parameters** to **outputs and loss**. **<sup>7</sup** 

The **nodes** in the graph represent **variables** which can be a **scalar, vector, matrix, tensor etc**. The **edges** in the graph correspond to **operations** that **transform one variable to another**.

For **recursive or recurrent computation**, such as those in an RNN, the computational graph can be **unfolded** into another computational graph that has a **repetitive structrue**, typically corresponding to a **chain of events**. 

The architectural notation of a basic RNN with **no output** is shown in **Fig 4** has a **feedback loop** (**Fig 4**)

<br>
  <div align="center">
    <img src="", alt="basic-rnn-architecture-notation-no-ouput"/>
    <p>
      <b>Fig 4</b> . <b><sup>7</sup></b>
    </p>
  </div>
<br>

The **left side** of **Fig 4** shows the **computational graph** of a **RNN with no outputs**. This RNN simply **processes input data %%x%%** by **incorportating it into the state $$h$$**. This state $$h$$ is then $$passed forward through time$$. The **black square** represents the **delay of a single time step**.

The **right side** of **Fig 4** shows the same RNN but as a **unfolded computational graph**, where **each input ($$x$$) and state ($$h$$) node** is now **associated with one particular time instance**. This unfolding simply means that we **represent the network as its complete sequence**.
   * For example, if the sequence being processed is a **sentence of 3 words**, the network would be **unfolded into a 3 time step neural network**, with **one time step for each word**.

<br>
  <div align="center">
    <img src="", alt="basic-rnn-architecture-notation-full"/>
    <p>
      <b>Fig 5</b> . <b><sup>7</sup></b>
    </p>
  </div>
<br>

Expanding this 

## 2.5 References
**[1]** Saigiridharan, L. et al. (2024) ‘AiZynthFinder 4.0: Developments based on learnings from 3 years of industrial application’, Journal of Cheminformatics, 16(1). <br><br>
**[2]** Fortunato, M.E. et al. (2020) ‘Data augmentation and pretraining for template-based retrosynthetic prediction in computer-aided synthesis planning’, Journal of Chemical Information and Modeling, 60(7), pp. 3398–3407. <br><br>
**[3]** Thakkar, A. et al. (2020) ‘Datasets and their influence on the development of computer assisted synthesis planning tools in the pharmaceutical domain’, Chemical Science, 11(1), pp. 154–168. <br><br>
**[4]** Genheden, S., Engkvist, O. and Bjerrum, E.J. (2020) 'A quick policy to filter reactions based on feasibility in AI-guided retrosynthetic planning.' <br><br>
**[5]** Chen, J. (no date) What is a neural network?, Investopedia. Available at: https://www.investopedia.com/terms/n/neuralnetwork.asp (Accessed: 30 September 2024). <br><br>
**[6]** Ibm (2024) What is a neural network?, IBM. Available at: https://www.ibm.com/topics/neural-networks (Accessed: 30 September 2024). <br><br>
**[7]**
