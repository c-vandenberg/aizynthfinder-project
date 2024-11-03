# 2. AiZynthFinder's Expansion Policy Neural Network

For **neural network-guided one-step retrosynthesis**, there are two primary methodologies used to **define the disconnection rules** and so **model the reverse reaction**: **<sup>1</sup>**
1. **Template-Based Methods**
2. **SMILES-Based (Template-Free) Methods**

AiZynthFinder primarily uses a **template-based method** during its **expansion policy** (*c.f.* **Section 1.3**).

In template-based retrosynthetic methods, a set of **predefined molecular transformations** are applied to the target molecule. These template rules can either be **bespoke rules written by in-house synthetic chemists**, or obtained by **data mining reaction databases**. **<sup>2</sup>**

The neural network is trained on these template rules, and the trained model is fed a **molecular representation of the target molecule** (in AiZynthFinder's case, an **ECFP4 bit string**) as input. The trained model then **predicts the most appropriate template to use**, giving a **set of reaction precursors**. This process is then **recursively employed** to **construct a retrosynthetic tree**. **<sup>1</sup>** **<sup>3</sup>**

## 2.1 What is AiZynthFinder's Expansion Policy Neural Network?

As its standard template-based expansion policy, AiZynthFinder employs a type of **feedforward neural network** called a **Multi-Layer Perceptron**. **<sup>4</sup>** This network is designed to predict the applicability of various reaction templates to a given target molecule during retrosynthetic planning.

The architecture effectively **maps molecular representations to reaction probabilities**, generating a **ranked list of reaction templates** representing the most feasible sets of reactions.

## 2.2 Neural Networks Overview

**Neural networks** are machine learning models inspired by the structure and function of **biological neural networks** in animal brains. They consist of **layers of nodes (artificial neurons)** that process input data to produce an output.

Each node/neuron can be thought of as a **linear regression model**, that involves **computing a weighted sum of inputs, plus a bias**. As such, each node/neuron consists of:
1. **Input Data**: The data the node receives.
   
2. **Weights**: Numerical parameters that determine the **strength and direction** of the **connection between neurons in adjacent layers**. Each input is **assigned a weight** which helps to **determine the correlation of each input to the output**.
   * **Positive Weights** indicate a **positive correlation between the input and the output**. With positive weights, as the **input increases**, the **neuron's activation tends to increase**.
   * **Negative Weights** indicate a **negative correlation between the input and the output**. With negative weights, as the **input increases**, the **neurons activation tends to decrease**.
     
3. **Biases**: These **shift the activation threshold**, enabling the **neuron to activate even when all input signals are zero**. This allows the model to better fit training data by allowing neurons to **activate in a broader range of scenarios**.
   
4. **Output data**: The output value passed to the next node/neuron in the adjacent layer if it **exceeds the activation threshold**.
   * Once **all inputs are multiplied by their respective weights and summed**, this value is **passed through an activation function**, which determines the output.
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
   * **Function**: Produces the final prediction or output for the given inputs.
   * **Structure**: Number of nodes/neurons in the output layer depends on the desired output format (e.g., classes for classification).
  
**N.B.** Although **deep learning and neural networks are often used interchangeably**, it is worth noting that the **"deep"** in deep learning simply refers to the **depth of the layers** in a neural network. Generally, a neural network that consists of **more than 3 layers** (i.e. an input layer, one hidden layer, and an output layer) can be considered a **deep learning neural network** (**Fig 2**).

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/04f5f505-b0c1-44af-8f63-23142b4e8d21", alt="deep-learning-neural-network"/>
    <p>
      <b>Fig 2</b> Deep learning neural network schematic. <b><sup>6</sup></b>
    </p>
  </div>
<br>
  
## 2.3 Feedforward Neural Networks (FNNs)

**Feedforward Neural Networks (FNNs)** are one of the **simplest type** of artificial neural networks. In FNNs, **data moves in only one direction - forward -** from the input nodes, to the hidden nodes (if any), and to the output nodes. There are **no cycles or loops** in the network.

The first type of neural network developed was called **single-layer perceptrons**. These consisted of only an **input layer** and an **output layer** and could only recognise/predict **linear patterns** between the input and output data, as there were no hidden layers (and so no associated activation functions) to **introduce non-linearity**.

FNNs were the first type of artificial neural network invented and are simpler than their counterparts like **recurrent neural networks** and **convolutional neural networks**.

Modern FNNs are called **multilayer perceptrons (MLPs)** and consist of **one or more hidden layers** as well as the input and output layer. As a result, they are able to recognise/predict **non-linear patterns** between the input and output data, due to **nonlinear activation functions** present within the hidden layers.

The training of MLP FNNs involves two main phases:
1. **Feedforward Phase/Forward Pass**:
   1. In this phase, the input data is fed into the network, and it **propagates forward through the network**.
   2. At each hidden layer, the **weighted sum of the inputs** from the previous layer is calculated and **passed through an activation function**, introducing non-linearity into the model.
   3. This process continues **until the output layer is reached**, and a **prediction is made**.
2. **Backpropagation Phase**:
   1. **Loss Calculation**: Once a prediction is made, the **error** (the **difference between the predicted output** and the **actual output**) is calculated using a **loss function**.
   2. **Backward Pass**: The **gradients of the loss with respect to weight** is then calculated by **applying the chain rule** and is **propagated back through the layers of the network**.
   3. **Weights Update**: Using these computed gradients, the **weights are adjusted to minimize the error**, typically using a **gradient descent optimization algorithm** such as **Stochastic Gradient Descent (SGD)** or **Adam**

This is an **iterative process** where the training dataset is **passed through the network multiple times**, and each time the **weights are updated to reduce the error in prediction**. This process is known as **gradient descent**, and it continues until the model reaches a **point of convergence (i.e. where the loss function is at a minimum)**, or another **stop criterion is reached** (**Fig 3**).

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/16d294fc-c27e-442f-986d-a8d2619c7473", alt="loss-function-gradient-descent"/>
    <p>
      <b>Fig 3</b> Gradient descent of error in prediction, calculated by a loss function. <b><sup>6</sup></b>
    </p>
  </div>
<br>

## 2.4 Recurrent Neural Networks (RNNs)

**Recurrent Neural Networks (RNNs)** are a special type of artificial neural network adapted to work for **time series data** (i.e. **data that involves sequences**). 
   * The general format of this sequential data is $$x(t) = x(1), . . . , x(\tau)$$, with a **time step index $$t$$** ranging from **$$1$$** to **$$\tau$$**.

RNNs are trained to process and convert **sequential data input into a specific sequential data output**.
   * Sequential data is data such as **words, sentences or time-series data** where **sequential components interrelate based on complex semantic and syntax rules**.

**Natural Language Processing (NLP)** is an example of a problem that involves sequential inputs. In an NLP problem, if you want to **predict the next word in a sentence**, it is important to **know the words before it**.

RNNs are called **recurrent** because they **perform the same task for every element of a sequence**, with the **output being dependent on the computations of the previous elements**. To put it another way, RNNs have a **"memory"** which **captures information about what has been calculated so far**. 

This "memory" is what **distinguishes it from FNNs** and is **passed between time steps in the RNN** in a **feedback loop**. This is represented by the **curly arrow** in **Fig 4**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/c1fb910c-81cc-4508-9ec4-ac51cb3cbacd", alt="recurrent-vs-feedforward-neural-networks"/>
    <p>
      <b>Fig 4</b> Recurrent Neural Networks (RNNs) vs Feedforward Neural Networks (FNNs). <b><sup>7</sup></b>
    </p>
  </div>
<br>

Another characteristic of RNNs that distinguishes them from FNNs is that they **share parameters across each time step within a layer, and across each layer of the network**:
* While FNNs have **different weights across each node**, RNNs **share the same weight parameter within each time step and layer of the network**.
* That said, while the weight parameter is shared across layers, the weights are **still adjusted through the processes of backpropagation and gradient descent** to **facilitate reinforcement learning**. Though as we will see later, the **backpropagation strategy employed by RNNs is different** to the standard backpropagation used by FNNs.

### 2.4.1 Recurrent Neural Network Architecture

**Computational graphs** are a way to **formalise the structure and set of given computations**, such as those involved in **mapping inputs and parameters** to **outputs and loss**. **<sup>7</sup>** 

The **nodes** in the graph represent **variables** which can be a **scalar, vector, matrix, tensor etc**. The **edges** in the graph correspond to **operations** that **transform one variable to another**.

For **recursive or recurrent computation**, such as those in an RNN, the computational graph can be **unfolded** into another computational graph that has a **repetitive structure**, typically corresponding to a **chain of events**. 

The architectural notation of a basic RNN with **no output** is shown in **Fig 5**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/442cb956-5e82-4eeb-b06d-10db49084939", alt="basic-rnn-architecture-notation-no-ouput"/>
    <p>
      <b>Fig 5</b> Time-unfolded computational graph representation of a basic RNN with no outputs. <b><sup>8</sup></b>
    </p>
  </div>
<br>

1. The **left side** of **Fig 4** shows the **computational graph** of a **RNN with no outputs**. This RNN simply **processes input data $$x$$** by **incorportating it into the state $$h$$**. This state $$h$$ is then $$passed forward through time$$. The **looping arrow** represents the **feedback loop** of the RNN and the **black square** represents the **delay of a single time step**.
2. The **right side** of **Fig 4** shows the same RNN but as a **unfolded computational graph**, where **each input ($$x$$) and state ($$h$$) node** is now **associated with one particular time instance**. This unfolding simply means that we **represent the network as its complete sequence**.
   * For example, if the sequence being processed is a **sentence of 3 words**, the network would be **unfolded into a 3 time step neural network**, with **one time step for each word**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/95107859-6976-4957-b0f9-1c9f205053b1", alt="basic-rnn-architecture-notation-loss"/>
    <p>
      <b>Fig 6</b> Time-unfolded computational graph of a training loss computation in a basic RNN. The RNN maps an <b>input sequence of <i>x</i> values</b> to a corresponding <b>output sequence of <i>o</i> values</b> <b><sup>8</sup></b>
    </p>
  </div>
<br>

Expanding this unfolded computational graph to represent the **loss calculation during training of an RNN** (**Fig 6**), we have:
1. **Input** - **$$x(t)$$** is the **input to the network at time step $$t$$**. For example, **$$x1$$** could be a **one-hot vector** corresponding to a **word in a sentence**.
2. **Hidden State** - **$$h(t)$$** represents a **hidden state** at **time $$t$$** and acts as the **"memory" of the network**. **$$h(t)$$** is calculated based on the **current input** and the **previous time steps hidden state** ($$h(t) = f(Ux(t) + Wh(t-1))$$. The **activation function $$f$$** is a **non-linear transformation** such as **tanh**, **ReLU** etc.
3. **Weights** - The RNN has **weights (U, W, V)** that are **shared across time**:
   * **Input-to-hidden connections**, parameterized by a **weight matrix $$U$$**
   * **Hidden-to-hidden connections**, paramaterized by a **weight matric $$W$$**
   * **Hidden-to-ouput connections**, paramaterized by a **weight matric $$V$$**
4. **Output** - **$$o(t)$$** is the **output of the network at time step $$t$$**
5. **Loss** - The **loss $$L(t)$$** measures **how far the output at time step $$t$$** is from the **corresponding training target $$y$$ at time step $$t$$**.

### 2.4.2 Backpropagation vs Backpropagation Through Time
As with FNNs, RNNs are trained by **processing input data and refining their performance**. 

The **nodes/neurons** have **weights** which give the **strength and direction** of the **connection between neurons in adjacent layers** when predicting the output. During training **these weights are adjusted to improve prediction accuracy**.

However, how the weights are adjusted **differs in RNNs compared to FNNs**. In FNNs, the weights are adjusted through **backpropagation**. In RNNs, the weights are adjusted through **backpropagation through time (BPTT)**.
   * FNNs use **standard backpropagation**, where the **gradients of the loss function are propagated backward only in the depth dimension (i.e. between layers)**.
   * In RNNs however, backpropagation is extended to **handle the temporal (sequential/time step) nature of the data** and so the gradients flow **both through the depth dimension (i.e. between layers)** and **through the temporal dimension (i.e. between time steps**).
   * In BPTT, the **loss gradients are summed/accumulated at each time step in all layers** because the **hidden states and weight parameter are shared/passed between each time step and each layer of the network**.
   * With FNNs, because they **don't share parameters across each layer**, they **do not need to sum/accumulate the loss gradients** and so **standard backpropagation** can be used.

BPTT is essential for **learning temporal dependencies and patterns** in sequential data, allowing the network to **adjust its weights** based on **how past inputs influence future outputs**.

### 2.4.3 Recurrent Neural Network Training

1. **Initialization**: Initialize the parameters of the RNN; initialize **weight matrices $$U$$, $$V$$ and $$W$$** using **random distribution** and initalize **bias vectors $$b$$ and $$c$$ as zero**.
2. **Unfolding the RNN**: The RNN is **expanded across all time steps in the input sequence**. This is known as **unfolding** and creates a **computational graph** that **spans both layers and time** (**Fig 6**).
3. **Forward Pass/Forward Propagation**: The input sequence data is processed sequentially, **maintaining and updating the hidden state at each time step**.
4. **Compute Loss**: The loss is typically computed **at each time step**, by measuring the **difference between the predicted and actual outputs** at that time step.
5. **Backpropagation Through Time (BPTT)**:
   1. Starting from the **final time step** of the **final layer**, the **gradients of the loss** are computed with respect to **outputs, hidden states and weights** at that time step using the **chain rule**.
   2. The **gradients are propagated backward through all time steps in that layer**, **accumulating the influence of the outputs, hidden states and weights across all time steps**.
   3. Simultaneously, the **gradients flow backward through the RNN layers** if the RNN has **multiple stacked layers**.
7. **Update weights and biases**: Following gradient computation, the **weights and biases are updated** based on the **accumulated gradients from all time steps**. The weights and biases are adjusted using **optimisation methods** such as **SGD** or **Adam** to **minimise the loss function**.
8. **Repeat Steps 3-7**: This is an **iterative process** where the training dataset is **passed through the network multiple times**, and each time the **weights are updated to reduce the error in prediction**. This continues until the model reaches a **point of convergence (i.e. where the loss function is at a minimum)**, or another **stop criterion is reached**.

### 2.4.4 Types of Recurrent Neural Networks

### i. Standard (Unidirectional) RNNs

The **most basic version** of an RNN, often referred to as **unidirectional RNNs**, process sequences in a **single direction** - typically from the **first element/token to the last (left to right)**. They **maintain a hidden state** that **captures and persists information about previous elements** in the sequence, allowing them to **model temporal dependencies**.

The **general architecture** of a standard/unidirectional RNN is as follows:
1. **Processing Direction**: Sequences are **process left to right** (also called **forward in time**).
2. **Hidden State Update**: At **each time step**, the **hidden state is updated** via the equation below, and **passed to the next time step** (and **to the next layer** if there is **more than one layer** in the network):

$$h_t = \phi(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$

where:
   * **$$h_t$$**: The **hidden state at time step $$t$$**
   * **$$x_t$$**: The **input at time $$t$$**
   * **$$W_{xh}x_t$$**: **Weight matrix multiplication** that **connects the input at time step $$t$$, to the hidden state at time step $$t$$**
   * **$$W_{hh}$$**: **Weight matrix multiplication** that **connects the hidden state at the previous time step ($$h_{t-1}$$)** to the **hidden state at the current time step ($$h_t$$)**.
        * Both the $$W_{xh}$$ and $$W_{hh}$$ matrices are crucial for **updating the current time step hidden state based on both the previous time step hidden state** and the **current input**.
   * **$$b_h**: The **bias vector**
   * **$$\phi$$**: The **activation function** (e.g. **tanh**, **ReLU** etc)

3. **Output Calculation**:

$$y_t = \phi(W_{hy} h_t + b_y)$$

where: 
   * **$$y_t$$**: The **output at time $$t$$
   * **$$W_{hy}$$**: **Weight matrix multiplication** that **connects the hidden state at time step $$t$$** to the **output at time step $$t$$**
   * **$$b_y$$**: The **bias vector**

The **strengths** of standard/unidirectional RNNs are:
1. **Simplicity**: They are easiest RNN to implement and understand.
2. **Short-Term Dependencies**: They excel in **simple tasks** with **short-term dependencies**, such as **predicting the next word in a short, simple sentence**, or the **next value in a simple time series**.

However, the **main limitations** of standard/unidirectional RNNs are:
1. **Long-Term Dependencies**: They **struggle with capturing long-term dependencies** due to issues like **vanishing and exploding gradients**.
2. **Directional Limitation**: They can **only utilise past information, not future context**, which may be limiting for certain tasks.

### ii. Bidirectional Recurrent Neural Networks (BRRNs)

While unidirectional RNNs can only **draw on previous inputs to make predictions about the current state**, **bidirectional RNNs (BRNNs)** enhance the standard RNN by **processing the input sequence in both forward and backward directions simultaneously**. This allows the network to **have access to both past (preceding) and future (succeeding) context at each time step**, which can **improve performance** on tasks where **context from both directions is beneficial**.

The **general architecture** of a bidirectional RNN is as follows:
1. **Processing Direction**: A BRNN has **dual hidden layers**
   * **Forward Layer** - Processes the sequence from **left to right**
   * **Backward Layer** - Processes the sequence from **right to left**
2. **Hidden State Update**:
   * **Foward Hidden State Update**:
     
     $$h_t^{\text{forward}} = \phi(W_{xh}^{\text{forward}} x_t + W_{hh}^{\text{forward}} h_{t-1}^{\text{forward}} + b_h^{\text{forward}})$$
     
   * **Backward Hidden State Update**:
     
     $$h_t^{\text{backward}} = \phi(W_{xh}^{\text{backward}} x_t + W_{hh}^{\text{backward}} h_{t+1}^{\text{backward}} + b_h^{\text{backward}})$$

   * **Combining Hidden States ($$;$$ Denotes Concatenation)**:
     
     $$h_t = [h_t^{\text{forward}} ; h_t^{\text{backward}}]$$

3. **Output Calculation**:

   $$y_t = \phi(W_{hy} h_t + b_y)$$

The **strengths** of bidirectional RNNs are:
1. **Enhanced Contextual Understanding**: Access to **future context** can **improve predictions**.
2. **Improved Performance**: BRNNs generally **achieve better performance** on sequence tasks compared to unidirectional RNNs.

However, the **main limitations** of bidirectional RNNs are:
1. **Increased Computational Cost**: BRNNs require **processing the sequence twice** (forward and backward), effectively **doubling the computation**.
2. **Unavailable Future Data**: BRNNs are **not suitable where future data is unavailable**.

### iii. Long Short-Term Memory (LSTM)

**Long Short-Term Memory (LSTM)** is a popular RNN architecture and were designed to **overcome the limitations of standard RNNs**, particularly the problems of **vanishing and exploding gradients**. This was achieved by designing LSTMs to be **capable of learning long-term dependencies in data**.
* With standard RNNs, if the **previous state that would be influencing the current prediction** is **not in the recent past**, a standard RNN would likely be **unable to accurately predict the current state**.
* For example, lets say we wanted to predict the italicized words in, “Alice is allergic to nuts. She can’t eat *peanut butter*.” The **context of a nut allergy** can help the RNN **anticipate that the food that cannot be eaten contains nuts**. However, if that **context was a few sentences prior**, then it would be **difficult or even impossible for the RNN to connect the information**.

To overcome this issue with learning long-term dependencies in sequences, LSTM networks have **cells** in the **hidden layers** which have **3 gates**:
1. **Forget Gate**
2. **Input Gate**
3. **Output Gate**

These gates **control the flow of information that is needed to predict the output in the network**.

The **general architecture** of an LSTM RNN is as follows:
1. **Processing Direction**: Can be either **unidirectional** or **bidirectional**.
2. **Core Components**:
   * **Cell State ($$C_t$$)** - This acts as a **conveyor belt**, carrying information **across time steps**.
   * **Gates** - Structures that **regulate the addition and removal of information from the cell state**
     1. **Forget Gate ($$f_t$$)**: Decides **what information to discard from the cell state**.
    
        $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

     2. **Input Gate ($$i_t$$)**: Determines **what new information to add to the cell state**.
    
        $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

     4. **Output Gate ($$o_t$$): Decides **what part of the cell state to output to the next time step**.

        $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

3. **Cell State Update ($$\odot$$ Denotes Element-Wise Multiplication)**:
   
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

5. **Hidden State Update**:

   $$h_t = o_t \odot \tanh(C_t)$$

The **strengths** of LSTM RNNs are:
1. **Handling Long-Term Dependencies**: LSTMs can **maintain and utilise information over long sequences effectively**.
2. **Mitigating Gradient Issues**: LSTM architecture helps **prevent vanishing and exploding gradients** during training.
3. **Flexibility**: LSTMs are suitable for a **wide range of sequential tasks**.

However, the **main limitations** of LSTM RNNs are:
1. **Complexity**: LSTMs have **more parameters** (**cell state $$C$$**) and are **more computationally expensive** compared to standard RNNs.
2. **Training Time**: This increased complexity results in **longer training times**.

### iv. Gated Recurrent Units (GRUs)

A **Gated Recurrent Unit (GRU)** is similar to an LSTM in that they **aim to capture long-term dependencies**, but they do so using a **simpler architecture**. GRUs **combine the forget and input gates** into a **single update gate** and **merge the cell and hidden states**. Similar to the gates within LSTMs, the **reset and update gates control how much and which information to retain**.

Because of the **simpler architecture** and **fewer parameters**, GRUs are **computationally more efficient**, making them **faster to train**.

The **general architecture** of a GRU RNN is as follows:
1. **Processing Direction**: Can be either **unidirectional** or **bidirectional**.
2. **Core Components**:
   1. **Update Gate ($$z_t$$)**: Determines **how much of the previous hidden state needs to be carried forward**.
  
      $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

   2. **Reset Gate ($$r_t$$)**: Decides **how to combine the new input with the previous memory**.
  
      $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

3. **Candidate Hidden State ($$\tilde{h_t}$$)**: Represents the **potential new hidden state** based on the **current input** and the **reset gate**.

   $$\tilde{h_t} = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$$

3. **Final Updated Hidden State ($$h_t$$)**: The updated hidden state after **considering both the previous state and the candidate state, modulated by the update gate**.

   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$


### v. Encoder-Decoder RNN

**Encoder-Decoder RNN Architecture** is the standard **neural machine translation (NMT)** approach and is used in models such as **sequence-to-sequence** models.

There are **three main components** in the encoder-decoder architecture.
1. **Encoder**
2. **Hidden Vector/Hidden State**
3. **Decoder**

At a **high level**:
* The encoder processes the **input sequence** into a **fixed-length, single-dimensional vector** called the **hidden vector**.
* The decoder then **converts the hidden vector** into an **output sequence**.

For a **lower level description**, we will use **Fig 7** as an illustration

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/33a0840e-420d-4729-ad32-74beb3d176f2", alt="encoder-decoder-architecture"/>
    <p>
      <b>Fig 7</b> Encoder-decoder sequence-to-sequence model. Both the encoder and decoder have a single layer of stacked RNN cells. <b><sup>9</sup></b>
    </p>
  </div>
<br>

**Encoder**
* For **every timestep (each input token) $$t$$**, the **hidden state/hidden vector $$h$$** is updated according to the **input at that timestep $$X[i]$$**.
* After **all the inputs are read by the encoder model**, the **final hidden state** of the encoder model represents the **context/summary of the whole input sequence**. This is why the hidden vector is also known as the **context vector**.
* For example, if we consider the input sequence **"I am a student"** to be encoded, there will be a total of **4 timesteps (4 tokens)** for the encoder model. At each timestep, the hidden state $$h$$ will be updated using the **previous hidden state** and the **current input**: <sup>9</sup>
  
1. **Timestep $$t_1$$**:
     * At the **first timestep $$t_1$$**, the **previous hidden state $$h_0$$** will set as **zero** or will be **randomly chosen**.
     * The **first RNN cell** will **update the current hidden state** with the **first input** and **$$h_0$$**.
     * Each layer outputs two things - the **updated hidden state** and the **output for each stage**.
     * The outputs at each stage are **rejected** and **only the hidden states** are propagated to the next layer.
     * At a given timestep, the hidden state is computed using the formula:
       
 $$
   h_t = f(W^{(hh)}h_{t-1} + W^{hx}x_t)
 $$

2. **Timestep $$t_2$$**:
   * At the **second timestep $$t_2$$**, the **hidden state $$h_1$$** and the **second input $$x_2$$** will be **given as input**, to the **next RNN cell in the layer**, and the **hidden state $$h_2$$** will be computed using these inputs and the formula above.

3. **Timestep $$t_n$$**:
   * This process is **repeated using the same RNN cell type** until either the **entire sequence has been processed**, or the **maximum sequence length is reached**.
  
4. **Single vs Stacked Encoder RNN Layers**
   * Typically, the encoder can be a **single layer**, or have **multiple, stacked layers**:
   * In a **single layer**, there is **one RNN cell type (e.g. LSTM)**, that is **reused across all time steps**. The hidden state is **passed from one time step to the next** within this single layer.
   * **Stacked RNN layers** however on the other hand involves stacking **mulitple RNN layers vertically** (i.e. in the **depth dimension**), to allow the model to **learn more complex representations**.
      * In stacked layers, **each layer processes the entire input sequence**, but they **operate on the output of the layer below them**.
      * Using the **"I am a student"** example, a **2-layer stacked RNN encoder** would look like:
```
Input Sequence: I → am → a → student

Layer 1 (Bottom Layer):
Time Step 1: I → H1_1
Time Step 2: am → H1_2
Time Step 3: a → H1_3
Time Step 4: student → H1_4

Layer 2 (Top Layer):
Time Step 1: H1_1 → H2_1
Time Step 2: H1_2 → H2_2
Time Step 3: H1_3 → H2_3
Time Step 4: H1_4 → H2_4
```
   1. **Layer 1 (Bottom Layer)**:
      * Processes each word **sequentially**, updating its hidden state at **each time step**.
   2. **Layer 2 (Top Layer)**:
      * Takes the **hidden states from layer 1** as its **inputs at each time step** and **updates its own hidden states accordingly**.
   3. **Final Hidden State**:
      * The **hidden state from layer 2 at the last time step (H2_4)** serves as the **context vector for the decoder**.

**Encoder Vector**
* This is the **final hidden state** produced from the **encoder layer** using the above formula.
* This **hidden state vector** aims to **encapsulate the information for all input elements** in order to help the **decoder make accurate predictions**.
* The encoder vector acts as the **initial hidden state of the decoder**. <sup>9</sup>

**Decoder**
* The decoder generates the output sequence by **predicting the next output $$y_t$$**, give the **hidden state $$h_t$$**.
* The **initial input** for the decoder is the **final hidden vector of the encoder**.
* At **each time step**, there will be **three inputs**, the **hidden state from the previous timestep $$h_{t-1}$$**, the **output from the previous timestep $$y_{t-1}$$**, and the **original hidden vector $$h$$ (i.e. the final hidden state of the encoder)**: <sup>9</sup>

1. **Timestep $$t_1$$**:
   * At the **first timestep $$t_1$$**, the **inputs** are an **empty hidden state $$h_{t-1}$$**, a **start token e.g. (`<START>`)**, and the **final hidden state of the encoder**.
   * The **outputs** are the **first token prediction $$y_1$$**, and **hidden state $$h_1$$**.
  
2. **Timestep $$t_2$$**:
   * At the **second timestep $$t_2$$**, the **inputs** are the **previous timestep hidden state $$h_{t-1}$$**, the **previous timestep output $$y_{t-1}$$**, and the **final hidden state of the encoder**.
   * The **outputs** are the **second token prediction $$y_2$$**, and **hidden state $$h_2$$**.
  
3. **Timestep $$t_n$$**:
   * This process is **repeated using the same RNN cell type** until either the **end token (e.g. `<END>`) is reached**, or the **maximum sequence length is reached**.
  
4. **Single vs Stacked Decoder RNN Layers**
   * Typically, the decoder can be a **single layer**, or have **multiple, stacked layers**:
   * In a **single layer**, there is **one RNN cell type (e.g. LSTM)**, that is **reused across all time steps**. The hidden state is **passed from one time step to the next** within this single layer.
   * **Stacked RNN layers** however on the other hand involves stacking **mulitple RNN layers vertically** (i.e. in the **depth dimension**), to allow the model to **learn more complex representations**.
      * In stacked layers, **each layer processes the entire input sequence**, but they **operate on the output of the layer below them**.
      * Using the **"I am a student"** example, a **2-layer stacked RNN decoder** that **predicts the French translation** of the sentence would look like:
```
Decoder:

Layer 1 (Bottom Layer):
Time Step 1: <SOS>  → H1_1
Time Step 2: "Je"   → H1_2
Time Step 3: "suis" → H1_3
Time Step 4: "étudiant" → H1_4
Time Step 5: <EOS>  → H1_5

Layer 2 (Top Layer):
Time Step 1: H1_1 → H2_1
Time Step 2: H1_2 → H2_2
Time Step 3: H1_3 → H2_3
Time Step 4: H1_4 → H2_4
Time Step 5: H1_5 → H2_5

Output:
Time Step 1: H2_1 → "Je"
Time Step 2: H2_2 → "suis"
Time Step 3: H2_3 → "étudiant"
Time Step 4: H2_4 → <EOS>
```
   1. **Layer 1 (Bottom Layer)**:
      * **Sequentially processes each input token**, updating its hidden states at each timestep
   2. **Layer 2 (Top Layer)**:
      * Takes the **hidden states from layer 1** at **each corresponding timestep** to update **its own hidden state**.
   3. **Output Layer**
      * The output generation at each time step **relies solely on the top layer's hidden state**.
     
**Output Layer**
* **Encoder-Decoder architecture** typically uses a **Softmax activation function** at the **output layer**.
* This is used to **produce the probability distribution** from a **vector of values**.
* The **output $$y_t$$** at **timestep $$t$$** is computed using the **hidden state at that timestep** together with the **respective weight $$W^S$$** using the formula:

$$
   y_t = \text{Softmax}(W^Sh_t)
$$

* Softmax is used to **create a probability vector** that will help in **determining the final output** (e.g. the **translated word**).

## 2.5 References
**[1]** Saigiridharan, L. et al. (2024) ‘AiZynthFinder 4.0: Developments based on learnings from 3 years of industrial application’, Journal of Cheminformatics, 16(1). <br><br>
**[2]** Fortunato, M.E. et al. (2020) ‘Data augmentation and pretraining for template-based retrosynthetic prediction in computer-aided synthesis planning’, Journal of Chemical Information and Modeling, 60(7), pp. 3398–3407. <br><br>
**[3]** Thakkar, A. et al. (2020) ‘Datasets and their influence on the development of computer assisted synthesis planning tools in the pharmaceutical domain’, Chemical Science, 11(1), pp. 154–168. <br><br>
**[4]** Genheden, S., Engkvist, O. and Bjerrum, E.J. (2020) 'A quick policy to filter reactions based on feasibility in AI-guided retrosynthetic planning.' <br><br>
**[5]** Chen, J. (no date) What is a neural network?, Investopedia. Available at: https://www.investopedia.com/terms/n/neuralnetwork.asp (Accessed: 30 September 2024). <br><br>
**[6]** Ibm (2024) What is a neural network?, IBM. Available at: https://www.ibm.com/topics/neural-networks (Accessed: 30 September 2024). <br><br>
**[7]** Stryker, C.S. (2024) What is a recurrent neural network (RNN)?, IBM. Available at: https://www.ibm.com/topics/recurrent-neural-networks (Accessed: 12 October 2024). <br><br>
**[8]** Goodfellow, I., Bendigo, Y. and Courville, A. (2016) Deep learning Ian Goodfellow, Yoshua Bengio, Aaron Courville. Cambridge ; Massachusetts ; London: MIT Press. <br><br>
**[9]** Encoders-decoders, sequence to sequence architecture. (2024) Medium. Available at: https://medium.com/analytics-vidhya/encoders-decoders-sequence-to-sequence-architecture-5644efbb3392 (Accessed: 03 November 2024). 
