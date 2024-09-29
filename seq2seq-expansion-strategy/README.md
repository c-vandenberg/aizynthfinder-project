# 2 AiZynthFinder's Expansion Policy Neural Network

## 2.1 What is AiZynthFinder's Expansion Policy Neural Network?

AiZynthFinder employs a **feedforward neural network**, specifically a **Multi-Layer Perception (MLP)**, as its standard expansion policy. **<sup>1</sup>** This network is designed to predict the applicability of various reaction templates to a given target molecule during retrosynthetic planning.

The architecture effectively **maps molecular representations to reaction probabilities**, generating a **ranked list of reaction templates** representing the most feasbile sets of reactions.

## 2.2 Feedforward Neural Networks (FNNs)

**Neural networks** are machine learning models inspired by the structure and function of **biological neural networks** in animal brains. They consist of **layers of nodes (articial neurons)** that process input data to produce an output.

Each node/neuron can be thought of as a **linear regression model**, that involves **computing a weighted sum of inputs, plus a bias**. As such, each node/neuron consists of:
1. **Input Data**: The data the node receives.
2. **Weights**: Numerical parameters that determine the **strength and direction** of the **connection between neurons in adjacent layers**. Each input is **assigned a weight** which helps to **determine the correlation of each input to the output**.
   * **Positive Weights** indicate a **positive correlation between the input and the output**. With positive weights, as the **input increases**, the **neurons activation tends to increase**.
   * **Negative Weights** indicate a **negative correlation between the input and the output**. With negative weights, as the **input increases**, the **neurons activation tends to decrease**.
4. **Biases**: These **shift the activation threshold**, enabling the **neuron to activate even when all input signals are zero**. This allows the model to better fit training data by allowing neurons to **activate in a broader range of scenarios**.
5. **Output data**: The output value passed to the next node/neuron in the adjacent layer if it **exceeds the activation threshold**.
   * Once **all inputs are multipied by their respective weights and summed**, this value is **passed through an activation function**, which determines the output.
   * If this output **exceeds the given activation threshold**, the node/neuron **fires (or activates)**, and the **output data is passed to a node/neuron in the next layer** in the network.
   * As a result, the **output of one node/neuron** becomes the **input of the next node/neuron**.
  
Mathematically, this can be represented as:

1. **Input Data**: The node/neuron receives inputs from the previous layer (or the input data itself if it is the input layer).
2. **Weighted Sum**: Each input is **multiplied by its corresponding weight**, and all these products are **summed together along with the bias**. This is why the bias can **shift the activation threshold**, because if it is a large enough value, it can cause the output to exceed the threshold even if all inputs are zero.

$$z =  \sum_{i=1}^{n} w_ix_i + b$$

where:
  * $$x_i$$ : Input from the $$i^{th}$$ neuron of the previous layer (or $$i^{th}$$ feature if input layer).
  * $wx_i$$ : Weight associated with the $$i^{th}$$ input.
  * $$b$$ : The bias term.
  * $$z$$ : The **weighted sum + bias term**. This is also known as the neuron's **pre-activation value**.

3. **Activation Function**: The **pre-activation value** $$z$$ is then **passed through an activation function** $$\sigma(z)$$ to produce the **node/neuron's output**:

$$ a = \sigma(z)$$

where:
  * $$a$$ : The **activated output of the neuron**.

Every neural network consists of **layers of nodes (artificial neurons)**. These layers are:

1. **Input Layer**
   * **Function**: Receives the input data.
   * **Structure**: Contains neurons corresponding to the number of input features.

2. **Hidden Layers**
   * **Function**: Perform intermediate computations and feature extraction.
   * **Structure**: One or more layers with neurons applying activation functions to weighted sums of inputs.
  
3. **Output Layer**
   * **Function**:
   * **Structure**:

**Feedforward Neural Networks (FNNs)** are the **simplest type** of a

## References

[1] Genheden, S., Engkvist, O. and Bjerrum, E.J. (2020) 'A quick policy to filter reactions based on feasibility in AI-guided retrosynthetic planning.'
