# 5. Project Retrosynthesis Sequence-to-Sequence Model

## 5.1 Data Preparation

The training, validation, and testing data for developing the seq2seq model in this project were derived from the *Liu et al.* model codebase. **<sup>1</sup>**

These data sets had already been processed as per the process described in [**4.2.1**](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#421-data-preparation), and split into:
1. **Train sources** (products)
2. **Train targets** (reactants)
3. **Validation sources** (products)
4. **Validation targets** (reactants)
5. **Test sources** (products)
6. **Test targets** (reactants)

As the ultimate goal of this project is to **incorporate this model into AiZynthFinder**, the **prepended reaction type token** in the source sequences **was removed** to leave just the split target molecule SMILES sequence. Additionally, the **spaces between the characters** were removed to give **raw canonical SMILES**.

Additionally, the sources and target datasets were **combined** so that they could be **split before each training run**. This would allow us to **control the split ratio** during the model development process.

## 5.2 Model Architecture

As this project is to be an introduction to seq2seq models, the model architecture was **not based on the open source library** provided by *Britz et al.*. Instead, a **custom model** was implemented based on the architecture and hyperparameters described by *Liu et al.* (**Table 1**), to act as a **baseline** for future model iterations.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/999ae54c-1d80-4f0a-8411-cb5d9391766e", alt="liu-et-al-model-hyperparameters"/>
    <p>
      <b>Table 1</b> Key hyperparameters of the seq2seq model by <i>Liu at al.</i> <b><sup>1</sup></b>
    </p>
  </div>
<br>

## 5.3 Model Optimisation

### 5.3.1 Deterministic Training Environment

**Determinism** when using machine learning frameworks is to have **exact reproducibility from run to run**, with a model's training run **yielding the same weights**, and a model's inference run **yielding the same prediction**. **<sup>3</sup>**

In the context of optimizing model performance, this is useful as it **reduces noise/random fluctuations in data** between training runs, ensuring any improvement or reduction in performance is solely the result of the hyperparameter change, change in model architecture etc.

Following the **NVIDIA documentation for Clara**, **<sup>3</sup>** the following steps were taken to ensure **deterministic training** in the [training environment set up](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/trainers/environment.py).
* Set environment variable for **Python's built-in has seed**.
* Set seeds for the **pseudo-random number generators** used in the model for reproducible random number generation.
* Enabling **deterministic operations** in TensorFlow.

Additionally, the environment set up gives the optional measure of **disabling GPU** and **limiting TensorFlow to single-threaded execution**. This is because modern GPUs and CPUs are designed to execute computations **in parallel across many cores**. This parallelism is typically managed **asynchronously**, meaning that the order of operations or the availability of computing resources can vary slightly from one run to another. 

It is this asynchronous parallelism that can introduce random noise, and hence, non-deterministic behaviour.

Setting up a custom deterministic training environment was used as an introduction to determinism in machine learning. Future models will use the [machine learning reproducibility framework package](https://github.com/NVIDIA/framework-reproducibility/tree/master) developed by NVIDIA.

### 5.3.2 Data Tokenization and Preprocessing Optimisation

Despite promising training, validation and test accuracy (~68%) and loss (~0.10) for a full training run of an early model version, BLEU score remained very low (~2%). Additionally, once the seq2seq model was integrated into AiZynthFinder, analysis of the retrosynthesis predictions showed that they were converging on SMILES strings containing **all carbons** (either `C` or `c`).

Debugging of tokenizer showed that space characters between the individual chemical characters were also being tokenized. This explains the relatively **high token-level accuracy**, but very **low sequence-level accuracy** (BLEU score). This also may explain why the model was **overfitting to the most frequent tokens (i.e. `C` and `c`)**.

In an attempt to resolve this issue, various new analytics and debugging tactics were employed for a **more granular analysis** of model performance. This included:
* Logging tokenizer **word index mappings** and **token frequency distribution** for both **tokenized products** and **tokenized reactants**.
* **Verifying tokenization process** by manually tokenizing and detokenizing known canonical SMILES strings, and logging random SMILES strings from all data sets throughout the training process.
* Adding **more validation metrics**, particularly **string and chemical validity metrics**.

### i. DeepChem Tokenizer
The initial tokenizer **fitted the tokenized SMILES list** on a `tensorflow.keras.preprocessing.text.Tokenizer` instance for **character-level tokenization**. This had the advantage of being **simple to implement**, and didn't introduce much **computational overhead** in the model training/inference runs.

However, not only was this resulting in spaec characters being tokenized, but it would also result in **loss of chemical semantic meaning** as there was no mechanism in place to account for **multi-character tokens** such as `Cl`, `Br` etc. As a result, these would be split into `C`, `l` and `B` `r` etc.

An alternative tokenizer was found in the documentation of (DeepChem)[https://deepchem.readthedocs.io/en/latest/]. This `deepchem.feat.smiles_tokenizer.BasicSmilesTokenizer` would be used to **generate the list of tokenized SMILES strings** while **preserving chemical information**,

### ii. TensorFlow TextVectorisation
During debugging, further research into the **TensorFlow Keras documentation** found that the `tensorflow.keras.preprocessing` module was **deprecated**. A more modern preprocessing TensorFlow module is the `tensorflow.keras.layers.TextVectorization` layer.

Therefore, an alternative strategy was employed whereby the **list of tokenized SMILES strings** generated by `deepchem.feat.smiles_tokenizer.BasicSmilesTokenizer` would be **adapted onto a `tensorflow.keras.layers.TextVectorization` layer instance**. This **`TextVectorization` layer** is a **more modern TensorFlow integration**, allowing for **better integration with the model graph**.

Analysis using the metrics described above showed that this new approach was vastly superior, with an **improvement of BLEU score to ~17%** even with **throttled hyperparameters**.

### 5.3.3 Loss Function Optimisation

### i. Categorical Cross-Entropy vs Sparse Categorical Cross-Entropy
When deciding on a loss function, both **Sparse Categorical Cross-Entropy** and **Categorical Cross-Entropy** were considered.
1. **Categorical cross-entropy**:
  * Categorical cross-entropy loss (also known as **softmax loss**), is a common loss function used for **multi-class classification tasks**.
  * It measures the **dissimilarity (or error)** between the **predicted probability distribution** and the **true probability distribution** of the target classes.
  * The **predicted probability distribution** are usually obtained by passing the outputs of the model through a **softmax function**, which converts the **model's raw output** into a **probability distribution across the target classes**.
  * The **true probability distribution** represents the **actual labels of the training (and validation) examples**, typically in the form of **one-hot encoded vectors**.
2. **Sparse categorical cross-entropy**:
  * Sparse categorical cross-entropy loss is another variant of the **cross-entropy loss function** used for **multi-class classification tasks**.
  * In contract to categorical cross-entropy loss, where the **true labels are represented as one-hit encoded vectors**, sparse categorical cross-entropy loss expects the **target labels to be integers indicating the class indices directly**.
  * The sparse categorical cross-entropy loss function works by **first converting the true labels into one-hot encoded vectors internally**, and then applying the **regular categorical cross-entropy loss calculation**.
  * Mathematically, this has the **same formula as cross-entropy loss**, it just **converts the true labels to one-hot encoded vecots first**.
  * Additionally, sparse cross-entropy loss takes the true labels as a **1D vector of integers**. For example **`[1,2,1,5]`**, not **`[[1], [2], [1], [5]]`**. **<sup>4</sup>**
 
The chosen preprocessing approach in `data.utils.preprocessing.SmilesDataPreprocessor` was to **map characters tokenized smiles strings** in the tokenized_smiles_list to **the integers that they correspond to in the smiles_tokenizers word index**. Given that this would give a **1D vector of integers**, a **sparse categorical cross-entropy loss function** was the appropriate choice.

### ii. Optimiser - Adam
For the **optimiser**, the **Adaptive Moment Estimation (Adam) optimiser** was chosen, in line with *Liu et al.* **<sup>2</sup>** (**Table 1**).
* **Role of Optimisers**: In machine learning, optimisers **adjust the weights and biases** of a neural network to **minimise the loss** calculated by the loss function.
* **Gradient Descent**: Most optimisers are based on **gradient descent principles**, where the idea is to **move in the direction opposite to the gradient of the loss function (i.e the negative gradient direction)** by **adjusting the model's parameters** (the **weights and biases**).
* There are various optimisers that use different strategies to **improve convergence speed**, **handle noisy gradients**, or **escape local minima**.
* **Adam** is one of the most popular and widely used optimisation algorithms in machine learning. It is a **combines two extensions of Stochastic Gradient Descent (SGD)** called **AdaGrad** and **RMSProp**, though this is beyond the scope of this project.

### iii. Weight Decay

### 5.3.4 Callbacks Optimisation

### i. Early Stopping

### ii. Dynamic Learning Rate

### iii. Checkpoints

### 5.3.5 Metrics Optimisation

### i. Perplexity

### ii. BLEU Score

### 5.3.6 Encoder Optimisation

Intial baseline model encoder architecture consisted of **2 bidirectional LSTM layers**, with hyperparameters matching those outlined by *Liu et al.* **<sup>1</sup>** (**Table 1**). However the **attention, encoder and decoder embedding dimensions**, as well as the **units** were all decreased first to **256**, then to **128** for efficient hardware usage while testing subsequent model versions.

The first siginificant encoder change implemented during the optimisation process was to **test 4 bidirectional LSTM layers**, as this was **missing in the analysis** by *Britz et al.*. This resulted in **marginal improvement**, but a **significant increase in computation**.

### i. Residual Connections
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
      <b>Fig 1</b> Residual connection in a FNN <b><sup>5</sup></b>
    </p>
  </div>
<br>

### 5.3.7 Decoder Optimisation

Initial baseline model decoder architecture consisited of **4 unidirectional LSTM layers** with hyperparameters matching those outlined by *Liu et al.* **<sup>1</sup>** (**Table 1**). However, **decoder embedding dimension** and **units** were decreased first to **256**, then to **128** for efficient hardware usage while testing subsequent model versions.

### i. Residual Connections
The first significant change was the **adddition of residual connections were added to the decoder** (**Fig 1**). This resulted in an **improvement in both accuracy and loss** for training, validation and testing. This was at odds to what was reported by *Britz et al.* (sections **[4.1.3](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#413-encoder-and-decoder-depth)** and **[4.1.4](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#414-unidirectional-vs-bidirectional-encoder)**). This need for residual connections between layers is likley due to the increased semantic complexity of SMILES strings.

### ii. Layer Normalisation
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
      <b>Fig 2</b> Section of a neural network with a Batch Normalisation Layer <b><sup>6</sup></b>
    </p>
  </div>
<br>

To get the output of any hidden layer `h` within a neural network, we pass the inputs through a **non-linear activation function**. To **normalise the neurons (activation) in a given layer (`k-1`)**, we can **force the pre-activations** to have a **mean of 0** and a **standard deviation of 1**. In batch normalisation this is achieved by **subtracting the mean from each of the input features across the mini-batch** and **dividing by the standard deviation**. **<sup>6</sup>**

Following the output of the **layer `k-1`**, we can add a **layer that performs this batch normalisation operation** across the **mini-batch** so that the **pre-activations at layer `k` are unit Gaussians** (**Fig 2**).

As a high-level example, we can consider a mini-batch with **3 input samples**, with each **input vector** being **four features long**. Once the **mean and standard deviation** is computed for **each feature in the batch dimension**, we can **subtract the mean** and **divide by the standard deviation** (**Fig 3**). **<sup>6</sup>**

In reality, forcing all pre-activations to have a **zero mean** and **unit standard deviation** can be **too restrictive**, so batch normalisation **introduces additional parameters**, but this is beyond the scope of this project.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/08e5dda1-8a59-474f-8793-b287424579b2", alt="how-batch-normlisation-works"/>
    <p>
      <b>Fig 3</b> How batch normalisation works <b><sup>6</sup></b>
    </p>
  </div>
<br>

**Layer normalisation** is a normalisation technique introduced to address some of the limitations of **batch normalisation**. In layer normalisation, **all neurons in a particular layer** effectively have the **same distribution across all features for a given input**.
* For example, if each input has **`d` features, it is a **d-dimensional vector**. If there are **`B` elements** in a batch, the normalisation is done **along the length of the d-dimensional vector** and **not across the batch of size `B`**. **<sup>6</sup>**

Normalising **across all features of each input removes the dependence on batches/batch statistics**. This makes layer normalisation **well suited for sequence models** such as seq2seq models, RNNs and transformers.

*Fig 4** illustrates the same example as earlier, but with **layer normalisation instead of batch normalisation**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/71187197-02ad-463a-934a-f15abd887344", alt="how-layer-normalisation-works"/>
    <p>
      <b>Fig 4</b> How layer normalisation works <b><sup>6</sup></b>
    </p>
  </div>
<br>

### 5.3.8 Attention Mechanism Optimisation

### i. Bahdanau Attention Mechanism
Initial baseline model used an **additive (Bahdanau) attention mechanism** in line with the mechanism used by *Liu et al.* **<sup>1</sup>**, with the **same dimension** (**Table 8**). However, **attention dimension** and **units** were decreased first to **256**, then to **128** for efficient hardware usage while testing subsequent model versions.

### 5.3.9 Inference Optimisation

### i. Beam Search

## 5.4 References
**[1]** Liu, B. et al. (2017) ‘Retrosynthetic reaction prediction using neural sequence-to-sequence models’, ACS Central Science, 3(10), pp. 1103–1113. <br><br>
**[2]** Pandegroup (2017) ‘Pandegroup/reaction_prediction_seq2seq’, GitHub. Available at: https://github.com/pandegroup/reaction_prediction_seq2seq/tree/master (Accessed: 09 October 2024). <br><br>
**[3]** Determinism (2023) NVIDIA Docs. Available at: https://docs.nvidia.com/clara/clara-train-archive/3.1/nvmidl/additional_features/determinism.html (Accessed: 17 October 2024). <br><br>
**[4]** Chand, S. (2023) Choosing between cross entropy and sparse cross entropy - the only guide you need!, Medium. Available at: https://medium.com/@shireenchand/choosing-between-cross-entropy-and-sparse-cross-entropy-the-only-guide-you-need-abea92c84662 (Accessed: 18 October 2024). <br><br>
**[5]** Wong, W. (2021) What is residual connection?, Medium. Available at: https://towardsdatascience.com/what-is-residual-connection-efb07cab0d55 (Accessed: 18 October 2024). <br><br>
**[6]** Priya, B. (2023) Build better deep learning models with batch and layer normalization, Pinecone. Available at: https://www.pinecone.io/learn/batch-layer-normalization/ (Accessed: 18 October 2024). <br><br>
