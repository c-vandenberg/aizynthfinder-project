# 5. Project Retrosynthesis Sequence-to-Sequence Model

## 5.1 Data Preparation

The training, validation, and testing data for developing the Seq2Seq model in this project were derived from the *Liu et al.* model codebase. **<sup>1</sup>**

These data sets had already been processed as per the process described in [**Section 4.2.1**](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#421-data-preparation), and split into:
1. **Train sources** (products)
2. **Train targets** (reactants)
3. **Validation sources** (products)
4. **Validation targets** (reactants)
5. **Test sources** (products)
6. **Test targets** (reactants)

As the ultimate goal of this project is to **incorporate this model into AiZynthFinder**, the **prepended reaction type token** in the source sequences **was removed** to leave just the split target molecule SMILES sequence. Additionally, the **spaces between the characters** were removed to give **raw canonical SMILES**.

Additionally, the sources and target datasets were **combined** so that they could be **split before each training run**. This would allow us to **control the split ratio** during the model development process.

## 5.2 Model Optimisation

As this project is to be an introduction to Seq2Seq models, the model architecture was **not based on the open source library** provided by *Britz et al.*. Instead, a **custom model** was implemented based on the architecture and hyperparameters described by *Liu et al.* (**Table 1**). This model was **iteratively optimised** over the course of the research project.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/999ae54c-1d80-4f0a-8411-cb5d9391766e", alt="liu-et-al-model-hyperparameters"/>
    <p>
      <b>Table 1</b> Key hyperparameters of the Seq2Seq model by <i>Liu at al.</i> <b><sup>1</sup></b>
    </p>
  </div>
<br>

### 5.2.1 Deterministic Training Environment

**Determinism** when using machine learning frameworks is to have **exact reproducibility from run to run**, with a model's training run **yielding the same weights**, and a model's inference run **yielding the same prediction**. **<sup>3</sup>**

In the context of optimizing model performance, this is useful as it **reduces noise/random fluctuations in data** between training runs, ensuring any improvement or reduction in performance is solely the result of the hyperparameter change, change in model architecture etc.

Following the **NVIDIA documentation for Clara**, **<sup>3</sup>** the following steps were taken to ensure **deterministic training** in the [training environment set up](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/trainers/environment.py).
* Set environment variable for **Python's built-in has seed**.
* Set seeds for the **pseudo-random number generators** used in the model for reproducible random number generation.
* Enabling **deterministic operations** in TensorFlow.

Additionally, the environment set up gives the optional measure of **disabling GPU** and **limiting TensorFlow to single-threaded execution**. This is because modern GPUs and CPUs are designed to execute computations **in parallel across many cores**. This parallelism is typically managed **asynchronously**, meaning that the order of operations or the availability of computing resources can vary slightly from one run to another. 

It is this asynchronous parallelism that can introduce random noise, and hence, non-deterministic behaviour.

Setting up a custom deterministic training environment was used as an introduction to determinism in machine learning. Future models will use the [machine learning reproducibility framework package](https://github.com/NVIDIA/framework-reproducibility/tree/master) developed by NVIDIA.

### 5.2.2 Data Tokenization and Preprocessing Optimisation

Despite promising training, validation and test accuracy (~68%) and loss (~0.10) for a full training run of an early model version, BLEU score remained very low (~2%). Additionally, once the Seq2Seq model was integrated into AiZynthFinder, analysis of the retrosynthesis predictions showed that they were converging on SMILES strings containing **all carbons** (either `C` or `c`).

Debugging of tokenizer showed that space characters between the individual chemical characters were also being tokenized. This explains the relatively **high token-level accuracy**, but very **low sequence-level accuracy** (BLEU score). This also may explain why the model was **overfitting to the most frequent tokens (i.e. `C` and `c`)**.

In an attempt to resolve this issue, various new analytics and debugging tactics were employed for a **more granular analysis** of model performance. This included:
* Logging tokenizer **word index mappings** and **token frequency distribution** for both **tokenized products** and **tokenized reactants**.
* **Verifying tokenization process** by manually tokenizing and detokenizing known canonical SMILES strings, and logging random SMILES strings from all data sets throughout the training process.
* Adding **more validation metrics**, particularly **string and chemical validity metrics**.

### i. DeepChem Tokenizer
The initial tokenizer **fitted the tokenized SMILES list** on a `tensorflow.keras.preprocessing.text.Tokenizer` instance for **character-level tokenization**. This had the advantage of being **simple to implement**, and didn't introduce much **computational overhead** in the model training/inference runs.

However, not only was this resulting in space characters being tokenized, but it would also result in **loss of chemical semantic meaning** as there was no mechanism in place to account for **multi-character tokens** such as `Cl`, `Br` etc. As a result, these would be split into `C`, `l` and `B` `r` etc.

An alternative tokenizer was found in the documentation of [DeepChem](https://deepchem.readthedocs.io/en/latest/). This `deepchem.feat.smiles_tokenizer.BasicSmilesTokenizer` would be used to **generate the list of tokenized SMILES strings** while **preserving chemical information**,

### ii. TensorFlow TextVectorisation
During debugging, further research into the **TensorFlow Keras documentation** found that the `tensorflow.keras.preprocessing` module was **deprecated**. A more modern preprocessing TensorFlow module is the `tensorflow.keras.layers.TextVectorization` layer.

Therefore, an alternative strategy was employed whereby the **list of tokenized SMILES strings** generated by `deepchem.feat.smiles_tokenizer.BasicSmilesTokenizer` would be **adapted onto a `tensorflow.keras.layers.TextVectorization` layer instance**. This **`TextVectorization` layer** is a **more modern TensorFlow integration**, allowing for **better integration with the model graph**.

Analysis using the metrics described above showed that this new approach was vastly superior, with an **improvement of BLEU score to ~17%** even with **throttled hyperparameters**.

### 5.2.3 Loss Function Optimisation

### i. Categorical Cross-Entropy vs Sparse Categorical Cross-Entropy
When deciding on a loss function, both **Sparse Categorical Cross-Entropy** and **Categorical Cross-Entropy** were considered.
1. **Categorical cross-entropy**:
  * Categorical cross-entropy loss (also known as **softmax loss**), is a common loss function used for **multi-class classification tasks**.
  * It measures the **dissimilarity (or error)** between the **predicted probability distribution** and the **true probability distribution** of the target classes.
  * The **predicted probability distribution** are usually obtained by passing the outputs of the model through a **softmax function**, which converts the **model's raw output** into a **probability distribution across the target classes**.
  * The **true probability distribution** represents the **actual labels of the training (and validation) examples**, typically in the form of **one-hot encoded vectors**.
2. **Sparse categorical cross-entropy**:
  * Sparse categorical cross-entropy loss is another variant of the **cross-entropy loss function** used for **multi-class classification tasks**.
  * In contrast to categorical cross-entropy loss, where the **true labels are represented as one-hot encoded vectors**, sparse categorical cross-entropy loss expects the **target labels to be integers indicating the class indices directly**.
  * The sparse categorical cross-entropy loss function works by **first converting the true labels into one-hot encoded vectors internally**, and then applying the **regular categorical cross-entropy loss calculation**.
  * Mathematically, this has the **same formula as cross-entropy loss**, it just **converts the true labels to one-hot encoded vectors first**.
  * Additionally, sparse cross-entropy loss takes the true labels as a **1D vector of integers**. For example **`[1,2,1,5]`**, not **`[[1], [2], [1], [5]]`**. **<sup>4</sup>**
 
The chosen preprocessing approach in `data.utils.preprocessing.SmilesDataPreprocessor` was to **map characters tokenized smiles strings** in the tokenized_smiles_list to **the integers that they correspond to in the smiles_tokenizers word index**. Given that this would give a **1D vector of integers**, a **sparse categorical cross-entropy loss function** was the appropriate choice.

### ii. Optimiser - Adam
For the **optimiser**, the **Adaptive Moment Estimation (Adam) optimiser** was chosen, in line with *Liu et al.* **<sup>2</sup>** (**Table 1**).
* **Role of Optimisers**: In machine learning, optimisers **adjust the weights and biases** of a neural network to **minimise the loss** calculated by the loss function.
* **Gradient Descent**: Most optimisers are based on **gradient descent principles**, where the idea is to **move in the direction opposite to the gradient of the loss function (i.e the negative gradient direction)** by **adjusting the model's parameters** (the **weights and biases**).
* There are various optimisers that use different strategies to **improve convergence speed**, **handle noisy gradients**, or **escape local minima**.
* **Adam** is one of the most popular and widely used optimisation algorithms in machine learning. It is a **combines two extensions of Stochastic Gradient Descent (SGD)** called **AdaGrad** and **RMSProp**, though this is beyond the scope of this project.

### iii. Weight Decay (L2 Regularisation)

**Weight decay** is a fundamental technique used in deep learning to **improve model performance**. It acts as a **regulariser** that **penalises large weights** in the network. This can lead to several benefits: **<sup>5</sup>**
1. **Reducing Overfitting**
   * Large weights can **lead to the model memorising the training data** and **failing to generalise to unseen examples**.
   * Weight decay **penalises large weights**, encouraging the model to **learn smaller weights** that **capture the underlying patterns in the data** rather than **memorising specific details**.
   * This leads to **better generalisation performance on unseen data**.
2. **Improving model stability**
   * Large weights can make training process **unstable** and **sensitive to noise** in the data.
   * Weight decay helps to **stabilise the training process** by **preventing the weights from becoming too large**.
   * This makes the model **less prone to overfitting** and **improves its overall robustness**.
3. **Promoting Feature Sharing**
   * Weight decay encourages the model to **learn weights that are similar across different nodes/neurons**.
   * This promotes **feature sharing**, where a **single feature is used by multiple nodes/neurons** in the network.
   * This can lead to a **more efficient model** with **fewer parameters**.
4. **Improving Generalisation in Overparameterised Models**
   * In modern deep learning, models often have **many more parameters than the amount of data they are trained on**.
   * This phenomenon is known as **overparameterisation**.
   * Weight decay helps to **control the complexity of overparameterised models** and **improves their generalisation performance**.
  
How weight decay works is that is **adds a penalty term to the loss function** that is proportional to the **sum of the squared weights** in the model. This penalty term **encourages the model to learn smaller weights** during training.

The weight decay penalty is typically implemented in **one of two ways**: **<sup>5</sup>**
1. **L2 Regularisation**: This **directly adds the penalty term to the loss function**. The penalty term is **proportional to the sum of the squared weights**.
2. **Weight Decay in the Optimiser**: This **modifies the update rule of the optimiser** to include a **delay factor** that **reduces the weights at each update step**.

The weight decay implementation used in this project is **L2 regularisation**. The implementation was designed in a way so that weight decay can be **fully controlled via the config file**. If the value for `weight_decay` in the config file is `null`, no weight decay is added, else the weight decay is passed to the **`kernal_regulariser`** in the **encoder and decoder LSTM layers**.

As of the **latest model version (V 21)**, small experimentation of weight decay has been carried out, with **simple cross-validation analysis**. 

However, choosing the right weight decay **depends on various factors**, such as the **model's size and complexity**, the **amount of training data** and the **learning rate**. Therefore, once the model architecture has been fully optimised, one of the following techniques will be used during **hyperparameter tuning** one of the following techniques will be used to **identify the best hyperparameter values for the model**, and hence the **optimal weight decay value**:
1. **Grid Search**
2. **Random Search**
3. **Bayesian Optimisation**

### 5.2.4 Callbacks Optimisation

**Callbacks** are powerful tools in **TensorFlow's Keras API** that allow you to customize and control the training process of your models. They are **Python objects** with methods that are **executed during training at given stages of the training procedure**, allowing you to **execute specific actions** at various stages of training (e.g. at the **end of an epoch**, **after a batch**, or **before training begins**).

Callbacks can be used for a wide range of purposes, such as:
1. **Monitoring training progress**
2. **Adjusting learning rates**
3. **Save model checkpoints**
4. **Generate logs**
5. **Create a TensorBoard**

TensorFlow Keras provides a **set of built-in callbacks**, but you can also **create custom callbacks** by **subclassing the `tf.keras.callbacks.Callback` class**. During training, Keras **calls specific methods of these callback objects** at different points. **Table 2** shows the methods within the  `tf.keras.callbacks.Callback` class.

<div style="display: flex;" align="center">
  <table border="1" cellspacing="0" cellpadding="5">
    <thead>
        <tr>
            <th>Method</th>
            <th>When It's Called</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>on_epoch_begin</td>
            <td>At the start of an epoch</td>
            <td>Initialize or reset variables related to epochs</td>
        </tr>
        <tr>
            <td>on_epoch_end</td>
            <td>At the end of an epoch</td>
            <td>Perform actions like logging or saving models</td>
        </tr>
        <tr>
            <td>on_batch_begin</td>
            <td>At the start of a batch</td>
            <td>Actions before processing a batch</td>
        </tr>
        <tr>
            <td>on_batch_end</td>
            <td>At the end of a batch</td>
            <td>Actions after processing a batch</td>
        </tr>
        <tr>
            <td>on_train_begin</td>
            <td>At the start of training</td>
            <td>Initialize resources or logging</td>
        </tr>
        <tr>
            <td>on_train_end</td>
            <td>At the end of training</td>
            <td>Cleanup resources or finalize logging</td>
        </tr>
        <tr>
            <td>on_predict_begin</td>
            <td>At the start of prediction</td>
            <td></td>
        </tr>
        <tr>
            <td>on_predict_end</td>
            <td>At the end of prediction</td>
            <td></td>
        </tr>
        <tr>
            <td>on_test_begin</td>
            <td>At the start of testing</td>
            <td></td>
        </tr>
        <tr>
            <td>on_test_end</td>
            <td>At the end of testing</td>
            <td></td>
        </tr>
    </tbody>
  </table>
  <p>
    <b>Table 2</b> Lifecycle methods of the `tf.keras.callbacks.Callback` class.
  </p>
</div>

### i. EarlyStopping

**EarlyStopping** is a built-in callback in Keras (`tensorflow.keras.callbacks.EarlyStopping`) that **monitors a specific metric** and **stops training when it stops improving**. 

In this project, `tensorflow.keras.callbacks.EarlyStopping` was used to **monitor validation loss**, and **stop training once it hasn't improved in over 5 consecutive epochs**. This helps to:
1. **Prevent overfitting**
   * Overfitting occurs when a model **learns the noise** and **specific patterns** in the training data to an extent that it **negatively impacts the model's performance on new, unseen data**.
   * **Training loss** measures how well the model **fits the training data**.
   * **Validation loss** measures how well the model **generalises to new data**.
   * As training progresses, training loss typically **decreases steadily**, whereas validation loss may **initially decrease**, but can **increase after a certain point**. This **indicates overfitting**.
2. **Ensures Computational Resources Aren't Wasted**
   * If the model's performance on the validation data **stops improving**, continued training may, at best, **lead to diminishing returns** or, at worse, **lead to overfitting**.
   * Early stopping by monitoring validation loss **conserves computational resources** by **preventing unnecessary epochs**, saving both **time and computational power**.

### ii. Dynamic Learning Rate (ReduceLROnPlateau)

In machine learning, **learning rate** is **one of the most critical hyperparameters**. It plays a pivotal role in **how effectively and efficiently a model learns from data**.

Learning rate ($$\eta$$) is a **scalar hyperparameter** that controls the **size of the steps taken during the optimisation process to minimise the loss function**. It dictates **how much the model's weights and biases are updated** in response to the estimated error each time a model processes a batch of data.

**ReduceLROnPlateau** is another built-in callback in Keras (`tensorflow.keras.callbacks.ReduceLROnPlateau`) that **monitors a specific metric**, and **reduces the learning rate by a specified factor when it stops improving**. 

In this model `tensorflow.keras.callbacks.ReduceLROnPlateau` was used to **monitor validation loss**. It would **reduce learning rate by a factor of 0.1** if the validation loss **didn't improve over 3 consecutive epochs**.

### iii. Checkpoints (ModelCheckpoint)

**Checkpoint callbacks** allow the **saving of the model regularly during training**, which is especially useful when training deep learning models which can take a **long time to train**. The callback monitors the training and **saves model checkpoints at regular intervals**, based on the metrics.

**ModelCheckpoint** is built-in callback in Keras (`tensorflow.keras.callbacks.ModelCheckpoint`) that can **save the whole model** or **just its weights** at **predefined intervals** (e.g. **after each epoch**), or **based on specific conditions** (e.g. an **improvement in validation loss**). This gives access to **intermediate versions** of the model that can be **restored later**, either to **resume training**, or to **deploy the best performing model**.

In this model, for greater customisability, a custom callback integrated with `from tensorflow.train.CheckpointManager` was used to save a **full-model checkpoint** when there was an **improvement in valdiation loss**, keeping only the **latest 5 checkpoints**, given the number of training runs needed to optimise the model.

### iv. Visualisation in TensorBoard (TensorBoard)

**TensorBoard** is TensorFlow's **comprehensive visualisation toolkit**, enabling developers to **gain insights into their machine learning model's training processes**. It provides tools to **visualise metrics** such as loss and accuracy, **analyse model architectures**, and **debug training processes**.

By **leveraging TensorBoard**, a developer can:
1. **Monitor Training Metrics**: Track how metrics evolve over time.
2. **Visualize Model Graphs**: Inspect the computational graph of the model.
3. **Analyze Distributions**: Examine weight and bias distributions, histograms etc.
4. **Project Embeddings**: Visualize high-dimensional data embeddings in lower-dimensional spaces.
5. **Profile Performance**: Identify bottlenecks and optimize performance.

**Tensorboard** is built-in callback in Keras (`tensorflow.keras.callbacks.TensorBoard`) that allows leveraging of TensorBoard and was **utilised extensively** in this project.

### v. Validation Metrics (ValidationMetricsCallback)

For a **more granular analysis** of model performance, specifically for **SMILES string predictions**, a custom **`callbacks.validation_metrics.ValidationMetricsCallback`** class was created, that would **handle a series of computations on the validation data predictions**. It was designed so that **further validation metrics could be easily integrated** in the future if needed.

As of the **latest model version (V 21)**, the following validation metric analytics have been added to the callback:
1. **Perplexity**
2. **BleuScore**
3. **SmilesStringMetrics**

### 5.2.5 Metrics Optimisation

### i. Perplexity

**Perplexity** is a widely used evaluation metric in **natural language processing (NLP)** and serves as an **indicator of how well the model predicts a sample**. It quantifies the **uncertainty of the model in predicting (i.e. assigning probabilities to) the next token in a sequence**.

In simpler terms, perplexity **evaluates the model's ability to assign high probabilities to the actual sequences it aims to predict**. A **lower perplexity value** indicates **better predictive performance**, meaning the model is **more confident and accurate in its predictions**.

A **custom `metrics.perplexity.Perplexity` metric** was passed to the model's `metrics` argument during **model compilation**.

### ii. BLEU Score

The **Bilingual Evaluation Understudy (BLEU) score** is a metric for assessing the **quality sequences generated by NLP models** by **measuring the overlap of n-gram (contiguous sequences of n words)** between the **candidate (machine-generated) sequence**, and a **reference (human-generated) sequence**.

In this model a **custom `metrics.bleu_score.BLEUScore` metric** was integrated into the **`callbacks.validation_metrics.ValidationMetricsCallback`** class, allowing for BLEU score calculation on the validation data predictions, and was integrated in to the model's **test data evaluation**. The BLEU score implementation chosen was **`nltk.translate.bleu_score.corpus_bleu`** with a **`SmoothingFunction`**.

While BLEU score useful metric for **quantifying how closely the predicted SMILES match the reference reactants** due to the **ease of computation**, it is important to **consider its limitations** in the context of **SMILES-based retrosynthesis prediction**:
1. **SMILES Syntax Sensitivity** - Due to the **strict syntax rules** that SMILES strings have, even a **minor error can represent a different molecule entirely**, **rendering a SMILES string invalid**.
2. **Chemical Correctness** - BLEU measures **surface-level similarity** and does not assess whether the predicted reactants can be used to **synthesise the target molecule**. Additionally, there is **no emphasis on functional group importance**.
3. **Multiple Valid Representations** - A single molecule can have **multiple valid SMILES representations**, which can **complicate BLEU's n-gram overlap assessment**. However, by **canonicalising SMILES during data preprocessing**, this can be accounted for.

### iii. SMILES String Metrics (Exact Match, Chemical Validity, Tanimoto Similarity, Levenshtein Distance)

As eluded to earlier, a series of **custom metrics** for **in-depth SMILES string analysis** were added to the custom **`ValidationMetricsCallback`** callback class.

1. **Exact SMILES string match**: Number of predicted exact SMILES string match as a percentage of total target SMILES.
2. **Chemical validity**: Measurement of whether the predicted SMILES string is chemically valid based on whether an **`rdkit.Chem.Mol` object** could be created from the predicted SMILES string. Metric is given the numner of chemically valid predicted SMILES string as a percentage of total target SMILES.
3. **Tanimoto Similarity**: The Tanimoto similarity algorithm provides a **measure of similarity between the molecular fingerprints of two molecules**. This is a commonly used metric in cheminformatics.
   * The Tanimoto similarity algorithm provides a **measure of similarity between the molecular fingerprints of two molecules**.
   * Usually the two molecular fingerprints are represented as **two sets of fingerprint 'bits'**, denoted as *A* and *B*.
   * The **Tanimoto coefficient**, *T(A,B)*, is calculated as the **ratio of the intersection of A and B to the union of A and B** (**Fig 1**) and is given by the equation:

$$
T(\mathbf{A}, \mathbf{B}) = \frac{|\mathbf{A} \cap \mathbf{B}|}{\|\mathbf{A}\| + \|\mathbf{B}\| - |\mathbf{A} \cap \mathbf{B}|}
$$

  * This can also be written in the form where the **union of A and B** is given as the **dot product of vectors A and B**:

$$
|\mathbf{A} \cap \mathbf{B}| = \mathbf{A} \cdot \mathbf{B}
$$

$$
T(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| + \|\mathbf{B}\| - \mathbf{A} \cdot \mathbf{B}}
$$

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/704fb554-f64f-43e3-8c25-2b26ad67e2cc", alt="tanimoto-coefficient-set-diagram"/>
    <p>
      <b>Fig 1</b> Tanimoto Coefficient Set Diagram
    </p>
  </div>
<br>

4. **Levenshtein Distance**: Also known as **Edit Distance**, Levenshtein distance measures the **minimum number of single-character edits** required to **change on string into another**.
   * The **allowable edits** are:
     * **Insertion**: Adding a character.
     * **Deletion**: Removing a character.
     * **Substitution**: Replacing one character with another.
   * Unlike other metrics such as BLEU, Levenshtein distance accounts for the **order of characters**, giving a **more granular view of sequence differences**.
   * This **sequential error quantification** is beneficial for tasks **requiring high precision**, such as **generating chemically valid SMILES strings**.
   * Additionally, Levenshtein distance gives **error type identification**. This helps in understanding whether the errors are primarily due to **insertions**, **deletions**, or **substitutions**, which can **inform model improvements**.

### 5.2.6 Encoder Optimisation

Initial baseline model encoder architecture consisted of **2 bidirectional LSTM layers**, with hyperparameters matching those outlined by *Liu et al.* **<sup>1</sup>** (**Table 1**). However the **attention, encoder and decoder embedding dimensions**, as well as the **units** were all decreased first to **256**, then to **128** for efficient hardware usage while testing subsequent model versions.

The first significant encoder change implemented during the optimisation process was to **test 4 bidirectional LSTM layers**, as this was **missing in the analysis** by *Britz et al.*. This resulted in **marginal improvement**, but a **significant increase in computation**.

### i. Residual Connections
The second significant encoder change was the implementation of **residual connections**. 
* Residual connections are **direct pathways** that allow the **output of one layer to be added to the output of a deeper layer in the network**.
* Instead of data flowing **strictly through a sequence of layers**, residual connections provide **shortcuts that bypass one or more layers**.

The benefits of residual connections include:
* **Mitigating the Vanishing/Exploding Gradient Problem**: Residual connections help this by **providing alternative pathways** for gradients to **flow backward through the network**, ensuring that gradients **remain sufficiently large** (mitigating vanishing gradients), while being **stable** (mitigating exploding gradients).
* **Enabling Identity Mappings**: Residual connections **apply identity mappings**, making it easier for **layers to learn identity functions** if necessary. This flexibility allows the network to **adaptively utilize or bypass certain layers**, enhancing its capacity to **model complex data**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/9082fa4e-0eb2-402b-a494-a29740efd7d4", alt="residual-connection"/>
    <p>
      <b>Fig 2</b> Residual connection in a FNN <b><sup>6</sup></b>
    </p>
  </div>
<br>

### ii. Layer Normalisation
The third significant change was to incorporate **layer normalisation** into the encoder.
* **Normalisation** works by **mapping all the values of a feature** to be in the **range [0,1]**.
* Normalisation techniques are employed in neural networks to:
  * **Stabilise training**: By **standardising inputs to layers**, they help to **maintain consistent activation scales**.
  * **Accelerate Convergence**: This enables the use of **higher learning rates** without the **risk of divergence**.
  * **Improve generalisation**: By acting as a form of **regularisation**, reducing overfitting.
  * **Mitigate Internal Covariate Shift**: By **reducing the change in the distribution of network activations** during training.

The first normalisation technique to consider is **batch normalisation**. In batch normalisation, the **inputs in each batch are scaled** so that they have a **mean of 0 (zero mean)** and a **standard deviation of 1 (unit standard deviation)**. Batch normalisation is applied **between the hidden layers of the encoder and/or decoder**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/6fdc7bd1-1f0f-450b-938e-83a2df51fb68", alt="batch-normalisation-overview"/>
    <p>
      <b>Fig 3</b> Section of a neural network with a Batch Normalisation Layer <b><sup>7</sup></b>
    </p>
  </div>
<br>

To get the output of any hidden layer `h` within a neural network, we pass the inputs through a **non-linear activation function**. To **normalise the neurons (activation) in a given layer (`k-1`)**, we can **force the pre-activations** to have a **mean of 0** and a **standard deviation of 1**. In batch normalisation this is achieved by **subtracting the mean from each of the input features across the mini-batch** and **dividing by the standard deviation**. **<sup>7</sup>**

Following the output of the **layer `k-1`**, we can add a **layer that performs this batch normalisation operation** across the **mini-batch** so that the **pre-activations at layer `k` are unit Gaussians** (**Fig 3**).

As a high-level example, we can consider a mini-batch with **3 input samples**, with each **input vector** being **four features long**. Once the **mean and standard deviation** is computed for **each feature in the batch dimension**, we can **subtract the mean** and **divide by the standard deviation** (**Fig 4**). **<sup>7</sup>**

In reality, forcing all pre-activations to have a **zero mean** and **unit standard deviation** can be **too restrictive**, so batch normalisation **introduces additional parameters**, but this is beyond the scope of this project.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/08e5dda1-8a59-474f-8793-b287424579b2", alt="how-batch-normlisation-works"/>
    <p>
      <b>Fig 4</b> How batch normalisation works <b><sup>7</sup></b>
    </p>
  </div>
<br>

**Layer normalisation** is a normalisation technique introduced to address some of the limitations of **batch normalisation**. In layer normalisation, **all neurons in a particular layer** effectively have the **same distribution across all features for a given input**.
* For example, if each input has **`d` features**, it is a **d-dimensional vector**. If there are **`B` elements** in a batch, the normalisation is done **along the length of the d-dimensional vector** and **not across the batch of size `B`**. **<sup>7</sup>**

Normalising **across all features of each input removes the dependence on batches/batch statistics**. This makes layer normalisation **well suited for sequence models** such as Seq2Seq models, RNNs and transformers.

**Fig 5** illustrates the same example as earlier, but with **layer normalisation instead of batch normalisation**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/71187197-02ad-463a-934a-f15abd887344", alt="how-layer-normalisation-works"/>
    <p>
      <b>Fig 5</b> How layer normalisation works <b><sup>7</sup></b>
    </p>
  </div>
<br>

### 5.2.7 Decoder Optimisation

Initial baseline model decoder architecture consisted of **4 unidirectional LSTM layers** with hyperparameters matching those outlined by *Liu et al.* **<sup>1</sup>** (**Table 1**). However, **decoder embedding dimension** and **units** were decreased first to **256**, then to **128** for efficient hardware usage while testing subsequent model versions.

### i. Residual Connections and Layer Normalisation
As with the encoder, the most significant decoder changes were the **addition of residual connections** (**Fig 2**) and the **incorporation of layer normalisation** (**Fig 4**). These changes resulted in an **improvement in both accuracy and loss** for training, validation and testing. 

Regarding residual connection, this improvement in model performance was at odds to what was reported by *Britz et al.* (sections **[4.1.3](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#413-encoder-and-decoder-depth)** and **[4.1.4](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#414-unidirectional-vs-bidirectional-encoder)**). This need for residual connections between layers is likely due to the increased semantic complexity of SMILES strings.

### 5.3.8 Attention Mechanism Optimisation

### i. Bahdanau Attention Mechanism
Initial baseline model used an **additive (Bahdanau) attention mechanism** in line with the mechanism used by *Liu et al.* **<sup>1</sup>**, with the **same dimension** (**Table 8**). However, **attention dimension** and **units** were decreased first to **256**, then to **128** for efficient hardware usage while testing subsequent model versions.

As with all attention mechanisms, the Bahdanau attention mechanism enables the Seq2Seq model to **dynamically focus on different parts of the input sequence** when **generating each element of the output sequence**. The high-level breakdown of this process is described in section [Section 3.4.3](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/README.md#343-attention-mechanism). 

The mechanism by which Bahdanau attention does this though is as follows:
1. **Mechanism**: Computes **alignment scores** by using a **feedforward neural network (FNN)** that **jointly considers the encoder hidden states** and the **decoder's previous hidden state**.
2. **Formula**:
   
$$
  e_{t,i} = v^T \tanh(W_1s_{t-1} + W_2h_i)
$$

where:
  * **$$W_1$$ and $$W_2$$** are **learnable weight matrices**
  * **$$v$$** is a **learnable weight vector**

### ii. Residual Connections and Layer Normalisation

Given the improvement in model performance after **integrating residual connections and applying layer normalisation into the encoder and decoder**, a further avenue for optimisation was to **integrate residual connections and applying layer normalisation around the attention mechanism**.

There is precedent for this in the literature; **Transformers heavily rely on residual connections and layer normalization** around their **multi-head attention mechanisms**. Given that transformers have achieved **state-of-the-art results** in various NPL tasks, this provides **strong empirical evidence** for the effectiveness of this approach.

By **integrating residual connections around attention**, in theory, the model is able to:
1. **Preserve Information Flow**:
   * The **original decoder output (`decoder_output`) is directly added to the attention-enhanced output (`context_vector`)**, ensuring that **essential information is not lost or distorted** through the attention layers.
2. **Facilitate Learning Complex Representations**:
   * Residual connections allow the model to **combine both raw (`decoder_output`)** and **attention-processed (`context_vector`) features**, leading to **richer and more nuanced representations**.
3. **Enhance Gradient Propagation**:
   * A with the encoder and decoder, **gradients flow through the attention layers during backpropagation through time (BPTT)**.
   * As such, residual connections **facilitate efficient gradient flow** by **providing alternative pathways** for gradients to **flow backward through the network**.
   * This ensures that gradients **remain sufficiently large** (mitigating vanishing gradients), while being **stable** (mitigating exploding gradients).

By **applying layer normalisation around attention**, in theory, the model is able to:
1. **Stabilise Activations**:
   * **Normalising the combined outputs of the attention and decoder** ensures that the **activations maintain a stable distribution**, preventing issues like **exploding or vanishing activations**.
2. **Improving Convergence**:
   * **Consistent activation scales** across different layers **facilitates faster and more reliable convergence** during training.
3. **Enhancing Generalisation**:
   * **Normalised features tend to generalise better**, reducing overfitting and improving performance on unseen data.

### 5.2.9 Inference Optimisation

### i. Greedy Decoding vs Beam Search

Once the model had the **majority of its optimisation features implemented**, a **beam search** was implemented with a **beam width of 5**, in line with *Liu et al.* **<sup>1</sup>**.

Beam search is a **heuristic search algorithm** that **explores a graph** by **expanding the most promising nodes in a limited set**. In the context of Seq2Seq models, it is used during the **decoding phase** to **generate the most probable output sequences based on the model's predictions**.

The **key characteristics** of beam search are:
1. **Breadth-First Exploration** - Unlike **greedy decoding**, which **selects the most probable token at each step**, beam search **maintains multiple hypotheses (beams) simultaneously**.
2. **Beam Width (Beam Size)** - This determines the **number of top candidate sequences to keep at each decoding step**. A larger beam width allows for a **more exhaustive search** but **increases computational complexity**.
3. **Probabilistic Approach** - Beam search **combines the probabilities of individual tokens** to **evaluate overall likelihood of entire sequences**.

### Greedy Search
Before we consider beam search, let us consider the simple **greedy search strategy**: **<sup>9</sup>**
- With **greedy search**, at any **time step $$t`$$**, we simply select the token with the **highest conditional probability from $$Y$$**. I.e.

$$
  y_{t'} = \text{argmax} P\left(y \mid y_1, \ldots, y{t'-1}, \mathbf{c}\right)
$$
 
 where:
   * $$y \in \mathcal{Y}$$

- This is a fairly reasonable strategy due it how **computationally undemanding it is**.
- However, putting aside efficiency, it may seem more reasonable to **search for the most likely sequence**, not the **sequence of (greedily selected) most likely tokens**. It turns out, **these two sequences can be quite different**.
- To illustrate this with an example, suppose there are four tokens 'A', 'B', 'C' and '<eos>' (end of sequence) in the **output dictionary (tokenizer word index)**. In **Fig 6**, the four numbers under each time step represent the **conditional probabilities of **generating 'A', 'B', 'C' and '<eos>' respectively, at that time step**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/e4aa34a8-375b-46fe-9eba-cc51a38b06a5", alt="greedy-search-probabilities"/>
    <p>
      <b>Fig 6</b> Greedy search selects the token with the highest conditional probability at each time step <b><sup>9</sup></b>
    </p>
  </div>
<br>

- Because greedy search **selects the token with the highest conditional probability at each time step**, the **predicted output sequence** will be **`['A', 'B', 'C', '<eos>']`**. The **conditional probability** of this output sequence is $$0.5\times0.4\times0.4\times0.6 = 0.048$$
- If we look at the same example, but **at time step 2 we instead select the token 'C'**, which has the **second highest conditional probability at that time step** (**Fig 7**)

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/4a4887a6-e43b-48e6-bb2f-e6e6e4239f8d", alt="greedy-search-probabilities-2"/>
    <p>
      <b>Fig 7</b> Greedy search selects the token with the highest conditional probability at each time step, except for time step 2 <b><sup>9</sup></b>
    </p>
  </div>
<br>

- Now the **output subsequences** at time steps 1 and 2 have changed from **'A', 'B'** to **'A', 'C'**.
- Because **each tokens conditional probability at each time step are dependent on the output sequences of all prior time steps (temporal dependency)**, now the **conditional probability of each token at time step 3 have changed**.
- If greedy search continues to select the token with the highest conditional probability, the **predicted output sequence** will be **`['A', 'C', 'B', '<eos>']`**. The **conditional probability** of this output sequence is $$0.5\times0.3 \times0.6\times0.6=0.054$$.
- As you can see, **this conditional probability is higher than that of the greedy search in Fig 6**. Therefore, the output sequence **`['A', 'B', 'C', '<eos>']`** obtained by the greedy search **is not optimal**.

### Exhaustive Search

If the goal is to obtain the **most likely sequence**, we may consider using **exhaustive search**. This involves **enumerating all the possible output sequences with their conditional probabilities**, and then output the one that **scores the highest probability**.

While this would certainly work, it comes at a **prohibitive computational cost**. The **Big-O complexity** of an exhaustive search would be:

**$$
  \mathcal{O}(\left|\mathcal{Y}\right|^{T'})
$$**

where:
* **$$\mathcal{Y}$$** is the **entire vocabulary set**
* **$$T'$$** is the **number of timesteps/the max output sequence length**

For example, even for a sequence with a **small vocabulary set $$\mathcal{Y}$$ = 10,000**, and **short max output sequence length $$T'$$ = 10**, an exhaustive search algorithm would need to evaluate **$$10000^10 = 10^40$$ sequences**. Even for this small, non-complex sequence, an exhaustive search is **beyond the capabilities of most/all computers**.

On the other hand, **Big-O complexity of greedy search** is:

**$$
   \mathcal{O}(\left|\mathcal{Y}\right|T')
$$**

As you can see, despite **not being optimal**, it is **computationally cheap**. For example for the **same sequence as earlier**, a greedy search algorithm would only need to evaluate **$$10000 \times 10 = 10^5$$ sequences**.

### Beam Search

If **sequence decoding strategies lay on a spectrum**, with **greedy search on one end** (**least computationally demanding**, but **least optimal**), and **exhaustive search on the other end** (**most computationally expensive**, but **most optimal**), then **beam search would lie in the middle**, striking a **compromise between greedy search and exhaustive search**.

The most straightforward version of beam search is **characterised by a single hyperparameter**, the **beam size $$k$$**, and works as follows:
1. **Time Step 1**:
   * At time step 1, we select the **$$k$$ tokens with the highest predicted probabilities**.
3. **Time Step 2**:
   * Each of these **$$k$$ candidates tokens from time step 1** will be the **first token of $$k$$ candidate output sequences** in **time step 2**.
   * Because **each tokens conditional probability at each time step are dependent on the output sequences of all prior time steps (temporal dependency)**, these tokens will **determine the conditional probabilies of all tokens in the $$k$$ candidate output sequences** in time step 2.
   * In time step 2, the **token chosen at each $$k$$ candidate output sequence** is the one that **gives the highest total conditional probability for that candidate sequence** (i.e. the **token with the highest conditional probability at that time step**)
5. **Iterative Process**: This process is **repeated for $$T'$$** time steps (the **max output sequence length**).
6. **Candidate Output Sequence Comparison**: At the end of the beam search, the **output sequence candidate with the highest conditional probability $$k|\mathcal{Y}|$$** is chosen as the **predicted sequence**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/da15729e-cfbf-464d-a213-bc2c6223ee1e", alt="beam-search"/>
    <p>
      <b>Fig 8</b> Beam search with <b>beam width, k = 2</b> and a <b>max sequence = 3</b>. The candidate output sequences chosen by beam search are <b>A, C, AB, CE, ABD</b> and <b>CED</b> <b><sup>9</sup></b>
    </p>
  </div>
<br>

**Fig 8** illustrates an example of beam search with an example. In this example, the **output vocabulary contains only five elements**: **$$Y = {A, B, C, D, E}$$** (where **one of them is `<eos>`**), a **beam width, $$k = 2$$** and a **maximum output sequence = 3**.
1. At **time step 1**, suppose that the **tokens with the highest conditional probabilities $$P(y_1 \mid \mathbf{c})$$** are **$$A$$** and **$$C$$**. Because we have a beam width = 2, these tokens are **chosen for each of the two beams**.
2. At **time step 2**, for all **$$y_2 \in \mathcal{Y},$$** (**tokens in output vocabulary set at time step 2**), we compute the below and **pick the largest two amount these ten value**. In **Fig 8** this is **$$P(A, B \mid \mathbf{c})$$** and **$$P(C, E \mid \mathbf{c})$$**:

$$
  \begin{split}\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\end{aligned}\end{split}
$$
$$
   \begin{split}\begin{aligned}P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}\end{split}
$$

3. At **time step 3**, for all **$$y_3 \in \mathcal{Y},$$** (**tokens in output vocabulary set at time step 2**), we compute the below and, again, **pick the largest two amount these ten value**. In **Fig 8** this is **$$P(A, B, D \mid \mathbf{c})$$** and **$$P(C, E, D \mid \mathbf{c})$$**:

$$
  \begin{split}\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\end{aligned}\end{split}
$$
$$
   \begin{split}\begin{aligned}P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}\end{split}
$$

4. As a result, we get **six candidate output sequences**:
   1. **$$A$$**
   2. **$$C$$**
   3. **$$A, B$$**
   4. **$$C, E$$**
   5. **$$A, B, D$$**
   6. **$$C, E, D$$**
5. At the **end of the beam search**, the algorithm **discards portions of these output sequences** (e.g. the **`<eos>` token** and **any tokens after it**) to obtain the **set of final six candidate output sequences**.
6. The algorithm then **chooses the output sequence** which **maximises the following score**;

$$
  \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}\mid \mathbf{c}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c});
$$
  * Here, **$$L$$** is the **length of the final candidate sequence** and **$$\alpha$$** is usually set to **$$0.75$$**.
  * Since **longer sequences have more logarithmic terms in the summation**, the term **$$L^\alpha$$** in the denominator **penalises long sequences**.

The **Big-O complexity** of beam seach is **$$\mathcal{O}(k\left|\mathcal{Y}\right|T')$$**. This is **between the Big-O complexities of greedy search and exhaustive search**.
  * **N.B.**: Greedy search can be thought of as a **special case of beam search arising when the beam size is set to 1**.

## 5.3 Model Architecture

### 5.3.1 Optimised Encoder Architecture

The encoder is a **custom TensorFlow Keras Layer** named **`StackedBidirectionalLSTMEncoder`**. The **bidirectional LSTM layers** are designed to **process input sequences** (**tokenized product SMILES strings**), and convery them into **rich, context-aware representations** that capture **both past and future information** in the sequence.

The **key components** of the encoder are:
1. **Embedding Layer**
2. **Masking Support**
3. **Stacked Bidirectional LSTM Layers**
4. **Layer Normalisation Layers**
5. **Dropout Layers**
6. **Residual Connections**
7. **Weight Decay (L2 Regularisation)**

### 1. Embedding Layer
  * **Purpose:** Converts **input token indices** into **dense vector embeddings**.
  * **Functionality:**
    * The embedding layer **transforms/maps each token** in the input sequence into a **continuous vector space** of a **specified dimension (`encoder_embedding_dim`)**.
    * Handles **padding tokens (`mask_zero=True`)** to prevent the model from considering them during training.
  * **Output Shape:** **`(batch_size, sequence_length, encoder_embedding_dim)`**.

### 2. Masking Support
  * **Purpose:** Handles **variable-length sequences with padding**.
  * **Functionality:**
    * The **embedding layer generates a mask (`encoder_mask`)** that **identifies padding tokens**.
    * The mask is **propagated through the LSTM layers** to **prevent the model from considering padded positions** during training.

### 3. Stacked Bidirectional LSTM Layers
  * **Purpose:** Captures **complex sequential dependencies** and **contextual information** from both **past (forward direction)** and **future (backward direction) tokens**.
  * **Functionality:**
    * Multiple **Bidirectional LSTM layers are stacked (`num_layers`)**, allowing the model to **learn hierarchical representations** of the input sequence.
    * Each Bidirectional LSTM layer consists of **two LSTMs**:
      * **Forward LSTM:** Processes the sequence from **start to end**.
      * **Backward LSTM:** Processes the sequence from **end to start.**
    * The outputs from both directions are **concatenated, doubling the number of units (`units * 2`)**.
  * **Outputs:**
    * **Encoder Output:** Sequence representations at **each time step**.
      * **Shape:** **`(batch_size, sequence_length, units * 2)`**.
    * **Final Hidden States (`final_state_h`):** Concatenated **forward and backward hidden states** from the last LSTM layer.
      * **Shape:** **`(batch_size, units * 2)`**.
    * **Final Cell States (`final_state_c`):** Concatenated **forward and backward cell states** from the last LSTM layer.
      * **Shape:** **`(batch_size, units * 2)`**
     
### 4. Layer Normalisation Layers
  * **Purpose:** Stabilizes and accelerates training by normalizing the outputs of each layer.
  * **Functionality:**
    * Applied **after each Bidirectional LSTM layer**.
    * **Normalises the activations** to have **zero mean and unit variance**, independently for **each sample in a batch**.
  * **Effect:** Reduces **internal covariate shift**, leading to **faster convergence**.
     
### 5. Dropout Layers
  * **Purpose:** **Prevents overfitting** by **randomly deactivating a subset of neurons** during training.
  * **Functionality:**
    * Applied **after each Bidirectional LSTM layer** (after layer normalisation).
    * The dropout rate is specified by the **`dropout_rate` hyperparameter** (default is **0.2**).
  * **Effect:** Helps the model **generalise better** by **reducing reliance on specific neurons**.

### 6. Residual Connections
  * **Purpose:** Facilitates **better gradient flow** and **mitigates vanishing gradient issues** in deep networks.
  * **Functionality:**
    * From the **second LSTM layer onwards**, a residual connection **adds the output of the previous layer** to the **current layer's output**.
    * This helps **preserve information from earlier layers**.
  * **Effect:** Allows the model to **learn identity mappings**, improving training of **deeper networks**.

### 7. Weight Decay (L2 Regularisation)
  * **Purpose:** Adds **regularisation to prevent overfitting**.
  * **Functionality:**
    * **L2 regularisation (`weight_decay`)** is applied to the **kernel weights** of the LSTM layers.
    * Encourages **smaller weights**, leading to **simpler models**.
   
### 5.3.2 Optimised Decoder Architecture

The decoder is a **custom TensorFlow Keras layer** named **`StackedLSTMDecoder`**. It is designed to **generate the target sequences** (**tokenized reactant SMILES strings**) in the Seq2Seq model by **leveraging the encoder's outputs** and **its own previous predictions** (or, in the case of **teacher forcing**, the **ground truth token**).

Within the Seq2Seq model, the decoder is used in the **following phases:**
1. **Traning Phase (Teacher Forcing)**
    * The decoder **receives the ground truth previous tokens** as input (**`decoder_input`**).
    * It uses the **encoder outputs** and its **own hidden states** to generate the **next token probabilities**.
    * The **loss is computed** between the **predicted probabilities** and the **actual target tokens**.
2. **Inference Phase**
    * The decoder **generates one token at a time**.
    * Starts with the **`<START>` token**.
    * At **each time step**, the decoder's **previous prediction** is used as the **next input**.
    * The **`single_step()`** method is utilised for **timestep-by-timestep decoding**.
3. **Attention Mechanism**
    * The attention mechanism **enhances the decoder's ability** to **focus on relevant parts of the input sequence** when **generating each token**.
    * This helps in **handling long sequences** and **complex dependencies**.

The **key components** of the encoder are:
1. **Embedding Layer**
2. **Masking Support**
3. **Stacked LSTM Layers**
4. **Layer Normalisation Layers**
5. **Dropout Layers**
6. **Residual Connections**
7. **Weight Decay (L2 Regularisation)**
8. **Attention Mechanism (Bahdanau Attention)**
9. **Residual Connection (Projection Layers) & Layer Normalisation Around Attention**
10. **Output Dense Layer**

### 1. Embedding Layer
  * **Purpose:** Transforms **decoder input token indices** (**`decoder_input`**) into **dense vector embeddings**.
  * **Functionality:**
    * **Converts each token** in the decoder input sequence into a **continuous vector space of a specified dimension** (**`decoder_embedding_dim`**).
    * Handles **padding tokens** (**`mask_zero=True`**) to **prevent the model from considering them during training**.
  * **Output Shape:** **`(batch_size, sequence_length_dec, decoder_embedding_dim)`**

### 2. Masking Support
  * **Purpose:** Handles **variable-length sequences with padding**.
  * **Functionality:**
    * The embedding layer **generates a mask (`decoder_mask`)** that **identifies padding tokens**.
    * The mask is **propagated through the LSTM layers** and **attention mechanism** to prevent the model from considering **padded positions**.
   
### 3. Stacked LSTM Layers
  * **Purpose:** Captures **sequential dependencies** and **patterns** in the target sequences.
  * **Functionality:**
    * **Multiple LSTM layers are stacked** (**`num_layers`**), allowing the decoder to **learn complex representations** of the output sequence.
    * Each LSTM layer **processes the embeddings** and **produces hidden states** that **encapsulate information** up to the **current time step**.
  * **Outputs:**
    * **Decoder Output:** Updated sequence representations at **each time step**.
  * **Shape:** **`(batch_size, sequence_length_dec, units)`**
  * **New States:** **Hidden and cell states** for each LSTM layer, used for **maintaining context across time steps**.

### 4. Layer Normalisation Layers
  * **Purpose:** **Stabilises and accelerates training** by **normalising the outputs** of each layer.
  * **Functionality:**
    * Applied **after each LSTM layer**.
    * **Normalises the activations** to have **zero mean** and **unit variance**, independently for each sample in a batch.
  * **Effect:** Reduces **internal covariate shift**, leading to **faster convergence**.

### 5. Dropout Layers
  * **Purpose:** **Prevents overfitting** by **randomly deactivating a subset of neurons** during training.
  * **Functionality:**
    * Applied **after each LSTM layer** (**after layer normalisation**).
    * The dropout rate is specified by the **`dropout_rate` hyperparameter** (default is **0.2**).
  * **Effect:** Helps the model **generalize better** by **reducing reliance on specific neurons**.

### 6. Residual Connections
  * **Purpose:** Facilitates **better gradient flow** and **mitigates vanishing gradient issues** in deep networks.
  * **Functionality:**
    * From the **second LSTM layer onwards**, a residual connection **adds the output of the previous layer** to the **current layer's output**.
    * This helps **preserve information from earlier layers**.
  * **Effect:** Allows the model to **learn identity mappings**, improving training of **deeper networks**.

### 7. Weight Decay (L2 Regularisation)
  * **Purpose:** Adds **regularization to prevent overfitting**.
  * **Functionality:**
    * **L2 regularization** (**`weight_decay`**) is applied to the **kernel weights** of the LSTM and Dense layers.
    * Encourages **smaller weights**, leading to **simpler models**.

### 8. Attention Mechanism (Bahdanau Attention)
  * **Purpose:** Allows the decoder to **focus on specific parts of the encoder's outputs** when generating each token, enhancing the model's ability to **handle long sequences** and **complex dependencies**.
  * **Functionality:**
    * Computes a **context vector** for **each decoder time step** using the **encoder's outputs**.
    * The context vector represents a **weighted sum of encoder outputs**, where the **weights are determined by the relevance of each encoder output to the current decoding step**.
  * **Outputs:**
    * **Context Vector:** Provides **relevant information** from the **encoder outputs** for **each decoding step**.
      * **Shape:** **`(batch_size, sequence_length_dec, enc_units)`**.
    * **Attention Weights:** The **alignment scores** between the **decoder hidden states** and **encoder outputs**.
      * **Shape:** **`(batch_size, sequence_length_dec, sequence_length_enc)`**.

### 9. Residual Connection (Projection Layers) & Layer Normalisation Around Attention
  * **Purpose:** **Transforms the decoder outputs and context vectors** to have **matching dimensions** before **adding them together in a residual connection**. Transformed decoder outputs are then **normalised**.
  * **Functionality:**
    * **`decoder_dense`:** A Dense layer that **projects the decoder outputs**.
    * **`context_dense`:** A Dense layer that **projects the context vectors**.
  * **Effect:** Ensures that the addition of decoder outputs and context vectors is **dimensionally compatible**, and then **normalises** the new transformed output.

### 10. Output Dense Layer
  * **Purpose:** Generates **probability distributions** over the target vocabulary for **each time step**.
  * **Functionality:**
    * A **Dense layer** with a **softmax activation function**.
    * **Maps the transformed decoder outputs** to **logits** over the vocabulary, which are then **converted to probabilities**.
  * **Output Shape:** **`(batch_size, sequence_length_dec, vocab_size)`**.

### 5.3.3 Optimised Attention Mechanism (Bahdanau Attention) Architecture

The attention mechanism is (**`BahdanauAttention`**) is an implementation of the **Bahdanau attention mechanism** proposed by **Dzmitry Bahdanau et al** in their 2014 paper. **<sup>10</sup>**

This attention mechanism allows the decoder to **dynamically focus on different parts of the encoder's outputs** by providing a **context vector** at **each decoding step**. By doing so, it helps the **model capture relevant contextual information** from the input sequence, improving the **quality and accuracy** of the generated output sequences.

The attention mechanism is **integrated within the Seq2Seq model's decoder** as follows:
1. **Decoder's Use of Attention**
  * At **each decoding time step**, the decoder uses the **context vector** to **inform its next prediction**.
  * The context vector provides a **summary of the encoder outputs** that are **most relevant to the current decoder state**.
2. **Flow within the Decoder**
  * **Before Attention:**
    * The decoder **processes its input tokens** through **embedding** and **stacked LSTM layers**.
  * **Applying Attention:**
    * The **decoder outputs** (**`decoder_output`**) and **encoder outputs** (**`encoder_output`**) are **passed to the attention mechanism**.
    * The attention mechanism computes the **context vector and attention weights**.
  * **After Attention:**
    * The context vector and decoder output are **transformed to a Dense layer** and **combined** for a **residual connection**, and then **normalised**.

The **key components** of the Bahdanau attention mechanism are:
1. **Dense Layers for Transformations**
2. **Score Calculation**
3. **Attention Weights Computation**
4. **Mask Handling**
5. **Context Vector Computation**

### 1. Dense Layers for Transformations
  * **Purpose:** Transform the **encoder and decoder outputs** into a **common feature space** to **compute attention scores**.
  * **Functionality:**
    * **`attention_dense1`:** A **Dense layer** that **processes the encoder outputs**.
      * Transforms the **encoder outputs** from **shape `(batch_size, seq_len_enc, enc_units)`** to **`(batch_size, seq_len_enc, units)`**.
    * **`attention_dense2`:** A **Dense layer** that **processes the decoder outputs**.
      * Transforms the **decoder outputs** from **shape `(batch_size, seq_len_dec, dec_units)`** to **`(batch_size, seq_len_dec, units)`**.
  * **Parameters:**
    * **units:** The **number of units** in the attention mechanism, determining the **dimensionality of the transformed features**.

### 2. Score Calculation
  * **Purpose:** Compute the **alignment scores** between the **encoder and decoder outputs**.
  * **Functionality:**
    * The transformed encoder and decoder outputs are **combined using a non-linear activation function (`tanh`)** to **capture the relevance** between **each encoder time step** and **each decoder time step**.
    * **Broadcasting and Expansion:**
      * The **encoder transformed outputs** are **expanded along the decoder time step dimension**.
        * **Shape:** **`(batch_size, 1, seq_len_enc, units)`**
      * The **decoder transformed outputs** are **expanded along the encoder time step dimension**.
        * **Shape:** **`(batch_size, seq_len_dec, 1, units)`**
  * **Combined Score Calculation:**
    * The **expanded encoder and decoder transformed outputs** are **added** and **passed through tanh activation**.
    * **Resulting shape:** **`(batch_size, seq_len_dec, seq_len_enc, units)`**
  * **Final Score Computation:**
    * **`attention_v`:** A **Dense layer** that **projects the combined scores** to a **scalar value** for **each encoder-decoder time step pair**.
    * **`Resulting shape`:** **`(batch_size, seq_len_dec, seq_len_enc, 1)`**
    * The **last dimension is squeezed** to get the **final score tensor**.
      * **Shape:** **`(batch_size, seq_len_dec, seq_len_enc)`**
     
### 3. Attention Weights Computation
  * **Purpose:** Convert the **alignment scores** into **normalized attention weights**.
  * **Functionality:**
    * **Softmax Activation:**
      * The alignment scores are **passed through a softmax function** along the **encoder time step axis (`axis=-1`)**.
      * This **converts the scores into probabilities** that **sum to 1** for **each decoder time step**.
      * **Shape:** **`(batch_size, seq_len_dec, seq_len_enc)`**

### 4. Mask Handling
  * **Purpose:** Ensure that the attention mechanism **does not consider padded positions in the encoder outputs**.
  * **Functionality:**
    * If an **encoder mask is provided** (**`encoder_mask`**), it is **expanded and applied to the alignment scores before softmax**.
    * **Masked positions** receive a **large negative value (`-1e9`)**, effectively **zeroing out their attention weights after softmax**.
     
### 5. Context Vector Computation
  * **Purpose:** Compute a context vector for **each decoder time step** as a **weighted sum of the encoder outputs**.
  * **Functionality:**
    * **Weighted Sum:**
      * The **attention weights** are used to **compute a weighted sum over the encoder outputs**
      * **Matrix Multiplication:**
        * **`context_vector = tf.matmul(attention_weights, encoder_output)`**
        * **Shape:** **`(batch_size, seq_len_dec, enc_units)`**
      * Each context vector **encapsulates relevant information** from the encoder outputs **specific to each decoder time step**.

## 5.4 Model Documentation

### 5.4.1 Model Training Pipeline

A **high-level overview** of the **main steps** in the model training pipeline is illustrated in **Fig 9**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/9bbf0643-76d8-4d9d-a080-60025852a4da", alt="seq2seq-model-train-pipeline"/>
    <p>
      <b>Fig 9</b> Retrosynthesis Seq2Seq model training pipeline.
    </p>
  </div>
<br>

### i. Configuration and Hyperparameter Loading

The training pipeline starts with the **initialisation of the `Trainer` class** in the `training.py` script. 

During initialisation, the training environment configuration and model hyperparameters are loaded from a **YAML file** via the **`Trainer.load_config()` method**, which is invoked in the **class constructor**. This data is stored in the `Trainer` class **`config` instance attribute**.

### ii. Data Loading, Tokenization, Preprocessing and Splitting

Another method that is **invoked in the `Trainer` class constructor**, is the **`Trainer.initialize_components()` method**. 

Within this method, the **`Dataloader` class is initialised** by passing in relevant data from the **`config` instance attribute** to the **constructor of the `Dataloader` class**. Once initialised, the `Dataloader` instance is stored in the `Trainer` class **`data_loader` instance attribute**.

Within the same `Trainer.initialize_components()` method, the **`Dataloader.load_and_prepare_data()` method is then called**. This method **orchestrates the entire data preparation pipeline**, including:
1. **Loading Raw Data** - Load **product and reactant SMILES sequences** from the specified data files.
    * **Train and Test Data**
      * **`products_file`**: Contains SMILES strings of product molecules (input sequences).
      * **`reactants_file`**: Contains SMILES strings of corresponding reactant molecules (output sequences).
    * **Validation Data**
      * **`products_valid_file`**: Contains SMILES strings for validation.
      * **`reactants_valid_file`**: Contains corresponding reactant SMILES strings for validation.
      * Validation is carried out at the **end of each epoch**.
    * Additional logic is present to **limit the number of samples** if specified, and to **ensure product and reactant data sets are of equal length**.
2. **Data Tokenization** - Converting SMILES strings into **tokenized sequences** suitable for **neural network input**.
    * **SMILES Tokenizer Initialisation**
      * Initialise a custom `SmilesTokenizer` class that handles the tokenization of SMILES strings
      * Default special tokens include **`<START>`**, denoting the **beginning of a sequence**, **`<END>`**, denoting the **end of a sequence**, and **`''`**, representing an **out-of-vocabulary token**.
    * **Tokenization Process**
      * Tokenize sequences using **`deepchem.feat.smiles_tokenizer.BasicSmilesTokenizer`** tokenizer.
      * For **input sequences (products)**, optionally **reverse the sequence** (commonly used to **improve performance of LSTM layers**), and add `<START>` and `<END>` tokens to each sequence.
      * For **output sequences (products)**, **do not reverse the sequence**, and add `<START>` and `<END>` tokens to each sequence.
    * **Vocabulary Building**
      * The `SmilesTokenizer` has a **Tensorflow `TextVectorization` layer** which is **adapted to the combined tokenized training data (products and reactants) to build the vocabulary**.
      * This ensures that **all tokens in the training data** are **recognised by the tokenizer**.
      * The `TextVectorization` layer is **not adapted to either the testing or validation data** to **prevent data leakage**.
      * This is not an issue for the testing data as it **comes from the same sourec as the training data**, but **additional steps** are carried out to **ensure the validation data does not contain tokens that aren't present in the training data**.
3. **Data Preprocessing** - Convert tokenized SMILES strings into **padded sequences of integers** and **prepare input-output pairs**.
    * **Sequence Conversion**
      * Use the tokenizer to **convert tokenized SMILES strings** into **sequences of integer tokens**.
      * **Map each token** to its **corresponding index in the vocabulary**.
    * **Padding and Truncation**
      * **Pad sequences** to a **fixed maximum length** (`max_seq_length`) for **uniformity**.
      * **Truncate sequences** longer than the maximum length.
    * **Input-Output Pair Preparation**
      * **Encoder Input**: Padded sequences of product SMILES tokens.
      * **Decoder Input**: Padded sequences of reactant SMILES tokens **shifted by one position to the right**.
      * **Decoder Target**: Actual reactant SMILES tokens used for **calculating loss** and for **teacher forcing**.
      * By **shifting the decoder input** compared to the **decoder target**, we **facilitate teacher forcing** by **providing the correct previous token (from the ground truth) as input**. This allows the model to **focus on learning the mapping from the current input to the next token**.
4. **Data Splitting** - Split the data into **training, validation, and test data sets**
    * **Train and Test Split**
      * Use **scikit-learn's `train_test_split()` method** to divide the data into training and test data sets based on the specified **`test_size`** from the **configuration YAML file**.
      * Ensure that the **splitting is random** but **reproducible** by passing a **`random_state`** parameter to `train_test_split()`.
    * **Validation Data Set**
      * The validation files have **already been split** prior to the training pipeline.
5. **TensorFlow Dataset Creation** - Create **TensorFlow datasets** for **efficient data loading** during training.
    * **Dataset Construction**
      * Convert the **preprocessed data** into **`tf.data.Dataset`** objects.
      * **Pair encoder and decoder inputs** with **decoder targets**.
    * **Shuffling and Batching**
      * Shuffle the training dataset to **ensure randomisation**.
      * **Batch the data sets** with a specified **`batch_size`** from the **configuration YAML file**.
      * Utilise **`tf.data.Dataset.prefetch(tf.data.AUTOTUNE)`** to **prefetch data for optimised performance**. This enables TensorFlow to **prepare data for the next training step** while the **current step is still executing**.

### iii. Training Environment Setup

Once the `Trainer` class has been initialised, the `Trainer.run()` method is invoked in the `training.py` script. 

This is the **entry point into the model training pipeline** and calls the following **`Trainer` class methods**:
1. **`TrainingEnvironment.setup_environment(Trainer.config)`**
2. **`Trainer.setup_model()`**
3. **`Trainer.build_model()`**
4. **`Trainer.setup_callbacks()`**
5. **`Trainer.train()`**
6. **`Trainer.model.summary()`**
7. **`Trainer.save_model()`**
8. **`Trainer.evaluate()`**

The `TrainingEnvironment.setup_environment()` method is used to **set up the training environment for deterministic (reproducible) training**. This is discussed in detail in [**Section 5.3.1**](https://github.com/c-vandenberg/aizynthfinder-project/edit/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#531-deterministic-training-environment).

### iv. Model Initialisation and Compilation

The **`Trainer.setup_model()`** method **orchestrates the initialisation and compilation** of the Seq2Seq model via:
1. **Model Initialisation** - Set up the Seq2Seq model encoder-decoder architecture with attention mechanisms.
    * **Model Architecture**
      * **Encoder**:
        * **Embedding layer** to **map each encoder input token** to a **dense embedding vectors** of a **fixed size (`encoder_embedding_dim`)**.
        * **LSTM recurrent neural network** to process the encoder input sequence.
        * Configurable **embedding dimension** and **number of layers and units**.
      * **Decoder**:
        * **Embedding layer** to **map each decoder input token** to a **dense embedding vectors** of a **fixed size (`decoder_embedding_dim`)**.
        * **LSTM recurrent neural network** with **attention mechanism** to process the decoder input sequence.
        * Configurable **embedding dimension**and **number of layers and units**.
      * **Attention Mechanism**:
        * Allows the decoder to **focus on different parts of the input sequence** at **each decoding step**.
        * Configurable **number of units**
    * **Hyperparameters**
      * **`encoder_embedding_dim`**: Dimension of the encoder's embedding layer.
      * **`decoder_embedding_dim`**: Dimension of the decoder's embedding layer.
      * **`units`**: Number of **units/neurons** in the **LSTM layers**.
      * **`attention_dim`**: Dimension of the attention layer (**number of units/neurons**).
      * **`encoder_num_layers`** and **`decoder_num_layers`**: Number of **encoder and decoder layers** respectively.
      * **`dropout_rate`**: Rate for **dropout regularization**.
      * **`weight_decay`**: Optional **weight decay for regularization**.
2. **Model Compliation** - Compile the Seq2Seq model with the **appropriate optimiser, loss function, and metrics**.
  * **Optimiser**
    * Use the **`Adam` optimiser** with a specified **`learning_rate` from the **configuration YAML file**.
    * Set **`clipnorm`** to **prevent gradient explosion**.
  * **Loss Function**
    * Use core **`tf.keras.losses.SparseCategoricalCrossentropy()` loss function**.
    * Ensure that **padded elements in the sequences** are **not taken into account in loss calculation**.
  * **Metrics**
    * Use core metrics specified in the **configuration YAML file** such as `accuracy`.
    * Use a custom **`Perplexity`** metric based on the **loss function**.

The **`Trainer.build_model()` method** is used to **initialise the model's weights and variables** by **running a sample input through the model**. 
  * When using **subclassed models**, the model's variables and weights are **not fully initialised until the model is called on some input data**.
  * This is because the **shapes and structures of the internal variables depend on the input data's shape**.

Therefore by **running a sample input through the model**, all of the **weights and variables are initialised** and the **computational graph is set up**, ensuring the model is **fully built and ready for training**.

### v. Callbacks Setup

The **`Trainer.setup_callbacks()` method** is used to **configure the various callbacks** specified in [**Section 5.3.4**](https://github.com/c-vandenberg/aizynthfinder-project/edit/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#534-callbacks-optimisation).

### vi. Model Training

Once the **data sets have been tokenized, preprocessed and prepared**, and the **model initialised and compiled**, the model is trained via the **`Trainer.train()` method**.
* **Training Procedure**:
  * The procedure uses the **`tf.keras.Model.fit()` method** with the **training and validation data sets** passed as arguments, along with the **callbacks**.
  * Additionally, the **number of epochs** is given as specified in the **configuration YAML file**.
* **Progress Monitoring**:
  * Both **training and validation accuracy, loss and metrics** are monitored, and the **callbacks** are used to **adjust training dynamically**.

### vii. Model Saving

Before the model is **evaluated with the test data**, it is **saved** using the **`Trainer.save_model()` method**. This method utilises **utility functions** to save the model in the following formats:
1. **Keras Format (.keras)**: Save model TensorFlow's **native Keras format**.
2. **HDF5 Format (.h5)**: Save **weights and architecture** in **HDF5 format for compatibility**.
3. **ONNX Format (.onnx)**: Export the model to **ONNX format** for **interoperability with other frameworks**.
4. **SavedModel Format**: Save the model in TensorFlow's historic **SavedModel format**.

### viii. Model Evaluation

The **final step** in the model training pipeline is to use the **`Trainer.evaluate()` method** to **evaluate the model on unseen (test) data**.
* **Evaluation Procedure**:
  * Use the **`tf.keras.Model.evaluate()` method** to compute **loss, accuracy and perplexity**.
  * **Generate predictions** on the test data set using **beam search decoding** as the **inference method**.
  * Convert predicted token sequences **back to SMILES strings** and **compute same metrics as validation metrics** listed in [**Section 5.3.5**](https://github.com/c-vandenberg/aizynthfinder-project/edit/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#535-metrics-optimisation).

### 5.4.2 Model Data Flow - ONNX Graph

The **flow of data** through the model's **encoder-decoder architecture** is shown in **Fig 10**.

<br>
  <div align="center">
    <a href="https://github.com/user-attachments/assets/94d3b08d-f39e-411a-b0cd-37d5f91a51e5" target="_blank">
      <img src="https://github.com/user-attachments/assets/94d3b08d-f39e-411a-b0cd-37d5f91a51e5", alt="retro_seq2seq_model_graph"/>
    </a>
    <p>
      <b>Fig 10</b> Retrosynthesis Seq2Seq Model Graph. Netron model visualisation tool <b><sup>11</sup></b> shows ONNX graph with nodes and operations for encoder (red), decoder (blue), and attention mechanism (green) (click to expand view).
    </p>
  </div>
<br>

### i. Flow of Data Through Encoder
1. **Encoder Input Processing**
    * **Input:** Sequences of **token indices** (**`encoder_input`**) with **shape `(batch_size, sequence_length)`**.
    * **Validation:** Checks that **`encoder_input`** is **not `None`** and is a **2D tensor**.
2. **Encoder Embedding Layer**
    * **Operation:** Converts/maps **encoder input (`encoder_input`) tokens** to **embeddings**.
    * **Output:** Outputs **`encoder_output`** with **shape `(batch_size, sequence_length, encoder_embedding_dim)`**
    * **Mask**: A mask (**`encoder_mask`**) is generated to **identify padding tokens**.
3. **Encoder Variable Initialisation**
    * **Residual Connections**: Initialise **`previous_output`** with **`encoder_output`** for residual connections later.
    * **Forward & Backward State Concatenation:** Initialise **final hidden state** (**`final_state_h`**) and **final cell state** (**`final_state_h`**) variables as **`None`** for later **concatenation of forward and backward hidden and cell states**.
4. **Encoder Stacked Bidirectional LSTM Layers Loop - For each layer (`i` from 0 to `num_layers - 1`)**
    * **Bidirectional LSTM Layer**
      * **Input:** The **`encoder_input`** from the **previous layer**
      * **Outputs:**
        * **Encoder Output:** The **updated sequence representations** (**`encoder_output`**).
          * **Shape:** **`(batch_size, sequence_length, units * 2)`**.
        * **Hidden States:** The **forward and backwards hidden states** (**`forward_h, backward_h`**).
          * **Shape:** **`(batch_size, units)`**.
        * **Cell States:** The **forward and backwards cell states** (**`forward_c, backward_c`**).
          * **Shape:** **`(batch_size, units)`**.
    * **Concatenate Hidden and Cell States**
      * **Forward & Backward Hidden States:** The `[forward_h, backward_h]`** variables are **concatenated along the last axis** to give the updated **`final_state_h`**.
        * **Shape:** **`(batch_size, units * 2)`**.
      * **Forward & Backward Cell States:** The `[forward_c, backward_c]`** variables are **concatenated along the last axis** to give the updated **`final_state_c`**.
        * **Shape:** **`(batch_size, units * 2)`**.
    * **Layer Normalisation**
      * Apply a **`tensorflow.keras.layers.LayerNormalization` layer (`layer_norm_layer()`)** to **`encoder_output`**.
    * **Residual Connections**
      * For **layers beyond the first layer (`i > 0`)**, add **`previous_output`** to **`encoder_output`** for **residual connections between layers**.
    * **Update `previous_output`**
      * Set **`previous_output`** to **current `encoder_output`** for **use in the next layer**.
    * **Dropout**
      * Apply a **`tensorflow.keras.layers.Dropout` layer** to **`encoder_output`**.
5. **Encoder Final Outputs:**
    * **`encoder_output`:** The **final sequence representations** after all layers.
    * **`final_state_h`:** The **last hidden states concatenated** from the **forward and backward directions**.
    * **`final_state_c`:** The **last cell states concatenated** from the **forward and backward directions**. 

### ii. Flow of Data Through Decoder
1. **Decoder Input Processing**
    * **Inputs:**
      * **Decoder Input:** Sequences of **target token indices** that have been **shifted for teacher forcing**.
        * **Shape:** **`(batch_size, sequence_length_dec)`**.
      * **Encoder Final States:** The **final hidden and cell states** from the encoder.
      * **Encoder Output**: Outputs from the encoder, used for attention.
        * **Shape:** **`(batch_size, sequence_length_enc, enc_units)`**
2. **Decoder Layer State Initialisation**
    * **First LSTM Layer State**
      * **Initial Hidden State**: The first LSTM layer hidden state (**`decoder_initial_state_h`**) is obtained by passing the encoder's final hidden state through a **`tensorflow.keras.layers.Dense` layer** (**`enc_state_h`**), and **mapping this to `decoder_initial_state_h`**.
      * **Initial Cell State**: The first LSTM layer cell state **`(decoder_initial_state_c)`** is obtained by passing the encoder's final cell state through a **`tensorflow.keras.layers.Dense` layer** (**`enc_state_c`**), and **mapping this to `decoder_initial_state_c`**.
    * **Subsquent LSTM Layer States**
      * Subsequent LSTM layer states are **initialised to zeros**.
3. **Decoder Embedding Layer**
    * **Operation:** Converts/maps **decoder input (`decoder_input`) tokens** to **embeddings**
    * **Output:** Outputs **`decoder_output`** with **shape `(batch_size, sequence_length_dec, decoder_embedding_dim)`**
    * **Mask**: A mask (**`decoder_mask`**) is generated to **identify padding tokens**.
4. **Decoder Stacked Bidirectional LSTM Layers Loop - For each layer (`i` from 0 to `num_layers - 1`)**
    * **LSTM Layer**
      * **Input:** The **`decoder_input`** from the **previous layer**.
      * **Initial States:** The **hidden and cell states** (`state_h`, `state_c`) for the current layer.
      * **Outputs:**
        * **Decoder Output:** The **updated sequence representations** (**`decoder_output`**).
          * **Shape:** **`(batch_size, sequence_length_dec, units)`**.
        * **Updated States:** The **updated hidden and cell states (`state_h`, `state_c`)** for the current layer.
    * **Layer Normalisation**
      * Apply a **`tensorflow.keras.layers.LayerNormalization` layer (`layer_norm_layer()`)** to **`decoder_output`**.
    * **Update `previous_output`**
      * Set **`previous_output`** to **current `decoder_output`** for **use in the next layer**.
    * **Dropout**
      * Apply a **`tensorflow.keras.layers.Dropout` layer** to **`decoder_output`**.
5. **Attention Mechanism**
    * **Compute Context Vector and Attention Weights**
      * **Inputs:** **`encoder_outout`** and **`decoder_output`**.
      * **Outputs:**
        * **Context Vector:** Vector that represents the **relevant encoder outputs for each decoder time step** (**`context_vector`**).
          * **Shape:** **`(batch_size, sequence_length_dec, enc_units)`**
        * **Attention Weights:** The **alignment scores** (**`attention_weights`**).
          * **Shape:** **`(batch_size, sequence_length_dec, sequence_length_enc)`**.
6. **Residual Connections and Layer Normalisation Around Attention**
    * **Projection Layers**
      * **Dense Decoder Output**: Pass **`decoder_output`** through a **`tensorflow.keras.layers.Dense` layer (**`decoder_dense`**) to give **`decoder_transformed`**.
      * **Dense Context Vector**: Pass **`context_vector`** through a **`tensorflow.keras.layers.Dense` layer (**`context_dense`**) to give **`context_transformed`**.
    * **Addition for Residual Connection**
      * Sum the **`decoder_transformed`** and **`context_transformed`** to give **`decoder_output`**.
    * **Layer Normalisation and Activation**
      * Apply a **`tensorflow.keras.layers.LayerNormalization` layer (`layer_norm_layer()`)**.
      * Use **`tensorflow.nn.relu`** on **`decoder_output`** for **activation**.
7. **Decoder Output Layer**
    * **Generate Output Probabilities**
      * Pass **`decoder_output`** through the **final Dense layer** with **softmax activation** to give **final `decoder_output`**.
        * **Shape:** **`(batch_size, sequence_length_dec, vocab_size)`**.
       
### iii. Flow of Data Through Attention Mechanism
1. **Inputs**
    * **Encoder Output:** Outputs from the encoder (**`encoder_output`**).
      * **Shape:** **`(batch_size, seq_len_enc, enc_units)`**.
    * **Decoder Output:** Outputs from the decoder prior to attention (**`decoder_output`**).
      * **Shape:** **`(batch_size, seq_len_dec, dec_units)`**.
    * **Mask (optional):** A mask (**`encoder_mask`**) is optionally used to **handle padding tokens**.
      * **Shape:** **`(batch_size, seq_len_enc)`**.
2. **Transformation of Encoder and Decoder Outputs**
    * **Encoder Transformation**
      * Pass **`encoder_output`** through a **`tensorflow.keras.layers.Dense` layer (**`self.attention_dense1`**) to give **`score_enc`**.
        * **Shape:** **`(batch_size, seq_len_enc, units)`**.
    * **Decoder Transformation**
      * Pass **`decoder_output`** through a **`tensorflow.keras.layers.Dense` layer (**`self.attention_dense2`**) to give **`score_dec`**.
        * **Shape:** **`(batch_size, seq_len_dec, units)`**.
3. **Expansion of Transformed Encoder and Decoder Outputs**
    * **Transformed Encoder Output Expansion**
      * Pass **`score_enc`** to **`tensorflow.expand_dims()`** to give **`score_enc_expanded`**.
        * **Shape:** **`(batch_size, 1, seq_len_enc, units)`**.
    * **Transformed Decoder Output Expansion**
      * Pass **`score_dec`** to **`tensorflow.expand_dims()`** to give **`score_dec_expanded`**.
        * **Shape:** **`(batch_size, seq_len_dec, 1, units)`**.
4. **Combined Score Calculation**
    * **Addition and Activation**
      * **Add `score_enc_expanded` and `score_dec_expanded` together** and pass to **`tensorflow.nn.tanh()`** for **activation** to give `score_combined`.
        * **Shape:** **`(batch_size, seq_len_dec, seq_len_enc, units)`**.
    * **Projection to Scalar Score**
      * Pass **`score_combined`** through a **`tensorflow.keras.layers.Dense` layer (**`self.attention_v`**) to give **`score`**.
        * **Shape:** **`(batch_size, seq_len_dec, seq_len_enc, 1)`**.
    * **Squeeze Out Last Dimension**
      * Pass **`score`** to **`tensorflow.squeeze()`** to **remove last dimension** and give final **`score`**.
        * **Shape:** **`(batch_size, seq_len_dec, seq_len_enc)`**.
5. **Applying Mask (if provided)**
    * **Expand Mask Dimensions to Match `score` Tensor**
      * Pass `encoder_mask` to `tensorflow.expand_dims()` to give `encoder_mask_expanded`.
        * **Shape:** **`(batch_size, 1, seq_len_enc)`**.
    * **Adjust Scores**
      * Add a **large negative value to masked positions** to **nullify their effect in softmax** via **`score += (1.0 - tf.cast(encoder_mask_expanded, score.dtype)) * -1e9`**.
6. **Attention Mechanism Computation**
    * **Softmax Over Encoder Sequence Length/Time Steps**
      * Pass **`score`** to **`tensorflow.nn.softmax()`** to give **`attention_weights`**.
        * **Shape:** **`(batch_size, seq_len_dec, seq_len_enc)`**.
7. **Context Vector Computation**
    * **Weighted Sum**
      * Compute context vector as **weighted sum of encoder outputs** by passing both **`attention_weights`** and **`encoder_output`** to **`tensorflow.matmul()`** to give **`context_vector`**.
        * **Shape:** **`(batch_size, seq_len_dec, enc_units)`**.

### 5.4.3 Model Debugging

### i. Data Tokenization and Preprocessing Debugging
1. **Analyse Data Set Token Frequency Distribution**:
   * Add `tensorflow.print()` statement **immediately after data set tokenization** in `expansion-policies.seq2seq-expansion-policy.src.data.utils/data_loader._tokenize_datasets`. For example, to check **tokenized products (input) data set**:
     
      ```
      train_test_products_all_tokens = ' '.join(self.tokenized_products_x_dataset).split()
      train_test_products_token_counts = Counter(train_test_products_all_tokens)
      print(
        f"Tokenized Train & Test Products x-Dataset Token Frequency Distribution: {train_test_products_token_counts.most_common(20)}\n"
      )
      ```
    
2. **Analyse Tokenizer Functionality**:
  * Check that tokenizer **reverses only the product (input/x-data) data set** by printing/logging random tokenized data set samples. For example:
    
      ```
      secure_random = random.SystemRandom()
      print(f"Tokenized Products x-Dataset Sample: {secure_random.choice(self.tokenized_products_x_dataset)}")
      print(f"Tokenized Reactants y-Dataset Sample: {secure_random.choice(self.tokenized_reactants_y_dataset)}")
      print(f"Tokenized Products Validation x-Dataset Sample: {secure_random.choice(self.tokenized_products_x_valid_dataset)}")
      print(f"Tokenized Products x-Dataset Sample: {secure_random.choice(self.tokenized_reactants_y_valid_dataset)}")
      ```
    
  * Check tokenizer word index **before and after adapting** with `tensorflow.print()` statement. Tokenizer is adapted to **tokenized products only** to prevent data leakage. This is carried **immediately after splitting data** in data loader class `expansion-policies.seq2seq-expansion-policy.src.data.utils/data_loader._split_datasets()`. For example:
    
      ```
      combined_tokenized_train_data = self.tokenized_products_x_train_data + self.tokenized_reactants_y_train_data
      tf.print(f"Tokenizer Word Index Before Adapting: \n{self.smiles_tokenizer.word_index}\n")
      self.smiles_tokenizer.adapt(combined_tokenized_train_data)
      tf.print(f"Tokenizer Word Index After Adapting: \n{self.smiles_tokenizer.word_index}\n")
      ```
    
  * **N.B.** Because environment setup has **set seeds for `random` psuedorandom number generator**, you will get the **same 'random' samples each time**.
3. **Analyse Data Preprocesser Functionality**:
  * Print/log random preprocessed data set samples and **cross reference integers with adapted tokenizer word index**. For example:
    
      ```
      secure_random = random.SystemRandom()
      tf.print(f"Preprocessed Training Data Sample: {secure_random.choice(self.train_data)}\n")
      tf.print(f"Preprocessed Testing Data Sample: {secure_random.choice(self.test_data)}\n")
      tf.print(f"Preprocessed Validation Data Sample: {secure_random.choice(self.valid_data)}\n")
      ```

### ii. General TensorFlow Debugging
1. **Analyse Tensor Shape**:
   * Add `tensorflow.print()` statement to print/log to **dynamically check tensor dimensions/shape**. For example:
     
       ```
       encoder_output: tf.Tensor = self.embedding(encoder_input)
       tf.print(encoder_output.shape) # Shape: (batch_size, seq_len, embedding_dim)
       ```

## 5.5 Results and Discussion

### 5.5.1 Analysis of Best Performing Model Architecture

As of **21/11/24**, the **top model architecture** has been evaluated using **two sets of hyperparameters**. These have been given the designations **Model V27** and **Model V28**.

Model V28 adopts a configuration similar to the Seq2Seq model developed by *Liu et al.*, however, it  was **highly computationally expensive** to train. Therefore, for **performance comparison**, Model V27 was configured with the **number of neurons/nodes (units)**, the **size of the token vector representations** for both the encoder and decoder, and the **attention vector dimensionality** all **reduced to 256**. All other hyperparameters are **consistent between the two models** (**Table 3**).

<div style="display: flex;" align="center">
  <table border="1" cellspacing="0" cellpadding="5">
    <thead>
        <tr>
            <th>Hyperparameter</th>
            <th>Model V27</th>
            <th>Model V28</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Batch Size</td>
            <td>32</td>
            <td>32</td>
        </tr>
        <tr>
            <td>Units</td>
            <td>256</td>
            <td>512</td>
        </tr>
        <tr>
            <td>Encoder Embedding Dimension</td>
            <td>256</td>
            <td>512</td>
        </tr>
        <tr>
            <td>Decoder Embedding Dimension</td>
            <td>256</td>
            <td>512</td>
        </tr>
        <tr>
            <td>Attention Dimension</td>
            <td>256</td>
            <td>512</td>
        </tr>
        <tr>
            <td>Number of Encoder Layers</td>
            <td>2</td>
            <td>2</td>
        </tr>
        <tr>
            <td>Number of Decoder Layers</td>
            <td>4</td>
            <td>4</td>
        </tr>
        <tr>
            <td>Max Encoder Sequence Length</td>
            <td>140</td>
            <td>140</td>
        </tr>
        <tr>
            <td>Max Decoder Sequence Length</td>
            <td>140</td>
            <td>140</td>
        </tr>
        <tr>
            <td>Dropout Rate</td>
            <td>0.8</td>
            <td>0.8</td>
        </tr>
        <tr>
            <td>Learning Rate</td>
            <td>1e-4</td>
            <td>1e-4</td>
        </tr>
        <tr>
            <td>Beam Width</td>
            <td>5</td>
            <td>5</td>
        </tr>
    </tbody>
  </table>
  <p>
    <b>Table 3</b> Model V27 and Model V28 hyperparameters.
  </p>
</div>

Both models were trained with using identical product and reactant datasets consisting of approximately **50,000 reactions**, derived from an **open source patent database** **<sup>12</sup>** by *Liu et al.*. These datasets were processed as described in [Section 5.1](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#51-data-preparation) and split into training and testing sets with a 7:3 ratio. The validation product and reactant datasets were processed in the same manner, but were pre-split by *Liu et al.*.

Both models were evaluated against test datasets using a **variety of metrics** (**Table 4**). These metrics included **standard machine learning and natural langauge processing (NLP) performance metrics** such as **loss**, **accuracy**, **perplexity** and **BLEU score**.

However, since these metrics **do not account for the chemical properties of the SMILES sequences**, it was necessary to incorporate **cheminformatics-based metrics** to provide a **more comprehensive assessment** of the model's predictions. This included the core cheminformatic metric, the **Tanimoto coefficient**, as well as a custom **chemical validity score** which measures the **ratio of chemically valid to non-valid SMILES strings predicted by the model**.

Additionally, **string metrics** such as **Levenshtein Distance** and **exact match accuracy** were employed to **broaden the range of sequence similarity evaluations**.

<div style="display: flex;" align="center">
  <table border="1" cellspacing="0" cellpadding="5">
    <thead>
        <tr>
            <th>Evaluation Metric</th>
            <th>Model V27</th>
            <th>Model V28</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Loss</td>
            <td>0.157</td>
            <td>0.157</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>0.984</td>
            <td>0.985</td>
        </tr>
        <tr>
            <td>Perplexity</td>
            <td>1.170</td>
            <td>1.170</td>
        </tr>
        <tr>
            <td>BLEU Score</td>
            <td>0.665</td>
            <td>0.688</td>
        </tr>
        <tr>
            <td>Average Levenshtein Distance</td>
            <td>32.163</td>
            <td>30.461</td>
        </tr>
        <tr>
            <td>Exact Match Accuracy</td>
            <td>0.105</td>
            <td>0.138</td>
        </tr>
        <tr>
            <td>Chemical Validity Score</td>
            <td>1.000</td>
            <td>1.000</td>
        </tr>
        <tr>
            <td>Average Tanimoto Coefficient</td>
            <td>0.869</td>
            <td>0.874</td>
        </tr>
    </tbody>
  </table>
  <p>
    <b>Table 4</b> Performance metrics of Model V27 and Model V28 on test datasets.
  </p>
</div>

For **test loss**:
* Test loss represents the **average error between the predicted sequences and the target sequences** on the **test dataset**.
* This metric is calculated using Tensorflow's **Sparse Categorical Cross Entropy Loss Function** (**`tf.keras.losses.SparseCategoricalCrossentropy`**), which measures the **discrepancy between the predicted probability distribution** and the **actual distribution of the target sequences**.
* Both **Model V27** and **Model V28** achieved a **test loss of 0.157**, which is **relatively low**. This indicates that the predictions made by both models are **closely aligned with the target sequences in terms of the loss function**, reflecting **high accuracy** and **effective learning** during training.

For **test accuracy**:
* Test accuracy refers to the **token-level accuracy**, measuring the **percentage of individual tokens predicted correctly** compared to the **ground truth tokens** at each timestep.
* Both **Model V27** and **Model V28** achieved **nearly identical test accuracies**, with Model V27 attaining an accuracy of **~98.4%** and Model V28 achieving **~98.5%**.
* These high accuracies suggest that the models **correctly predict individual tokens in approximately 98.4–98.5% of timesteps**, indicating **robust performance in token prediction** and **effective learning** during training.

For **test perplexity**:
* Perplexity is a measure of **how well a probability model predicts a sample**, calculated as the **exponential of the cross-entropy loss**. A **lower perplexity** indicates **better predictive performance**.
* In the context of retrosynthetic SMILES predictions, low perplexity signifies that the model can **accurately predict the sequence of tokens representing valid retrosynthetic transformations**.
* Both **Model V27** and **Model V28** achieved a **test perplexity of 1.170**, which is **excellent**. This demonstrates that both models are **highly confident and accurate in their token predictions**, effectively **capturing the underlying patterns** in the data.

For **test BLEU score**:
* The **BLEU (Bilingual Evaluation Understudy) score** measures the **similarity between the predicted sequences and the reference sequences**, focusing on **matching n-grams**. Scores **range from 0 to 1**, with **higher scores indicating closer matches**.
* In the context of retrosynthetic SMILES predictions, a higher BLEU score signifies that the model's generated sequences **exhibit substantial n-gram overlap with the target sequences**.
* **Model V27** achieved a **BLEU score of 0.665**, while **Model V28** attained a **BLEU score of 0.688**. These scores indicate that the predictions made by both models **share substantial n-gram overlap with the target sequences**.
* **Model V28's slightly better BLEU score** suggets its predictions are **marginally more similar to the targets** in terms of **n-gram overlap**. This improvement, although modest, highlights the **enhanced capability of Model V28** in generating sequences that **more closely align with the reference data**.

For **test Average Levenshtein Distance**:
* Levenshtein Distance measures the **minimum number of single-character edits** required to **change one sequence into another**.
* **Model V27** has an **average Levenshtein Distance of 32.163**, while **Model V28** achieved an **average distance of 30.461**.
* This indicates that, on average, approximately **32 edits** are needed to **convert Model V27's predicted sequences into the target sequences**, compared to **~30 edits for Model V28**.
* A **lower average Levenshtein Distance** suggests that **Model V28's predictions are closer to the target sequences** with **fewer edits required**, highlighting its **superior performance** in generating accurate retrosynthetic SMILES predictions.

For **test exact match accuracy**:
* Exact match accuracy measures the **percentage of predictions that exactly match the target sequences in their entirety**.
* **Model V27** achieved an **exact match accuracy of ~10.50%**, while **Model V28** attained an **exact match accuracy of ~13.80%**. These **relatively low percentages** highlight the **significant semantic complexity inherent in SMILES-based retrosynthesis predictions**.
* **Even a single character difference** in a SMILES string can result in a **completely different or invalid molecule**. Consequently, the **low exact match percentages** present a **significant challenge** for integrating SMILES-based retrosynthesis models into AiZynthFinder as an expansion policy. Overcoming this hurdle is crucial to ensure the **reliability and accuracy** of the predicted retrosynthetic pathways.

For **chemical validity score**:
* Chemical validity score represents the proportion of predicted SMILES strings that **correspond to valid chemical structures when parsed by RDKit**. This metric ensures that the generated SMILES strings translate into chemically viable molecules, which is essential for retrosynthesis predictions.
* Both Model V27 and Model V28 achieved a **perfect chemical validity score of **1.000**, indicating that **all predicted SMILES strings are chemically valid**.
* This **100% validity** underscores the models' ability to **generate syntactically correct and chemically feasible molecular structures**, thereby ensuring reliable retrosynthetic predictions when integrated into AiZynthFinder.

For **average Tanimoto coefficient**:
* The **Tanimoto coefficient** (also known as the **Jaccard index**) measures the **similarity between two chemical structures based on their fingerprints**, ranging from **0 (no similarity)** to **1 (identical structures)**.
* The **threshold for similarity** is **intrinsically subjective** and is **highly dependent on the molecular fingerprint used**. For instance, with **Daylight fingerprints**, two structures are **typically considered similar** if **$$T > 0.85$$** **<sup>13</sup>**.
* Our Seq2Seq model utilized **Morgan fingerprints** via **`rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect`** to calculate the Tanimoto coefficient. In future iterations, **comparative Tanimoto coefficients** based on **different molecular fingerprints** will be implemented to **enhance similarity assessments**.
* **Model V27** has an **average Tanimoto coefficient of 0.869**, while **Model V28** achieved an **average coefficient of 0.874**. This indicates that the predictions from both models are **very structurally similar to the target molecules**, with **Model V28** demonstrating **marginally greater similarity**.

### 5.5.2 Integrating Seqeuence-to-Sequence Model into AiZynthFinder

When evaluating the performance metrics of the two models, it became evident that despite showing **identical token-level metrics**, **Model V28 outperformed Model V27 in sequence-level metrics** (**Table 3**). This improvement can be attributed to several factors:
1. **Greater Model Capacity**
    * **More units and larger embeddings** allow the model to **capture complex patterns** by **learning intricate dependencies**.
    * **Higher-dimensional embeddings** for the encoder and decoder enable the model to **capture more semantic information about the tokens**.
2. **Improved Generalisation**
    * The **larger capacity** of Model V28 **reduces the likelihood of underfitting**, potentially **enhancing validation and test performance** (**up to a point**).
3. **Enhanced Attention Mechanism**
    * A **larger attention vector** allows for **more precise computation of alignment scores between the encoder and decoder states**.
    * This improvement enables the attention mechanism to **focus more effectively on relevant parts of the input** when generating each output token.

However, when each model was **incorporated into AiZynthFinder** as the **expansion policy**, **Model V28 predicted correct precursors less frequently than Model V27**. Additionally, **Model V28 did not account for stereochemistry in its predictions** when tested with **chiral molecules**, whereas **Model V27 did**. Finally, **Model V27 provided more route options more consisitently during inference** than Model V28.

There are **two related explanations** for the is discrepancy between **improved sequence-level metrics** and **reduced generalisation and less diverse prediction options**:
1. **Overfitting of Model V28**
  * Overfitting occurs when a **model learns the training data too well**, including its **noise and outliers**, at the expense of its **ability to generalise to new, unseen data**. As a result, the model may **memorise training examples rather than generalisable patterns**.
  * To **mitigate overfitting**, the training regime utilises **TensorFlows Early Stopping Callback (`tensorflow.keras.callbacks.EarlyStopping`)**. This would **terminate training early** if the **validation loss did not improve over five consecutive epochs**
  * Model V28 has **significantly more parameters** than Model V27, due to its **larger units, as well as embedding and attention dimensions**. This **increases the risk of overfitting**, especially if the **training data is not sufficiently large or diverse**.
2. **Model Complexity vs Dataset Size**
  * Model V28 has **significantly more parameters** than Model V27 (**Table 5**), due to its **larger units, as well as embedding and attention dimensions**.
  * It is well known that **increased model complexity leads to overfitting on smaller training sets**. Therefore, if **training dataset size does not increase relative to the complexity of the model**, there is a **significant risk of overfitting**.

<div style="display: flex;" align="center">
  <table border="1" cellspacing="0" cellpadding="5">
    <thead>
        <tr>
            <th>Layer (Type)</th>
            <th>Model V27 Parameters</th>
            <th>Model V28 Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>stacked_bdirectional_lstm_encoder (StackedBidirectionalLSTMEncoder)</td>
            <td>2,650,368</td>
            <td>10,543,616</td>
        </tr>
        <tr>
            <td>stacked_lstm_decoder (StackedLSTMDecoder)</td>
            <td>2,543,962</td>
            <td>10,068,570</td>
        </tr>
        <tr>
            <td>enc_state_h (Dense)</td>
            <td>131,328</td>
            <td>524,800</td>
        </tr>
        <tr>
            <td>enc_state_c (Dense)</td>
            <td>131,328</td>
            <td>524,800</td>
        </tr>
    </tbody>
  </table>
  <p>
    <b>Table 5</b> Number of parameters in each layer type for Models V27 and V28.
  </p>
</div>

**Table 5** gives the **number of parameters in each layer type** for both Model V27 and Model V28. However, when considering **model complexity vs dataset size**, we **mainly need to focus on trainable parameters**.
* Trainable parameters are the parameters in a neural network that are **learned and updated during the training process through backpropagation/backproagation through time**.
* They include **weights and biases** of the layers in the model.
* These parameters are **adjusted to minimise the loss function** by **optimising the network's performance on the training data**.

<div style="display: flex;" align="center">
  <table border="1" cellspacing="0" cellpadding="5">
    <thead>
        <tr>
            <th>Model</th>
            <th>Trainable Parameters</th>
            <th>Non-Trainable Parameters</th>
            <th>Optimiser Parameters</th>
            <th>Total Parameters</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Model V27</td>
            <td>5,456,986</td>
            <td>0</td>
            <td>10,913,974</td>
            <td>16,370,960</td>
        </tr>
        <tr>
            <td>Model V28</td>
            <td>21,661,786</td>
            <td>0</td>
            <td>43,323,574</td>
            <td>64,985,360</td>
        </tr>
    </tbody>
  </table>
  <p>
    <b>Table 6</b> Comparison of the number of trainable parameters, non-trainable parameters, optimizer parameters, and total parameters for Models V27 and V28.
  </p>
</div>

As you can see from **Table 6**,

As a result, when **incorporated as the AiZynthFinder expansion policy**, Model V28 **produced superior retrosynthetic SMILES predictions**. These performance gains **more than compensated for the increased computational cost**, with **little indication of diminishing returns**.

## 5.6 Future Model Optimisations

Given that Seq2Seq models have **largely been superseded by transformer architectures**, the primary future priority for this research project is to **incorporate the encoder, decoder and attention mechanism of this model** into a **transformer-based expansion policy**. However, there are some features to add that could improve the performance of **both the Seq2Seq model**, and the **future transformer model**.

### 5.6.1 Increased Training Dataset Size

### 5.6.2 Layer-wise Learning Rate Decay
  * Layer-wise learning rate decay **adjusts the learning rate for different layers**, typically using **smaller learning rates for lower layers** and **larger ones for higher layers**. This approach can help **stabilise training**, especially in **very deep networks**.
  * Given that this Seq2Seq model has an **2-layer encoder** and a **4-layer decoder**, it is classified as a **deep neural network** and thus **could benefit from layer-wise learning rate decay**.

### 5.6.3 Scheduled Sampling
  * Scheduled Sampling is a technique used to **mitigate exposure bias during training** in NLP models.
  * Instead of **always using the ground truth tokens as inputs to the decoder during training**, the model **occasionally uses its own predictions as inputs** for the next time step.
  * This strategy helps the model become **more robust to its own errors during inference**.

### 5.6.4 High Throughput Testing of Model Expansion Policy Performance

## 5.7 References
**[1]** Liu, B. et al. (2017) ‘Retrosynthetic reaction prediction using neural sequence-to-sequence models’, ACS Central Science, 3(10), pp. 1103–1113. <br><br>
**[2]** Pandegroup (2017) ‘Pandegroup/reaction_prediction_seq2seq’, GitHub. Available at: https://github.com/pandegroup/reaction_prediction_seq2seq/tree/master (Accessed: 09 October 2024). <br><br>
**[3]** Determinism (2023) NVIDIA Docs. Available at: https://docs.nvidia.com/clara/clara-train-archive/3.1/nvmidl/additional_features/determinism.html (Accessed: 17 October 2024). <br><br>
**[4]** Chand, S. (2023) Choosing between cross entropy and sparse cross entropy - the only guide you need!, Medium. Available at: https://medium.com/@shireenchand/choosing-between-cross-entropy-and-sparse-cross-entropy-the-only-guide-you-need-abea92c84662 (Accessed: 18 October 2024). <br><br>
**[5]** Mudadla, S. (2023) Weight decay in deep learning., Medium. Available at: https://medium.com/@sujathamudadla1213/weight-decay-in-deep-learning-8fb8b5dd825c (Accessed: 20 October 2024). <br><br>
**[6]** Wong, W. (2021) What is residual connection?, Medium. Available at: https://towardsdatascience.com/what-is-residual-connection-efb07cab0d55 (Accessed: 18 October 2024). <br><br>
**[7]** Priya, B. (2023) Build better deep learning models with batch and layer normalization, Pinecone. Available at: https://www.pinecone.io/learn/batch-layer-normalization/ (Accessed: 18 October 2024). <br><br>
**[8]** Kalra, R. (2024) Introduction to transformers and attention mechanisms, Medium. Available at: https://medium.com/@kalra.rakshit/introduction-to-transformers-and-attention-mechanisms-c29d252ea2c5 (Accessed: 21 October 2024). <br><br>
**[9]** Modern Recurrent Neural Networks - Beam Search (2020) 10.8. Beam Search - Dive into Deep Learning 1.0.3 documentation. Available at: https://d2l.ai/chapter_recurrent-modern/beam-search.html (Accessed: 21 October 2024). <br><br>
**[10]** Bahdanau, D. et al. (2014) ‘Neural Machine Translation by Jointly Learning to Align and Translate’, International Conference on Learning Representations.<br><br>
**[11]** Roeder, L. (2021) Lutzroeder/netron: Visualizer for Neural Network, Deep Learning and Machine Learning Models, GitHub. Available at: https://github.com/lutzroeder/netron (Accessed: 05 November 2024). <br><br>
**[12]** Lowe, D. M. (2012) ‘Extraction of Chemical Structures and Reactions from the Literature’; University of Cambridge. <br><br>
**[13]** Maggiora, G. et al. (2013) ‘Molecular similarity in Medicinal Chemistry’, Journal of Medicinal Chemistry, 57(8), pp. 3186–3204. <br><br>
