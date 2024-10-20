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

### iii. Weight Decay (L2 Regularisation)

**Weight decay** is a fundamental technique used in deep learning to **improve model performance**. It acts as a **regulariser** that **penalises large weights** in the network. This can lead to several benefits: **<sup>5</sup>**
1. **Reducing Overfitting**
   * Large weights can **lead to the model memorising the training data** and **failing to generalise to unseen examples**.
   * Weight decay **penalises large weights**, encouraging the model to **learn smaller weights** that **capture the undelying patterns in the data** rather than **memorising specific details**.
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

### 5.3.4 Callbacks Optimisation

**Callbacks** are powerful tools in **TensorFlow's Keras API** that allow you to customize and control the training process of your models. They are **Python objects** with methods that are **executed during training at given stages of the training procedure**, allowing you to **exectute specific actions** at various stages of training (e.g. at the **end of an epoch**, **after a batch**, or **before training begins**).

Callbacks can be used for a wide range of purposes, such as:
1. **Monitoring training progress**
2. **Adjusting learning rates**
3. **Save model checkpoints**
4. **Generate logs**
5. **Create a TensorBoard**

TensorFlow Keras provides a **set of built-in callbacks**, but you can also **create custom callbacks** by **subclassing the `tf.keras.callbacks.Callback` class**. During training, Keras **calls specific methods of these callback objects** at different points. **Table 1** shows the methods within the  `tf.keras.callbacks.Callback` class.

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
    <b>Table 1</b> Lifecycle methods of the `tf.keras.callbacks.Callback` class.
  </p>
</div>

### i. Early Stopping (EarlyStopping)

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

In this model `tensorflow.keras.callbacks.ReduceLROnPlateau` was used to **monitor validation loss**, and **reduce learning rate by a factor of 0.1** when validation loss **hadn't improved over 3 epochs**.

### iii. Checkpoints (ModelCheckpoint)

**Checkpoint callbacks** allow the **saving of the model regularly during training**, which is especially useful when training deep learning models which can take a **long time to train**. The callback monitors the training and **saves model checkpoints at regular intervals**, based on the metrics.

**ModelCheckpoint** is built-in callback in Keras (`tensorflow.keras.callbacks.ModelCheckpoint`) that can **save the whole model** or **just its weights** at **predefined intervals** (e.g. **after each epoch**), or **based on specific conditions** (e.g. an **improvement in validation loss**). This gives access to **intermediate versions** of the model that can be **restored later**, either to **resume training**, or to **deploy the best performing model**.

In this model, for greater customisability, a custom callback integrated with `from tensorflow.train.CheckpointManager` was used to save a **full-model checkpoint** when there was an **improvement in valdiation loss**, keeping only the **latest 5 checkpoints**, given the number of training runs needed to optimise the model.

### iv. Visualisation in TensorBoard (TensorBoard)

**TensorBoard** is TensorFlow's **comprehensive visualisation toolket**, enabling developers to **gain insights into their machine learning model's training processes**. It provides tools to **visualise metrics** such as loss and accuracy, **analyse model architectures**, and **debug training processes**.

By **leveraging TensorBoard**, a developer can:
1. **Monitor Training Metrics**: Track how metrics evolve over time.
2. **Visualize Model Graphs**: Inspect the computational graph of the model.
3. **Analyze Distributions**: Examine weight and bias distributions, histograms etc.
4. **Project Embeddings**: Visualize high-dimensional data embeddings in lower-dimensional spaces.
5. **Profile Performance**: Identify bottlenecks and optimize performance.

**Tensorboard** is built-in callback in Keras (`tensorflow.keras.callbacks.TensorBoard`) that allows leveraging of TensorBoard and was **utilised extensively** in this project.

### v. Validation Metrics (ValidationMetricsCallback)

For a **more granular analysis** of model performance, specifically for **SMILES string predictions**, a custom **`callbacks.validation_metrics.ValidationMetricsCallback`** class was created, that would **handle a series of computations on the validation data predictions**. It was designed so that **further valdiation metrics could be easily integrated** in the future if needed.

As of the **latest model version (V 21)**, the following validation metric analytics have been added to the callback:
1. **Perplexity**
2. **BleuScore**
3. **SmilesStringMetrics**

### 5.3.5 Metrics Optimisation

### i. Perplexity

**Perplexity** is a widely used evaluation metric in **natural langauge processing (NLP)** and serves as an **indicator of how well the model predicts a sample**. It quantifies the **uncertainty of the model in predicting (i.e. assigning probabilities to) the next token in a sequence**.

In simpler terms, perplexity **evalautes the model's ability to assign high probabilities to the actual sequences it aims to predict**. A **lower perplexity value** indicates **better predictive performance**, meaning the model is **more confident and accurate in its predictions**.

A **custom `metrics.perplexity.Perplexity` metric** was passed to the model's `metrics` argument during **model compilation**.

### ii. BLEU Score

The **Bilingual Evaluation Understudy (BLEU) score** is a metric for assessing the **quality sequences generated by NLP models** by **measuring the overlap of n-gram (contiguous sequences of n words)** between the **candidate (machine-generated) sequence**, and a **reference (human-generated) sequence**.

In this model a **custom `metrics.bleu_score.BLEUScore` metric** was integrated into the **`callbacks.validation_metrics.ValidationMetricsCallback`** class, allowing for BLEU score calculation on the validation data predictions, and was integrated in to the model's **test data evaluation**. The BLEU score implementation chosen was **`nltk.translate.bleu_score.corpus_bleu`** with a **`SmoothingFunction`**.

While BLEU score useful metric for **quanitfying how closely the predicted SMILES match the reference reactants** due to the **ease of computation**, it is important to **consider its limitations** in the context of **SMILES-based retrosynthesis prediction**:
1. **SMILES Syntax Sensitivity** - Due to the **strict syntax rules** that SMILES strings have, even a **minor error can represent a different molecule entirely**, **rendering a SMILES string invalid**.
2. **Chemical Correctness** - BLEU measures **surface-level similarity** and does not assess whether the predicted reactants can be used to **synthesise the target molecule**. Additionally, there is **no emphasis on functional group importance**.
3. **Multiple Valid Representations** - A single molecule can have **mutiple valid SMILES representations**, which can **complicate BLEU's n-gram overlap assessment**. However, by **canonicalising SMILES during data preprocessing**, this can be accounted for.

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
   * Unlike other metrics such as BLEU, Levenshtein distance accouts for the **order of characters**, giving a **more granular view of sequence differences**.
   * This **sequential error quantification** is beneficial for tasks **requiring high precision**, such as **generating chemically valid SMILES strings**.
   * Additionally, Levenshtein distance gives **error type identification**. This helps in understanding whether the errors are primarily due to **insertions**, **deletions**, or **substitutions**, which can **inform model improvements**.

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
      <b>Fig 2</b> Residual connection in a FNN <b><sup>6</sup></b>
    </p>
  </div>
<br>

### iii. Layer Normalisation
The third significant change was to incorporate **layer normalisation** into the encoder.
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

Normalising **across all features of each input removes the dependence on batches/batch statistics**. This makes layer normalisation **well suited for sequence models** such as seq2seq models, RNNs and transformers.

**Fig 5** illustrates the same example as earlier, but with **layer normalisation instead of batch normalisation**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/71187197-02ad-463a-934a-f15abd887344", alt="how-layer-normalisation-works"/>
    <p>
      <b>Fig 5</b> How layer normalisation works <b><sup>7</sup></b>
    </p>
  </div>
<br>

### 5.3.7 Decoder Optimisation

Initial baseline model decoder architecture consisited of **4 unidirectional LSTM layers** with hyperparameters matching those outlined by *Liu et al.* **<sup>1</sup>** (**Table 1**). However, **decoder embedding dimension** and **units** were decreased first to **256**, then to **128** for efficient hardware usage while testing subsequent model versions.

### i. Residual Connections and Layer Normalisation
As with the encoder, the most significant decoder changes were the **addition of residual connections** (**Fig 2**) and the **incorporation of layer normalisation** (**Fig 4**). These changes resulted in an **improvement in both accuracy and loss** for training, validation and testing. 

Regarding residual connection, this improvement in model performance was at odds to what was reported by *Britz et al.* (sections **[4.1.3](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#413-encoder-and-decoder-depth)** and **[4.1.4](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#414-unidirectional-vs-bidirectional-encoder)**). This need for residual connections between layers is likley due to the increased semantic complexity of SMILES strings.

### 5.3.8 Attention Mechanism Optimisation

### i. Bahdanau Attention Mechanism
Initial baseline model used an **additive (Bahdanau) attention mechanism** in line with the mechanism used by *Liu et al.* **<sup>1</sup>**, with the **same dimension** (**Table 8**). However, **attention dimension** and **units** were decreased first to **256**, then to **128** for efficient hardware usage while testing subsequent model versions.

As with all attention mechanisms, the Bahdanau attention mechanism enables the seq2seq model to **dynamically focus on different parts of the input sequence** when **generating each element of the output sequence**. The high-level breakdown of this process is described in section [3.4.3](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/README.md#343-attention-mechanism). 

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

There is precedent for this in the literature; **Transformers heavily rely on residual connections and layer normalization** around their **multi-head attention mechanisms**. Given that transformers have achieved **state-of-the-art results** in various NPL tasks, this provides **strong emperical evidence** for the effectiveness of this approach.

By **integrating residual connections around attention**, in theory, the model is able to:
1. **Preserve Information Flow**:
   * The **original decoder output (`decoder_output`) is directly added to the attention-enhanced output (`context_vector`)**, ensuring that **essential information is not lost or distorted** through the attention layers.
2. **Facilitate Leaning Complex Representations**:
   * Residual connections allow the model to **combine both raw (`decoder_output`)** and **attention-processed (`context_vector`) features**, leading to **richer and more nuanced representations**.
3. **Enchance Gradient Propagation**:
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

### 5.3.9 Inference Optimisation

### i. Beam Search

## 5.4 Model Documentation

### 5.4.1 Model Debugging

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

## 5.5 References
**[1]** Liu, B. et al. (2017) ‘Retrosynthetic reaction prediction using neural sequence-to-sequence models’, ACS Central Science, 3(10), pp. 1103–1113. <br><br>
**[2]** Pandegroup (2017) ‘Pandegroup/reaction_prediction_seq2seq’, GitHub. Available at: https://github.com/pandegroup/reaction_prediction_seq2seq/tree/master (Accessed: 09 October 2024). <br><br>
**[3]** Determinism (2023) NVIDIA Docs. Available at: https://docs.nvidia.com/clara/clara-train-archive/3.1/nvmidl/additional_features/determinism.html (Accessed: 17 October 2024). <br><br>
**[4]** Chand, S. (2023) Choosing between cross entropy and sparse cross entropy - the only guide you need!, Medium. Available at: https://medium.com/@shireenchand/choosing-between-cross-entropy-and-sparse-cross-entropy-the-only-guide-you-need-abea92c84662 (Accessed: 18 October 2024). <br><br>
**[5]** Mudadla, S. (2023) Weight decay in deep learning., Medium. Available at: https://medium.com/@sujathamudadla1213/weight-decay-in-deep-learning-8fb8b5dd825c (Accessed: 20 October 2024). <br><br>
**[6]** Wong, W. (2021) What is residual connection?, Medium. Available at: https://towardsdatascience.com/what-is-residual-connection-efb07cab0d55 (Accessed: 18 October 2024). <br><br>
**[7]** Priya, B. (2023) Build better deep learning models with batch and layer normalization, Pinecone. Available at: https://www.pinecone.io/learn/batch-layer-normalization/ (Accessed: 18 October 2024). <br><br>
**[8]** Kalra, R. (2024) Introduction to transformers and attention mechanisms, Medium. Available at: https://medium.com/@kalra.rakshit/introduction-to-transformers-and-attention-mechanisms-c29d252ea2c5 (Accessed: 21 October 2024). <br><br>
