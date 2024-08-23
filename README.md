# AiZynthFinder Project

## 1 Retrosynthesis with AiZynthFinder - Overview

### 1.1 Basics of Retrosynthesis 

**Retrosynthetic analysis** involves the **deconstruction of a target molecule** into **simpler precursor structures** in order to **probe different synthetic routes** to the target molecule and **compare the different routes** in terms of synthetic viability.

Retrosynthesis involves:
1. **Disconnection**: The **breaking of a chemical bond** to give a **possible starting material**. This can be thought of as the reverse of a synthetic reaction.
<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/292e43d1-9b0e-47ea-ae8e-0c1bda3fe67b", alt="retrosynthesis-disconnection"/>
    <p>
      <b>Fig 1</b> Disconnection in retrosynthesis.
    </p>
  </div>
<br>

2. **Synthons**: These are the **fragments produced by the disconnection**. Usually, a single bond disconnection will give a **negatively charged, nucleophilic synthon**, and a **positively charged, electrophilic synthon**. However, other times the disconnection will give **neutral fragments**. Classical examples of this are **pericyclic reactions** such as **Diels-Alder reactions**.
<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/ee9ed362-1d6d-4de9-a1c8-ad6d8084ea9a", alt="retrosynthesis-synthons-ionic"/>
    <p>
      <b>Fig 2</b> Ionic synthons in retrosynthesis.
    </p>
  </div>
<br>

   However, other times the disconnection will give **neutral fragments**. Classical examples of this are **pericyclic reactions** such as **Diels-Alder reactions**.
<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/0715c148-cbb2-4ba2-b1f3-ece94738276c", alt="retrosynthesis-synthons-neutral"/>
    <p>
      <b>Fig 3</b> Neutral synthons in retrosynthesis.
    </p>
  </div>
<br>

4. **Synthetic Equivalients**: Synthons are not species that exist in reality due to their **reactivity**, and so a synthetic equivalent is a **reagent carrying out the function of a synthon** in the synthesis.
<br>

5. **Functional Group Interconversion (FGI)**: If a **disconnection is not possible** at a given site, **FGI can be used**. FGI is an operation whereby **one functional group is converted into another** so that a **disconnection becomes possible**. A common FGI is the **oxidation of an alcohol to a carbonyl, or amine to nitro group**
<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/321dc552-e38a-422f-885a-104fbe11d56f", alt="functional-group-interconversion"/>
    <p>
      <b>Fig 4</b> Functional group interconversion (FGI) in retrosynthesis.
    </p>
  </div>
<br>
     
6. **Functional Group Addition (FGA)**: Similar to FGI, **FGA** is the **addition of a functional group to another** to make it **suitable for disconnection**.
<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/e56e3767-d38e-40a1-a963-0dad681f4078", alt="functional-group-addition"/>
    <p>
      <b>Fig 5</b> Functional group addition (FGA) in retrosynthesis.
    </p>
  </div>
<br>

### 1.2 Retrosynthetis Search Tree

Typically, the retrosynthetic analysis of a target molecule is an **iterative process** whereby the **subsequent fragments are themselves broken down** until we **reach a stop criterion**. This stop criterion is typically when we reach **precursors that are commerically available/in stock**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/bb4d5580-028b-4610-b95e-8b2f4fc234ba", alt="retrosynthetic-tree"/>
    <p>
      <b>Fig 6</b> Chemical representation of a retrosynthesis search tree. <b><sup>1</sup></b>
    </p>
  </div>
<br>

This iterative process results in a **retrosynthesis tree** where the **bredth is incredibly large**, but the **depth is quite small/shallow**. In comparison to the search trees for games such as chess and Go (**Fig 6**), the **bredth of a retrosynthesis search tree is incredibly large** because you could **theoretically break any bonds** in the target molecule, and the subsequent fragments. This leads to an **explosion in child nodes** from the **first few substrees**.

The **depth** of a retrosynthesis search tree is **small/shallow** on the other hand, as it only takes a **few disconnections before viable precursors are found**. This is ideal since we don't want **linear synthetic reactions** with an **excessive number of steps**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/6c3efa40-bf2c-4124-bd13-4ad9f5a5d6b0", alt="retrosynthetic-search-tree-bredth-depth"/>
    <p>
      <b>Fig 6</b> Retrosynthesis search tree bredth and depth compared to the search trees in chess and Go. <b><sup>2</sup></b>
    </p>
  </div>
<br>

For **effective retrosynthetic analysis**, a retrosynthesis program must:
1. **Define the disconnection rules clearly and efficiently** in order to **reduce the bredth** of the retrosynthesis search tree.
2. **Traverse the retrosynthesis search tree efficiently** using an **effective search algorithm**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/06068a46-d07c-4aa0-b0dd-9e82725bda74", alt="retrosynthetic-tree"/>
    <p>
      <b>Fig 7</b> Retrosynthesis search tree. The search tree starts from the target molecule; each node in the tree is either discarded or picked up for expansion; a leaf node ends when it has no child node to expand or arrives at the starting materials.
    </p>
  </div>
<br>

### 1.3 AiZynthFinder Template-Based Retrosynthesis Model

AiZynthFinder uses a **template-based retrosynthesis model** to **define the disconnection rules**. This approach utilises a **curated database** of **transformation rules** that are **extracted from external reaction databases** that are then **encoded computationally into reaction templates** as **SMIRKS**. 
* **SMIRKS** is a form of **linear notation** used for **molecular reaction representation**. It was developed by **Daylight** and can be thought of as a hybrid between **SMILES** and **SMARTS**

These reaction templates can then used as the **disconnection rules** for decomposing the target molecule into simpler, commercially available precursors.

However, before they are used, AiZynthFinder uses a **simple neural network (Expansion policy)** to **predict the probability for each template given a molecule** (**Fig 8**)

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/8e60a52c-d9b1-474d-ad72-faf6f8592626", alt="template-neural-network"/>
    <p>
      <b>Fig 8</b> Reaction template ranking using Expansion policy neural network. <b><sup>2</sup></b>
    </p>
  </div>
<br>

This **expansion policy neural network template ranking** works as follows:
1. **Encoding of query molecule/target molecule**: The query molecule/target molecule is encoded as an **extended-connectivity fingerprint (ECFP) bit string**, specifically an **ECFP4 bit string**.
2. **


## References

**[1]** Zhao, D., Tu, S. and Xu, L. (2024) ‘Efficient retrosynthetic planning with MCTS Exploration Enhanced A* search’, Communications Chemistry, 7(1). <br><br>
**[2]** Genheden, S. (2022) 'AiZynthFinder', AstraZeneca R&D Presentation. Available at: https://www.youtube.com/watch?v=r9Dsxm-mcgA (Accessed: 22 August 2024). <br><br>
