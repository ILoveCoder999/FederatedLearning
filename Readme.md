Based on the notes you provided, here is the organized content structured in clear, readable Markdown.

---

# Federated Learning & Task Arithmetic: A Paradigm Shift

## 1. The Federated Learning (FL) Paradigm

Federated Learning is a **decentralized learning paradigm** that functions like an *orchestra*. Instead of gathering data in one place, it coordinates dispersed clients to learn together.

### The Iterative Two-Step Process

1. **Local Training:** A large number of clients train models locally using their own **private data**.
2. **Aggregation:** These local models are sent to a server where they are aggregated into a **single shared global model**.

### Core Characteristics

* **Privacy:** FL has an inherent **privacy-preserving nature** because it relies on restricted data sharing (raw data never leaves the client).
* **The Cost Challenge:** The process requires **repeated multi-round communication** between clients and the server, creating a bottleneck.

---

## 2. The Solution: Task Arithmetic & Model Editing

To address the communication costs and integration challenges, we introduce concepts from **Task Arithmetic** and **Model Editing**.

### What is Model Editing?

Model editing refers to the task of modifying, correcting, or improving the functionality of a **pre-trained model** without retraining it from scratch.

### The Process

* **Input:** Multiple fine-tuned models (each meeting specific user requirements).
* **Action:** These models are **merged**.
* **Output:** A single, coherent, and performant model.
* **Constraint:** This must be achieved **without retraining from scratch**.

### Goals & Risks

* **Goal:** Minimize direct access to raw data while combining capabilities.
* **Risk (Interference):** Updates from different sources can conflict. This parameter conflict can lead to **model quality degradation**.

---

## 3. Theoretical Support: Shared Low-Sensitivity Parameters

How do we merge models without causing interference? The answer lies in identifying and utilizing **Shared Low-Sensitivity Parameters**.

### The Theory

Research suggests that within a specific parameter subspace, changing the weights has an **almost linear effect** on the model output.

### The Mechanism

Because of this linearity, if you have several fine-tuned models:

1. You can focus on subsets of model weights (low-sensitivity parameters).
2. You can directly **"add" their weight changes together**.
3. **Result:** You incorporate new information from multiple sources with **minimal interference**, ensuring the model's functionality is not compromised.

---
