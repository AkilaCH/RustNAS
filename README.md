# RustNAS: Evolutionary Neural Architecture Search

![Rust](https://img.shields.io/badge/Rust-1.84+-orange?logo=rust)
![Burn](https://img.shields.io/badge/Deep%20Learning-Burn-fire)
![Rayon](https://img.shields.io/badge/Concurrency-Rayon-blue)

**RustNAS** is a high-performance engine that uses **Genetic Algorithms (GA)** to evolve optimal Deep Learning architectures automatically. 

Unlike traditional Python-based NAS which often suffers from GIL (Global Interpreter Lock) bottlenecks, this project leverages **Rust's fearless concurrency** to evaluate entire populations of neural networks in parallel.

##  Key Features

* **    Genetic Evolution:** Implements Selection, Crossover, and Mutation to "breed" better neural networks over generations.
* **    Dynamic Graph Generation:** Uses the **Burn** framework to construct valid, trainable neural networks at runtime based on random "DNA" vectors.
* **    Parallel Fitness Evaluation:** Utilizes **Rayon** to train/evaluate multiple candidate models simultaneously across all CPU cores.
* **   Type-Safe Architecture:** Ensures that generated network topologies are mathematically valid at compile-time constraints.

##  Tech Stack

* **Language:** Rust 
* **DL Framework:** [Burn](https://github.com/tracel-ai/burn) (Torch/WGPU backend)
* **Parallelism:** Rayon
* **Serialization:** Serde

##  Project Structure

```text
src/
├── main.rs            #  The Genetic Algorithm Orchestrator
├── genetics/          #  Evolutionary Logic
│   ├── genome.rs      # Defines the "DNA" (layers, learning rate)
│   └── mod.rs
└── model/             #  Deep Learning Implementation
    ├── network.rs     # Dynamic MLP Builder (Burn)
    ├── trainer.rs     # Fitness Evaluation Logic
    └── mod.rs