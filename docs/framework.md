## **A Unified Framework and Practical Blueprint for Representative Subset Selection in Energy Systems Modeling**

### **Preamble: Why Bother With a Framework?**

So, you need to pick a few representative weeks from a year of time-series data. You dive into the literature and find a dozen different methods: k-means, k-medoids, MILP optimization, autoencoders, chronological clustering, and more. They all sound plausible, but it's a tangled mess. How do you organize this chaos into a coherent, modular software package?  
This isn't about contributing to a grand scientific theory. This is a pragmatic guide for the person building the tool. It's a mental model for deconstructing any selection method into a set of logical, interchangeable parts. It’s a blueprint for designing a package that is flexible, extensible, and easy to reason about.  
Synthesizing decades of academic research—a task that once took months of painstaking work—is now achievable in a fraction of the time. This document is the result: a clear, modular framework designed to help build a useful, modern package for this exact problem.

Inspired by unified frameworks in optimization, we can define the representative subset selection problem not by a single method, but by a set of four fundamental, interchangeable components. Any specific methodology is simply one concrete implementation of this abstract structure.  
The problem is to find an optimal selection x\* that minimizes an objective O, where the selection is drawn from a defined space S, found using a search algorithm A, and used to approximate the full dataset via a representation model R.  
$$x^* = argmin_{x ∈ S} O(R(x, D_{full}))$$ solved by A, each slice represented by feature vector F during the selection process  

### **The Five Pillars of a Modular Design**

At its core, any method for selecting representative periods can be broken down into five distinct jobs. Thinking of them as separate components is the key to a flexible software architecture.

#### **Pillar 1: The Feature Space (F): How We See the Data**

Before we can compare weeks, we need to decide what they "look like" to our algorithm. This is the foundational step of transforming raw time-series data into a structured mathematical object. Your FeatureEngineerProtocol is the perfect home for this.

* **F\_direct (The Raw Vector):** Just use the raw hourly data. Simple, but can be slow and noisy in high dimensions.  
* **F\_engineered (The Smart Summary):** This is where the real work happens.  
  * **Statistical Features:** Instead of 168 hourly points, describe a week by its mean, standard deviation, max/min values, ramp rates, etc..1 This is fast and reduces noise.  
  * **Latent Space Features (PCA/Autoencoders):** Use machine learning to automatically find the most important patterns and compress the week into a dense, low-dimensional vector.2 Autoencoders are particularly powerful here as they can capture complex non-linear relationships.3  
  * **Model-Informed Features (The Pro Move):** This is the most advanced technique. You run a simplified version of your final energy model for *every single week* of the year. You then take the *outputs* of that model (like generator dispatch, storage usage, or electricity prices) and use them as features.5 This makes your feature space "problem-aware"—it learns to group weeks that pose similar  
    *operational challenges*, not just weeks that look statistically similar.5

#### **Pillar 2: The Objective (O): How We Score a "Good" Selection**

Once we can represent our weeks, we need a way to score a potential selection of 8 of them. This is the job of your ObjectiveSet and ScoreComponent protocols.

* **O\_stat (Statistical Fidelity):** This is the most common approach. We score a selection based on how well its statistical properties match the full year. This is a proxy for our real goal. Your components (WassersteinFidelity, CorrelationFidelity, etc.) are perfect examples. They measure how well the selection preserves things like the overall distribution of values, the correlation between variables, and the average daily shape.6  
* **O\_model (Model Outcome Fidelity):** This is the true goal. The best selection is the one that causes our final energy model to produce results (total cost, technology mix) that are closest to the results we'd get from running the full, 8760-hour model.7 This is very hard to measure directly during the selection process, which is why we usually rely on statistical proxies. However, some advanced methods, like identifying extreme events using model infeasibilities (slack variables), are a direct attempt to optimize for this.8

#### **Pillar 3: The Selection Space (S): What We're Actually Picking**

What is the *output* of our selection process? It's not always just a list of weeks.

* **S\_subset (Historical Subset):** The output is a list of k actual weeks that occurred in the original data. This is what your package currently does, and it's the most common and intuitive approach.  
* **S\_synthetic (Generated Archetypes):** The output is a set of k *new*, artificial weeks that are the centroids (averages) of clusters. These "average weeks" may never have actually occurred, but they represent the center of a group of similar weeks.  
* **S\_chronological (Contiguous Segments):** The output is a series of k contiguous periods of *variable length* that preserve the original timeline of the year.10 This is crucial for modeling long-term storage but is a very different kind of output.

#### **Pillar 4: The Representation Model (R): How the Selection Represents the Whole**

Once we have our 8 weeks, how do they represent the other 44? This is the job of your ResponsibilityWeightingProtocol.

* **R\_hard (Hard Assignment):** Each of the 52 weeks is assigned to the single "closest" representative week. The weight of each representative is simply its cluster size. This is what PCAKMedoidsClusterSizeWeights does. It's simple and intuitive.  
* **R\_soft (Blended Representation):** This is a more powerful idea. Every single one of the 52 original weeks is represented as a unique *weighted combination* of all 8 selected weeks.11 For example, "Week 15" might be represented as  
  (0.7 \* RepWeek\_1) \+ (0.2 \* RepWeek\_3) \+ (0.1 \* RepWeek\_8). This provides a much more accurate approximation of the full year but requires a different kind of output from your weighting module and a different formulation in the final energy model.

#### **Pillar 5: The Search Algorithm (A): The Engine That Finds the Solution**

This is the core logic, the engine that puts the other pieces together to find the best selection.

* **A\_combinatorial (Generate-and-Test):** This is the paradigm your package is built on. It uses a CombinationGenerator to create candidate solutions and an OptimizationStrategy (like ExhaustiveOptimization or GeneticOptimization) to test them against the objective function and find the best one.13  
* **A\_constructive (Iterative Building):** This approach builds the solution from the ground up.  
  * **Clustering:** Algorithms like k-means, k-medoids, or hierarchical clustering don't test pre-made combinations; they *construct* the groups based on similarity.15 The selection is an  
    *output* of the algorithm itself.  
  * **Greedy Selection:** Algorithms like Hull Clustering iteratively add the one period that gives the biggest bang-for-the-buck at each step until k periods are selected.  
* **A\_optimization (Mathematical Programming):** This approach formulates the entire selection problem as one big optimization model (e.g., a Mixed-Integer Linear Program) and lets a solver find the single best answer directly.17  
* **A\_hybrid (Multi-stage):** This combines algorithms, for example, using clustering to find 6 "typical" weeks and then a separate model-based method to find 2 "critical stress" weeks to ensure the final system is robust.9

### **A Blueprint for a Modern 'RepSet' Package**

Your current package is an excellent and robust implementation of one specific path through this framework:  
F\_engineered → O\_stat → S\_subset → R\_hard → A\_combinatorial  
It provides a fantastic toolkit for users who want to define a custom objective function from statistical metrics and then search for the best historical subset.  
So, where would this architecture need to stretch to truly become a "unified" package?

1. **The Search Algorithm is the Main Hurdle:** The current design, centered on a CombinationGenerator and an OptimizationStrategy that evaluates those combinations, is a powerful pattern for the A\_combinatorial family. However, it's fundamentally incompatible with other search algorithms.  
   * **Challenge:** You can't implement k-means clustering within this structure. K-means doesn't evaluate externally provided combinations; its internal logic *is* the search.  
   * **Path Forward:** The architecture could be generalized by making the OptimizationStrategy more abstract. An ExhaustiveStrategy would use a CombinationGenerator, but a KMeansStrategy would ignore it and run the k-means algorithm on the feature set directly.  
2. **The Feature Space Needs More Power:** The feature\_engineer.fit\_transform() interface works perfectly for statistical features and basic PCA. It would be strained by the "Model-Informed" approach.  
   * **Challenge:** Generating model-informed features requires a complex, multi-step process that involves configuring and running an external optimization model 52 times. This is more of a preliminary workflow or a "meta-feature-engineer" than a simple fit\_transform call.  
   * **Path Forward:** The package could define a more advanced protocol for feature engineering that accommodates these complex, model-in-the-loop workflows, perhaps by allowing a user to provide a function that executes the necessary steps.  
3. **The Output Needs More Flexibility:** The package is currently designed to output a SelectionResult containing a list of historical periods and a single set of responsibility weights (R\_hard).  
   * **Challenge:** It cannot currently produce synthetic, averaged periods (S\_synthetic) or the full matrix of weights needed for a blended representation (R\_soft).  
   * **Path Forward:** The output objects could be generalized. An OptimizationStrategy could return different types of Result objects—a SubsetResult, a SyntheticResult, or a BlendedResult—each containing the appropriate data structure for the chosen method.

By using this five-pillar framework as a mental model, you have a clear roadmap for evolving your package. You can see which new methods would require fundamental architectural changes (like supporting clustering) and which could be added more easily as new implementations of existing protocols (like a new ScoreComponent). This blueprint turns the chaotic world of academic methods into a structured and actionable plan for building a powerful, flexible, and genuinely useful tool.