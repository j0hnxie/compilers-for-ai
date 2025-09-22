# Summarize Ansor paper

**User:** John Xie (johnjxie@gmail.com)  
**Link:** [https://chatgpt.com/c/68d17b79-dff4-832f-b9e8-f8bef6304125](https://chatgpt.com/c/68d17b79-dff4-832f-b9e8-f8bef6304125)  

## Prompt:
Ansor.pdf

PDF

Summarize this paper and explain the main ideas using analogies for someone with only basic ML knowledge to understand

## Response:
Got it — let’s break down the **Ansor** paper into a summary that uses analogies so someone with only basic ML knowledge can understand it.

* * *

🌟 Big Picture
--------------

Deep learning models are made up of many **building blocks** (like convolution, matrix multiplication). To run these blocks fast on CPUs, GPUs, and other hardware, we need **tensor programs** (low-level code that tells the computer exactly how to do the math). Writing these programs by hand for every hardware type is like handcrafting custom race cars for each racetrack — powerful but very expensive and time-consuming.

Ansor is a system that **automatically builds and tests different “race car designs”** (tensor programs), figures out which ones run fastest on each track (hardware), and keeps improving them.

* * *

🚧 The Problem with Existing Methods
------------------------------------

*   **Manual templates (like TVM)** → Imagine writing instructions for how to cook pasta for every single pot, stove, and recipe. Works, but requires a lot of effort and can’t cover all recipes.
*   **Sequential search (like Halide auto-scheduler)** → Imagine building a Lego set one brick at a time, but having to guess halfway whether the castle will look good. Errors pile up, and you miss many better designs.

Both approaches miss a lot of potentially great programs.

* * *

🛠️ Ansor’s Key Ideas
---------------------

Ansor fixes this with three main tricks:

1.  **Hierarchical Search Space (Sketch + Annotation)**
    *   Think of it like house blueprints.
    *   First you decide the high-level **layout** (sketch: “3 bedrooms, 2 floors”).
    *   Then you fill in details (annotation: “this room is blue, that wall has a window”).
    *   This way, Ansor can cover **billions of possible designs** without being stuck in a narrow set of templates.
        Ansor
2.  **Evolutionary Search with Cost Model**
    *   Imagine breeding race cars: take two good designs, mix features (crossover), tweak parts (mutation), then test which runs faster.
    *   Instead of physically testing all cars (too slow), Ansor uses a **cost model** (like a simulator) to guess which cars are likely fast, then only tests the best few on real hardware.
        Ansor
3.  **Task Scheduler**
    *   A neural network has many subgraphs (sections). Not all are equally important.
    *   The task scheduler is like a **coach deciding which player to train harder** — it spends more time improving the subgraphs that affect overall performance the most.
        Ansor

* * *

📊 Results
----------

*   **Speedups:** Up to **3.8× on Intel CPUs, 2.6× on ARM CPUs, and 1.7× on NVIDIA GPUs** compared to the best existing methods.
*   **Coverage:** Finds optimized programs that older systems couldn’t even consider.
*   **Efficiency:** Gets results in _less_ time because it doesn’t waste effort on unimportant parts.
    Ansor

* * *

🧩 Analogy Recap
----------------

*   **Tensor programs** = custom race cars (low-level optimized code).
*   **Templates (old methods)** = recipe books that only cover certain dishes.
*   **Ansor** = an **AI chef & car designer** that can invent new recipes/designs, mix and match, and figure out which ones perform best — without needing human engineers to handcraft everything.

* * *

👉 **In short**: Ansor makes it possible to automatically generate super-optimized code for deep learning on any hardware, beating both manual effort and previous automated methods by being broader (covers more possibilities), smarter (evolves solutions), and strategic (focuses on important parts).

Would you like me to also **go section by section** (background, design, experiments, limitations) in plain language with analogies, so you have a more structured walkthrough of the paper?

## Prompt:
Go more in depth this was too high level. Break it down based on the sections of the paper and go in depth on each section of the paper

## Response:
Thought for 1m 38s

Here’s a section-by-section deep-dive of **Ansor: Generating High-Performance Tensor Programs for Deep Learning**, with plain-language explanations and a few lightweight analogies to keep the ideas accessible.

* * *

1) Introduction — what problem Ansor solves
===========================================

Deep nets are graphs of operators (matmul, conv, etc.). To run fast on different hardware, each operator needs a carefully optimized **tensor program** (the tight loop code). Hand-writing these for every operator × every device is costly and slows down innovation. Prior automated methods either (a) depend on **manual templates** (narrow coverage) or (b) **prune early** while building programs step-by-step (they guess too soon and often discard late-blooming winners). Ansor’s goal: **cover a much larger space of possible programs and still search it efficiently** so it can discover implementations others miss. (See the problem framing in §1; the **search-strategy comparison** diagram on p.3, Fig. 2.)

Ansor

_Analogy:_ Think of designing race cars for many tracks (CPUs/GPUs). Templates are like a few preset car blueprints. Step-by-step search is assembling a car while constantly discarding half-built cars based on rough guesses. Ansor instead **samples full cars**, then **evolves** the best ones.

Ansor

* * *

2) Background — how operators become code & why search is hard
==============================================================

Frameworks like TVM/Halide let you write **math-like definitions** of operators, then search for low-level loop transformations (tiling, reordering, vectorization, parallelization). Two mainstream search styles:

*   **Template-guided (e.g., AutoTVM/FlexTensor):** human-written templates expose a few knobs. Good when templates exist, but **limited coverage** and high engineering cost.
    Ansor
*   **Sequential construction (e.g., Halide auto-scheduler):** build programs by making decisions in a fixed order; to stay tractable, it **prunes** after each step using a cost model trained on completed programs. That model predicts poorly on **incomplete** programs, so promising paths get dropped. The **pairwise accuracy / top-K recall** curves for partial programs (p.3, Fig. 3) show this sharply: predictions are weak until programs are nearly complete.
    Ansor

_Why that matters:_ If you can’t score half-finished cars, pruning mid-build throws away designs that would’ve finished fast.

Ansor

* * *

3) Design overview — three moving parts
=======================================

Ansor’s pipeline (p.4, Fig. 4):

1.  **Program Sampler:** builds a **hierarchical search space** and **samples complete programs** from it.
2.  **Performance Tuner:** **evolves** sampled programs using mutations/crossover guided by a **learned cost model**, then **measures** the best few on real hardware and retrains the model.
3.  **Task Scheduler:** if you have many subgraphs across one/many models, it **allocates tuning time** to the subgraphs that most improve end-to-end latency.
    Ansor

_Analogy:_ Blueprints → sampler; wind-tunnel sim + pit-crew tweaks → tuner; head coach deciding who trains more → scheduler.

Ansor

* * *

4) Program sampling — the hierarchical space
============================================

4.1 Sketch generation (high-level structure)
--------------------------------------------

Ansor separates **high-level structure** (“sketch”) from **low-level details** (“annotations”). Sketches are created by **derivation rules** applied to the operator DAG in topological order; rules decide whether to inline, tile, fuse, add caches, or factor reductions. See **Table 1 (p.5)** for rules like:

*   **Always Inline** simple element-wise ops into consumers.
*   **Multi-level tiling** (CPU “SSRSRS” pattern: Space–Space–Reduce–Space–Reduce–Space) for compute-intensive ops.
*   **Fuse Consumer** when possible (e.g., conv+ReLU).
*   **Add Cache** when no fusible consumer, then later fuse the cache write.
*   **Reduction factorization (rfactor)** to expose parallelism in reductions.
    Ansor

The enumeration collects all terminal sketches (usually **<10 per subgraph**), each describing loop-nest topology, fusion, caching, but **without** picking tile sizes or pragmas yet. **Figure 5 (p.7)** shows examples from math → naive loops → DAG → sketches → sampled programs.

Ansor

_Why it helps:_ A handful of structural patterns summarize billions of fine-grained choices and **cover compositions across multiple ops** (fusion), which single-op templates often miss.

Ansor

4.2 Random annotation (low-level details)
-----------------------------------------

For each sketch, Ansor **randomly** (uniform over valid choices):

*   sets **tile sizes**, chooses **which loops to parallelize/vectorize/unroll**,
*   tweaks **computation locations** for movable nodes (e.g., padding),
*   **rewrites constant weight layouts** to match tiling for cache-friendliness (big impact in inference).
    Ansor

This yields **complete programs**; loops of length 1 are simplified out (so a sketch can produce many concrete programs).

Ansor

4.3 GPU specifics
-----------------

GPU uses a deeper tiling pattern **“SSSRRSRS”** and **binds** the outer space tiles to `blockIdx`, a “virtual thread” layer (to mitigate bank conflicts), then `threadIdx`; it adds rules for **shared-memory caching** and **cross-thread reduction**.

Ansor

* * *

5) Performance fine-tuning — evolutionary search + learned cost model
=====================================================================

5.1 Evolutionary search
-----------------------

Across iterations, Ansor maintains a **population**: freshly sampled programs plus prior good ones. It selects parents **proportional to predicted fitness** (throughput), then applies:

*   **Tile-size mutation:** redistribute factors across tile levels while keeping product equal to the original loop extent.
*   **Parallel mutation:** fuse/split adjacent loops to adjust parallel granularity.
*   **Pragma mutation:** tweak codegen pragmas, e.g., `auto_unroll_max_step`.
*   **Computation-location mutation:** move flexible nodes to different attach points.
*   **Node-based crossover:** merge **rewriting histories** per DAG node from two parents (validated for dependence correctness).
    Ansor

_Key advantage:_ These operations can **edit programs out-of-order** (not bound to the sequential decision order), avoiding the pitfalls of early pruning.

Ansor

5.2 Learned cost model
----------------------

To cheaply score many candidates, Ansor trains a **gradient-boosted decision tree** to predict throughput. It extracts rich features per **innermost non-loop statement** and sums predictions across statements:

*   arithmetic & memory access counts,
*   vectorize/unroll/parallel/thread-binding facts,
*   an **arithmetic-intensity curve**,
*   per-buffer reuse/stride/cache-line features, etc. (Appendix B lists ~164 features).  
    It uses a **weighted squared error** loss that **emphasizes faster programs**. The model is **retrained each iteration** using newly measured data, so it quickly adapts to the target device. (See §5; model quality in §7.5 and **Fig. 11 (p.13)**.)
    Ansor

* * *

6) Task scheduler — allocating tuning time across subgraphs
===========================================================

Tuning whole models produces many **tasks** (one per unique subgraph). Ansor lets you define objectives (Table 2, §6.1), e.g.:

*   **f₁**: minimize total latency (weighted sum across subgraphs).
*   **f₂**: meet per-model latency targets (don’t overspend on models already fast enough).
*   **f₃**: maximize **geometric mean speedup** vs a reference.
*   **f₄**: add **per-task early-stopping** based on progress history.
    Ansor

It then uses a **gradient-descent-style** allocator (§6.2): at each step it estimates which task would give the biggest objective drop, using (i) recent measured improvement for that task, and (ii) an “optimistic” predictor plus **similar-task priors** (uses FLOPs `Cᵢ` and best observed performance `Vₖ` among similar tasks `N(i)`), with an **ε-greedy** sprinkle for exploration. (Derivation in **Appendix A**.)

Ansor

_Intuition:_ Spend time where we’ve either (a) recently seen fast gains or (b) expect big gains by analogy to similar subgraphs.

Ansor

* * *

7) Evaluation — how well does Ansor work?
=========================================

**Hardware:** Intel server CPU, NVIDIA V100 GPU, ARM Cortex-A53 (Raspberry Pi 3b+). **Datatype:** FP32. Search frameworks get equal measurement budgets per case (typically 1,000 per test). (See §7 setup.)

Ansor

7.1 Single operators (10 ops × 4 shapes × 2 batch sizes)
--------------------------------------------------------

Against PyTorch (MKL-DNN), Halide auto-scheduler, FlexTensor, AutoTVM, **Ansor is best or tied in all settings**, with **1.1×–22.5×** gains over search baselines depending on op. Reasons include:  
• discovering **reduction-parallel** schedules (e.g., 2-norm),  
• better tiling/unrolling for **transposed conv** (removes zero-multiplies). See **Fig. 6 (p.11)**.

Ansor

_Ablation (Fig. 7, p.11):_

*   **Limited space** (template-like) underperforms—best programs lie **outside** narrow spaces.
*   **Beam-search (early pruning)** misses winners due to poor partial-program estimates.
*   **No fine-tuning** (random sampling only) lags; fine-tuning is crucial.
    Ansor

7.2 Subgraphs (ConvLayer; TBS from attention)
---------------------------------------------

On CPU and GPU, Ansor beats libraries and search baselines by **1.1×–14.2×** and handles **fusion** that template systems struggle with. (**Fig. 8, p.11**.)

Ansor

7.3 End-to-end networks (ResNet-50, MobileNet-V2, 3D-ResNet-18, DCGAN, BERT)
----------------------------------------------------------------------------

Across CPU/GPU/ARM, Ansor is best or tied, improving over the best alternative by **up to 3.8× (Intel CPU), 2.6× (ARM), 1.7× (GPU)**. Notably, **layout rewrites of constant weights** add ~**40%** for ResNet-50 in their setup. Ansor shines on **small batches and uncommon shapes** where vendor libraries are weaker. (**Fig. 9, p.12**.)

Ansor

**Auto-tuning curves & ablations (Fig. 10, p.12):** With MobileNet-V2 alone and MobileNet-V2+ResNet-50, the **task scheduler** boosts both **convergence speed and final performance**; “limited space” and “no fine-tuning” underperform.

Ansor

* * *

8) Search time & practicality
=============================

Ansor matches AutoTVM’s end-to-end results with **far fewer measurements** and **much less wall-clock time** (thanks to the scheduler, effective fine-tuning, and efficient measurement). **Table 3 (p.13)** shows, e.g., 8.6×–88.6× less wall time across models on Intel CPU. Tuning a whole network typically takes **hours on one machine** and parallelizes easily.

Ansor

* * *

9) Cost model quality
=====================

Measured vs. predicted throughputs (**Fig. 11, p.13**) show good fit (RMSE ≈ 0.079, R² ≈ 0.958, pairwise accuracy ≈ 0.851, recall@30 ≈ 0.624). Points at zero correspond to invalid/time-out programs during measurement—useful negative signals.

Ansor

* * *

10) Related work (high level)
=============================

*   **Scheduling-language systems:** Halide (auto-schedulers), TVM/AutoTVM, FlexTensor. Ansor differs by **sampling full programs** from a **derivation-based**, multi-op space and **evolving** them without early pruning.
    Ansor
*   **Polyhedral compilers:** Tiramisu, TensorComprehensions—powerful but struggle with some DL kernels and implicit cost models.
    Ansor
*   **Graph-level optimizations:** layout search, fusion, constant folding; largely complementary to Ansor’s operator/subgraph-level codegen.
    Ansor

* * *

11) Limitations & future work
=============================

*   **Static shapes only** (no dynamic/symbolic shapes).
*   **Dense operators only** (sparse ops need a redesigned space).
*   Relies on downstream codegens for **platform-specific instructions** (e.g., Tensor Cores, VNNI).
    Ansor

* * *

Core takeaways (one-liners)
---------------------------

*   **Space:** Derivation-based **sketches** + random **annotations** cover far more implementations, including multi-op fusion patterns.
*   **Search:** **Evolutionary** mutations/crossover + a **learned cost model** sidestep the “incomplete-program prediction” trap.
*   **Focus:** A **gradient-guided scheduler** pours effort into the subgraphs that matter for end-to-end speed.
*   **Result:** Consistent SOTA performance with **less tuning time**, across CPUs/GPUs/edge devices. (See Figs. 6–10, Table 3.)
    Ansor

* * *

If you’d like, I can make a **cheat-sheet** that maps each derivation rule in **Table 1** to an everyday analogy and highlight when it triggers (with examples from **Fig. 5**), or sketch a quick **flowchart** of the evolutionary operators and when to use each.

Ansor

## Prompt:
How do TVM and Ansor differ?

## Response:
Great question — TVM and Ansor are closely related, but they approach the problem of generating optimized tensor programs **very differently**. Let’s unpack the key differences.

* * *

🌐 TVM (with AutoTVM / FlexTensor)
==================================

TVM is a deep learning compiler stack. For low-level optimization, it traditionally relies on **templates** and **parameter tuning**:

*   **Templates = Handwritten blueprints**:  
    Experts write “schedules” (loop tilings, unrolling, vectorization strategies) for each operator (conv2d, matmul, etc.). These templates expose **knobs** (tile size, unroll factor) that AutoTVM can tune.
*   **Search = Parameter tuning inside templates**:  
    AutoTVM (and FlexTensor) explores only within the **template-defined search space**. It can find good parameter values but cannot go outside what the human template writer imagined.
*   **Limitations:**
    *   Heavy engineering effort (TVM repo has **15K+ lines** of template code)
        Ansor
        .
    *   Narrow coverage: if an operator/hardware combination has no template, performance can be poor.
    *   Struggles to capture **multi-operator optimizations** (like operator fusion).

* * *

⚡ Ansor
=======

Ansor is a **next-generation auto-scheduler** that removes the need for manual templates. It does three big things differently:

1.  **Hierarchical Search Space (Sketch + Annotation)**
    *   **Sketch** = high-level structure (like loop tiling pattern, fusion, caching).
    *   **Annotation** = low-level details (tile sizes, unrolls, parallel pragmas).
    *   Generated **automatically** from the operator’s math definition — no human templates needed.
        Ansor
    👉 Result: **Much larger and more flexible search space** than TVM templates.
2.  **Sampling Complete Programs**
    *   Unlike Halide/TVM’s step-by-step construction (which evaluates incomplete programs and prunes too early), Ansor **samples only full programs**.
    *   It then fine-tunes them with **evolutionary search** (mutations + crossover) guided by a **learned cost model**.
        Ansor
    👉 Result: Avoids the inaccurate prediction problem for half-built programs.
3.  **Task Scheduler**
    *   When optimizing an entire DNN, Ansor uses a gradient-descent-based scheduler to **focus tuning effort on bottleneck subgraphs** (like 3×3 conv layers in ResNet).
        Ansor
    *   TVM/AutoTVM tunes subgraphs one by one with a fixed budget, even if some are unimportant.
    👉 Result: **Faster convergence and less wasted tuning time.**

* * *

📊 Performance Difference
=========================

*   **Coverage:** Ansor finds optimized implementations that AutoTVM templates can’t even represent (outside their search space).
*   **Efficiency:** Ansor matches or beats AutoTVM’s results with **8–80× less search time** in some cases (Table 3, p.13)
    Ansor
    .
*   **Results:** On CPUs, GPUs, and ARM, Ansor improves DNN execution performance by up to **3.8× vs AutoTVM**
    Ansor
    .

* * *

🧩 Analogy
==========

*   **TVM (AutoTVM):** Like a cooking contest where chefs are only allowed to adjust spice levels and cooking times inside **preset recipes**.
*   **Ansor:** Like a system that can **invent entirely new recipes**, then mutate/crossbreed them until it finds the tastiest — and spends more time perfecting the dishes that matter most for the final meal.

* * *

✅ **In short:**

*   TVM depends on **human-designed templates**; Ansor eliminates templates and explores a **broader search space automatically**.
*   Ansor uses **evolutionary search + cost model** instead of narrow parameter tuning.
*   Ansor adds a **task scheduler** to optimize whole networks more efficiently.

Would you like me to also show you a **visual side-by-side workflow diagram** (TVM vs. Ansor) so you can see how the pipeline differs step by step?

## Prompt:
What is tilling?

## Response:
Good question — **tiling** (sometimes called _blocking_) is one of the most common optimizations in compilers for deep learning and high-performance computing. Let’s break it down simply:

* * *

🧱 What Tiling Is
=================

Imagine you have to multiply two huge matrices. The naïve way is to loop over all rows and columns directly — but that means constantly pulling data from **slow memory (RAM)**.

**Tiling** breaks the problem into **smaller rectangular chunks (tiles)** that can **fit in fast memory (cache)**. You compute on one tile at a time, reusing it while it’s in cache, before moving to the next tile.

* * *

📦 Analogy
==========

Think of moving books from a library to your desk:

*   **Without tiling:** You walk back and forth for _every single book_ you need. Lots of wasted trips.
*   **With tiling:** You bring a **box of books** (a tile) to your desk, do all the work with those books, then return the box and grab another. Much more efficient.

* * *

🔄 How It Looks in Code
=======================

Without tiling (simple triple loop for matrix multiply):

```python
for i in range(N):       # rows
    for j in range(M):   # cols
        for k in range(K):   # inner dimension
            C[i,j] += A[i,k] * B[k,j]
```

With tiling (split into smaller blocks):

```python
for i0 in range(0, N, tile_size_i):     # outer loop over row tiles
    for j0 in range(0, M, tile_size_j): # outer loop over col tiles
        for k0 in range(0, K, tile_size_k):
            for i in range(i0, i0+tile_size_i):
                for j in range(j0, j0+tile_size_j):
                    for k in range(k0, k0+tile_size_k):
                        C[i,j] += A[i,k] * B[k,j]
```

The **outer loops** control which tile you’re working on.  
The **inner loops** operate within a tile.

* * *

🚀 Why Tiling Helps
===================

1.  **Cache efficiency:** Data fetched into cache gets reused many times before eviction.
2.  **Parallelism:** Tiles can often be given to different threads/cores.
3.  **Vectorization:** The compiler can optimize fixed-size inner loops better.

* * *

🔎 In Ansor
===========

Ansor uses **multi-level tiling**:

*   CPUs: “SSRSRS” pattern (Space–Space–Reduce–Space–Reduce–Space).
*   GPUs: “SSSRRSRS” to match block/grid/thread hierarchy.
    Ansor

This structured tiling pattern is what gives Ansor flexibility to optimize across both hardware and operator types.

* * *

✅ **In short:** Tiling = breaking large computations into cache-friendly blocks, so the computer reuses data efficiently and runs much faster.

Do you want me to also explain the difference between **tiling for CPUs vs GPUs**, since Ansor actually handles them differently?

## Prompt:
Ansor talks a lot about how the kernels it generates are faster? Is my understanding correct? If so, how does the generation time of these kernels compare to other methods?

## Response:
Yes — your understanding is correct 👍.

Ansor’s whole purpose is to automatically generate **faster kernels** (the low-level code that implements operators like `conv2d`, `matmul`, etc.) by exploring a much larger design space than older systems (like AutoTVM templates). The paper repeatedly shows that the best kernels Ansor finds are often **outside the search space** of previous methods

Ansor

.

* * *

⚡ 1. Kernel _Execution_ Speed
=============================

*   On **single operators**, Ansor outperforms Halide, FlexTensor, AutoTVM by **1.1×–22.5×** depending on the op
    Ansor
    .
*   On **subgraphs** (e.g., conv+batchnorm+ReLU, or attention softmax blocks), it outperforms by **1.1×–14.2×**
    Ansor
    .
*   On **end-to-end networks** (ResNet, MobileNet, BERT, etc.), Ansor beats AutoTVM and vendor libraries by up to **3.8× (Intel CPU), 2.6× (ARM), 1.7× (GPU)**
    Ansor
    .

So yes: the generated kernels are faster once produced.

* * *

⏱️ 2. Kernel _Generation_ (Search) Time
=======================================

This is where the story gets interesting.  
You might think “larger search space” = “slower tuning,” but Ansor is actually **more efficient in wall-clock search time**:

*   **Why?**
    1.  **Samples complete programs** → avoids wasted time on half-programs.
    2.  **Evolutionary search + cost model** → quickly hones in on promising candidates.
    3.  **Task scheduler** → prioritizes bottleneck subgraphs instead of wasting equal time everywhere.
*   **Measured comparisons (Table 3, p.13):**  
    To match AutoTVM’s performance on Intel CPU benchmarks:
    *   **ResNet-50:** Ansor needs **3.3× fewer measurements** (21,220 → 6,403) and **8.6× less wall time** (39,250s → 4,540s).
    *   **MobileNet-V2:** **16.5× fewer measurements**, **88.6× less wall time**.
    *   Other models (3D-ResNet, DCGAN, BERT) show **2.7×–12.5× fewer measurements** and **3.3×–45× less wall time**
        Ansor
        .
*   **Typical cost:** Tuning a full DNN takes **a few hours on a single machine**, which the authors consider acceptable since it’s a **one-time cost before deployment** (and trivially parallelizable).
    Ansor

* * *

🧩 Analogy
==========

Think of it like:

*   **AutoTVM:** small toolbox, but it tries every screwdriver size carefully on every bolt (lots of time, limited results).
*   **Ansor:** giant toolbox, but it uses a **good simulator (cost model)** + **evolution (mutation/crossover)** + **coach (scheduler)** to only try tools that actually look promising — so it explores more possibilities _and_ finishes sooner.

* * *

✅ **Bottom line:**

*   Yes, Ansor’s generated kernels are consistently faster at runtime.
*   Despite searching a **much bigger space**, its generation time is usually **1 order of magnitude faster** (8–80× wall-clock savings) than AutoTVM and similar systems.
    Ansor

* * *

Would you like me to also make a **timeline comparison chart** (AutoTVM vs Ansor) showing _generation time vs final kernel speed_ so you can see the tradeoff clearly?

## Prompt:
What are schedule space templates?

## Response:
Great question — the term **“schedule space templates”** shows up a lot in the TVM/AutoTVM vs. Ansor discussion. Let’s unpack it.

* * *

🧾 First: What’s a “schedule”?
==============================

In deep learning compilers like **TVM** or **Halide**, you start with a **mathematical definition** of an operator (say, matrix multiply).  
That definition says _what_ to compute, but not _how_.

A **schedule** is the _recipe_ for _how_ to implement it efficiently:

*   How to **reorder loops**.
*   Whether to **tile loops** (break into blocks).
*   Which loops to **parallelize**.
*   Which to **vectorize**.
*   Where to **store data** (registers, shared memory, cache).

* * *

📦 What are “schedule templates”?
=================================

A **schedule template** is a **handwritten blueprint** that defines a _restricted search space_ of possible schedules for one operator.

*   Example: For `conv2d`, a template might say:
    > “Tile the spatial loops into {8, 16, 32}, try unroll factors {2, 4, 8}, and maybe vectorize across the channel dimension.”
*   AutoTVM then **searches within that template** by tuning the exposed parameters (tile size, unroll factor, etc.).
*   Templates are **specific**: each operator (conv2d, matmul, depthwise conv, etc.) often requires its own custom template. In TVM, there are **15K+ lines of template code** just to cover common operators
    Ansor
    .

* * *

⚠️ The Problem with Templates
=============================

1.  **Labor-intensive**: Writing good templates takes expert knowledge of both hardware and operator behavior.
2.  **Narrow coverage**: Templates only explore what the human imagined. If the best implementation uses an unusual tiling pattern, caching trick, or fusion, it’s **outside the template space** and will never be found.
3.  **Hard to scale**: Every new operator or hardware backend needs new templates.

_Analogy:_ A template is like a **menu** at a restaurant — you can only order dishes listed there, maybe with slight tweaks (like spice level). But if you want sushi at a pizza place, you’re out of luck.

* * *

🌟 How Ansor Differs
====================

Ansor **throws away manual templates**. Instead of predefined search spaces:

*   It automatically constructs a **hierarchical search space** from the operator’s math (sketch + annotation).
*   It explores _all possible schedules_ that are derivable, not just the ones a human wrote down.
*   This lets Ansor discover **new kernels that templates miss**, including **multi-operator fusion patterns**.
    Ansor

* * *

✅ **In short:**

*   **Schedule space templates** = human-written blueprints that define _where AutoTVM is allowed to search_.
*   They make tuning easier but restrict coverage and require lots of expert effort.
*   **Ansor** replaces templates with an **automatic, general-purpose schedule space** so it can search more broadly and find better-performing kernels.

* * *

Do you want me to give you a **concrete code snippet comparison** (e.g., what a TVM schedule template for `conv2d` looks like vs. what Ansor would auto-generate)?

## Prompt:
How does Ansor create the hierarchical search space?

## Response:
Great — this is the **core innovation of Ansor**. Let’s go step by step on how Ansor builds its **hierarchical search space**.

* * *

🔑 The Problem
==============

We want to generate **many possible implementations** (schedules) of a tensor operator (e.g., conv2d, matmul).

*   If we try to directly enumerate _all possible low-level loop transformations_ (tile sizes, vectorization, unrolling, parallelization, etc.), the space is astronomically huge.
*   If we only use **templates** (TVM/AutoTVM), the space is tiny and misses many good schedules.

Ansor’s solution: **split the search into two levels** → **sketch** + **annotation**

Ansor

.

* * *

1️⃣ Step One: Sketch Generation (High-Level Structure)
======================================================

Think of this as drawing the **blueprint** of the program without filling in all the details.

*   Ansor starts from the **DAG** of the operator/subgraph (nodes = computations, edges = dependencies).
*   It applies a small set of **derivation rules** (Table 1, p.5) recursively to build different **skeletons (sketches)**.
*   Each sketch captures the **big structural choices**:
    *   Inline or keep an operator separate?
    *   Tile loops, and how many levels?
    *   Fuse consumer ops (like conv + ReLU)?
    *   Insert caching nodes (so data is reused from faster memory)?
    *   Factor reductions to expose more parallelism?

👉 Example:  
For matrix multiply `C[i,j] = sum_k A[i,k] * B[k,j]`, a sketch might decide to:

*   Use a 6-level tiling loop nest (“SSRSRS” for CPUs).
*   Fuse in a ReLU consumer.
*   Add a write-cache before main memory.

Each operator typically yields **<10 sketches** (manageable size).

Ansor

_Analogy:_ Sketches are like **floor plans of a house**: you decide “3 bedrooms, 2 floors, a garage,” but you haven’t chosen wall colors or furniture yet.

* * *

2️⃣ Step Two: Random Annotation (Low-Level Details)
===================================================

Now we take a sketch and **fill in the fine-grained decisions**:

*   **Tile sizes:** How big is each block in the tiling structure?
*   **Loop pragmas:** Which loops to parallelize, vectorize, unroll, and with what factors?
*   **Computation location:** Where to compute some nodes (e.g., fuse padding inside/outside)?
*   **Memory layouts:** Rewrite constant weights (like conv filters) to match tiling for cache-friendliness.

Ansor **samples these details randomly** (uniformly over valid values), so every schedule has a chance to appear. Later, the **evolutionary tuner** improves them.

Ansor

_Analogy:_ Annotation is like **furnishing the house**: once the floor plan is fixed, you decide tile sizes = room sizes, parallelization = who uses which room, and vectorization/unrolling = furniture arrangements.

* * *

3️⃣ Why This Hierarchy Helps
============================

*   **Compact high-level space:** Only a few sketches per op → manageable.
*   **Explosive low-level diversity:** Each sketch can expand into billions of annotated schedules.
*   **Coverage:** Can represent patterns _outside_ templates, including multi-operator fusions.
*   **Sampling complete programs:** Unlike Halide’s sequential search (which evaluates incomplete programs), Ansor **always samples full programs**, so the cost model predictions are reliable.
    Ansor

* * *

4️⃣ CPU vs GPU Differences
==========================

*   **CPU tiling:** “SSRSRS” (Space–Space–Reduce–Space–Reduce–Space).
*   **GPU tiling:** “SSSRRSRS,” matching the block/thread hierarchy:
    *   Outer space tiles → `blockIdx`
    *   Next → “virtual threads” (to reduce bank conflicts)
    *   Next → `threadIdx`
*   Extra rules: GPU caching (shared memory) and cross-thread reductions.
    Ansor

* * *

✅ In Short
==========

Ansor creates its hierarchical search space by:

1.  **Generating sketches** (high-level loop/fusion/cache structures) via recursive rules.
2.  **Randomly annotating** them with low-level details (tile sizes, vectorization, parallelization, unrolling).

This **sketch + annotation** hierarchy balances **coverage** (huge space) and **efficiency** (structured exploration). It lets Ansor find fast kernels that template systems can’t even represent.

* * *

Would you like me to draw a **diagram-style walkthrough** (math → DAG → sketch → annotated program) to visualize this pipeline? It’s basically what Figure 5 in the paper shows, but I can make it more intuitive.

## Prompt:
How does ansor determine what sketches to use and make

## Response:
Great follow-up — this gets right into the **“sketch generation”** step of Ansor’s hierarchical search space. Let’s unpack how Ansor _decides what sketches to generate_.

* * *

1\. Start From the Computation DAG
==================================

*   Each operator (e.g., conv2d, matmul) or subgraph is represented as a **DAG**:
    *   **Nodes** = computations (element-wise add, matmul, conv).
    *   **Edges** = data dependencies.
*   Ansor traverses the DAG in **topological order** (from outputs backward to inputs).
    Ansor

* * *

2\. Use **Derivation Rules**
============================

Ansor doesn’t “guess” sketches randomly — it **systematically applies a fixed set of rules** (Table 1 in the paper) to each node in the DAG.

Some important rules (summarized):

*   **Inline simple ops:** If the node is element-wise (like ReLU, add), inline it into its consumer.
*   **Multi-level tiling:** If the node has heavy data reuse (like matmul, conv2d), create a multi-level tiling structure.
*   **Fuse consumers:** If the node has a fusible consumer (e.g., conv + ReLU), fuse them into one sketch.
*   **Add cache stage:** If the node has no fusible consumer (e.g., final output), insert a cache write node for efficiency.
*   **Reduction factorization:** If a reduction loop could expose more parallelism (e.g., matrix 2-norm), split it using rfactor.
    Ansor

Each rule has a **condition** (e.g., “HasDataReuse(node) = True”) and an **action** (e.g., “apply MultiLevelTiling(node)”).

👉 The process is recursive: for every state `(sketch_so_far, current_node)`, Ansor applies all valid rules, creating new states. This continues until no nodes remain.

* * *

3\. Enumerate All Possible Sketches
===================================

*   The derivation rules are applied **systematically** (not randomly).
*   Each path through the derivation tree produces one **sketch**.
*   The process ends when all nodes have been processed → the set of sketches.
*   In practice, for most subgraphs, this yields **fewer than 10 sketches** total.
    Ansor

_Analogy:_ Think of it like LEGO instructions. At each step, you choose whether to:

*   snap two pieces together (fusion),
*   keep them separate (no fusion),
*   add a supporting plate (cache),
*   build multiple floors (multi-level tiling).  
    By applying these rules, you enumerate all reasonable LEGO castle blueprints.

* * *

4\. Example
===========

From **Figure 5 (p.7)** in the paper:

*   Input: a simple DAG with `A × B → C → ReLU(D)`.
*   Derivation path could be:
    *   Rule 4: fuse ReLU into matmul,
    *   Rule 3: multi-level tile the matmul,
    *   Rule 1: skip nodes without special structure.
*   This yields one sketch.
*   Applying a different sequence (e.g., adding a cache stage instead of fusing) yields another sketch.

* * *

5\. Why This Matters
====================

*   Instead of having humans write templates (like TVM), Ansor can **derive sketches automatically** for _any_ operator from its math definition.
*   Sketches summarize **high-level strategies** (fusion, tiling, caching) without committing to details yet.
*   The small set of sketches per subgraph keeps the exploration tractable, while later annotations expand each sketch into billions of full programs.
    Ansor

* * *

✅ **In short:**  
Ansor determines sketches by **recursively applying derivation rules** to the computation DAG. Each rule encodes a high-level optimization decision (inline, tile, fuse, cache, factor reductions). By systematically applying all valid rules, Ansor enumerates all possible **sketches** — typically <10 per operator — that capture the big structural choices before fine-grained tuning.

* * *

Would you like me to also **map each derivation rule (Table 1)** to a **real-life analogy** (e.g., “inline” = cooking directly in the same pot vs. “cache” = using a serving tray), so you can remember what each does intuitively?

## Prompt:
Is the second step of random annotation / low level details at the same level of what TVM does with templates?

## Response:
Yes — that’s a sharp observation 👍

Let’s compare directly:

* * *

🧾 TVM (AutoTVM / templates)
============================

*   TVM uses **handwritten schedule templates** for each operator.
*   Each template encodes a **high-level structure** (like a loop nest with tiling, or a memory layout strategy) and exposes some **knobs** (tile sizes, unroll factors, vectorization flags).
*   AutoTVM then tunes by searching over these knobs.
*   In other words: **the template already fixes the big structure**; you only tweak the low-level details inside it.
    Ansor

* * *

⚡ Ansor (sketch + annotation)
=============================

*   Ansor **separates** the problem:
    1.  **Sketch generation** = automatically derive the _high-level structure_ (loop tiling pattern, fusion, cache, reduction factorization).
    2.  **Annotation (random sampling)** = fill in the _low-level details_ (tile sizes, parallelization, unrolling, vectorization, computation locations, layout rewrites).
*   That **second step (annotation)** is indeed conceptually **at the same level as TVM templates**:
    *   TVM templates hard-code a single sketch + knobs.
    *   Ansor annotations fill knobs for _many possible sketches_.

👉 So Ansor’s annotation ≈ “TVM template tuning,” except:

*   It comes **after** an automatic sketch generator (no manual templates).
*   It’s **much broader**: each sketch can explode into billions of annotated schedules.
*   It doesn’t just tweak knobs from one template; it spans across many structural alternatives.
    Ansor

* * *

📊 Analogy
==========

*   **TVM templates:** Imagine a chef hands you a fixed recipe and says, “You can only adjust the salt, pepper, and cooking time.”
*   **Ansor:** First, a **blueprint generator** invents _several different recipes_ for the same dish (stir-fry, soup, casserole). Then the **annotation step** lets you tweak the salt, pepper, cooking time _for each recipe_.

So yes — **the annotation step is where Ansor does what TVM templates do**. But Ansor goes further because it generates the structural blueprints automatically, instead of requiring experts to handcraft them.

* * *

✅ **In short:** The **annotation step in Ansor ≈ template knob tuning in TVM**. The big difference is that Ansor’s sketches are generated automatically, so the annotations apply to a much richer and more flexible search space than TVM’s human-written templates.

* * *

Would you like me to show you a **side-by-side pseudo-code example**:

*   a TVM schedule template for `conv2d` (with tunable knobs),
*   versus what an Ansor sketch + annotation would look like?

## Prompt:
How are the rules in Ansor generated? Also does Ansor create more or less sketches than TVM has templates?

## Response:
Great pair of questions — they go right to the heart of what makes Ansor different from TVM. Let’s break them down.

* * *

1\. 🔧 How are the rules in Ansor generated?
============================================

The **rules** in Ansor (Table 1 in the paper, p.5) are not learned or discovered automatically. Instead:

*   They are a **small, fixed library of derivation rules** designed by the Ansor authors, based on well-known compiler and HPC loop optimizations.
*   Each rule has:
    *   a **condition** (e.g., “is this node simple element-wise?” or “does this op have data reuse?”), and
    *   an **action** (e.g., inline it, tile it, fuse it, add a cache stage, factorize the reduction).

Examples:

*   **Always Inline:** If an operator is element-wise, always inline it into its consumer.
*   **Multi-level tiling:** If the operator has heavy data reuse (matmul, conv), create a standard tiling structure (“SSRSRS” for CPU, “SSSRRSRS” for GPU).
*   **Fuse Consumer:** If a node has a fusible consumer (like conv+ReLU), fuse them.
*   **Add Cache:** If the final node writes directly to memory, add a cache write stage first.
*   **Reduction Factorization:** If the reduction dimension offers parallelism, split it with `rfactor`.

👉 These rules are **manually written**, but unlike TVM templates, they are **generic and reusable** across many operators.  
👉 Users can **extend** them if new hardware features or algorithms demand special tiling (e.g., Winograd convolution, Tensor Cores)

Ansor

.

_Analogy:_ Think of them like **laws of the road** — not custom driving instructions for each car, but general rules (stop at red lights, yield to pedestrians) that apply to all driving situations.

* * *

2\. 📊 Does Ansor create more or fewer sketches than TVM has templates?
=======================================================================

*   **TVM templates:**
    *   Each operator type (conv2d, matmul, depthwise conv, etc.) typically requires **one or more templates**.
    *   The TVM repo already has **15K+ lines of template code** covering dozens of operators.
    *   Templates are **manual, operator-specific**, and new ops or hardware need new templates
        Ansor
        .
*   **Ansor sketches:**
    *   For each operator/subgraph, Ansor generates **sketches automatically** by applying its rules to the operator DAG.
    *   Typically, the number of sketches per subgraph is **small (<10)**
        Ansor
        .
    *   But each sketch can be expanded into **billions of full programs** after annotation.

👉 So in absolute numbers:

*   **TVM** has **more templates overall** (dozens/hundreds across ops and backends).
*   **Ansor** produces **fewer sketches per operator**, but they are generated on the fly and cover a much **larger and more diverse search space** than all TVM templates combined.

_Analogy:_

*   TVM templates = a **large recipe book**, each recipe written by a chef for a specific dish.
*   Ansor sketches = a **small set of cooking rules** (chop, stir-fry, bake, add spice) that can generate endless new recipes. Fewer starting points, but way more variety.

* * *

✅ In Short
==========

*   **Rules in Ansor**: Manually written once, but generic; they’re applied automatically to any operator’s DAG.
*   **Sketch count vs. templates**: Ansor generates _fewer_ sketches (<10 per op) than TVM has templates, but each sketch covers far more possible implementations.
*   **Coverage**: Ansor’s search space is _much larger_ than TVM’s, despite having fewer starting sketches.

* * *

Would you like me to also show you a **side-by-side example** of:

*   what a **TVM conv2d template** looks like, versus
*   what **Ansor’s derivation rules** would do to the same conv2d DAG (step by step into sketches)?

## Prompt:
Can you explain the performance tuner more?

## Response:
Perfect — the **performance tuner** is the “engine room” of Ansor. It takes the huge set of possible full programs (generated by **sketch + annotation**) and **hunts for the fastest ones**. Let’s go step by step.

* * *

🌟 What the Performance Tuner Does
==================================

The sampled programs (from sketches + random annotations) have **good coverage** of the space but their quality is random. The tuner is in charge of:

1.  **Improving them iteratively** (making them faster),
2.  **Guided by a learned cost model**,
3.  **Validating on real hardware** to ensure correctness and speed.

👉 Think of it as a “genetic algorithm coach” that takes random athletes (programs), trains them, breeds the best features, and tests them in races (measurements).

* * *

🔄 The Evolutionary Search
==========================

Ansor uses **evolutionary search** — inspired by biological evolution:

### 1\. Population Initialization

*   Start with:
    *   Freshly sampled random programs,
    *   Plus the best ones found in earlier iterations.

### 2\. Fitness Scoring

*   Use the **learned cost model** (a fast predictor) to estimate which programs are “fit” (likely to run fast).
*   Fitness = predicted throughput.

### 3\. Selection

*   Programs with higher predicted performance are **more likely to be chosen** as parents.

### 4\. Evolution Operators

Applied to parents to create children:

*   **Tile size mutation** → split/fuse tile sizes differently.
*   **Parallel mutation** → change how loops are parallelized.
*   **Pragma mutation** → tweak unroll/vectorization hints.
*   **Computation-location mutation** → move an op to compute earlier or later.
*   **Node-based crossover** → merge “rewriting histories” of two parents at the DAG-node level, producing a new valid program.
    Ansor

### 5\. Iteration

*   Generate the next “generation” of programs.
*   Keep the top ones.
*   Repeat for several rounds.

* * *

🧠 The Learned Cost Model
=========================

Why not just measure every program on real hardware? → Too slow.

*   **Cost model = fast predictor** of program performance.
*   Implemented as a **gradient-boosted decision tree (GBDT)** trained on measured data.
*   Features include:
    *   Arithmetic counts (adds, muls, divs),
    *   Memory access patterns,
    *   Loop structures (vectorization, unrolling, parallelization),
    *   Arithmetic intensity curve,
    *   Buffer access details (stride, reuse, cache lines).  
        (Full list in Appendix B — about 164 features!)
        Ansor
*   Loss function weights **fast programs more heavily**, so the model learns to distinguish “winners.”
*   Model is retrained continuously: each new batch of real measurements improves accuracy.

👉 Benefit: Ansor can **score thousands of candidates in seconds**, then only measure a few hundred promising ones on hardware.

* * *

⏱️ Hardware Measurement
=======================

*   After cost-model ranking, Ansor **compiles and measures** a small batch of top candidates on the real CPU/GPU.
*   Actual runtimes are fed back to:
    1.  Retrain the cost model,
    2.  Update the evolutionary population with ground-truth winners.

* * *

📊 Why This Works Better Than TVM / Halide
==========================================

*   **TVM templates:** only tweak pre-defined knobs; limited variety.
*   **Halide auto-scheduler:** prunes incomplete programs early, leading to bad predictions and missed winners.
*   **Ansor:** always evolves _complete programs_ and uses its cost model on valid schedules → much more accurate guidance.

Result: Ansor finds kernels **faster** _and_ makes them **faster at runtime**.

* * *

🔎 Analogy
==========

Imagine breeding racing dogs:

*   **Random sampling** = adopting a bunch of puppies without training.
*   **Evolutionary tuner** = pick the fastest ones, breed them, introduce mutations, repeat.
*   **Cost model** = a vet who can predict which pups are likely to be fast just by looking at their bones/muscles (so you don’t have to race every pup).
*   **Hardware measurement** = the actual race, to see who wins for real.

Over generations, you get a kennel of champions.

* * *

✅ **In short:**  
The performance tuner in Ansor combines **evolutionary search** (mutate + crossover), a **learned cost model** (fast prediction), and **real hardware feedback** (ground truth). This loop ensures that from billions of possibilities, Ansor hones in on kernels that are both **valid** and **very fast**, much more efficiently than older systems.

* * *

Do you want me to also **draw out one full “generation cycle”** (population → cost model scoring → mutations/crossover → hardware measure → retrain → repeat), almost like a flowchart, so you can see the loop clearly?

## Prompt:
How is the learned cost model created?

## Response:
Great question — the **learned cost model** is one of the most important parts of Ansor’s tuner, because it lets Ansor explore billions of possible programs without having to actually run them all on hardware. Let’s go step by step.

* * *

1\. 🎯 Goal of the Cost Model
=============================

*   Purpose: **quickly predict the runtime (throughput)** of a tensor program without actually executing it.
*   Why: measuring every candidate on real hardware is way too slow. Ansor wants to filter down from thousands → a few hundred promising schedules to test for real.
    Ansor

* * *

2\. 🧱 What Data the Model Uses
===============================

*   Ansor works with **tensor programs**, which are mostly loop nests with compute statements.
*   The model predicts the score (throughput) of a program by looking at its **innermost statements in loop nests** (e.g., the math happening inside the innermost loop).
*   For each innermost statement, Ansor extracts a **feature vector** that describes both the arithmetic and memory context.

### Features (Appendix B in the paper, ~164 in total):

*   **Arithmetic counts:** # of additions, multiplications, divisions, comparisons, etc.
*   **Loop features:** how many loops, their extents, which are vectorized/unrolled/parallelized, arithmetic intensity curve.
*   **Memory access patterns:** stride, reuse distance, cache line usage, number of bytes touched, read/write type.
*   **GPU-specific features:** blockIdx, threadIdx binding lengths.
*   **Buffer features:** reuse type (serial, loop multiple read, none), unique bytes, reuse counter, etc.
    Ansor

👉 Together, these features capture both _compute workload_ and _memory behavior_, which are the two main drivers of performance.

* * *

3\. 🏗️ Model Architecture
==========================

*   Ansor uses a **Gradient Boosted Decision Tree (GBDT)** model (specifically XGBoost).
*   Why GBDT instead of deep neural nets?
    *   Works well with tabular, structured features.
    *   Easy to train quickly on small datasets.
    *   Good interpretability and robustness.

* * *

4\. 🏋️ Training the Cost Model
===============================

*   During search, Ansor is constantly measuring programs on hardware anyway.
*   Each measured program gives:
    *   Input: program features.
    *   Label: actual measured throughput.
*   Ansor trains the GBDT on this data.

### Training Details:

*   **Objective:** Weighted squared error loss.
    *   Programs with higher throughput are given more weight.
    *   Why? Because we mostly care about distinguishing the _fastest programs_ in the search space.
*   **Normalization:** For each DAG, program throughputs are normalized into \[0,1\] so the model is comparable across tasks.
*   **Retraining:** A new model is trained from scratch every iteration (not incrementally). Training is fast because the dataset is small (<30k measured programs typically).
    Ansor

* * *

5\. 📊 Model Usage During Search
================================

*   For each candidate program in evolutionary search:
    *   Extract features,
    *   Use cost model to predict throughput,
    *   Rank candidates by score.
*   Only the **top predicted candidates** are actually compiled and measured.
*   Their real runtimes then update the training data → retrain model → improve predictions in the next round.

This creates a **feedback loop**:  
cost model guides search → measurements refine cost model → cost model gets more accurate → search gets more efficient.

* * *

6\. ⚡ How Well It Works
=======================

*   In evaluation, the cost model had **R² = 0.958**, pairwise accuracy ≈ 0.85, recall@30 ≈ 0.62 for top-30 programs
    Ansor
    .
*   That means: it’s good enough to consistently bubble up the fastest candidates for measurement.

* * *

🔎 Analogy
==========

Imagine you’re breeding racehorses (tensor programs).

*   The **cost model** is like a vet who can examine bone structure, muscles, and stride (features) and **predict which horses will be fastest**.
*   You don’t send every horse to the racetrack (hardware), just the ones the vet says are promising.
*   Every time you do race a horse, you give that info back to the vet, so their predictions get sharper over time.

* * *

✅ **In short:**  
Ansor’s learned cost model is a **GBDT predictor** trained on program features (arithmetic + memory + loop structure). It is continuously retrained with real hardware measurements, and it emphasizes fast programs in its loss. This lets Ansor explore a huge search space while only paying the measurement cost for a small subset of promising candidates.

* * *

Would you like me to also **show a concrete feature vector example** (e.g., for a tiled matrix multiply loop) so you can see what numbers the cost model actually consumes?

## Prompt:
How many iterations does the performance tuner run?

## Response:
Good question — the paper doesn’t hard-code a single fixed number of iterations for the performance tuner, because Ansor is designed to run **iteratively until the tuning budget is used up**. Let me explain in detail.

* * *

🔄 How the Iterative Tuning Works
=================================

1.  **Initialize**
    *   Start with a batch of random programs (from sketch + annotation) and some promising ones from earlier rounds.
2.  **Evolutionary Search Cycle**
    *   Use the **learned cost model** to predict fitness of candidates.
    *   Select parents proportional to predicted fitness.
    *   Apply mutations/crossover to create new children.
    *   Pick a batch of high-scoring candidates to actually measure on hardware.
    *   Update dataset with real runtimes.
    *   Retrain the cost model.
3.  **Repeat**
    *   Each such loop is **one iteration**.
    *   Continue until you exhaust the **search budget** (either a time budget or a measurement budget).
        Ansor

* * *

📏 How Many Iterations in Practice?
===================================

*   Ansor’s experiments typically use **up to 1,000 measurement trials per test case** for operator/subgraph benchmarks
    Ansor
    .
*   For full networks, they allocate up to `1000 × n` measurements, where `n` = number of subgraphs in the model
    Ansor
    .
*   Since each iteration measures a _batch_ of candidates (tens or hundreds), the total number of iterations usually ends up in the **tens to low hundreds**.
*   The key point: it’s **budget-driven**, not fixed — Ansor keeps iterating until the allotted measurements/time are used.

* * *

⏱️ Practical Runtime
====================

*   Typically, **a few hours on a single machine** is enough to fully tune an end-to-end DNN (e.g., ResNet-50, MobileNet)
    Ansor
    .
*   Ansor is much faster than AutoTVM: it can reach equivalent performance with **8×–80× less wall-clock search time** (Table 3, p.13)
    Ansor
    .

* * *

✅ In Short
==========

The **number of iterations** the performance tuner runs is **not fixed** — it’s determined by the measurement/time budget. Each iteration = sample, evolve, predict with cost model, measure, retrain.

*   For single ops: ~1000 measurements → tens of iterations.
*   For networks: `1000 × n` measurements (n = subgraphs) → hundreds of iterations spread across subgraphs.
*   Practically: a few hours per DNN on one machine.

* * *

Would you like me to make a **numerical worked example** (e.g., “tuning ResNet-50 with 29 subgraphs → 29,000 measurement budget → ~100 iterations with batches of 300 programs each”), so you can see how the numbers play out?

## Prompt:
Finally can you explain the task scheduler more?

## Response:
Yes! The **task scheduler** is Ansor’s “project manager.” It decides **which parts of a DNN get tuning effort** at each step. Without it, the system might waste tons of time tuning pieces that don’t actually affect end-to-end performance. Let’s break it down.

* * *

🧩 Why a Task Scheduler?
========================

*   A deep network = many **subgraphs** (e.g., conv+batchnorm+ReLU blocks).
*   Each subgraph = one **task** (find the best kernel implementation for it).
*   If you gave **equal time** to every subgraph (what AutoTVM does), you’d waste effort:
    *   Some subgraphs are **rarely executed** → little impact.
    *   Some already run near-optimally → little room for improvement.

👉 The scheduler solves this by **prioritizing bottlenecks** — subgraphs where tuning can most reduce end-to-end latency.

Ansor

* * *

⚙️ How It Works
===============

### 1\. Formulating the Objective

*   For each subgraph `i`:
    *   Define `g_i(t)` = best latency achieved so far, given `t` units of tuning effort.
*   End-to-end cost = function of all subgraph latencies.
*   Several objectives are possible (Table 2 in the paper):
    *   **f₁:** Minimize total latency = Σ (weight × gᵢ).
    *   **f₂:** Respect latency targets (stop tuning once a model meets requirements).
    *   **f₃:** Maximize geometric mean of speedups vs. reference baselines.
    *   **f₄:** Early stopping for tasks where progress has plateaued.
        Ansor

* * *

### 2\. Gradient-Based Scheduling

*   Ansor approximates the **gradient** ∂f/∂tᵢ for each task:
    *   If tuning task `i` further will likely reduce end-to-end latency a lot, its gradient magnitude is high.
    *   If not, it’s low.
*   It uses:
    *   **Recent history** (improvements per unit time).
    *   **Optimistic guess** (what if latency could drop to 0).
    *   **Similarity with other tasks** (reuse knowledge from subgraphs with similar FLOP patterns).
*   At each iteration, Ansor picks the task with the **largest estimated gradient**.
*   Occasionally (ε-greedy), it explores a random task to avoid tunnel vision.
    Ansor

* * *

### 3\. Workflow

*   Start: round-robin “warm-up” to gather initial data.
*   Then:
    1.  Scheduler picks task i.
    2.  Generate/evolve candidate kernels for that subgraph.
    3.  Measure them on hardware.
    4.  Update gᵢ(t) and the cost model.
    5.  Repeat until budget/time runs out.
        Ansor

* * *

📊 Results
==========

*   Without a scheduler, tuning assigns equal effort → many measurements wasted.
*   With Ansor’s scheduler:
    *   **Reaches strong performance faster** (Fig. 10, p.12).
    *   Prioritizes, e.g., 3×3 convolutions in ResNet (true bottlenecks).
    *   Matches AutoTVM’s performance with **8×–80× less wall time** (Table 3, p.13).
        Ansor

* * *

🔎 Analogy
==========

Imagine you’re training a sports team (a DNN):

*   Every player (subgraph) could improve.
*   But some are already stars, some barely touch the ball, and some are critical but underperforming.
*   A **bad coach (AutoTVM)** trains everyone equally.
*   A **good coach (Ansor)** spends more time on the bottlenecks: the goalie who keeps missing saves, or the striker who takes most shots. That’s how the whole team wins games faster.

* * *

✅ In Short
==========

*   The **task scheduler** allocates tuning effort intelligently.
*   It uses **gradient-based optimization + exploration** to spend more time on subgraphs that yield the biggest end-to-end performance gains.
*   This is a key reason Ansor is both **faster to tune** and produces **faster kernels** than template-based systems.

* * *

Would you like me to also **walk through a concrete example** (say, tuning ResNet-50 with its 29 subgraphs) and show how the scheduler would shift effort dynamically over time?

## Prompt:
What is a subgraph?

## Response:
Great question — **subgraphs** show up a lot in Ansor (and in deep learning compilers generally). Let’s make it clear.

* * *

🔹 What is a Subgraph?
======================

*   A deep neural network (DNN) can be represented as a **computation graph**:
    *   **Nodes** = operators (conv2d, matmul, ReLU, batchnorm, etc.).
    *   **Edges** = data dependencies (tensor outputs flowing to inputs).

A **subgraph** is simply a **connected portion of that larger graph**.

*   Instead of optimizing the entire DNN as one giant monolith (which would explode the search space), compilers **partition the graph into smaller pieces** — each piece is a subgraph.
*   Then they generate and tune a kernel (low-level code) for each subgraph.

* * *

🔹 Examples
===========

1.  **Single Operator Subgraph**
    *   A lone `conv2d` op could be treated as a subgraph.
2.  **Fused Subgraph**
    *   If you have `conv2d → batchnorm → ReLU`, Ansor might fuse these into one subgraph and generate one fused kernel for it.
    *   This is better than tuning them separately, because it avoids writing intermediate results back to memory.
3.  **Attention Subgraph**
    *   In transformers, a typical subgraph is `transpose → batch_matmul → softmax`.

* * *

🔹 Why Subgraphs Matter in Ansor
================================

*   **Optimization boundary:** Each subgraph is the unit of tuning (Ansor generates sketches, samples, and tunes programs at the subgraph level).
*   **Scheduler granularity:** The **task scheduler** decides which subgraph (task) to allocate tuning time to.
*   **Reusability:** Subgraphs often repeat (e.g., the same 3×3 conv block appears many times in ResNet-50). Ansor can tune once and reuse results.

* * *

🔎 Analogy
==========

Think of a DNN as a **factory assembly line**:

*   Each machine (operator) does a step.
*   A **subgraph** is a group of machines that work closely together (e.g., drilling + polishing + painting all fused into one workstation).
*   Optimizing the workstation (subgraph) makes the factory run much faster than just tweaking one tiny machine at a time.

* * *

✅ **In short:**  
A **subgraph** is a smaller section of the overall computation graph (one operator or a fused set of operators). Ansor generates and tunes kernels at the subgraph level, then uses a task scheduler to focus effort on the most critical subgraphs for end-to-end performance.

* * *

Do you want me to also show you how **Relay (TVM’s high-level IR)** partitions a DNN into subgraphs before Ansor takes over? That’s the step where “operator fusion” creates these subgraphs automatically.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)