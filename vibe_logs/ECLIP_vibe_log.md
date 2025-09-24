# Summarize ECLIP Paper

**User:** John Xie (johnjxie@gmail.com)  
**Created:** 9/24/2025 15:53:15  
**Updated:** 9/24/2025 17:30:57  
**Exported:** 9/24/2025 17:34:43  
**Link:** [https://chatgpt.com/c/68d44c2b-14fc-8330-b9fa-1bb62d53782d](https://chatgpt.com/c/68d44c2b-14fc-8330-b9fa-1bb62d53782d)  

## Prompt:
ECLIP.pdf

PDF

Summarize this paper and explain the main ideas using analogies for someone with only basic ML knowledge to understand

## Response:
Got it — let me break this paper (_ECLIP: Energy-efficient and Practical Co-Location of ML Inference on Spatially Partitioned GPUs_) down for you in simple terms with analogies, so even if you only have basic ML knowledge, the ideas will feel approachable.

* * *

**Big Picture**
---------------

Running ML models on GPUs is a lot like using a big kitchen with many chefs (GPU cores). But often, one recipe (inference task) doesn’t need all the chefs at once — many just stand idle. That wastes energy and slows things down.

ECLIP is like a smart kitchen manager who figures out how to **share the chefs among multiple recipes** without wasting time reorganizing them too often. This makes the kitchen both faster and more energy-efficient.

* * *

**Problem**
-----------

1.  **Underutilization**:
    *   A single ML model (like ResNet or BERT) rarely uses all GPU cores efficiently.
    *   It’s like hiring 60 chefs but only asking 20 to chop onions while the other 40 just stand around.
2.  **Spatial partitioning (splitting GPU into sections)**:
    *   Past solutions tried dividing the kitchen into fixed groups (e.g., one half for pasta, one half for dessert).
    *   But reorganizing the chefs for each step (kernel) takes too much time — like reassigning chefs every 30 seconds.
3.  **Overheads**:
    *   Changing partitions at every kernel is more costly than the cooking itself (sometimes the reshuffling takes longer than actually chopping the onions).
    *   Some methods even required changing the kitchen’s architecture (hardware mods), which isn’t practical.

* * *

**ECLIP’s Solution**
--------------------

ECLIP is like a **kitchen scheduling system** that finds a middle ground:

1.  **Pre-allocated groups of chefs** (pre-defined GPU streams):
    *   Instead of assigning chefs one by one each time, ECLIP sets up “teams” in advance (like 15-chef, 30-chef, and 45-chef teams).
    *   This avoids wasting time reshuffling every single step.
2.  **Runtime scheduler (kitchen manager)**:
    *   It intercepts each task (kernel) and assigns it to the right pre-allocated team.
    *   It makes sure the order of steps (dependencies) is respected — so the salad isn’t served before the vegetables are chopped.
3.  **Resource allocation optimizer (planning the menu)**:
    *   It profiles each recipe step and figures out the **minimum number of chefs needed without slowing down**.
    *   Then it groups steps together to minimize switching costs.
    *   It also ensures fairness when multiple recipes are cooked at once (so one dish doesn’t hog all the chefs while others starve).

* * *

**Results**
-----------

From experiments (see _Figures 4–6 on pages 5–6_):

*   **Throughput**: Up to **21% faster** overall.
*   **Energy efficiency**: On average **25% better**, sometimes as much as **35%**.
*   **Latency**: Kept within acceptable limits — even though splitting workers usually risks making some tasks wait longer.

So the kitchen produces more dishes per hour and uses less electricity, without making customers wait too long.

* * *

**Analogy Recap**
-----------------

*   **GPU cores = chefs in a big kitchen.**
*   **Inference kernels = recipe steps.**
*   **Old methods** = constantly reassigning chefs step-by-step (too much overhead) or dividing the kitchen in half for the entire meal (wasteful).
*   **ECLIP** = pre-formed chef teams + smart scheduling + fair sharing → faster, cheaper, greener meal prep.

* * *

✅ **Main Idea**:  
ECLIP makes GPUs more energy-efficient and practical for ML inference by **smartly grouping tasks and reusing partitions**, avoiding the waste of constant reshuffling or idling cores

ECLIP

.

* * *

Would you like me to also create a **section-by-section breakdown** (Intro, Background, Framework, Evaluation, etc.) in more detail, similar to how you asked me for other papers like TorchBench and Tenplex?

## Prompt:
Go more in depth this was too high level. Break it down based on the sections of the paper and go in depth on each section of the paper

## Response:
Thought for 2m 35s

Here’s a section-by-section, plain-English breakdown of **ECLIP: Energy-efficient and Practical Co-Location of ML Inference on Spatially Partitioned GPUs (ECLIP)**, with light analogies to keep it intuitive.

* * *

Abstract & Introduction — what’s the paper about?
=================================================

**Problem.** Single ML inference jobs often don’t use all of a GPU. Think of a kitchen with 60 chefs (GPU cores): many stand idle during parts of a recipe (a kernel), but you still pay to keep them around and powered. Prior ways to split the kitchen (GPU spatial partitioning like MPS/MIG) either split too coarsely (per whole model) or cost too much to reconfigure frequently (per kernel). Both waste energy or time.

ECLIP

ECLIP

**Idea.** ECLIP gets the benefits of fine-grained partitioning **without** paying the heavy reconfiguration cost each kernel: it (1) **pre-allocates** pools of “streams” already tied to specific compute-unit (CU) subsets, and (2) **optimizes** which kernels (or groups of kernels) use which stream sizes. Result: **+13% throughput on average** and **+25% energy efficiency** (up to **+21%** and **+35%**), while keeping tail latency acceptable.

ECLIP

* * *

II. Background — hardware & why energy is wasted
================================================

**AMD GPU layout.** AMD GPUs are built from **Compute Units (CUs)** (NVIDIA’s SMs), grouped into **Shader Engines (SEs)**—on MI50, 60 CUs across four 15-CU SEs. AMD’s **CU Masking** lets you steer a stream’s kernels to chosen CUs (finer than process-level MIG/MPS).

ECLIP

**Why under-utilization = wasted energy.** Lowering clock/voltage (DVFS) helps dynamic power, but **idle CUs still leak**; production GPUs can’t power-gate at fine granularity (usually just clock-gating or gating the whole GPU). The best practical way to save energy is to **increase utilization via co-location**.

ECLIP

**Limits of existing partitioning.**

*   **Model-grain right-sizing**: pick a fixed partition per model (find the “knee” of the latency vs. resources curve; often sized to ~3× isolated tail latency). But kernels inside a model need different amounts of CUs—so you still leave chefs idle.
    ECLIP
*   **Kernel-grain right-sizing**: better fit, but **repartitioning every kernel** can cost **more than the kernel runtime**; compiler or hardware changes used by prior work (ElasticRoom, KRISP) are impractical for real deployments.
    ECLIP
*   **libsmctrl** (NVIDIA): can set SM masks, but would still need a call **before every kernel** and framework changes—again, not practical.
    ECLIP

**Key challenge.** Make kernel-level partitioning practical by **managing reconfiguration overheads**.

ECLIP

* * *

III. Analyzing Hardware Characteristics — what overheads actually look like
===========================================================================

On an AMD **MI50** (60 CUs; four SEs), the authors instrument ROCm to inspect CU masking costs.

ECLIP

**Obs. 1 — per-kernel CU-mask changes are expensive and noisy.** The IOCTL that updates masks can be **~55.4 µs**, more than **2×** a 19.8 µs kernel—so “reassigning chefs every step” can take longer than the step itself.

ECLIP

**Obs. 2 — too many masked streams hurt latency (queue oversubscription).** Performance stays stable up to **7 CU-masked streams**; beyond that tail latency spikes because software queues outnumber hardware queues and must be time-multiplexed. Design rule: **at most 7 masked streams** (plus the default = 8 hardware queues).

ECLIP

* * *

IV. ECLIP Framework — how ECLIP works
=====================================

**High-level plan.** 1) **Pre-allocate** a small pool of CU-masked streams, 2) a **runtime scheduler** redirects each kernel to the right stream while maintaining dependencies, 3) a **resource-allocation optimizer** chooses CU sizes for **groups of kernels** (not every kernel) to cut switch overhead. See the block diagram in **Fig. 2**.

ECLIP

A. Runtime Scheduler
--------------------

*   **Pre-allocate seven masked streams** (per Obs. 2), in **15-CU increments** aligned with SEs (15, 30, 45 ...), so kernels run on whole SEs and co-located workers don’t step on each other’s toes. (Example worker-to-SE layouts provided.)
    ECLIP
    ECLIP
*   **Redirect kernel dispatch**: the HIP backend intercepts kernel launches, looks up the pre-computed CU size for that kernel (from the optimizer), and **moves** the kernel to the appropriate masked stream.
    ECLIP
*   **Enforce dependencies across streams**: when kernels from one user stream get split across hardware queues, ECLIP injects **HSA barrier packets** as needed (only if a prior dependent kernel hasn’t finished). Over-barriering hurts, so ECLIP budgets these.
    ECLIP
    ECLIP

**Mental model.** Pre-make “chef teams” (15, 30, 45 chefs). For each cooking step you **hand it to the right team** and place a **“don’t start until this finishes”** sticky note only when needed.

B. Resource Allocation Optimizer
--------------------------------

*   **Minimum CU threshold profiling.** For each kernel, find the **smallest CU count** that avoids noticeable slowdown (see Fig. 3; different kernels prefer very different sizes).
    ECLIP
*   **Group-level switching.** Switching CU size forces a barrier; too many switches dominate runtime. So ECLIP **groups kernels** and limits switches (the **barrier packet budget**).
    ECLIP
*   **Fairness across workers.** If one model finishes way earlier, others keep the GPU partially idle (since you can’t power-gate fine-grained CUs). The optimizer **balances progress** so CUs aren’t left leaking.
    ECLIP
*   **ILP formulation (solved offline).** Decision variable  $x_{k,c}$  selects one CU config  $c\in\{15,30,45,60\}$  per kernel; objective minimizes total (fair) execution time. Slowdown from co-location is modeled via **average CU overlap** across workers; a **switch budget** caps stream changes. Solutions are stored in a **lookup table** used at runtime.
    ECLIP
    ECLIP
    ECLIP
    ECLIP

**Mental model.** Plan the menu **before** the dinner rush: decide which steps need 15/30/45/60 chefs, **limit how often** you reshuffle teams, and keep all dishes moving so no chefs are stuck idle.

* * *

V. Evaluation — does it work in practice?
=========================================

**Setup.** Dual AMD EPYC 7302, MI50 (60 CUs in four 15-CU SEs), Ubuntu 20.04, ROCm 5.2.0, PyTorch 1.12; power via rocm-smi. Models: **albert, densenet201, alexnet, resnet15, resnext101, shufflenet, vgg19**. Seven **co-location mixes** at max RPS.

ECLIP

ECLIP

**Compared strategies.**

*   **Baseline** (no partitioning, all 60 CUs).
*   **Model-Wise** (fixed partition per model).
*   **Kernel-Wise\_Prealloc** (switch nearly every kernel using pre-allocated streams).
*   **ECLIP** (like above, but **with a max ~14 switches/request** from the barrier budget). Kernel-Wise\_IOCTL is omitted in graphs due to extreme slowdowns; its switch counts are huge.
    ECLIP

**Switch counts (why ECLIP is practical).** Table I shows per-request switch totals across workers—e.g., Mix 1: **1319** (IOCTL) vs **413** (Kernel-Wise\_Prealloc) vs **42** (ECLIP). Similar gaps across mixes.

ECLIP

**Throughput.** ECLIP wins: **up to 21%** (Mix 7), **+13% average**; Kernel-Wise\_Prealloc underperforms due to too many switches. (Fig. 4)

ECLIP

**Tail latency (95th).** Fine-grain partitioning can inflate tails; ECLIP’s **switch budget** keeps tails in check vs Kernel-Wise\_Prealloc. Over-coarse Model-Wise can also hurt (e.g., Mix 3 where **vgg19 often needs all 60 CUs**, forcing others to contend). (Fig. 5)

ECLIP

**Energy efficiency (Requests/Watt).** ECLIP improves energy efficiency consistently—**up to 35%** (Mix 7), **+25% average**—by keeping CUs busy **without** paying prohibitive switch costs. (Fig. 6)

ECLIP

* * *

VI. Related Work — what’s new here?
===================================

Earlier GPU partitioning work focused on how to split resources; some mentioned overheads, but with simplified models or no deep analysis. ECLIP contributes **real-hardware** characterization of partition overheads and a **practical** runtime + optimizer that works in stock ROCm/PyTorch setups.

ECLIP

* * *

VII. Conclusion — the one-liner
===============================

**Pre-allocated stream pools + dependency-aware scheduling + ILP-based CU sizing for kernel groups** ⇒ **+13% throughput** and **+25% energy efficiency on average** (peaks: **+21%**, **+35%**) with acceptable tails—**no model or PyTorch code changes required**.

ECLIP

* * *

Quick “how to use this” takeaways (for practitioners)
-----------------------------------------------------

*   If you co-locate models, **don’t re-mask per kernel**; **pre-allocate** a **small** set of masked streams (≤7).
    ECLIP
*   **Profile** kernels to find **minimum CU thresholds**; **group** kernels to limit switches.
    ECLIP
    ECLIP
*   **Budget barriers/switches** (≈ dozens, not hundreds) to keep tail latency and overheads sane. Table I shows the difference.
    ECLIP

If you want, I can turn this into a 1–2 page PDF or a slide with the key figures called out (Fig. 2 overview, Fig. 4–6 results, Table I).

2/2

## Prompt:
What does it mean to reconfigure a kernel?

## Response:
Great question — this is at the heart of what ECLIP is trying to fix. Let’s unpack it.

* * *

**What a “kernel” is here**
---------------------------

On a GPU, a **kernel** is basically a small program (a function) that runs in parallel across many GPU cores (CUs/SMs). For example, a convolution kernel in a CNN computes one layer of the network.

Each kernel has:

*   **Workload size** (how much math it needs to do).
*   **Resource needs** (how many GPU cores/CUs it could use efficiently).

* * *

**What “reconfiguring a kernel” means**
---------------------------------------

Reconfiguring a kernel means **changing which part of the GPU (and how many cores) it runs on** before it executes.

*   Normally, a kernel just launches across **all available GPU cores** (like telling all 60 chefs in the kitchen to help).
*   But many kernels don’t need that many cores — some may run just as fast with 15 or 30.

So “reconfiguring” means **adjusting the GPU’s spatial partitioning mask** (e.g., CU masking on AMD, SM masking on NVIDIA) to give that kernel only a subset of cores.

This requires a system call (IOCTL) to tell the GPU driver:

> “Hey, for this next kernel, only use these 15 CUs, not all 60.”

* * *

**Why it’s costly**
-------------------

Each reconfiguration:

1.  **Sends commands to the GPU driver** (CU mask update).
2.  **Flushes queues** and enforces dependencies so kernels don’t break order.
3.  Can take **tens of microseconds** — which is longer than the runtime of many ML inference kernels! (e.g., 19.8 µs kernel vs 55.4 µs reconfig time)
    ECLIP
    .

So if you reconfigure for _every_ kernel, you spend more time **reassigning chefs** than actually **cooking**.

* * *

**ECLIP’s trick**
-----------------

Instead of reconfiguring on-the-fly for each kernel:

*   **Pre-allocate streams** with fixed CU masks (like teams of 15, 30, 45 chefs ready to go).
*   **Redirect kernels** to the right preconfigured team instead of reshuffling chefs mid-meal.

This avoids the huge per-kernel overhead, while still letting small kernels use fewer resources.

* * *

✅ **In short:**  
Reconfiguring a kernel = telling the GPU to change how many cores (and which ones) that kernel can run on. It saves energy when done wisely, but if done naively every kernel, the “paperwork” of reconfiguration costs more than the actual computation.

* * *

Do you want me to illustrate with a **numerical example** (say, kernel takes 20 µs compute but 50 µs reconfig cost) to show exactly why naive per-kernel reconfiguration is worse?

## Prompt:
What is a spatial partitioning mask?

## Response:
Good question — let’s make this concrete.

* * *

**Spatial Partitioning in GPUs**
--------------------------------

A GPU is like a big factory floor with many identical workstations (CUs in AMD, SMs in NVIDIA). Normally, when you launch a kernel, it can use _all_ the workstations.

**Spatial partitioning** is the act of **splitting the GPU into smaller chunks** so different jobs (kernels or models) can run _side by side_ without interfering too much.

* * *

**What the “mask” is**
----------------------

The _mask_ is basically a **bitmask (on/off switches)** telling the GPU **which compute units (CUs/SMs) are allowed** to execute a given stream or kernel.

*   Example: An AMD MI50 has 60 CUs.
*   A 60-bit mask might look like:
    ```
    111111111111111111111111111111111111111111111111111111111111
    ```
    (all 60 CUs enabled).
*   If you want only 15 CUs active (say, the first SE), the mask might look like:
    ```
    111111111111111000000000000000000000000000000000000000000000
    ```
    (first 15 on, the rest off).

The GPU driver uses this mask to **route kernels only to those enabled cores**.

* * *

**Why it matters**
------------------

*   **Performance isolation**: Two ML models can run on separate masks (disjoint CUs), avoiding resource fights.
*   **Right-sizing**: A lightweight kernel that only needs 15 CUs doesn’t waste the whole GPU.
*   **Energy efficiency**: Idling CUs can in principle be clock-gated if they aren’t in the mask.

* * *

**The downside**
----------------

Every time you change the mask (say, one kernel needs 15 CUs, the next needs 45), you incur an **IOCTL reconfiguration cost** — tens of microseconds. For tiny inference kernels, this cost can exceed the kernel’s runtime. That’s why ECLIP avoids frequent remasking by **pre-allocating multiple masked streams**.

* * *

✅ **In short:**  
A spatial partitioning mask = a binary on/off map that says **which GPU cores are active for a job**. It’s like taping off sections of a kitchen so only certain chefs can cook that dish.

* * *

Do you want me to also show how NVIDIA’s **MIG (Multi-Instance GPU)** uses a _coarser-grained_ spatial mask compared to AMD’s fine-grained CU masking? That contrast makes the idea even clearer.

## Prompt:
How does ECLIP knows which streams to map together?

## Response:
Great, now we’re drilling into the heart of **ECLIP’s scheduling + optimizer logic**. Let’s unpack step by step.

* * *

**Streams in ECLIP**
--------------------

*   A **stream** in ROCm (or CUDA) is like a “lane” where kernels are queued to run.
*   ECLIP creates a **fixed pool of streams**, each with a pre-set **CU mask** (e.g., 15, 30, 45 CUs). Think of them as “teams of chefs” already reserved.

The big question:  
👉 _When multiple kernels arrive, how does ECLIP decide which kernel goes to which stream/team?_

* * *

**Step 1 — Kernel Profiling (offline)**
---------------------------------------

Before runtime, ECLIP profiles each kernel **in isolation** to find the **minimum CU threshold** that doesn’t cause slowdown.

*   Example:
    *   Kernel A needs ≥15 CUs.
    *   Kernel B needs ≥30 CUs.
    *   Kernel C needs all 60 CUs to stay fast.

This tells ECLIP the “sweet spot” resource size per kernel

ECLIP

.

* * *

**Step 2 — Grouping Kernels (to reduce switching)**
---------------------------------------------------

If you reconfigure every kernel, overhead kills you. So instead, ECLIP:

*   Groups consecutive kernels that have **similar CU needs**.
*   Assigns the whole group to the same stream.
*   Limits the **number of stream switches** (via a _barrier packet budget_).

This is like saying: instead of swapping chef teams every single dish, keep one team for a batch of similar dishes, then switch.

* * *

**Step 3 — Co-location Fairness**
---------------------------------

When multiple models (workers) run together, ECLIP balances assignments so **no one model hogs the GPU**.

*   It tracks **average CU usage per worker**.
*   If workers’ CU masks overlap too much, they both slow down.
*   The optimizer penalizes overlapping streams to **spread them fairly**.

So if Model A and Model B both want 30 CUs, ECLIP tries to give them **different 30-CU streams** that don’t overlap across Shader Engines

ECLIP

.

* * *

**Step 4 — Resource Allocation Optimizer (ILP-based)**
------------------------------------------------------

ECLIP formulates this as an **Integer Linear Programming (ILP)** problem:

*   **Decision variable**  $x_{k,c}$ : does kernel  $k$  use configuration  $c \in \{15,30,45,60\}$ ?
*   **Objective**: minimize total execution time across all workers.
*   **Constraints**:
    *   Each kernel picks _one_ config.
    *   Total number of switches ≤ budget.
    *   Slowdown from overlap modeled explicitly.

Solutions are computed **offline** for different co-location scenarios and stored in a **lookup table**.

At runtime, the scheduler just **looks up the pre-computed answer** and redirects kernels accordingly

ECLIP

.

* * *

**So how does it know which streams to map together?**
------------------------------------------------------

*   **It knows each kernel’s “minimum CU need” from profiling.**
*   **It groups kernels with similar needs** to minimize reconfiguration.
*   **It uses fairness + overlap modeling** to avoid giving two workers the same CUs.
*   **It pulls the decision from a pre-computed optimizer lookup table.**

In other words:  
ECLIP maps kernels to streams by **consulting an offline-optimized plan** that balances speed, fairness, and limited switching, then applies it on-the-fly through pre-allocated CU-masked streams.

* * *

✅ **Analogy:**  
Imagine you’re running a wedding kitchen with multiple dishes (models).

*   You’ve tested each dish before and know how many chefs it _really_ needs.
*   You’ve pre-formed teams of 15, 30, 45, 60 chefs.
*   You batch dishes that need similar chef counts.
*   You make sure no two dishes are stealing the same chefs.
*   And you follow a pre-written playbook (lookup table) for how to assign dishes to teams.

* * *

Do you want me to **walk through a concrete example** with, say, two models (Albert + VGG19) and show exactly how ECLIP would map their kernels to streams?

2/2

## Prompt:
Can you explain the overlap modeling and ILP more? This is still confusing

## Response:
Perfect — let’s slow down and go into **overlap modeling** and the **ILP (Integer Linear Program)** inside ECLIP, since that’s the core mathy part.

* * *

1\. Why overlap matters
=======================

Imagine two models (workers) sharing the GPU.

*   Model A is using 30 CUs.
*   Model B is also using 30 CUs.

If both happen to use **the same 15 CUs**, they fight over them → slowdown.  
If they’re assigned **disjoint sets of CUs**, they can run more smoothly.

👉 So, **overlap modeling** = quantifying “how much two workers’ CU assignments collide” and how much that slows them down

ECLIP

.

* * *

2\. How overlap is modeled
==========================

The paper does this in a simplified way:

*   Each kernel  $k$  is assigned a CU count  $c$  (chosen from {15, 30, 45, 60}).
*   For each worker  $w$ , they compute:
    
$$
CUAverage_w = \frac{\sum_{k \in w} \sum_c x_{k,c} \cdot c}{\# \text{kernels in } w}
$$

This is the **average CU usage** per kernel for worker  $w$ .

*   Then they define:
    
$$
CUOverlap_w = CUAverage_w + \sum_{w' \neq w} CUAverage_{w'}
$$

This is a rough way to say: “The slowdown worker  $w$  experiences is proportional to how many CUs it overlaps with the others.”

Finally, they model kernel slowdown as:

\\alpha\_k = \\frac{CUOverlap\_w}{\\text{Total # of CUs}}

So if overlap is large (many shared CUs), slowdown factor  $\alpha_k$  is bigger → longer runtimes.

* * *

3\. ILP decision variables
==========================

The ILP decides, for each kernel  $k$ :

*   **Which CU configuration to pick** (15, 30, 45, or 60).

It uses a binary variable:

$$
x_{k,c} = \begin{cases} 1 & \text{if kernel } k \text{ uses config } c \\ 0 & \text{otherwise} \end{cases}
$$

Constraint:

$$
\sum_{c \in configs} x_{k,c} = 1 \quad \forall k
$$

(Each kernel must get exactly one configuration.)

* * *

4\. Modeling switching (barrier cost)
=====================================

Every time you change configs between consecutive kernels, you need a **barrier packet** (dependency enforcement). Too many barriers → overhead.

So they define:

$$
switchTotal_w = \sum_{k \in w} \sum_{c \in configs} |x_{k,c} - x_{k-1,c}| / 2
$$

This counts how many times worker  $w$  changes CU configuration between kernels.  
Constraint:

$$
switchTotal_w \leq switchMax
$$

(where `switchMax` is the barrier packet budget, e.g. ~14 per request).

* * *

5\. Execution time estimation
=============================

For each kernel  $k$ :

$$
e_k = \beta_k \cdot (1 + \alpha_k)
$$
*    $\beta_k$  = base runtime from profiling (depends on CU count).
*    $\alpha_k$  = slowdown factor from overlap.

So this gives the **estimated runtime** of that kernel under current stream assignment.

* * *

6\. The ILP objective
=====================

Finally, the ILP minimizes total runtime across all workers, with fairness:

$$
\forall w, \quad \min \sum_{k \in w} e_k
$$

Each worker’s time is equally weighted, so no model gets starved while others speed up.

This creates a **balanced mapping** that reduces overlap, limits switches, and keeps runtimes short.

* * *

7\. Solving & Using
===================

*   Solving the ILP online would be too slow.
*   So they solve it **offline** for different “scenarios” (e.g., mix of 2 or 3 models) using an ILP solver (like Gurobi).
*   The solutions are stored in a **lookup table**.
*   At runtime, the scheduler just looks up the right CU assignments and maps kernels to the pre-allocated streams.

* * *

✅ **Analogy version:**  
Think of it as seating people at tables in a cafeteria:

*   Each kernel needs a table of a certain size (15, 30, 45, 60 seats).
*   If two groups sit at the same table (overlap), they slow each other down.
*   You don’t want people to keep switching tables every time (barrier overhead).
*   The ILP figures out the best seating plan in advance: who sits at which table, how often they move, and making sure everyone finishes eating around the same time.

* * *

Do you want me to **walk through a tiny toy example** with 2 workers and 3 kernels each, showing how the ILP variables get set and how overlap affects the solution? That might make the math click.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)