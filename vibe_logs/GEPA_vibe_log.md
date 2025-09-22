# GEPA paper summary

**User:** John Xie (johnjxie@gmail.com)  
**Link:** [https://chatgpt.com/c/68d16e62-b7ec-8326-bac6-1e3c5c647283](https://chatgpt.com/c/68d16e62-b7ec-8326-bac6-1e3c5c647283)  

## Prompt:
GEPA.pdf

PDF

Summarize this paper and explain the main ideas using analogies for someone with only basic ML knowledge to understand

## Response:
Here’s a summary of the **GEPA paper** (“Reflective Prompt Evolution Can Outperform Reinforcement Learning”) in plain, analogy-driven terms, aimed at someone with only basic ML knowledge:

* * *

🌟 Big Picture
--------------

Most current ways to improve AI systems (like ChatGPT) use **reinforcement learning (RL)**. Think of RL as training a dog: you give it treats (rewards) when it does the right thing. But this can be very slow—dogs need lots of repetitions before they learn.

The GEPA method says: _Instead of just giving treats, why not let the AI **talk about what it did right or wrong** in plain language, and then rewrite its own “instructions” (prompts) to do better next time?_

This approach is like a student keeping a diary after each test:

*   _“I forgot to check the last question, so next time I’ll double-check before submitting.”_  
    By writing reflections, the student learns faster than if they only got a grade number back.

* * *

🧩 Core Ideas
-------------

### 1\. **Reflective Prompt Evolution**

*   Instead of changing the AI’s inner brain (weights), GEPA changes the **instructions it follows** (the prompt).
*   After each attempt (“rollout”), GEPA looks at the step-by-step reasoning, successes, and failures written out in text.
*   It then rewrites the prompt with lessons like: _“Be more specific about missing details,”_ or _“Don’t repeat the question.”_  
    👉 Analogy: Imagine a chef trying new recipes. Instead of just hearing “6/10 taste score,” the chef keeps notes like: _“Too salty—reduce salt next time.”_

* * *

### 2\. **Genetic Evolution + Pareto Frontier**

*   GEPA evolves prompts like nature evolves species: making mutations (small changes) and combining different good prompts.
*   It keeps a “Pareto frontier”—the set of best prompts, each good in different ways (fast, accurate, privacy-safe, etc.).  
    👉 Analogy: Think of building a soccer team. You don’t just keep the single “best” player. You keep strikers, defenders, and goalies—each the best in their role—and mix their strengths.

* * *

### 3\. **Sample Efficiency**

*   RL needs **tens of thousands of trials** to improve (like practicing 100,000 basketball shots).
*   GEPA often learns from just a few trials, because natural-language feedback gives richer clues than a single number score.  
    👉 Analogy: It’s the difference between a teacher saying _“B+”_ versus _“You lost points for skipping step 3—focus on that next time.”_

* * *

### 4\. **Performance**

*   On benchmarks like HotpotQA, HoVer, IFBench, and PUPA, GEPA:
    *   Outperformed RL methods (GRPO) by ~10–20% while using up to **35× fewer rollouts**
        GEPA
        .
    *   Beat the leading prompt optimizer MIPROv2 by over 10%.
*   Also showed promise for **code optimization tasks**: improving CUDA kernels and chip-specific code at inference time.

* * *

📊 Example
----------

On page 3, the paper shows how GEPA improved a **multi-hop QA system**:

*   The old prompt just said: “Make a query.”
*   GEPA rewrote it to say things like: _“Don’t just repeat the question. Instead, look at what’s missing from the first document and search for the bigger context.”_  
    👉 This is like teaching a detective: instead of re-asking the same question to a witness, ask about the _next missing clue_ to solve the case.

* * *

🚀 Why It Matters
-----------------

*   GEPA shows we can often optimize AI **without costly reinforcement learning** or retraining models.
*   It uses **language as the medium for improvement**—more interpretable, cheaper, and often more effective.
*   This makes AI adaptation more practical in real-world settings with limited time, money, or compute.

* * *

✅ **In one line:**  
GEPA teaches AI to learn like a reflective student—writing notes, revising its instructions, and combining the best ideas—rather than like a dog trained by trial-and-error rewards.

* * *

Would you like me to also break this down **section by section** (Intro, Methods, Results, Limitations, etc.) so you have a structured deep-dive summary you could use for class notes or presentations?

## Prompt:
Go more in depth this was too high level. Break it down based on the sections of the paper and go in depth on each section of the paper

## Response:
Thought for 2m 43s

GEPA paper — section-by-section deep dive
=========================================

Below is a structured summary of **“GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.”** I follow the paper’s section flow and call out key figures/tables by page. Citations refer to your uploaded PDF.

GEPA

* * *

Abstract (what the paper claims)
================================

*   **Problem.** Adapting LLM systems with RL (e.g., GRPO) is powerful but needs _lots_ of rollouts—often tens of thousands—because the reward is a single scalar at the end.
*   **Idea.** Use the rich **natural-language traces** that LLM systems already produce (module instructions, chain-of-thought, tool errors, evaluator logs) to **reflect** and **rewrite prompts**—then **evolve** these prompts with a multi-objective (Pareto) search.
*   **Result.** Across four tasks, **GEPA** beats GRPO by ~**10% on average** (up to ~20%) using up to **35× fewer** rollouts, and beats the prompt optimizer **MIPROv2** by >10%. It also shows promise as an inference-time search method for code optimization. See **Figure 1 (p.1)** for learning curves.
    GEPA

* * *

1\. Introduction (why this matters)
===================================

*   RL with verifiable rewards is common (e.g., **GRPO**), but it collapses rich process information into a single number, which is **sample-inefficient**.
*   Observation: compound LLM systems generate **serializable text traces** (reasoning steps, tool calls, evaluator messages), which are **interpretable by LLMs** themselves.
*   **GEPA** leverages this: it **reflects in language** over full system trajectories, **proposes prompt updates**, and **combines** complementary lessons along a **Pareto frontier** to avoid local optima.
*   Evaluated on **HotpotQA, IFBench, HoVer, PUPA** with **Qwen3-8B** and **GPT-4.1 mini**; gains are large and fast. **Figure 2 (p.3)** shows how a seed prompt for multi-hop retrieval is rewritten into a much more effective “second-hop query” instruction.
    GEPA

* * *

2\. Problem Statement (formal setup)
====================================

*   A **compound AI system** is formalized as **Φ = (M, C, X, Y)**:
    *   **M**: language modules  $M_i = (\pi_i,\; \theta_i,\; X_i,\; Y_i)$  with **prompt  $\pi_i$ ** and **weights  $\theta_i$ **.
    *   **C**: control flow connecting modules and tools.
    *   **X, Y**: global I/O schemas.
*   Given task instances  $(x,m)$  and a metric **µ** (EM/F1/pass-rate, etc.), optimize over prompts and/or weights  $\langle \Pi,\Theta\rangle$  to maximize expected performance.
*   **Sample-efficiency constraint:** the optimizer has a rollout budget **B** and must extract **maximal learning** per rollout. This motivates learning from **textual feedback** (not just scalar µ).
    GEPA

* * *

3\. GEPA: Reflective Prompt Evolution (how it works)
====================================================

**High-level loop.** Maintain a **pool of candidate systems** (different prompts across modules). Repeatedly:

1.  select a promising candidate; 2) select a module to improve; 3) run a **minibatch** of rollouts; 4) **reflect** over traces and evaluator feedback; 5) **rewrite** that module’s prompt; 6) keep the new candidate if it improves; 7) record validation scores and ancestry; 8) repeat. **Figure 3 (p.4)** diagrams the pipeline; **Algorithm 1/2 (p.5)** give pseudocode.
    GEPA

### 3.1 Genetic optimization loop

*   Start with the baseline system as the only candidate. Each successful mutation **appends a child** to the search tree, tracking **ancestry** so GEPA can later **merge** lineages (crossover).
*   New variants are accepted if they **beat their parent** on a minibatch; promising ones are **fully evaluated** on a Pareto validation set (**Dₚₐᵣₑₜₒ**). This keeps compute focused.
    GEPA

### 3.2 Reflective prompt mutation (credit assignment in text)

*   **What it uses.** GEPA inspects **module IO + reasoning traces** _and_ **evaluator traces** (e.g., compiler errors, unit-test details) to perform **implicit credit assignment**: it identifies which module prompt likely caused success/failure.
*   **How it updates.** A **meta-prompt** (Appendix B) instructs an LLM to rewrite the module’s instruction by:
    *   Inferring **task format** and **niche rules** from examples + feedback,
    *   Distilling **high-level strategies**, not just examples,
    *   Producing compact, declarative instructions.
*   **Granular feedback** via a modified metric **µ\_f** passes back textual signals (e.g., “missing doc X at hop 2”), enabling **targeted prompt changes** instead of blind global edits. **Figure 5 (p.7)** visualizes an **iterative prompt refinement** path on the privacy task (PUPA), where each step adds concrete rules (e.g., “ban names/codes,” “justify abstractions”).
    GEPA

### 3.3 Pareto-based candidate selection (explore without getting stuck)

*   **Problem.** Always picking the current “best” candidate causes **stalling** (local optimum).
*   **Solution.** GEPA forms a **Pareto frontier** over **individual training instances**: the set of candidates that are **non-dominated** (each “wins” on at least one instance). It **samples** from these with probability proportional to how often a candidate appears on the frontier—an “illumination” strategy.
*   **Why it helps.** Preserves **diverse winning strategies** (e.g., different modules improved), escaping local optima while controlling search breadth. **Figure 6 (p.11)** contrasts the stuck “SelectBestCandidate” tree vs. GEPA’s broader, higher-scoring tree.
    GEPA

* * *

4\. Evaluation Setup (what they tested)
=======================================

**Data protocol.** Standard **train/val/test** splits; optimizers see **training content** and **validation scores** but **not** validation content; test is fully held-out.

GEPA

**Benchmarks + systems.**

*   **HotpotQA** multi-hop QA: modify last hop to answer; feedback tells which gold docs remain to retrieve. Splits: 150/300/300.
*   **IFBench** instruction following with **hard output constraints**; 2-stage program (answer → constraint-rewrite) with feedback on satisfied/violated constraints. Splits: 150/300/294.
*   **HoVer** claim verification via multi-hop document retrieval (up to 3 hops; 2 query writers + 2 summarizers); feedback reports which gold docs are found/missing. Splits: 150/300/300.
*   **PUPA** privacy-conscious delegation with trusted/untrusted models; feedback decomposes quality vs. PII leakage. Splits: 111/111/221. (See §4.1, pp.7–8.)
    GEPA

**Models + decoding.**

*   **Qwen3-8B** (open): temp 0.6, top-p 0.95, top-k 20.
*   **GPT-4.1 mini** (commercial): temp 1.0. (See §4.2, p.8.)
    GEPA

**Baselines / optimizers.**

*   **Baseline:** unoptimized program.
*   **MIPROv2:** Bayesian search over **instructions + few-shots** (TPE), “auto=heavy” (~18 instruction candidates and 18 few-shot sets per module), total rollouts ranged ~2.3k–6.9k depending on task.
*   **GRPO:** 24,000 rollouts fixed; **LoRA** finetuning (rank 16, α 64, dropout 0.05), group size 12 (4 instances/step), bf16, grad-accum 20, gradient clip 0.1, warmup scheduler; trained on A100/H100.
*   **GEPA:** minibatch size **b=3**; **Merge** (system-aware crossover) used up to **5×** per run when enabled; **compute-budget matched** to MIPROv2 within ~10% to ensure fair comparison. (§4.3, pp.8–9.)
    GEPA

* * *

5\. Results & Analysis (what happened)
======================================

**Main scores.** **Table 1 (p.9)**:

*   On **Qwen3-8B**, GEPA improves aggregate score to **61.28** vs **51.14** for GRPO and **55.11** for MIPROv2; per-task gains are similar (e.g., **HotpotQA: 62.33** vs 43.33 GRPO / 55.33 MIPROv2).
*   On **GPT-4.1 mini**, GEPA achieves **66.97** aggregate vs **59.71** (MIPROv2). **GEPA+Merge** can be even higher on GPT-4.1 mini (**68.69**). **Figure 9 (p.20)** shows final test bars.
    GEPA

### Observation 1 — **Sample efficiency & outperformance of RL**

*   **Rollout efficiency:** GEPA **matches or beats** GRPO’s **best val scores** with orders-of-magnitude fewer training rollouts (e.g., HotpotQA **~402** vs 24k; IFBench **~330**; HoVer **~1179**; PUPA **~306**).
*   **Train-only rollouts** to reach _final_ best: HotpotQA 737; IFBench 79; HoVer 558; PUPA 269.
*   **Curves** (Figures **10–13**, pp.21–22) and **Figure 1 (p.1)** confirm steeper early gains and higher endpoints than GRPO.
    GEPA

### Observation 2 — **Instruction-only beats instruction+few-shot**

*   Despite optimizing **only instructions**, GEPA **consistently beats MIPROv2** across models/tasks (e.g., +11.1% GPT-4.1 mini; +10.3% Qwen3-8B).
*   **Generalization gap** analysis (Figure 14, p.22) shows **reflectively evolved instructions** now **generalize better**, likely because recent LLMs follow instructions and reflect more robustly.
*   GEPA’s prompts are **declarative rule-sets** rather than quasi-examples; see **Figure 2 (p.3)** and Appendix I for side-by-side prompts.
    GEPA

### Observation 3 — **Pareto sampling matters (search strategy)**

*   Replacing Pareto with “select best” hurts final quality; the **ablation** (Table 2, p.10) shows up to **8.17%** drop.
*   The **search trees** (Figure 6, p.11) visualize why: best-only gets stuck after 1–2 profitable mutations; Pareto continues exploring **diverse lineages** and converges **higher** within the same budget.
    GEPA

### Observation 4 — **Cost/latency: shorter prompts, better perf**

*   GEPA’s final prompts are **much shorter** (often **×2–×9** smaller) than MIPROv2, which packs many few-shot exemplars (Figures **15–16**, pp.22–24).
*   Shorter inputs reduce **token cost** and **latency** at serving time while **raising** accuracy.
    GEPA

### Observation 5 — **System-aware crossover (Merge) can help**

*   **GEPA+Merge** sometimes outperforms GEPA (e.g., on GPT-4.1 mini), by **combining modules** from **complementary lineages** (Appendix F, Algorithm 4 on p.25).
*   Timing matters: invoke **after** sufficiently diverse lineages emerge; suboptimal timing hurt Qwen3-8B on some tasks.
    GEPA

* * *

6\. GEPA for inference-time search (overfitting the task set on purpose)
========================================================================

*   Treat the **inference task list** as both train and Pareto sets; let GEPA **overfit**—iteratively refining prompts per task—_without_ stochastic sampling (cache disables temperature effects).
*   **NPU kernels (AMD XDNA2 / NPUEval):** a simple iterative agent baseline gets **4.25%** mean vector utilization; with **RAG** 16.33%; with **RAG+MIPROv2** 19.03%. **GEPA** drives several kernels to **~70%** utilization; mean **30.52%**, and even a **single** GEPA-prompt lifts the agent to **26.85%** (Figure 7, p.12; Figure 25 prompt example on p.76–77).
*   **CUDA kernels (KernelBench/V100):** **fast₁** (fraction faster than PyTorch eager by ≥1×) surpasses **20%** as budget grows (Figure 8, p.13).
*   **Key trick:** use **µ\_f** to **retrieve** relevant manual snippets based on **compiler errors**, injecting domain expertise into reflection.
    GEPA

* * *

7\. Related Work (what’s new here)
==================================

*   **Prompt optimization:** from manual tricks (CoT) to LLM-driven optimizers; GEPA differs by using **environment textual feedback** + **Pareto-aware evolutionary** search across **modules** in compound systems.
*   **Evolutionary LLMs:** prior work mutates prompts/code but often with **random** mutations or no environment-aware credit; GEPA **reflects** with **task-specific traces** and **retains** all instance-wise “wins.”
*   **Reflection/feedback & RL:** language-feedback RL exists, but still **weight-training heavy**; GEPA shows **prompt-space** learning can be **more sample-efficient** in many practical regimes.
*   **Compound system optimization:** DSPy/MIPROv2/Optimas optimize programs but mainly via **global rewards**; GEPA emphasizes **module-level textual feedback** and **instance-wise Pareto tracking.**
    GEPA

* * *

8\. Limitations & Future Work (what’s not settled)
==================================================

*   **Prompt vs weight updates.** With **abundant data**, weight-space learning might win; GEPA excels when rollouts are **expensive**. Current GRPO baseline used **LoRA**; **full-parameter** GRPO (very costly, 100k–500k+ rollouts) merits study.
*   **Validation budget dominates** GEPA’s counts—future work can shrink **Dₚₐᵣₑₜₒ** or **subsample** validation adaptively.
*   **Feedback engineering.** Which evaluator/system traces are most valuable for reflection?
*   **Hybrid methods.** Use GEPA’s **language-level lessons** to **guide** RL rollouts or combine with finetuning.
    GEPA

* * *

9\. Conclusion (take-home)
==========================

*   **Language is a learning substrate.** By reflecting in natural language over rich traces, an LLM can **rewrite and evolve** its own prompts and **outlearn** RL that sees only scalar rewards.
*   **Pareto fronts preserve diversity** and prevent premature convergence.
*   **In practice:** better accuracy, **vastly fewer rollouts**, and **cheaper prompts**—plus promising results for **code optimization** as an inference-time search strategy.
    GEPA

* * *

Pictorial anchors you can cite in slides
========================================

*   **Figure 1 (p.1):** Learning curves—GEPA vs MIPROv2 vs GRPO on HotpotQA/HoVer.
*   **Figure 2 (p.3):** Seed vs GEPA-optimized prompt for second-hop retrieval.
*   **Figure 3 + Algorithms 1–2 (pp.4–5):** System diagram + pseudocode.
*   **Figure 5 (p.7):** Step-by-step prompt evolution on PUPA (scores labeled).
*   **Table 1 (p.9):** Main results across tasks/models.
*   **Figure 6 (p.11):** Search trees—why Pareto sampling matters.
*   **Figures 7–8 (pp.12–13):** NPU/CUDA kernel improvements.
*   **Figures 14–16 (pp.22–24):** Generalization gaps and prompt size vs score.
    GEPA

* * *

If you want, I can also extract the **exact meta-prompt** GEPA uses for reflection (Appendix B) and produce a **one-page cheat-sheet** you can reuse to tune your own multi-module pipelines (retrievers, planners, callers, verifiers).

GEPA

2/2

## Prompt:
Can you explain the compound ai system representation? Go more in depth about the M language modules.

## Response:
Thought for 1m 36s

Here’s a deep dive into the paper’s **compound AI system representation** and, especially, what the **language modules  $M$ ** are and how they work.

* * *

1) The formal picture:  $\Phi = (M, C, X, Y)$ 
==============================================

The paper models any multi-step LLM “program” (agent, tool-using pipeline, multi-module workflow) as a **compound AI system**  $\Phi$  with four parts

GEPA

:

*   ** $M=\langle M_1,\dots,M_{|M|}\rangle$ ** — the **language modules** (explained in §2 below).
*   ** $C$ ** — the **control flow** (the orchestration logic: order, looping, conditionals, tool calls, passing outputs from one module to the next).
*   ** $X$ ** and ** $Y$ ** — the **global input and output schemas** (what the whole system expects as input and ultimately returns).

At run time,  $C$  may invoke modules multiple times, in any order, and interleave them with **external tools/APIs**; the whole trajectory is a sequence of **textual and tool traces** (reasoning, queries, tool outputs, evaluator logs) that you can serialize and read like a transcript

GEPA

.

**Why this matters for GEPA:** because those traces are language, the system can _reflect in language_ over what happened and use that to improve modules (their prompts) without touching model weights

GEPA

.

* * *

2) What exactly is a language module  $M_i$ ?
=============================================

Each module is defined as a 4-tuple:

$$
M_i = (\pi_i,\, \theta_i,\, X_i,\, Y_i)
$$

with concrete meanings in the paper

GEPA

:

1.  ** $\pi_i$ ** — the **prompt** (the “instruction sheet”): the system message, constraints, and possibly demonstrations that steer how this module behaves.
2.  ** $\theta_i$ ** — the **model weights** the module uses. In most experiments here, **all modules share the same base LLM** (e.g., Qwen3-8B or GPT-4.1-mini), so  $\theta_i$  is the same across modules; you can conceptually keep them separate because, in other settings, you _could_ assign different models per module
    GEPA
    ; see also the model instantiation note that every program is built once per model and all modules rely on it
    GEPA
    .
3.  ** $X_i$ ** — the **module’s input schema** (the fields it expects).  
    Examples from the paper’s systems include fields like `question`, `summary_1`, `claim`, `passages`, or constraint descriptions for instruction following.
4.  ** $Y_i$ ** — the **module’s output schema** (the fields it returns).  
    Examples include `query` (for a retrieval hop), `summary` (for a summarizer hop), `answer` (final QA), or a redacted request in a privacy system.

> **Intuition:** A module is a _typed function_ implemented by an LLM call:
> 
> $$
> M_i: X_i \rightarrow Y_i \quad\text{(parametrized by the prompt \(\pi_i\) and model \(\theta_i\))}
> $$
> 
> The compound program wires these typed functions together via  $C$ .

**How GEPA uses modules.**

*   GEPA’s primary optimization **edits  $\pi_i$ ** (the module’s instructions), _not_  $\theta_i$  (weights). It runs rollouts, reflects on the traces, and proposes a new  $\pi_i'$  for a selected  $M_i$ , creating a new candidate program  $\Phi'$  with only that module changed
    GEPA
    .
*   In contrast, the GRPO baseline in the paper **updates  $\theta$ ** (via LoRA) for the shared LLM and uses a scalar reward signal across many rollouts
    GEPA
    .

* * *

3) How modules are orchestrated by  $C$  (with real examples)
=============================================================

The control flow  $C$  connects module outputs to later module inputs, optionally looping for multiple “hops”:

*   **Multi-hop retrieval (HoVer / HotpotQA).** The program includes **query-writer modules** and **document summarizer modules**; it can perform up to **3 hops** by alternating between writing a new query and summarizing what came back, before a final answer step. HoVer uses **2 query writers + 2 summarizers** as explicit modules
    GEPA
    .
    *   Example module signatures:
        *   `create_query_hop2: (question, summary_1) → (query)`
        *   `summarize1: (question, passages) → (summary)`
    *   The **diagram on p.4 (Fig. 3)** shows this iterative proposal–evaluate loop at the optimizer level, while **the prompt box on p.3 (Fig. 2)** shows how GEPA rewrites the _second-hop query module’s_ instruction from a vague template to a detailed rule-set (“don’t paraphrase the original question; target the _missing_ context”)
        GEPA
        .
*   **Instruction following (IFBench).** A 2-stage pipeline:
    1.  **Answer generator module** produces a draft.
    2.  **Constraint enforcer module** rewrites it to satisfy precise output constraints (e.g., exact words, counts, casing). Feedback identifies which constraints were violated or satisfied, feeding the _module-specific_ update
        GEPA
        .
*   **Privacy-aware delegation (PUPA).** Two modules on a trusted model:
    1.  **Redaction module** converts a user query into a **privacy-preserving request** to an untrusted model.
    2.  **Response rewriter** cleans the untrusted reply before returning to the user. The evaluator decomposes the reward into **quality vs PII-leakage**, a very useful module-level signal for updates
        GEPA
        .

In all of these,  $C$  can **re-invoke modules**, branch, and interleave tool calls (e.g., retrieval). The important point is that **every step leaves a _textual trace_** (LLM thoughts, tool outputs, evaluator messages) that GEPA can read to perform **credit assignment** (“which module’s instruction should change, and _how_?”)

GEPA

.

* * *

4) What goes inside  $\pi_i$  (the module prompt) and how GEPA changes it
=========================================================================

A module’s prompt  $\pi_i$  typically contains:

*   **Role & task framing**: “You are hop-2 query writer in a multi-hop retriever…”
*   **I/O contract**: which input fields matter; what output field(s) to produce (e.g., only `query`, or produce `summary` not an answer).
*   **Domain rules / strategies**: the _know-how_ distilled from reflection (e.g., “do not simply restate the question; aim at the _broader entity_ hinted by `summary_1`”).
*   **Formatting constraints**: “Output exactly one line named `query` …”.

**GEPA’s mutation step** uses traces from recent rollouts and a **meta-prompt** (Appendix B) to rewrite  $\pi_i$  so it explicitly encodes _lessons learned_ (“key observations and pitfalls,” “how to build the query,” etc.). Compare the seed vs optimized hop-2 prompt on **p.3 (Fig. 2)**—the optimized version adds concrete tactics for finding _missing_ documents and warns against paraphrasing

GEPA

. The paper also shows a **step-by-step prompt evolution** in PUPA (privacy task) where successive nodes add stricter redaction rules, justification requirements, and bans on partial redaction (see **p.7, Fig. 5** and Appendix G extracts)

GEPA

.

* * *

5) Inputs/outputs  $X_i, Y_i$ : typed, small, and task-specific
===============================================================

The module interface is deliberately **small and typed**, so modules compose cleanly:

*   **Retrieval hop writer:** `X = {question, summary_1}` → `Y = {query}`
*   **Summarizer hop:** `X = {question, passages}` → `Y = {summary}`
*   **Final answerer:** `X = {question, summary_1, summary_2}` → `Y = {answer}`
*   **IFBench enforcer:** `X = {query, draft_response}` → `Y = {corrected_response}`
*   **PUPA redactor:** `X = {user_query}` → `Y = {redacted_request}`

This typing is visible throughout the program descriptions in §4.1 and in the optimized prompt boxes in Appendix I (per-module prompts are listed with their I/O fields)

GEPA

.

* * *

6) Weights  $\theta_i$ : shared vs per-module, and how the paper handles them
=============================================================================

*   In this work, **all modules in a program share the same LLM** and decoding settings per experiment (e.g., Qwen3-8B with temp 0.6, or GPT-4.1-mini with temp 1.0)
    GEPA
    .
*   **GEPA** edits ** $\pi$ ** across modules; **GRPO** baseline updates ** $\theta$ ** via **LoRA** (fixed 24k rollouts)
    GEPA
    .
*   The framework is general: you _could_ mix prompt and weight updates; the paper calls the boundaries between prompt-space and weight-space learning an open area (see Limitations/Future Work)
    GEPA
    .

* * *

7) How modules receive feedback (and why this makes optimization sample-efficient)
==================================================================================

The evaluator metric  $\mu$  is expanded to a **feedback function  $\mu_f$ ** that returns **scores + textual feedback**, often **at the module’s granularity**:

*   In multi-hop retrieval, feedback includes **which gold docs are still missing at each hop**, helping credit the _query writer_ rather than, say, the summarizer
    GEPA
    .
*   In IFBench, feedback lists **which constraints were satisfied/violated**, ideal for the **constraint-enforcer module**
    GEPA
    .
*   In PUPA, the metric decomposes into **quality vs PII leakage**, steering **redaction** and **rewrite** modules precisely
    GEPA
    .

That module-aware, text-rich feedback is the primary reason GEPA can learn strong rules from **just a handful of rollouts**, instead of needing tens of thousands of scalar rewards

GEPA

.

* * *

8) Summary: mental model of a module
====================================

*   Think of **each module as a typed LLM tool** with a small API; its **behavior is governed by  $\pi_i$ **, and it runs on shared ** $\theta$ ** (the backbone LLM).
*   The **program**  $\Phi$  wires these tools, possibly looping, branching, and calling external APIs/tools, **leaving a rich text trace**.
*   **GEPA** repeatedly picks a module, reads the trace, **rewrites  $\pi_i$ ** with the distilled rule(s), and keeps the change only if it improves validation—while keeping a **Pareto set** of the best diversified variants across training instances to avoid local optima
    GEPA
    .

If you want, I can sketch a **mini typed spec** for one of the paper’s pipelines (e.g., HoVer 3-hop retriever + summarizers + answerer) showing  $X_i, Y_i$  for each  $M_i$ , and an example **before/after** of a real prompt mutation pulled from the figures to make it concrete.

## Prompt:
This was too complicated simplify it with analogies

## Response:
Got it 👍 — let’s throw away the heavy math and make it super intuitive with analogies.

* * *

🧩 Compound AI System = A Team of Specialists
=============================================

Imagine you run a **newspaper newsroom**.

*   The **compound AI system** is the _whole newsroom_.
*   Each **language module  $M_i$ ** is like one **specialist journalist** with a role:
    *   One person is the **fact-checker**.
    *   Another is the **headline writer**.
    *   Another is the **summary writer**.
    *   Another is the **privacy editor** who redacts names.
*   The **control flow  $C$ ** is like the **editor-in-chief** who decides:  
    “First let the fact-checker check sources, then send it to the summary writer, then pass it to the privacy editor before publishing.”

So the newsroom takes some **input** (a story idea) and outputs a **finished article**.

* * *

🔍 What’s inside each module (the journalists)
==============================================

Each module has four parts (the paper calls it  $(\pi, \theta, X, Y)$ , but let’s stay with the analogy):

1.  **Prompt ( $\pi$ ) = Job description.**
    *   Like telling the journalist: _“You are the fact-checker. Your job is to verify names and dates.”_
    *   For the summary writer: _“Write a short 3-sentence summary of the key facts.”_
2.  **Weights ( $\theta$ ) = The journalist’s brain.**
    *   In practice, all the journalists may share the **same brain** (the same LLM), but the _job description_ changes how they behave.
3.  **Inputs ( $X$ ) = What they get on their desk.**
    *   Fact-checker gets: the claim + some retrieved documents.
    *   Summary writer gets: the question + a few passages.
4.  **Outputs ( $Y$ ) = What they hand back.**
    *   Fact-checker hands back: “True/False” plus evidence.
    *   Summary writer hands back: a 3-sentence summary.

* * *

🔄 How GEPA changes the newsroom
================================

Normally, reinforcement learning (RL) is like the editor just giving each journalist a **score at the end**:

*   “Final article = 6/10.”
*   No one knows _who messed up_ or _how to improve_.

GEPA instead lets the newsroom **talk through what happened**:

*   “The fact-checker forgot to check dates.”
*   “The summary was just a paraphrase, not new info.”

Then it rewrites the job description ( $\pi$ ) for the journalist:

*   Instead of _“Check facts,”_ it says: _“Specifically check dates and make sure to use the missing document, not just repeat the question.”_

* * *

🌱 Why this matters
===================

*   **RL dog-training analogy:** RL is like training a dog — you only give it a treat or not. It takes **tons of repetitions**.
*   **GEPA student analogy:** GEPA is like a student keeping a diary — after each test, they write: _“I lost points for forgetting to label graphs. Next time, I’ll add labels.”_ That’s **fewer tries, faster learning**.

* * *

✅ **In short:**  
A compound AI system is a **team of mini-experts (modules)** working together under an **editor (control flow)**. GEPA improves them by **rewriting their job descriptions based on reflections** instead of just giving a single end-score.

* * *

Would you like me to also make a **diagram-style walkthrough** of one real example from the paper (like the multi-hop QA system), showing each “journalist” and how GEPA rewrote their job descriptions step by step?

## Prompt:
Can you explain what the Pareto frontier scoring is doing and how it works? Are they getting running each model on multiple different tasks and comparing the scores?

## Response:
Great question! Let’s make Pareto frontier scoring simple with an analogy and then tie it back to what GEPA is doing.

* * *

🍎 Analogy: Cooking Contest With Many Judges
============================================

Imagine you’re running a **cooking contest** with **four judges**.

*   Judge A cares about _taste_.
*   Judge B cares about _presentation_.
*   Judge C cares about _healthiness_.
*   Judge D cares about _originality_.

Each chef (candidate system) submits a dish (a prompt configuration).

*   One chef might score **10/10 on taste** but only **5/10 on presentation**.
*   Another might score **8/10 on taste** but **8/10 on presentation**.

Now — which chef is “best”?

👉 There’s no single answer. Instead, you keep the **Pareto frontier**: the set of chefs who are the _best at something_ without being completely worse than someone else.

*   If Chef 1 beats Chef 2 in _every_ category, Chef 2 is thrown out.
*   If Chef 2 is the best in at least one category, they’re kept on the frontier.

This way you preserve **diverse winning strategies** (spicy but ugly vs. beautiful but bland) and avoid getting stuck on just one style.

* * *

⚙️ How GEPA uses Pareto Frontier
================================

Back to AI systems:

*   GEPA runs **many candidate prompt variants** across **all training examples** (think of these examples as the judges).
*   Each example gives a **score** (like EM, F1, pass/fail).
*   For each training example, GEPA records: _Which candidate did best here?_

👉 The set of “best-so-far” candidates across all examples is the **Pareto frontier**.

*   Example: Candidate A might be best on _task instance 1_, Candidate B might be best on _task instance 2_. Both stay in the frontier.

**Then GEPA samples from this frontier** to decide which candidate to mutate next.

*   It doesn’t just always pick the global best average (which risks local optima).
*   Instead, it stochastically picks from this **diverse pool of local winners**, ensuring the search keeps exploring different strategies.

* * *

📊 So yes — multiple tasks/examples
===================================

To your question: **Yes, they are running each candidate system on multiple different training tasks/examples and comparing scores.**

*   Each **task instance** = one training example (e.g., one HotpotQA question, one IFBench instruction, etc.).
*   A candidate program is run on many such instances, producing a **score vector**.
*   GEPA uses this score matrix to build the Pareto frontier: for each instance, which candidates are non-dominated (i.e., no one else is strictly better across all instances)?

* * *

🎯 Why this helps
=================

*   Prevents “tunnel vision”: If you only pick the average-best candidate, you might overfit one style and miss better general strategies.
*   Encourages diversity: Each candidate that’s the “winner” on _some_ instance gets a chance to contribute its lessons.
*   Keeps search efficient: You don’t explore _everything_, only the non-dominated “winners.”

* * *

✅ **In short:**  
The Pareto frontier in GEPA is like keeping a roster of the best performers across different tasks/examples. Even if one candidate isn’t the absolute top overall, if it shines on _some_ example, it stays in the pool. GEPA then mutates from this diverse pool to steadily evolve stronger and more general prompts.

* * *

Do you want me to also **sketch a toy example with numbers** (like 3 candidates × 3 tasks, showing who ends up on the Pareto frontier)? That could make the mechanics crystal clear.

## Prompt:
How does it choose from the GEPA samples on the frontier to mutate next if there are multiple samples that performed well on different metrics?

## Response:
Awesome follow-up — you’re getting right to the heart of how GEPA balances _exploration vs. exploitation_. Let’s walk through it step by step, using a simple analogy and then the actual mechanics from the paper.

* * *

🍊 Analogy: Picking Athletes for Training
-----------------------------------------

Imagine you run a sports academy:

*   One athlete is great at **sprinting** but average at **long jump**.
*   Another is best at **long jump** but weak at **sprinting**.
*   A third is solid at both, but not the best at either.

If you always train just the “average best” athlete, you risk missing the superstar sprinter or jumper.  
So instead, you:

1.  Keep both sprinter and jumper (the **Pareto frontier**).
2.  Count how often each athlete is the “best” across different events.
3.  Pick athletes to train with probability proportional to how often they are winners.

This way, the all-rounder doesn’t dominate, and the niche specialists still get attention.

* * *

⚙️ GEPA’s Actual Process
------------------------

When GEPA has a pool of candidates (prompt sets), it evaluates them across all training instances:

*   For each instance, find the **top score achieved so far**.
*   Collect all candidates that achieved that top score for that instance.
*   This builds an **instance-wise Pareto set** — the roster of “winners” for each example.

Now, to choose who to mutate next:

1.  **Remove dominated candidates**: if Candidate B does as well or better than Candidate A on _every_ instance, Candidate A is dropped.
2.  **Count appearances**: For each remaining candidate, count how many instances it is a winner for.
3.  **Stochastic sampling**: Pick a candidate with probability proportional to this count.

👉 So if Candidate X is best on 7 examples and Candidate Y is best on 2, Candidate X is ~3.5× more likely to be chosen, but Candidate Y _still has a chance_.

* * *

🎯 Why This Works
-----------------

*   **Avoids tunnel vision**: Doesn’t just pick the current global average best.
*   **Encourages diversity**: Keeps candidates that win on niche examples.
*   **Balances fairness**: Stronger candidates are more likely, but weaker “specialists” aren’t ignored.

* * *

✅ **In short:**  
GEPA samples from the Pareto frontier by weighting candidates based on **how often they were the best across different training tasks**. This way, it mutates both the general all-stars and the niche experts, steadily combining their strengths into stronger prompt lineages.

* * *

Would it help if I cooked up a **toy example with 3 candidates × 5 tasks and showed exactly how the counts and probabilities are computed**? That might make the selection step super concrete.

## Prompt:
What is the ultimate model at the end then?

## Response:
Great question — this gets at what GEPA is actually “building” by the end of optimization. Let’s break it down.

* * *

🏗️ GEPA’s Ultimate Output
--------------------------

At the end of the optimization loop, GEPA produces **a single compound AI system  $\Phi^*$ ** that has:

*   The **same underlying LLM weights** ( $\theta$ ) you started with (e.g., GPT-4.1-mini or Qwen-8B).
*   A set of **optimized prompts** ( $\pi_i^*$ ) for each module in the system.
*   The **same control flow**  $C$  (the orchestration logic doesn’t change).

So the _ultimate model_ is **the original LLM system, but with much better prompts for each module, chosen from the evolutionary search**

GEPA

.

* * *

🧩 How It Chooses the Final Version
-----------------------------------

1.  During optimization, GEPA generates a **pool of candidates** (different prompt sets for the modules).
2.  Each candidate is tested on a held-out validation set (**Dₚₐᵣₑₜₒ**).
3.  At the end, GEPA returns the **candidate with the best aggregate performance on validation**
    GEPA
    .

So it’s not just keeping multiple variants forever — it ends up with **one best prompt configuration** across modules.

* * *

🍳 Analogy: Cooking Contest Again
---------------------------------

Think of GEPA as a “cooking bootcamp” for chefs (modules):

*   Each week, chefs try new recipes (prompt versions).
*   You keep track of which dishes score best across judges (instances).
*   At the end, you don’t serve all the past experiments — you serve the **winning recipe** that consistently did best overall.

* * *

⚙️ Example: Multi-Hop QA System
-------------------------------

Suppose the system has three modules:

1.  Query-writer (hop 1)
2.  Query-writer (hop 2)
3.  Answer synthesizer
    *   GEPA evolves prompts for each of these.
*   The final model is the **same three-module pipeline**, but with prompts rewritten like:
    *   Query-writer (hop 2): “Don’t paraphrase the original question; instead target missing, logically linked documents.”
    *   Answerer: “Use summaries 1 and 2, compare consistency, and give a short yes/no or name.”

This tuned set of instructions makes the system much stronger — without ever retraining the LLM itself.

* * *

✅ **So the ultimate model = the same backbone LLM, wired into the same system, but with a set of “evolved job descriptions” (prompts) for each module.**

* * *

Would you like me to also contrast this with the **RL baseline (GRPO)** so it’s clearer how GEPA’s “final model” differs from RL’s “final model” (i.e., optimized weights vs optimized prompts)?

## Prompt:
How does GEPA know what is the final prompt if it is evaluating prompts on multiple tasks?

## Response:
Great question — this is exactly the tricky part: if GEPA is juggling **lots of different prompts that are each good on different tasks**, how does it decide on **one final prompt set** at the end?

* * *

🧩 Step 1: During training (Pareto frontier stage)
--------------------------------------------------

*   While GEPA is exploring, it **keeps many candidates** (different prompt configurations).
*   Each candidate is scored on **every training task instance**.
*   The Pareto frontier is used only to decide _which candidates to evolve next_, so that niche winners aren’t thrown away.  
    👉 This is the _exploration phase_ — diversity matters.

* * *

🏁 Step 2: At the end of training (choosing the winner)
-------------------------------------------------------

*   GEPA doesn’t keep multiple prompts for deployment.
*   Instead, it evaluates every surviving candidate on the **validation set** (a separate set of examples it hasn’t been trained on).
*   It then picks the **single candidate (prompt set across all modules)** that has the **highest average validation performance**
    GEPA
    .

So while Pareto scoring is used during the search to balance exploration, the **final decision is made by a simple “who scores best on validation overall” rule.**

* * *

🍎 Analogy: School Exams
------------------------

Think of it like a student trying different study strategies:

*   Some strategies are great for math, others better for writing.
*   During practice, the teacher keeps them all around so the student can mix and match lessons.
*   But for the **final exam**, the student picks the one overall study plan that seems to work best across _all subjects_.

* * *

⚙️ In GEPA’s case
-----------------

*   The **final prompt** is not per-task.
*   It’s the **single system-wide prompt set** (one instruction per module) that had the **best balance of performance** across all validation tasks.

That’s why the paper reports a single **GEPA score per benchmark** (e.g., HotpotQA or HoVer) — because in the end, you get one “best evolved program.”

* * *

✅ **In short:**  
GEPA uses the Pareto frontier to keep diversity during _search_, but for the _final output_ it simply picks the **single best-performing candidate prompt set across the validation tasks**.

* * *

Would you like me to build a **tiny toy example with numbers** (say, 3 prompt candidates × 3 tasks) showing how some stay on the frontier, then how GEPA finally chooses the “winner” on validation? That could make the process super concrete.

## Prompt:
Is the Pareto frontier data and validation data separate?

## Response:
Yes — they are separate in GEPA.

* * *

📂 How the paper splits data
----------------------------

For each benchmark (HotpotQA, IFBench, HoVer, PUPA), the dataset is split into three parts

GEPA

:

1.  **Train set** → used for generating rollouts and feedback (GEPA “learns” here).
2.  **Validation set** → used for _Pareto scoring_ (to decide which candidates are “winners” on which examples during the search).
3.  **Test set** → completely held out until the end, used only for final reporting.

* * *

🧩 Why this matters for GEPA
----------------------------

*   **Feedback set (train)** = where GEPA gets its “diary notes.”
    *   It runs minibatches here, collects traces, and reflects to propose better prompts.
    *   This is where the _mutation step_ happens.
*   **Pareto set (validation)** = where GEPA measures which candidates are good at which instances.
    *   It doesn’t rewrite prompts here, it just _scores_ them.
    *   Used to build the Pareto frontier and decide which candidates to evolve next.
*   **Test set** = untouched until the end, to see if the final prompts generalize.

* * *

🍎 Analogy: Practice vs. Mock Exams vs. Final Exam
--------------------------------------------------

*   **Practice problems** (train set): the student works through them, makes mistakes, and writes reflection notes (“oops, I forgot to check units”).
*   **Mock exams** (validation/Pareto set): every strategy (candidate prompt) is tested on the mock exam to see which ones do best on each question. The Pareto frontier keeps the strongest ones alive.
*   **Final exam** (test set): the very last evaluation, no touching during training.

* * *

✅ **So yes:** GEPA separates the _feedback data_ (used for reflective learning) and the _Pareto validation data_ (used to score and select candidates). Both are distinct, and then the final model is evaluated only on the _test set_.

* * *

Would you like me to **sketch a concrete toy example** (say, 10 training questions, 5 validation questions, 5 test questions) to show how GEPA uses each split differently step by step?

## Prompt:
How do they split up the Pareto validation data into different tasks?

## Response:
Great question — the wording in the paper can be confusing because “tasks” can mean two things: **different benchmarks** (HotpotQA, IFBench, HoVer, PUPA) or **individual problem instances** inside one benchmark. GEPA’s Pareto frontier operates at the _instance level_ inside a single benchmark run.

* * *

🧩 How the Pareto validation data is organized
----------------------------------------------

When GEPA runs on, say, **HotpotQA**, it splits the dataset like this

GEPA

:

*   **Train / feedback set** → e.g. 150 examples. Used for generating rollouts and reflection.
*   **Validation / Pareto set (Dₚₐᵣₑₜₒ)** → e.g. 300 examples. Used for scoring candidates.
*   **Test set** → e.g. 300 examples. Held out for final evaluation.

Within the **Pareto set**, each **example (question)** is treated as its _own mini-task_.

* * *

⚙️ What “tasks” means in Pareto scoring
---------------------------------------

*   For each validation example, GEPA looks at all candidate systems and records **who got the best score** on that example.
*   That example acts like a “judge” in the cooking analogy — awarding a “gold medal” to whichever candidate did best there.
*   A candidate can win on some examples but lose on others.

👉 The **Pareto frontier** is then built by keeping every candidate that is a winner on _at least one validation example_.

* * *

🍎 Analogy: Sports League
-------------------------

Think of the validation set as a **league season** with 300 matches.

*   Each candidate prompt set is like a team.
*   In each match (validation example), whichever team scores highest is marked as a “winner” for that match.
*   At the end of the season, the **Pareto frontier** is the collection of all teams that were league-leaders in at least one match.

* * *

✅ Summary
---------

*   The **Pareto validation set** is not split into categories like “reasoning” vs. “retrieval” vs. “privacy.”
*   Instead, it is a **collection of individual examples** from the benchmark.
*   Each example is a “task instance,” and candidates are compared across all of them.
*   Candidates that win on at least one instance survive on the Pareto frontier; dominated ones (never the best anywhere) get discarded.

* * *

Do you want me to mock up a **tiny table (say 3 candidates × 5 validation examples)** showing exactly how GEPA builds the Pareto frontier and picks which candidates to keep? That would make the mechanics really concrete.

## Prompt:
Can you explain the GEPA inference section more? I'm confused about that

## Response:
Perfect — the **GEPA inference-time section** is a bit different from the rest of the paper, so let’s slow it down and use analogies.

* * *

🌱 What the earlier GEPA sections were about
============================================

*   Normally, GEPA is used like **training or tuning**:
    *   It has a **train set** → to reflect and propose new prompts.
    *   It has a **validation set (Pareto set)** → to decide which candidates to keep evolving.
    *   Then it outputs the single best program and tests it on a **held-out test set**.
*   The goal there: **generalize** to new, unseen problems.

* * *

🚀 What the “Inference-time GEPA” section changes
=================================================

Here, they flip the idea:

*   Suppose you don’t care about generalization to new data.
*   Instead, you just want to **squeeze the best possible performance on a fixed set of tasks right now**.

So GEPA is run at **inference-time** — on the _exact tasks you want solved_.

👉 It’s like a student who normally studies _practice problems_ to prepare for an exam.  
But in inference-time GEPA, the “student” is allowed to **practice on the actual exam questions**, iteratively improving answers until they’re great.

* * *

⚙️ How it works at inference
============================

*   You give GEPA the **tasks you want solved** (e.g., “generate CUDA kernels for this list of 35 functions”).
*   GEPA treats those tasks as _both_ the training set and the Pareto set.
*   It runs multiple rollouts → reflects → rewrites prompts → tests again → keeps improvements.
*   Over iterations, prompts accumulate useful rules like _“use shared memory,”_ or _“avoid unnecessary global memory accesses.”_
*   At the end, you get a final prompt (or a set of prompts) that performs much better on those exact tasks.

* * *

🖥️ Examples from the paper
===========================

### 1\. **NPU kernels (AMD XDNA2, NPUEval benchmark)**

GEPA

*   A baseline “sequential refinement agent” (just retrying code 10 times) barely got **4.25% utilization**.
*   Add retrieval-augmented generation (RAG) from manuals → **16.33%**.
*   Add RAG + MIPRO → **19.03%**.
*   **GEPA prompts alone** (no RAG) → mean of **30.52%**, with some kernels hitting **70% utilization**!

👉 GEPA figured out better coding strategies just by **reflecting on compiler errors and profiling outputs** and encoding that knowledge into new prompts.

* * *

### 2\. **CUDA kernels (KernelBench, NVIDIA V100)**

GEPA

*   Task: generate efficient CUDA kernels that run faster than PyTorch’s baseline.
*   With GEPA at inference-time, the fraction of tasks where the generated kernel was faster (**fast₁ score**) climbed above **20%**.
*   Without GEPA, GPT-4o alone was basically near 0%.

👉 Again, by reflecting on code errors and feedback, GEPA distilled rules like _“vectorize loops”_ or _“check memory alignment”_ into prompts.

* * *

🍎 Analogy: Coach During the Game
=================================

Normally: a coach trains athletes before the competition.  
Inference-time GEPA: the coach is **allowed to whisper advice during the actual match**, iteratively refining plays based on mistakes until the team wins.

* * *

✅ Key differences from training use
===================================

*   **Training GEPA:** optimize prompts on train data, validate on held-out Pareto set, and hope it generalizes.
*   **Inference-time GEPA:** directly optimize on the _actual target tasks_. It may “overfit” (that’s fine here, because you only care about those tasks).
*   Uses the same loop: reflection, mutation, Pareto selection — but applied directly to the tasks at hand.

* * *

👉 **In short:** GEPA inference-time mode is like running the optimizer _live_ on the problems you care about, iteratively rewriting prompts using the task feedback until performance improves — almost like an interactive search for the best instructions.

* * *

Would you like me to **sketch a mini toy example** (say, writing a sorting function where GEPA iteratively rewrites the prompt after seeing compiler/runtime errors) so you can see how inference-time GEPA plays out step by step?

## Prompt:
How does GEPA know what is good and what is bad at inference type? Is it based on a reward policy like RL? But I thought GEPA wasn't RL?

## Response:
Excellent question 🙌 — this is where GEPA is subtle. You’re right: **GEPA is not RL** in the sense of doing policy gradient updates on weights. But it _still needs feedback_ to know what is “good” vs. “bad” at inference time.

* * *

⚖️ Where GEPA’s feedback comes from
===================================

GEPA uses a **feedback function**  $\mu_f$ , which is basically an **evaluation metric plus any textual signals**.

*   For QA: metrics like **Exact Match / F1** against the gold answer.
*   For code: whether the code **compiles**, runs correctly on **unit tests**, or how fast it runs.
*   For privacy: whether there’s **PII leakage** plus quality of response.
*   For instruction-following: which constraints were satisfied or violated.

👉 Crucially, these metrics **don’t have to be scalar only**. They often include **textual traces** (like compiler error messages, “constraint X failed,” “document Y missing”), which GEPA feeds back into the reflection step

GEPA

.

* * *

🧩 Difference vs. RL
====================

*   **RL:** Compresses all of that into a **single reward number** and backpropagates through policy gradients to update weights. Needs thousands of rollouts.
*   **GEPA:** Keeps the rich **language feedback** intact. The model reads:
    *   “Your query just paraphrased the original question, missing new info.”
    *   “Compiler error: mismatched data type float vs int.”
    *   “Constraint failed: must include the word ‘yes’ three times.”  
        and then rewrites the prompt accordingly.

So instead of a scalar reward → gradient step → new weights, it’s:  
**textual feedback → prompt rewrite → new candidate.**

* * *

🍎 Analogy
==========

Think of two ways to train a chef:

*   **RL style:** You give them only a number after each dish: “7/10.” The chef has to guess what to improve. Takes forever.
*   **GEPA style:** You tell them _why_: “Too salty, plating was messy, dessert was perfect.” The chef can immediately adjust the recipe and instructions.

* * *

🚀 At inference-time specifically
=================================

When GEPA is used at inference-time (e.g., code generation):

*   It runs the candidate prompt on the _actual tasks_.
*   The **environment produces feedback** (e.g., compiler says “syntax error in line 12,” or profiler says “only 4% vector utilization”).
*   GEPA reflects on this feedback in natural language and proposes new prompt versions.
*   The Pareto frontier ensures it keeps any candidate that “wins” on at least one task, instead of converging too quickly.

So:

*   It’s not reinforcement learning.
*   It’s more like **search + reflection, guided by textual feedback from the environment**.

* * *

✅ **In short:**  
GEPA knows what’s good vs. bad based on **evaluation metrics + textual feedback from the environment**, not a scalar reward policy. That feedback is directly fed into reflection to rewrite prompts, rather than into gradient updates like in RL.

* * *

Do you want me to show you a **step-by-step toy example** (say, GEPA trying to generate a piece of code, failing compilation, then rewriting the prompt based on the error message)? That might make the inference-time loop click.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)