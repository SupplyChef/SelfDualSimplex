# Self-Dual Simplex — Algorithm Blueprint

## 1. How SDS Works (in Words)

### The Core Idea

We want to solve the LP: **min c·x  s.t. Ax ≤ b, x ≥ 0**.

The Self-Dual Simplex embeds this into a one-parameter family of LPs:

> **min (c + t·p)·x  s.t. Ax ≤ b + t·q, x ≥ 0**

where `p` (dual perturbation) and `q` (primal perturbation) are random positive vectors.
At **t = 0** this is the original LP.  
At **large t** the slack variables form an optimal+feasible basis (slacks are free in cost so they're optimal; `b + t·q ≥ 0` for large enough t so they're feasible).

The algorithm starts with the slack basis at large t and pivots to decrease t toward 0.  
At each step the basis remains **simultaneously** primal and dual feasible for the current t.  
When t ≤ ε the basis is feasible and optimal for the original problem.

### The State

At any basis B the state consists of four vectors maintained across iterations:

| Variable | Meaning | Formula |
|---|---|---|
| `b_hat` | Basic variable values | `B⁻¹ b` |
| `perturbation_b_hat` | Basic perturbation values | `B⁻¹ q` |
| `c_hat` | Reduced costs (all columns) | `c - cᴮ B⁻¹ A` |
| `perturbation_c_hat` | Perturbed reduced costs | `p - pᴮ B⁻¹ A` |

The parametric state at parameter t is:
- `b_hat(t)[i] = b_hat[i] + t · perturbation_b_hat[i]` — basic variable value
- `c_hat(t)[j] = c_hat[j] + t · perturbation_c_hat[j]` — reduced cost

**Invariants maintained at all times**: both `b_hat(t) ≥ 0` (primal) and `c_hat(t) ≥ 0` (dual) hold for every nonbasic nonbasic variable. Because both hold simultaneously, t itself acts as the objective function: driving t to 0 solves the original LP.

### The Parametric Parameter t

`t = max(t_b, t_c)` where:
- `t_b = max_i { −b_hat[i] / perturbation_b_hat[i] : b_hat[i] < 0 }` — the largest t at which any basic variable is still (parametrically) primal-infeasible
- `t_c = max_j { −c_hat[j] / perturbation_c_hat[j] : c_hat[j] < 0, j nonbasic }` — same for dual

When `t ≤ ε`: all infeasibilities are within tolerance → return the solution.

### Step Type Selection

**Dual step** (when `t_b > t_c`): the binding constraint on t comes from primal infeasibility at row `leaving`. Row `leaving` has `b_hat[leaving] + t_b · perturbation_b_hat[leaving] = 0`: it would become primal feasible exactly at `t = t_b`. We pivot `basis[leaving]` OUT and bring in a new column `j`. The column `j` is chosen by the dual ratio test so that dual feasibility is preserved as t decreases through `t_b`.

**Primal step** (when `t_c ≥ t_b`): the binding constraint comes from dual infeasibility at column `j`. Column `j` has `c_hat[j] + t_c · perturbation_c_hat[j] = 0`. We bring `j` INTO the basis. The leaving row is chosen by the primal ratio test so that primal feasibility is preserved.

### The Ratio Tests

**Dual ratio test** (find entering j given leaving row):
- Compute direction `Δc = B⁻ᵀ eₗ` scattered through A (the dual direction)
- For each nonbasic j where `Δc[j] < 0`, entering j would decrease c_hat[j]. The ratio `(c_hat[j] + t · perturbation_c_hat[j]) / (−Δc[j])` is the step size before c_hat(t)[j] hits 0.
- Select `j = argmin` of this ratio (tightest dual constraint).

**Primal ratio test** (find leaving row given entering j):
- Compute direction `Δb = B⁻¹ aⱼ` (the primal direction)
- For each basic row i where `Δb[i] > 0`, the value `b_hat(t)[i]` decreases as the step size grows. The ratio `(b_hat[i] + t · perturbation_b_hat[i]) / Δb[i]` is the step before that basic variable hits 0.
- Select `leaving = argmin` (tightest primal constraint).

### Basis Update

After selecting leaving/entering:

**Primal (b_hat) update** — one column of B changed:
```
t_step = b_hat[leaving] / Δb[leaving]
b_hat ← b_hat − t_step · Δb      (all rows shift)
b_hat[leaving] ← t_step           (leaving row set to ratio)
```
Same formula applied identically to `perturbation_b_hat`.

**Dual (c_hat) update** — one row of B⁻¹A changed:
```
s_step = c_hat[j] / Δc[j]
c_hat ← c_hat − s_step · Δc       (all columns shift)
c_hat[basis[leaving]] ← −s_step   (old leaving variable gets new reduced cost)
```
Same formula applied identically to `perturbation_c_hat`.

**Cleanup**: Basic variables must have `c_hat = 0` (by definition of reduced cost). Numerical drift is corrected by zeroing all `c_hat[basis[i]]` after each pivot.

### Factorization Maintenance (PFI)

The full LU factorization (`B = LU` via UMFPACK) is done infrequently. Between factorizations, each pivot appends an ETAMatrix η to the chain. Solving `Bx = v` becomes:
```
ftran!: U⁻¹ L⁻¹ η₁⁻¹ η₂⁻¹ … ηₖ⁻¹ v
btran!: v' ηₖ⁻ᵀ … η₁⁻ᵀ L⁻ᵀ U⁻ᵀ
```
Refactorization is triggered when `total_eta_nnz > 10m` OR `k ≥ pivot_cap`, resetting the chain and recomputing all four state vectors from scratch.

---

## 2. Pseudocode

```
SDS(A, b, c):
  # --- Initialization ---
  tA = Aᵀ                          # precompute for efficient scatter in Δc
  q = random_scaled(n_constraints)  # primal perturbation
  p = random_scaled(n_cols)         # dual perturbation

  basis = last n_constraints columns (slacks)
  B = LU(A[:, basis])               # initially identity
  b_hat     = B⁻¹ b                 # = b for slack basis
  c_hat     = c − cᴮ B⁻¹ A         # = c for slack basis
  pb_hat    = B⁻¹ q                 # = q for slack basis
  pc_hat    = p − pᴮ B⁻¹ A         # = p for slack basis

  # --- Main loop ---
  loop:
    # PRICING
    t_b, leaving = max_i { −b_hat[i] / pb_hat[i] : b_hat[i] < 0 }
    t_c, j       = max_j { −c_hat[j] / pc_hat[j] : c_hat[j] < 0, j nonbasic }
    t            = max(t_b, t_c)

    if t_b ≤ ε and t_c ≤ ε: return {basis[i] → b_hat[i]}

    if t_b > t_c:   # DUAL STEP
      # Dual direction: scatter row `leaving` of B⁻¹ through A
      el = B⁻ᵀ e_leaving          # btran
      Δc[j] = aⱼᵀ · el  ∀ j      # sparse scatter

      # Dual ratio test
      j = argmin { (c_hat[j] + t · pc_hat[j]) / (−Δc[j])
                   :  j nonbasic, Δc[j] < −ε }
      if no such j: throw Infeasible/Unbounded

      # Primal direction for update
      Δb = B⁻¹ a_j                 # ftran

    else:           # PRIMAL STEP
      # Primal direction
      Δb = B⁻¹ a_j                 # ftran

      # Primal ratio test
      leaving = argmin { (b_hat[i] + t · pb_hat[i]) / Δb[i]
                         :  Δb[i] > ε }
      if no such leaving: throw Infeasible/Unbounded

      # Dual direction for update
      el = B⁻ᵀ e_leaving           # btran
      Δc[j] = aⱼᵀ · el  ∀ j       # sparse scatter

    # DEGENERATE PIVOT RECOVERY
    if Δb[leaving] == 0 or Δc[j] == 0:
      regenerate (p, q) scaled by 1/100
      recompute (pb_hat, c_hat, pc_hat) via ftran/btran/mul
      continue

    # BASIS UPDATE
    η = ETAMatrix(leaving, B⁻¹ a_j sparse)
    push η to PFI chain

    t_step = b_hat[leaving] / Δb[leaving]
    b_hat  ← b_hat  − t_step · Δb  ;  b_hat[leaving]  = t_step
    pb_hat ← pb_hat − t_step_p · Δb  ;  pb_hat[leaving] = t_step_p
       where t_step_p = pb_hat[leaving] / Δb[leaving]

    s_step = c_hat[j] / Δc[j]
    c_hat  ← c_hat  − s_step · Δc  ;  c_hat[old_basis_leaving]  = −s_step
    pc_hat ← pc_hat − s_step_p · Δc  ;  pc_hat[old_basis_leaving] = −s_step_p
       where s_step_p = pc_hat[j] / Δc[j]

    basis[leaving] = j
    c_hat[basis[i]] = 0, pc_hat[basis[i]] = 0  ∀ i  (cleanup drift)

    # REFACTORIZATION
    if eta_fill > 10m or eta_count > cap:
      B = LU(A[:, basis])
      b_hat  = B⁻¹ b  ;  pb_hat = B⁻¹ q    (ftran)
      c_hat  = c − cᴮ B⁻¹ A  ;  pc_hat = p − pᴮ B⁻¹ A   (btran + mul)
```

---

## 3. Code Map (pseudocode → `SelfDualSimplex.jl`)

| Pseudocode section | Code location | Notes |
|---|---|---|
| `tA = Aᵀ` | line 169 `tA = sparse(Js, Is, ...)` | Precomputed once; used by `computeΔc2!` and `mul!` |
| Perturbation init | lines 196–200 | `c_scale`, `b_scale` set scale. Both structural and slack columns get perturbation. |
| Initial LU | line 189 `LUdecomposition(A, basis)` | Slack basis = identity; UMFPACK solves reduce to no-ops |
| `b_hat = B⁻¹ b` | line 193 `b_hat = copy(b)` | Exact since initial B = I |
| `c_hat = c − …` | line 194 `c_hat = copy(c)` | Exact since initial cᴮ = 0 (slack costs are 0) |
| **Pricing t_b** | lines 224–229 | `pb[i] = -b_hat[i] / perturbation_b_hat[i]` |
| **Pricing t_c** | lines 231–237 | `pc[i] = -c_hat[i] / perturbation_c_hat[i]` with `!is_basic[i]` guard |
| `t_b, leaving` | line 238 `max_argmax(pb)` | |
| `t_c, j` | line 239 `max_argmax(pc)` | |
| **Dual direction** | lines 267 `computeΔc2!(...)` | btran + sparse scatter via `tA`; slack cols handled separately |
| **Dual ratio test** | lines 272–279 | `pc[i] = (c_hat[i] + t*pc_hat[i]) / -Δc[i]` |
| **Primal direction** | lines 286–287 `get_column! + ftran!` | Dense column fetch + forward solve |
| **Primal ratio test** | lines 294–300 | `pb[i] = (b_hat[i] + t*pb_hat[i]) / Δb[i]` |
| **Primal direction** (primal step) | lines 291–292 | Same as dual step primal direction |
| **Dual direction** (primal step) | line 309 `computeΔc2!(...)` | After leaving is known |
| **Degenerate recovery** | lines 318–339 | `/100` rescale + full recompute via btran+mul |
| **ETAMatrix push** | lines 342–346 | `sparsevec + droptol(1e-10) + ETAMatrix` |
| **b_hat update** | line 348 `updateBasicVariables(b_hat, Δb, leaving)` | Dense axpy |
| **pb_hat update** | line 349 `updateBasicVariables(perturbation_b_hat, Δb, leaving)` | Dense axpy |
| **c_hat update** | line 351 `updateDualVariables(c_hat, Δc, j, leaving, basis)` | Dense axpy |
| **pc_hat update** | line 352 `updateDualVariables(perturbation_c_hat, ...)` | Dense axpy |
| **Cleanup c_hat** | lines 376–380 `for i in basis: c_hat[i] = 0` | Corrects drift |
| **Refactorization** | lines 383–415 | Trigger: `total_eta_nnz > 10m` OR `k ≥ cap` |
| `b_hat = B⁻¹ b` (refactor) | lines 391–393 `ftran!(pfi, b_hat)` | Full reset |
| `c_hat = c − cᴮ B⁻¹ A` | lines 401–410 `btran + mul!` | Two btrans + two `mul!` calls |

### `computeΔc2!` internals (lines 94–129)

```
fill(el, 0); el[leaving] = 1
btran!(pfi, el)                         # el ← B⁻ᵀ e_leaving

fill(Δc, 0)
for i where |el[i]| > eps:             # skip near-zero rows
  for (k, v) in tA column i:           # tA[j, i] = A[i, j]
    Δc[k] += el[i] * v                 # scatter into Δc

for i in slack range:
  Δc[i] = el[i - (n-m)]               # slack Δc = direct el value
```

Note the redundant `if i > c - b` check in the slack loop — the loop starts at `c-b+1` so this is always true.

---

## 4. Accuracy and Performance Analysis by Section

### A. Initialization

**Accuracy:**
- For the slack (identity) basis, `B⁻¹ = I`, so copying `b` and `c` directly is exact. No numerical error introduced.
- The variable shadowing on line 184 (`for b in basis`) shadows the parameter `b` (the RHS vector). Works by accident since the loop variable is used only as an index, but is confusing.
- Perturbation scaling by `b_scale = norm(b)/sqrt(m)` and `c_scale = norm(c)/sqrt(c_nz)` is sound: it makes the perturbation proportional to the problem magnitude so pricing ratios `b_hat[i] / perturbation_b_hat[i]` are O(1).

**Performance:**
- `LUdecomposition` on the initial slack basis is unnecessary: UMFPACK factorizes an identity matrix. This costs O(m) time and space instead of O(1). Consider skipping the LU for the initial slack basis and initializing PFI with an empty eta chain + `luf = nothing`.
- `tA` construction is one-time and cheap.

---

### B. Pricing

**Accuracy:**
- `perturbation_b_hat[i]` and `perturbation_c_hat[i]` must be **strictly positive** for the pricing ratios to have correct signs. Both start positive (scaled `1.0 + 5.0*rand()`), but the update formulas `axpy!(-t_step_p, Δb, perturbation_b_hat)` can make them negative over many iterations.
- If `perturbation_b_hat[i] < 0` and `b_hat[i] > 0` (primal feasible row), then `pb[i] = -b_hat[i] / perturbation_b_hat[i] > 0`. This row would be treated as primal-infeasible and selected as `leaving`, which is incorrect. **This is a primary cause of cycling.**
- If `perturbation_c_hat[j] < 0` and `c_hat[j] < 0` (dual-infeasible column), the ratio is `pc[j] > 0` — possibly selected, but the SDS ratio tests also depend on `perturbation_c_hat[j] > 0` being correct.
- The perturbation change (Section E) regenerates positive perturbations but only triggers on **exact-zero** pivots, not when perturbations drift negative.
- **Fix candidate**: After each pivot update, clip `perturbation_b_hat[i]` and `perturbation_c_hat[j]` to `max(value, small_positive)`. Or trigger regeneration when any perturbation value goes below a threshold.

**Performance:**
- Two O(m) and O(n) loops with conditional writes. Already fast.
- Could maintain separate lists of currently-infeasible rows and dual-infeasible nonbasic columns, updated incrementally after each pivot. For degenerate problems where most rows are feasible, this reduces pricing from O(m)+O(n) to O(# infeasible).

---

### C. Dual Direction (`computeΔc2!`)

**Accuracy:**
- The `abs(el[i]) > eps` guard (eps = 1e-8) in the scatter loop drops elements of `el` smaller than `1e-8`. This introduces a controlled approximation in `Δc`. For columns `j` that only receive contributions from these near-zero rows of `el`, `Δc[j]` may be computed as 0 when it should be small-but-nonzero. This could cause those columns to be excluded from the dual ratio test, potentially selecting a suboptimal entering variable.
- The slack column handling `Δc[i] = el[i - (n-m)]` **overwrites** any previously accumulated value. For slack columns this is correct (the A matrix column of slack i is `eᵢ`, so `Δc_slack[i] = el[i]`), but the code does this unconditionally, discarding any scatter contribution.

**Performance:**
- `btran!` is the dominant cost: one UMFPACK transpose-solve plus traversal of the entire eta chain (k matrices × avg nnz per eta).
- `fill!(Δc, 0.0)` zeros all n elements every iteration. Since `Δc` is set sparsely (via scatter + slack copy), most of these writes are redundant. **Maintaining an active-set index list** and zeroing only those entries would save ~n writes per iteration (for DEGEN3: 6660 × 6464 ≈ 43M wasted writes).
- The scatter loop `for j in tA.colptr[i]` exploits column-major structure of `tA = Aᵀ`. This is the correct access pattern for the row-of-A-times-vector product.
- The `if i > c - b` check in the slack loop at line 116 is dead code; the loop starts at `c-b+1`.

---

### D. Ratio Tests

**Accuracy (dual ratio test):**
- Uses `t = max(t_b, t_c) = t_b` in the dual branch. The ratio `(c_hat[j] + t_b · pc_hat[j]) / (-Δc[j])` correctly measures "how far can we decrease t before c_hat(t)[j] hits 0 in the current step direction." This is correct.
- The `Δc[j] < -eps` filter: only columns where the dual variable moves towards infeasibility participate. Columns with `|Δc[j]| < eps` are excluded. Since `eps = 1e-8`, this correctly avoids near-zero denominators.

**Accuracy (primal ratio test):**
- `Δb[i] > eps` filter: only rows where the basic variable decreases. Numerically stable since we need `Δb[i]` to be meaningfully positive.
- `b_hat[i] + t · pb_hat[i]` should be ≥ 0 for all basic rows at the current t. If `pb_hat[i] < 0` (drift), this could be negative even for primal-feasible rows (b_hat[i] > 0, small t), producing a ratio that's negative. `min_argmin` would pick the most-negative ratio as the leaving row. This is another manifestation of the perturbation drift bug.

**Performance (ratio tests):**
- Each ratio test is O(n) or O(m). No obvious improvement beyond maintaining infeasibility sets.
- The `fill!(pc, Inf64)` before the dual ratio test (and `fill!(pb, +Inf64)` before the primal ratio test) initialize the entire array. These could be avoided by tracking which entries were set, but the benefit is small.

---

### E. Degenerate Pivot Recovery

**Accuracy:**
- Only triggers on `Δb[leaving] == 0 || Δc[j] == 0` — exact zeros only. Near-zero pivots (|pivot| ∈ (0, 1e-8)) are not caught. A near-zero pivot would make the ETAMatrix ill-conditioned (small `eta_pivot`) and cause large factors in btran/ftran.
- The `/100` rescaling makes perturbation values smaller. After the regeneration, `t_b` and `t_c` can **increase dramatically** because t_b = b_hat[i] / new_pb_hat[i] where new_pb_hat is ~100x smaller. This means t can increase after a perturbation change, and the algorithm needs many more steps to drive t back to 0. If perturbation changes happen frequently, t oscillates upward — this is visible in DEGEN2's output.
- **Better approach**: Perturbation change should maintain the current t value or decrease it. One way: scale the new perturbation so that the new `t_b_new = t_b_old` (i.e., choose the new perturbation_b proportional to b_hat, making all infeasible rows at the same t ratio). This keeps t from jumping up.
- After the perturbation change, `c_hat` is recomputed from scratch (btran + mul), which is correct and also a useful numerical refresh. However, `b_hat` is NOT recomputed — it retains accumulated drift. An asymmetry that's probably intentional (b_hat is the "solution" state).

**Performance:**
- Rare event, but involves 2 btrans + 1 ftran + 2 muls. Acceptable cost.

---

### F. Basis Update

**Accuracy:**
- **`updateBasicVariables` uses DENSE `Δb`** (not the sparsified `sΔb` used for the ETAMatrix). This is correct and critical: the b_hat update must use the full numerical Δb. The droptol on `sΔb` only affects the ETAMatrix quality (future solves), not the b_hat values themselves.
- **Lesson from the failed 1e-8 droptol**: When the sparse update was applied to b_hat using sΔb, small-but-real entries of Δb were dropped from b_hat updates. This accumulated O(n_pivots × droptol × max_Δb) error in b_hat, enough to corrupt pricing and cause cycling (DEGEN2) and singular bases (BRANDY).
- `droptol!(sΔb, 1e-10)` for the ETAMatrix: safe because:
  - Accumulated b_hat error via eta chain = O(k × 1e-10 × ||b||) ≈ 1.5e-7 over 150 pivots — within tolerance
  - Refactorization corrects all errors from scratch
- The cleanup `c_hat[i] = 0 for i in basis` (O(m) scatter) corrects numerical drift in basic reduced costs each iteration. This is important: without it, c_hat[basis[i]] would gradually accumulate nonzero values from floating-point rounding in `updateDualVariables`.
- `perturbation_c_hat[i] = 0 for i in basis` similarly prevents perturbation drift for basic variables.

**Performance:**
- **`updateBasicVariables` × 2**: Two full `axpy!(−t, Δb, b_hat)` calls, O(m) each. The dense Δb is required for accuracy — **cannot be sparsified**. The two calls could be fused into one pass (`b_hat[i] -= t · Δb[i]` and `pb_hat[i] -= t_p · Δb[i]` in the same loop), saving one O(m) read of Δb and one write to b_hat. Cache-friendly, ~2x speedup for this step.
- **`updateDualVariables` × 2**: Two full `axpy!(−s, Δc, c_hat)` calls, O(n) each. For DEGEN3 (n = 6660): 2 × 6660 × 8 bytes = 107KB of writes per iteration, ~323ms over 6464 iterations. The two calls can be fused similarly. Δc is dense (no droptol applied), so this must remain O(n) — **cannot be sparsified without tracking which Δc entries are nonzero**.
  - **Safe improvement**: fuse the two axpy calls into one O(n) loop that updates both c_hat and pc_hat simultaneously. This reads Δc once instead of twice. For memory-bandwidth-limited problems, ~1.5–2× speedup for this step.
- **`sparsevec(Δb)`**: Allocates a new sparse vector every iteration. With 6464 iterations for DEGEN3, this is 6464 allocations × ~2220 × 8 bytes = ~115MB of allocation, creating GC pressure. Consider pre-allocating a reusable sparse vector workspace.
- **ETAMatrix push**: The `push!(pfi.eta_matrices, η)` grows the vector. With `refactor_pivot_cap = max(100, n/4)`, the max pre-refactorization count is large. Pre-sizing with `sizehint!` on initialization could avoid repeated array growth.
- **`total_eta_nnz` counter**: Already maintained as a running sum — good, avoids the O(k) generator sum that was there previously.

---

### G. Refactorization

**Accuracy:**
- Full UMFPACK factorization resets all accumulated numerical errors in b_hat, c_hat, perturbation_b_hat, perturbation_c_hat.
- `perturbation_c_hat[i] = abs(v) < eps ? 0.0 : v` (line 409): epsilonizes near-zero entries in `perturbation_c_hat`. **Potential bug**: if `perturbation_c_hat[j] = 0` for a nonbasic j with `c_hat[j] < 0`, then `pc[j] = -c_hat[j] / 0 = +Inf`, which `max_argmax` selects, giving `t_c = +Inf`. The stopping criterion `t_c ≤ eps` is never satisfied and the algorithm cycles or runs forever.
  - In practice, the perturbation is generated as `1.0 + 5.0 * rand()` (always ≥ 1), so `perturbation_c_hat[j]` after B⁻¹A transformation would be near-zero only if the corresponding row of B⁻¹ is near-zero — indicating a near-singular basis. This is a rare but real edge case.
  - **Fix candidate**: instead of zeroing near-zero perturbation_c_hat, keep the raw computed value. If the value is genuinely near-zero due to B⁻¹ structure, it reflects a near-degenerate column and pricing with a large ratio is informative, not catastrophic.
- `@assert count(x -> isnan(x), b_hat) == 0` (line 393): checks for NaN after refactorization but not for unexpectedly large values. A useful additional check would be `@assert all(isfinite, b_hat)`.

**Performance:**
- **UMFPACK LU** is the dominant cost: O(m^{1.5} to m^2) depending on matrix density. For DEGEN3 (m = 2220), this likely takes 1–10ms per call. With ~44 refactorizations over 6464 pivots, LU contributes 44–440ms.
- **Refactorization trigger**: Currently `10m` fill threshold. For DEGEN3, this gives ~147 pivots between refactorizations. Increasing the threshold reduces LU frequency but increases eta chain length (slower btran/ftran). The current 10m is already tuned well; further tuning requires problem-specific profiling.
- **Post-refactorization recomputation**: 2 ftrans + 2 btrans + 2 `mul!` calls. With fresh LU (no eta chain), these are fast (pure UMFPACK solves). The `mul!` calls are O(nnz(A)) and fast.
- The loop `for i in basis` after refactorization (line 412) has an empty body (assertion commented out). Dead code; can be removed.

---

## 5. Summary: Highest-Impact Opportunities

| Priority | Change | Section | Expected Impact | Risk |
|---|---|---|---|---|
| **1** | Fix perturbation drift: clip/regenerate when `perturbation_b_hat[i] < threshold` | B (pricing) | Eliminate cycling in degenerate problems | Medium — need to carefully re-price |
| **2** | Fuse paired axpy loops: `updateBasicVariables`×2 and `updateDualVariables`×2 | F | ~1.5× speedup for update step; safe | Low |
| **3** | Track active Δc indices; reset only those instead of `fill!(Δc, 0)` | C | Save n=6660 writes per iteration | Low — additive bookkeeping |
| **4** | Better degenerate recovery: rescale perturbation to maintain current t | E | Prevent t from jumping up on regeneration | Medium — algorithmic change |
| **5** | Skip initial LU for slack basis (set `luf = nothing`) | A | Eliminate one unnecessary UMFPACK call | Low |
| **6** | Pre-allocate sparse Δb workspace to reduce GC pressure | F | Reduce allocation cost ~6k iter | Low |
| **7** | Remove dead `if i > c - b` check and empty refactorization loop | C, G | Readability only | None |
