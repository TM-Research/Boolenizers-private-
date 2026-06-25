# Why booleanization determines Tsetlin Machine performance (literature synthesis)

(Deep-research synthesis with primary sources — see inline citations.)

**Core mechanism.** A TM learns propositional formulas over Boolean inputs `{0,1}`. Each
variable yields two literals (`x_k`, `¬x_k`), each governed by a Tsetlin Automaton that
learns Include/Exclude via Type I (combat false negatives) and Type II (combat false
positives) feedback. A clause is an AND of included literals; class score = Σ positive −
Σ negative clauses. **The input must be Boolean**, and what a clause can discriminate is
bounded by what the literals express (Granmo 2018, arXiv:1804.01508).

**Thermometer > one-hot.** Threshold/thermometer encoding emits one cumulative bit per
threshold (`x > v_w`). A single literal is then a half-line test, and ANDing two
thermometer bits (with negation) yields an interval `v_a < x ≤ v_b` — exactly the
conditions a conjunctive clause expresses cheaply. One-hot destroys order and forces a
clause to OR many bins (impossible for one conjunction) → wasted clause budget
(Abeyrathna et al. 2019, arXiv:1905.04199).

**Bit budget = bias/variance.** Each threshold adds 2 literals (2 TAs). Too few → wide
intervals, underfit; too many → spurious narrow intervals, overfit + memory/latency cost.
Granularity interacts with `s`, `T`, clause count (Multigranular TM, arXiv:1909.07310;
Weighted TM, arXiv:1911.12607).

**Quantile > equal-width.** Equal-width bins leave empty bins = dead literals on
skewed/heavy-tailed data; quantile (equal-frequency) bins carry ~equal information per bit
and are robust to skew — even 1–2 quantile bins/feature can be highly discriminative
(audio-KWS TM, arXiv:2101.11336).

**Implications for streaming IDS:** thermometer (not one-hot) + adaptive **quantile-like**
cut-points (streaming, drift-aware) + a conservative, drift-aware bit budget.

Sources: arXiv:1804.01508, 1905.04199, 1905.04206, 1909.07310, 1911.12607, 2005.05131,
2101.11336, 2406.00704; *An Introduction to Tsetlin Machines* (Granmo).
