# SAC with Teacher: Demo, BC, Reward Shaping and Priority Sampling

---

**Algorithm 1** **SAC with demonstrations, behavior cloning, reward shaping and priority sampling**

---

**Input:** Expert demonstration env $\mathcal{E}_E$; training env $\mathcal{E}$; teacher method $\phi$ (e.g. spectral, betweenness, CI); buffer capacity $M$; batch size $B$; max steps $T$; demo flag $\texttt{demo}$, BC flag $\texttt{bc}$, reward-shaping flag $\texttt{reward\_shaping}$, priority type $\texttt{priority\_type}$; BC steps $K_{\text{bc}}$; shaping coefficient $\beta_0$, decay steps $T_{\beta}$, shaping method $\eta \in \{\texttt{KL}, \texttt{betweenness}\}$; policy $\pi_\theta$, Q-networks $Q_{\psi_1}, Q_{\psi_2}$.

---

**Phase 0 — Demonstration (if $\texttt{demo}$):**

1. Initialize replay buffer $\mathcal{B}$ (with teacher tags and optional TD-error / policy-gradient storage).
2. **while** $|\mathcal{B}_{\text{teacher}}| < N_{\text{demo}}$ **do**
   - Sample batch of states $\{s\}$ from $\mathcal{E}_E$.
   - Get teacher actions $a^* = \phi(s)$ (e.g. spectral / betweenness / CI).
   - Execute $a^*$, get $(s', r, d)$; store $(s, a^*, s', r, d)$ in $\mathcal{B}$ with $\texttt{from\_teacher}=\texttt{True}$, $\texttt{fixed}=\texttt{True}$.
3. **end while**

**Phase 1 — Behavior cloning (if $\texttt{bc}$):**

4. **for** $k = 1$ **to** $K_{\text{bc}}$ **do**
   - Sample batch $(s, a^*, s', r, d) \sim \mathcal{B}$ (e.g. uniform over buffer).
   - Compute $\mathcal{L}_{\text{bc}} = -\hat{\mathbb{E}}_{(s,a^*)}\big[\log \pi_\theta(a^*|s)\big]$.
   - Update $\theta$ with $\nabla_\theta \mathcal{L}_{\text{bc}}$.
5. **end for**

**Phase 2 — Main SAC loop (for $t = 1$ to $T$):**

6. Sample actions $a_t \sim \pi_\theta(\cdot|s_t)$ (or random if $t \le t_{\text{learn}}$).
7. Execute $a_t$ in $\mathcal{E}$, get $s_{t+1}, r_t, d_t$.
8. **If $\texttt{reward\_shaping}$:** compute shaping reward $r^{\text{sh}}(s_t, a_t)$:
   - If $\eta = \texttt{KL}$: $r^{\text{sh}} = -\frac{1}{n_{\text{init}}}\,\text{KL}\big(\pi_{\phi}(\cdot|s_t) \,\|\, \pi_\theta(\cdot|s_t)\big)$ (per-graph sum of KL, teacher $\pi_\phi$ from $\phi$ with temperature).
   - If $\eta = \texttt{betweenness}$: $r^{\text{sh}}(s_t, a_t) = \text{betweenness}(a_t) / \max_v \text{betweenness}(v)$.
   - Decay: $\beta(t) = \beta_0 \cdot \big(1 - \min(t/T_{\beta}, 1)\big)$.
   - Set $r_t \leftarrow r_t + \beta(t)\, r^{\text{sh}}(s_t, a_t)$.
9. Store $(s_t, a_t, s_{t+1}, r_t, d_t)$ in $\mathcal{B}$ with $\texttt{from\_teacher}=\texttt{False}$. Optionally add teacher transitions (teacher_distill) with $\phi$.
10. **If** $t > t_{\text{learn}}$ **then** **for** $u = 1$ **to** $U$ **do**
    - **Priority sampling:** sample batch indices by priority type $\texttt{priority\_type}$:
      - **TDE:** $p_i \propto |\delta_i| + \varepsilon$ with TD-error $\delta_i = r_i + \gamma V(s'_i) - Q(s_i, a_i)$.
      - **DDPGfD:** $p_i \propto \big(|\delta_i| + \lambda \|\nabla_\theta \log\pi_\theta(a_i|s_i)\| + D_{\text{teacher},i} + \varepsilon\big)^\alpha$.
      - **TEACHER:** $p_i \propto \mathbb{1}[\text{from\_teacher}(i)] + 0.1$.
      - **Else:** $p_i = 1$ (uniform).
    - Sample $\mathcal{D}_u = \{(s_j, a_j, s'_j, r_j, d_j)\}_{j=1}^B \sim \mathcal{B}$ according to $\{p_i\}$, with $\texttt{return\_indices}=\texttt{True}$.
    - **Critic update:** compute target $y_j = r_j + (1-d_j)\gamma\, \mathbb{E}_{a'\sim\pi_\theta}[Q' - \alpha \log\pi_\theta(a'|s'_j)]$, update $Q_{\psi_1}, Q_{\psi_2}$ toward $y_j$; optionally update stored $|\delta_i|$ for sampled indices.
    - **Actor update:** update $\theta$ with SAC policy gradient; optionally update stored policy gradient norms for sampled indices (for DDPGfD).
    - (Optional) Update target networks and $\alpha$.
11. **end for** **end if**
12. $s_t \leftarrow s_{t+1}$ (or reset from env).
13. **end for** $t$

---

*Notation:* $\hat{\mathbb{E}}$ denotes empirical average over the current batch; $\pi_\phi$ is the teacher policy induced by method $\phi$; $n_{\text{init}}$ is initial graph size; $\lambda, \alpha, \varepsilon$ are priority hyperparameters.
