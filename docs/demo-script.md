# Three-minute presentation

Open `http://localhost:3000`. Keep the synthetic-demo disclosure visible. Select **The sleeper account**, reset, and use **Next event** while narrating; Play advances every two seconds at 1×. The underlying simulated clock advances independently.

1. **0:00–0:30 — Context.** Five accounts are monitored. Select Maya Chen and point out admission risk, the current-risk curve, three signals and the policy threshold.
2. **0:30–1:00 — Onboarding.** Advance to Avery Martin's arrival; the account appears Screening, then Monitoring. Jordan Lee subsequently fails admission and remains visible as Rejected at onboarding.
3. **1:00–1:40 — Behavior shift.** Advance Morgan's transactions through repeated transfers. Watch transaction and sequence signals rise and fused risk cross the line before the cash-out.
4. **1:40–2:20 — Investigation.** Pause, read the observed changes, enter analyst notes and select Review & Escalate. The account remains visible with an auditable human disposition.
5. **2:20–3:00 — Policy.** Open Risk policy, show the separate admission/monitoring thresholds, and explain the friction versus missed-risk tradeoff. Existing rejection history is preserved. End with how verified external models plug into the same runner interface.

Optional: use **An unusual, legitimate day**, inspect the synthetic large purchases and dismiss the case with notes. This demonstrates that anomaly is not proof of fraud. **Everyday banking** remains below the default threshold. **Connected applications** shows shared synthetic devices in onboarding evidence.

For rehearsal: Play/Pause, 1×/5×/20×, Next event and Reset are real server controls. Reset preserves old run audit data. The browser tests record a backup in `apps/web/test-results/`; the walkthrough spec saves desktop/mobile captures and a video. Do not describe synthetic rule scores as trained-model results.
