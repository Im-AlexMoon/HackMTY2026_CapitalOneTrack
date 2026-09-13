import assert from "node:assert/strict";
import { test } from "node:test";
import { availableScore, clock, filterAccounts, riskTone, score } from "../lib/format";
import type { Account, Prediction } from "../lib/types";

test("missing and invalid scores never appear as zero risk", () => {
  assert.equal(score(null), "—");
  assert.equal(score(Number.NaN), "—");
  assert.equal(score(0), "0");
  assert.equal(score(0.82), "82");
  assert.equal(availableScore({ status: "error", risk_score: 0.8 } as Prediction), null);
  assert.equal(availableScore({ status: "demo", risk_score: 0.8 } as Prediction), 0.8);
});

test("risk severity follows the active policy including the boundary", () => {
  assert.equal(riskTone(0.75, 0.75), "danger");
  assert.equal(riskTone(0.74, 0.75), "warning");
  assert.equal(riskTone(0.75, 0.95), "neutral");
  assert.equal(riskTone(null, 0.75), "muted");
});

test("account filters preserve input order and distinguish screening from review", () => {
  const accounts = [
    { id: "a", alias: "River Lane", status: "monitoring" },
    { id: "b", alias: "River Field", status: "review_required" },
    { id: "c", alias: "Cedar", status: "rejected" },
  ] as Account[];
  assert.deepEqual(filterAccounts(accounts, " river ", "all").map((item) => item.id), ["a", "b"]);
  assert.deepEqual(filterAccounts(accounts, "", "attention").map((item) => item.id), ["b"]);
  assert.deepEqual(filterAccounts(accounts, "", "rejected").map((item) => item.id), ["c"]);
});

test("the replay clock uses UTC and rejects malformed dates", () => {
  assert.equal(clock("2026-09-12T12:30:00Z"), "12:30:00");
  assert.equal(clock("invalid"), "—");
});
