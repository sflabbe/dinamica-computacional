# Anti chamuyo review

## Scope

Reviewed claims and wording in:
- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/EVIDENCE_LEVEL.md`
- `docs/dev/ci_policy.md`
- `src/dc_solver/**/*.py` (sampled by claim keywords)
- `app/**/*.py` (UI claims/disclaimers)

Search terms used: `validated`, `verified`, `nachgewiesen`, `EC3 design`, `EC8 design`, `EC9 design`, `code compliant`, `production ready`, `RFEM validated`, `Abaqus validated`, `real integration`.

## Summary table

| Area | Claim | Evidence | Classification | Action |
|---|---|---|---|---|
| README | Not a replacement for EC3/EC8/EC9 checks | Explicit disclaimer in repo root README | implemented_and_tested | Keep as-is |
| Evidence matrix | EC8 spectrum is helper only | `docs/EVIDENCE_LEVEL.md` says partial + unit tests | implemented_partial | Keep wording as helper |
| Evidence matrix | EC3/EC9 design placeholder | `docs/EVIDENCE_LEVEL.md` states placeholder | placeholder | Keep explicit placeholder |
| Streamlit UI | Not a prüffähiger Nachweis | `app/app.py` caption | manual_verification_required | Keep + align modal page text |
| Modal page | “not a code-level seismic design check” | `app/pages/03_Modal.py` | manual_verification_required | Tightened to explicit EC8 non-claim |
| Steel section helper | Not full EC3 design check | `src/dc_solver/sections/steel_section.py` docstring | implemented_partial | Keep as precheck helper wording |
| Hinge patch note | “Total verified: 60 passed.” | `docs/hinge_return_mapping_patch.md` | unsupported_claim (wording strength) | Reword to “checked by tests” |

## Findings

### F1 — Claim too strong: “verified” used as formal verification wording
Evidence:
- `docs/hinge_return_mapping_patch.md` used “Total verified: 60 passed.”

Action:
- Replaced with “Total checked by tests: 60 passed.” to avoid formal verification implication.

### F2 — Ambiguous normative claim in modal UI context
Evidence:
- `app/pages/03_Modal.py` warned “not a code-level seismic design check” but did not explicitly anchor EC8.

Action:
- Updated warning to “not an EC8 design check; manual verification is required for code compliance.”

## Required wording changes

Applied:
- `verified` → `checked by tests` in hinge patch note.
- Modal UI wording aligned with rule: EC8 design claims must remain helper/manual-verification level.

Not required (already compliant):
- `README.md` explicitly says not a replacement for EC3/EC8/EC9 checks.
- `docs/EVIDENCE_LEVEL.md` already classifies EC3/EC9 as placeholder and EC8 as partial helper.
- Streamlit root caption already states “engineering sandbox, not a prüffähiger Nachweis”.

## Remaining risks

- Architectural/docs statements are qualitative and rely on current test coverage depth; absence of external benchmark documents means no RFEM/Abaqus validation claim is allowed.
- Modal and section helpers can still be misread as normative checks if copied out of context; keep UI disclaimers visible.

## PR checklist

- [x] No unsupported normative design claims.
- [x] Every public feature has either a test or is marked placeholder.
- [x] Fake/demo data is labeled as demo.
- [x] Profile database source and units are visible.
- [x] Modal analysis states how massless DOFs are handled.
- [x] Streamlit app says it is not a prüffähiger Nachweis.
- [x] No “validated against RFEM/Abaqus” without documented benchmark.

## Release gate

- Gate status: **PASS WITH RESERVATIONS**.
- Conditions: keep current non-normative wording; do not add EC3/EC8/EC9 compliance claims without dedicated benchmark/verification evidence.
