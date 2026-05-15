# ATEENA Localization Pack

This package contains a complete working set for localization planning, review, and QA for the current ATEENA app build.

## Included files

1. **01_locale_matrix.xlsx**
   - Master locale matrix
   - Browser/device auto-detect mapping
   - Coverage notes

2. **02_glossary_master.xlsx**
   - Core glossary
   - Product taxonomy
   - High-risk phrases
   - Do-not-translate rules

3. **03_lqa_checklist.xlsx**
   - LQA checklist
   - Severity scale
   - Bug log template
   - Sign-off matrix

4. **04_screen_inventory.xlsx**
   - Screen inventory
   - High-risk strings
   - Source strings

5. **05_translation_template.xlsx**
   - Source strings
   - Blank target rows for each locale

6. **06_tone_guide_and_review_playbook.docx**
   - Tone of voice rules
   - Non-medical / body-sensitive language rules
   - Locale-specific review rules
   - In-country review workflow
   - Sign-off criteria

## Scope

The package is aligned to the current locale matrix embedded in the app, including:
- regional variants (for example `pt-BR`, `pt-PT`, `es-ES`, `es-419`)
- script variants (for example `zh-Hans-CN`, `zh-Hant-TW`, `sr-Cyrl-RS`, `sr-Latn-RS`)
- RTL locales (Arabic, Persian, Urdu, and Arabic-script Punjabi / Pashto variants)

## Recommended workflow

1. Freeze the source strings for a release.
2. Use **01_locale_matrix.xlsx** to confirm target locales and ownership.
3. Use **02_glossary_master.xlsx** before translating UI copy.
4. Fill targets in **05_translation_template.xlsx**.
5. Review localized screenshots with **04_screen_inventory.xlsx**.
6. Run native LQA with **03_lqa_checklist.xlsx**.
7. Approve or reject locale release readiness in the sign-off matrix.

## Notes

- The translation workbook is a **template**, not a final translated build.
- Several locales may intentionally share a base language at first, but should still go through separate screenshot review where locale, script, or tone differs.


## v2 additions

This v2 pack extends v1 with execution-oriented rollout planning:

- **07_rollout_prioritization.xlsx** — locale / market rollout waves, weighted prioritization model, and GTM notes.
- **08_reviewer_cost_model.xlsx** — editable budgeting model for native reviewers, screenshot review, and PM coordination.
- **09_go_live_checklist.xlsx** — master checklist, locale sign-off tracker, and bug log template for launch readiness.

### Important note about costs
The cost model uses **editable working assumptions**, not vendor quotes or guaranteed market rates. Replace them with your real freelancer / agency quotes before approval.

### Recommended order of use
1. Review `01_locale_matrix.xlsx`
2. Lock terminology in `02_glossary_master.xlsx`
3. Review screens using `04_screen_inventory.xlsx`
4. Prioritize rollout in `07_rollout_prioritization.xlsx`
5. Budget review resources in `08_reviewer_cost_model.xlsx`
6. Track go-live readiness in `09_go_live_checklist.xlsx`


## v3 additions
- `10_market_region_fashion_ecommerce_priority.xlsx` — region-first rollout planning with fashion-category focus
- `11_vendor_comparison_matrix.xlsx` — vendor shortlist matrix with source URLs and scorecard template
- `12_executive_summary.docx` — concise decision-ready summary for stakeholders
- `13_locale_expansion_addendum.xlsx` — additional locale tracks added in v3 and routing rules

## Suggested reading order for v3
1. `12_executive_summary.docx`
2. `10_market_region_fashion_ecommerce_priority.xlsx`
3. `11_vendor_comparison_matrix.xlsx`
4. `13_locale_expansion_addendum.xlsx`
5. Existing v2 files (`07`, `08`, `09`) for rollout costing and go-live readiness
