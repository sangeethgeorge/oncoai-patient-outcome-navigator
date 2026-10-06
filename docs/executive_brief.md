# Executive brief: 30-day mortality for cancer patients in the ICU

*October 2026. Numbers come from [reports/quality_measures.md](../reports/quality_measures.md) and `models/metrics.json`. For research and education only.*

## The question

Which critically ill cancer patients are most likely to die within 30 days of reaching the ICU, and can routine lab results from the first two days flag them early?

## Who we looked at

**2,671 adults with cancer whose first ICU stay lasted at least two days.** The data is from one Boston teaching hospital, 2001–2012 (MIMIC-III).
**765 of them (28.6%) died within 30 days.**

## What we found

1. **Risk depends heavily on the type of cancer.**

   | Cancer group | 30-day deaths |
   | :-- | --: |
   | Cancer that has spread (metastatic) | 35.9% |
   | Cancer that hasn't spread | 19.3% |
   | Blood cancers overall | 29.1% |
   | Acute leukemia | 40.7% |

   "Blood cancer" is not one risk group. Acute leukemia is the highest-risk group in the cohort, and lymphoma, myeloma and chronic leukemia are closer to average.
2. **Two days of routine labs and vitals sort patients into risk groups that hold up.** Patients in the lowest third of predicted risk died 9.8% of the time, against 10.9% predicted.
   The highest third died 49.8% of the time, against 51.6% predicted. On held-out patients the model separates those who died from those who survived much better than age alone (AUC 0.77 vs 0.55).
3. **The model under-predicts for metastatic and lung cancer.** It has no information about cancer stage. Metastatic patients had 18% more deaths than predicted, and lung and chest cancers 31% more.
   Adding the cancer group to the model is the clearest next improvement.
4. **ICU death rates differ by unit, mostly because the units admit different patients.** The medical ICU's rate is 38.6% and the cardiac surgery ICU's is 14.0%.
   These figures are not adjusted for how sick patients were, so they must not be read as a ranking of care quality.
5. **Readmission to the ICU within 48 hours is rare, about 2% of patients who left the ICU alive.**

## Data-quality issue found and fixed

The original cohort rule picked out cancer patients by searching diagnosis titles for "malignant" or "neoplasm". It was wrong in both directions:

- **It included 390 patients with benign or pre-cancerous growths.**
- **It missed 387 patients with leukemia, lymphoma, myeloma or melanoma**, whose diagnosis titles use neither word.

Measured 30-day mortality rose from 25.1% to 28.6% once the rule was corrected. The cohort now uses the standard Charlson cancer code definitions, and automated tests stop the error from coming back.

## What this data can't tell us

- **Cancer stage, treatment and goals of care.** Do-not-resuscitate and comfort-care orders weren't used, and they strongly affect deaths in oncology ICUs.
- **When conditions began.** Diagnosis codes are assigned at discharge, so the data can't show what was known when the patient arrived.
- **Whether the results still hold.** This is one hospital, the data is 14–25 years old, and patients who died within two days are excluded by design.
- **Which unit gives better care.** That needs a separate model that adjusts for how sick patients were on arrival.

## Recommended next steps

1. **Add cancer group, admission type and code status to the risk model**, then re-check calibration for metastatic and lung cancer patients.
2. **Build a separate case-mix model from the first 24 hours** (a standard ICU severity score plus age, admission type and comorbidities) before comparing units on mortality.
3. **Agree on risk thresholds with oncology and ICU clinicians** that trigger a specific action, such as an early goals-of-care conversation, instead of using fixed thirds.
4. **Validate on newer data.** MIMIC-IV uses ICD-10 and is also available in FHIR format.
