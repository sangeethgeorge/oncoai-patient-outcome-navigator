# Business rules

The rules that turn raw MIMIC-III rows into the cohort, the curated tables and the published measures.
Each rule names the file where it is implemented, so the documentation and the code can be checked against each other.

## 1. Cohort

Implemented in `src/oncoai_prototype/data_loading/02_define_onco_cohort.sql`. Tests: `tests/test_db_utils.py` and `dbt/models/staging/_staging.yml`.

| Rule | Definition | Why |
| :-- | :-- | :-- |
| Unit of analysis | One row per ICU stay; the patient's **first** ICU stay in MIMIC | Repeat stays of one patient are correlated; one stay per patient keeps every observation independent |
| Cancer admission | The hospital admission carries at least one qualifying ICD-9 code (§2), at any diagnosis position | MIMIC codes diagnoses at discharge, so position doesn't reliably show why the patient was in the ICU |
| Minimum ICU stay | `outtime - intime >= 48 h` | The 48 h feature window must end before the outcome can happen in the ICU |
| Age | 18–89 at hospital admission | Adults only; MIMIC shifts ages over 89 to about 300 |
| Outcome | `mortality_30d = 1` if `patients.dod <= intime + 30 days` | Anchored at ICU admission, which is the prediction time. `dod` includes deaths after discharge (Social Security records) |

## 2. ICD-9 cancer codes

MIMIC stores ICD-9-CM codes without the decimal point (`1749` is 174.9 and `185` is 185). Rules match on the leading digits.
The cohort uses the **Charlson comorbidity definitions** (Deyo's ICD-9-CM algorithm as tabulated by Quan et al., *Med Care* 2005):

- **Any malignancy, including lymphoma and leukemia, except malignant neoplasm of skin:** 140–172, 174–195.8, 200–208.
- **Metastatic solid tumor:** 196–199.1.

To these we add the neuroendocrine codes created in FY2010, which follow AHRQ/HCUP's Elixhauser assignment.
Categories come from the seed `dbt/seeds/icd9_cancer_category.csv`.

| ICD-9-CM range | Meaning | In cohort? | Site category |
| :-- | :-- | :-- | :-- |
| 140–149 | Lip, oral cavity, pharynx | Yes | head_and_neck |
| 150–159 | Digestive organs, peritoneum | Yes | digestive |
| 160–165 | Respiratory and intrathoracic | Yes | respiratory_thoracic |
| 170–172, 176 | Bone, connective tissue, melanoma, Kaposi sarcoma | Yes | bone_soft_tissue_melanoma |
| 174–175 | Breast | Yes | breast |
| 179–189 | Genitourinary | Yes | genitourinary |
| 190–192 | Eye, brain, other nervous system | Yes | cns_and_eye |
| 193–195 | Thyroid, other endocrine, ill-defined sites | Yes | other_solid |
| 196–198, 199.0–199.1 | Secondary, disseminated, site unspecified (Charlson "metastatic") | Yes | sets `metastatic_flag` |
| 200–208 | Lymphoma, leukemia, myeloma | Yes | hematologic |
| 209.0–209.3 | Malignant neuroendocrine tumors (HCUP: solid tumor) | Yes | neuroendocrine |
| 209.7 | Secondary neuroendocrine tumors (HCUP: metastatic) | Yes | sets `metastatic_flag` |
| 173 | Other malignant neoplasm of skin (basal and squamous cell) | **No** | Excluded by Charlson; rarely relevant to ICU outcomes |
| 230–234 | Carcinoma in situ | **No** | Not a Charlson malignancy |
| 209.4–209.6, 210–229 | Benign neoplasms | **No** | |
| 235–238 | Uncertain behavior (incl. MDS 238.7x, polycythemia vera 238.4, plasmacytoma 238.6) | **No** | See limitation below |
| 199.2, 239 | Malignancy in transplanted organ; unspecified nature | **No** | Not in Charlson; too rare or vague |

**Cancer group** (`int_stay_cancer_dx.cancer_group`) is the main stratifier, because critical-care oncology studies compare outcomes this way:

| Group | Rule |
| :-- | :-- |
| hematologic | Any 200–208 code. A patient with both a hematologic and a solid code counts as hematologic; `hematologic_and_solid_flag` records how many |
| solid_metastatic | Otherwise, any metastatic code (196–199.1, 209.7) |
| solid_nonmetastatic | All other stays |

**Hematologic subtype** (hematologic stays only; the first match wins): acute leukemia (204.0, 205.0, 206.0, 207.0, 208.0, including in remission),
then lymphoma (200–202), myeloma (203), chronic leukemia (204.1–208.1) and other. Studies of hematologic patients in the ICU usually separate acute leukemia,
because its outcomes differ sharply from those of lymphoma and myeloma.

**Primary site** is secondary detail: the site category of the primary-site code with the lowest `seq_num`, or `unknown_primary` when the only cancer codes are secondary.

**Known limitations:**

- **MDS, myeloproliferative neoplasms and plasmacytoma (238.4, 238.6, 238.7x) are excluded.** SEER treats them as reportable cancers, but Charlson doesn't count them as malignancies. Including them would add roughly 170 stays. It's a sensitivity analysis to run if hematology stakeholders want them in.
- **Codes are assigned at discharge and MIMIC-III has no present-on-admission flag.** A cancer first diagnosed during the stay counts the same as known cancer. That's fine for defining the cohort, but it's another reason not to use these codes as admission-time risk adjusters.

### Change record: October 2026 cohort correction

The earlier cohort used ICD-9 140–239 **and** a diagnosis title containing "malignant" or "neoplasm". That rule had two errors.

1. **It admitted non-malignant neoplasms.** Range 140–239 includes benign, uncertain and in-situ codes, and their titles contain "neoplasm".
   Restricting to the Charlson malignancy codes **removed 390 stays (26 deaths, 7%)**. Most were benign-only; 23 had only non-melanoma skin cancer or carcinoma in situ.
2. **It silently dropped most blood cancers and melanoma.** Titles such as "Acute myeloid leukemia…", "Multiple myeloma…",
   "Hodgkin's disease…" and "Melanoma of skin…" contain neither word. **387 stays (119 deaths, 31%) were added.**

| | Stays | 30-day deaths | Rate |
| :-- | --: | --: | --: |
| Previous cohort | 2,674 | 672 | 25.1% |
| Corrected cohort | 2,671 | 765 | 28.6% |

The rule now matches code ranges only; titles aren't used. A dbt test (`assert_every_stay_has_cancer_code`) and pytests
(`test_cohort_has_only_charlson_malignancies`, `test_cohort_keeps_hematologic_cancers`) keep the cohort definition and the category seed in agreement.

## 3. Measurement window and extraction

Implemented in `03a_extract_all_labs_48h.sql` and `03b_extract_all_vitals_48h.sql`. Test: `assert_measurements_within_48h_window`.

- Window: `[intime, intime + 48 h)`. The start is inclusive and the end is exclusive.
- Labs (`labevents`) are joined to the stay through `subject_id` + `hadm_id`, which includes labs drawn before ICU transfer if they fall inside the window. Chart vitals (`chartevents`) are joined through `icustay_id`.
- Only numeric results (`valuenum IS NOT NULL`) are kept.

## 4. Standard coding (LOINC)

Implemented in `int_measurements_48h` and `dim_measurement_item`. Test: `assert_feature_eligible_items_have_loinc`.

- **Labs:** LOINC from MIMIC's `d_labitems.loinc_code`. All lab items that qualify as model features (recorded in ≥ 70% of stays) are coded.
  The codes behind the selected model features were checked against the NLM Clinical Tables LOINC service.
- **Chart vitals:** MIMIC-III chart items carry no LOINC. The seed `chart_item_loinc.csv` maps the CareVue and MetaVision items for heart rate (8867-4),
  respiratory rate (9279-1) and hemoglobin (718-7). Each code is verified, with the source and date recorded in the seed.
  Codes are never filled in from memory.
- CareVue and MetaVision record the same measurement under different item IDs (heart rate is 211 and 220045). Feature engineering pools them by label,
  and the LOINC code is what makes that pooling auditable.
- **Open question for clinical review:** MIMIC codes Anion Gap as 1863-0, which LOINC describes as the four-ion gap (including potassium).
  BIDMC may report the three-ion gap. That would change reference ranges but not the model, which uses the value as recorded.

## 5. Units, plausibility and missing data

- **Units:** no conversions are applied. `warn_one_unit_per_item` lists items recorded in more than one unit, ignoring case.
  None of the model's features are on that list. Triage of the current list:
  - **Same unit, different spelling** (safe to pool after normalizing the label): PT (`sec`/`SECONDS`), TSH (`uU/mL`/`uIU/mL`) and ascites cell counts (`#/cu mm`/`#/uL`).
    Urobilinogen (`mg/dL`/`EU/dL`) is roughly equivalent at 1 EU ≈ 1 mg/dL.
  - **Real conflicts** (need a conversion or exclusion before use): C-reactive protein (`mg/L` vs `mg/dL`, a 10× difference), FSH and LH (`mIU/mL` vs `mIU/L`, 1,000×),
    and measured osmolality (`mOsm/L` is osmolarity and `mOsm/kg` is osmolality, which are different measures).
- **Monitor flags:** the current flags are vital-sign coverage in the 11 stays whose documentation spans both CareVue and MetaVision (`dbsource = 'both'`).
  These stays straddle MIMIC's 2008 system change. The slice is too small to act on, and the published report suppresses it.
- **Plausibility:** `measurement_plausible_range.csv` holds wide physiologic limits. These are analyst judgement, not clinical reference ranges.
  Values outside the limits are counted in `dq_measurement_coverage`, not removed.
- **Missing data:** a feature missing in more than 20% of stays is dropped (`feature_engineering.py`). Remaining gaps are filled with the training-split median,
  computed separately for each training split and cross-fitting fold, so held-out rows never influence it.

## 6. Risk stratification and model calibration

Implemented in `src/oncoai_prototype/analytics/quality_measures.py`. Tests: `tests/test_quality_measures.py`.

- **Predicted risk:** cross-fitted, out-of-fold probabilities (`model_training.cross_fitted_predictions`). There are 5 folds, grouped by patient.
  Imputation, feature selection and the model fit are repeated inside each fold, so every stay is scored by a model that never saw it.
  The published model and test metrics use a separate 80/20 split.
- **Risk tiers:** tertiles of cross-fitted risk, recomputed on each retrain. They are descriptive. A tier that triggers an action (for example, a goals-of-care conversation)
  needs an absolute threshold agreed with clinicians and checked for net benefit (decision-curve analysis), not a quantile.
- **Calibration:** observed against mean predicted mortality by decile of predicted risk.
- **Observed/predicted by subgroup:** observed deaths ÷ the sum of predicted probabilities, with Byar's 95% CI, by cancer group, primary site and ICU type.
  The label "model under-predicts" or "over-predicts" is applied only when the CI excludes 1.

### Why this is *not* a standardized mortality ratio

An SMR compares units or providers after **case-mix adjustment**: a model that estimates each patient's risk from their condition at presentation.
ICU benchmarking systems (APACHE, SAPS II, ICNARC) use the worst values in the **first 24 h** together with age, admission type (medical, elective surgery or emergency surgery) and chronic health.
The 30-day model falls short of that in four ways:

1. **Beyond physiology it sees only age, and only when feature selection keeps it (the current model does). There's no admission type, comorbidity, cancer group or code status.** Units that admit post-operative patients (SICU, CSRU) will look "better" than the MICU simply because of who they admit.
2. **It uses 48 h of physiology.** Hours 24–48 already reflect the ICU's treatment, so adjusting for them partly adjusts away the quality being measured.
3. **The cohort excludes stays under 48 h.** Early deaths are left out, so a unit with many early deaths would look better than it is.
   A quality-measure cohort would include all adult first ICU stays.
4. **Code status is ignored.** Patients admitted with DNR or comfort-care orders (MIMIC chartevents items 128 and 223758) die for reasons that aren't failures of care.
   This matters most in oncology ICUs.

These are separate design choices for a prediction cohort and a quality-measure cohort, and both are legitimate.
The 48 h, ≥ 48 h design matches the standard MIMIC-III mortality benchmark (Harutyunyan et al., *Sci Data* 2019) and suits prediction. A case-mix-adjusted SMR by ICU type would need:

- a first-24 h severity score (SAPS II or OASIS, which needs the MIMIC `services`, `outputevents` and ventilation data not loaded here);
- age, admission type, cancer group, Elixhauser comorbidities and code status at admission;
- the full cohort.

## 7. Utilization

Implemented in `int_icu_utilization`. Test: `assert_readmission_only_for_icu_survivors`.

| Measure | Definition |
| :-- | :-- |
| ICU death | In-hospital death no later than 6 h after ICU discharge (MIMIC sometimes closes a stay shortly after death is charted) |
| ICU length of stay | `outtime - intime` of the cohort stay, in days (≥ 2 by construction). Reported separately for hospital survivors and decedents, because death shortens a stay |
| Hospital length of stay | `dischtime - admittime`, in days, split the same way |
| ICU readmission within 48 h | **Denominator:** patients discharged alive from the ICU. **Numerator:** another ICU stay in the same hospitalization starting within 48 h of ICU discharge. This is the indicator the SCCM Quality Indicators Committee ranks first for ICU quality |
| Any ICU readmission | Same denominator, any later ICU stay in the same hospitalization (kept in the mart for reference) |

## 8. Publication and governance

- **Row-level data** (MIMIC tables, parquet files, the DuckDB warehouse, MLflow runs) stays on the local machine and is git-ignored.
  This follows the PhysioNet Credentialed Health Data License.
- **Published outputs** are aggregate only: the README, `models/metrics.json`, `reports/quality_measures.md` and `reports/figures/`.
  Observed rates by ICU type are labeled as not risk-adjusted. Subgroup observed/predicted ratios are labeled as model calibration, not quality.
- **Small-cell suppression:** a table row is blanked when its stay count, event count or non-event count is between 1 and 10 (`suppress_small_cells`).
  This follows the CMS cell-size convention, and the complement is suppressed too so a count can't be recovered by subtraction.
- **Database access:** the pipeline runs under a read-only Postgres role. dbt attaches Postgres `read_only`.
- **Data health monitor:** `dq_measurement_coverage` compares each measurement's coverage and implausible-value rate across ICU types and documentation systems (CareVue/MetaVision).
  MIMIC shifts dates per patient, so calendar-time trends can't be monitored. Flags appear as dbt warnings.

## References

- Quan H, et al. Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data. *Med Care.* 2005;43(11):1130–9. Charlson code table as reproduced in the [LSU Charlson coding document](https://medschool.lsuhsc.edu/orthopaedics/docs/Charlson%20Comorbidities%20-%20Coding%20Algorithms%20for%20ICD-9-CM%20and%20ICD-10.pdf).
- AHRQ HCUP. Elixhauser Comorbidity Software, FY2010 ICD-9-CM code additions (neuroendocrine codes 209.x).
- NCI SEER. [ICD-9-CM casefinding list](https://seer.cancer.gov/tools/casefinding/case2011short.html) (reportable neoplasms, including 238.4 and 238.6–238.79).
- Harutyunyan H, et al. Multitask learning and benchmarking with clinical time series data. *Sci Data.* 2019. (In-hospital mortality from the first 48 h; ages 18–89; stays < 48 h excluded.)
- Taccone FS, et al. [Characteristics and outcomes of cancer patients in European ICUs](https://link.springer.com/doi/10.1186/cc7713). *Crit Care.* 2009;13:R15. (Outcomes by hematologic vs solid cancer.)
- SCCM Quality Indicators Committee: ICU readmission within 48 h as the top-ranked ICU quality indicator; see the review at [PMC3359937](https://pmc.ncbi.nlm.nih.gov/articles/PMC3359937).
- Spiegelhalter DJ. Funnel plots for comparing institutional performance. *Stat Med.* 2005;24(8):1185–202. (Relevant only once a case-mix-adjusted SMR exists.)
- CMS. [Present-on-admission indicator reporting](https://www.cms.gov/Medicare/Medicare-Contracting/ContractorLearningResources/Downloads/JA5499.pdf).
