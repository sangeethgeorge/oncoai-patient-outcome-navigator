DROP MATERIALIZED VIEW IF EXISTS onco_cohort;

CREATE MATERIALIZED VIEW onco_cohort AS
SELECT DISTINCT
    d.subject_id,
    d.hadm_id,
    i.icustay_id,
    d.icd9_code,
    dd.long_title AS diagnosis_title,
    p.gender,
    p.dob,
    p.dod,
    a.ethnicity,
    a.admittime,
    a.dischtime,
    i.intime,
    i.outtime
FROM diagnoses_icd AS d
INNER JOIN d_icd_diagnoses AS dd 
    ON d.icd9_code = dd.icd9_code
INNER JOIN admissions AS a 
    ON d.hadm_id = a.hadm_id
INNER JOIN patients AS p 
    ON d.subject_id = p.subject_id
INNER JOIN icustays AS i 
    ON i.hadm_id = a.hadm_id
WHERE
    -- Include only numeric ICD-9 codes (exclude V and E codes)
    d.icd9_code ~ '^[0-9]{3}'
    AND
    -- Neoplasm range: 140–239
    CAST(SUBSTRING(d.icd9_code FROM 1 FOR 3) AS INTEGER) BETWEEN 140 AND 239
    AND
    -- Ensure clinically relevant diagnoses
    dd.long_title ILIKE ANY (ARRAY[
        '%malignant%',
        '%neoplasm%',
        '%carcinoma%',
        '%lymphoma%',
        '%leukemia%'
    ])
WITH DATA;
