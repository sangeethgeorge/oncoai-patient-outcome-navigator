DROP MATERIALIZED VIEW IF EXISTS onco_vitals;

CREATE MATERIALIZED VIEW onco_vitals AS
WITH filtered_chartevents AS (
    SELECT
        ce.subject_id,
        ce.hadm_id,
        ce.icustay_id,
        ce.itemid,
        ce.charttime,
        ce.valuenum,
        ce.valueuom,
        di.label AS item_label
    FROM chartevents ce
    INNER JOIN d_items di ON ce.itemid = di.itemid
    WHERE EXISTS (
        SELECT 1
        FROM onco_cohort AS oc
        WHERE oc.icustay_id = ce.icustay_id
    )
    AND ce.valuenum IS NOT NULL
    AND coalesce(ce.warning, 0) = 0
    AND coalesce(ce.error, 0) = 0
)
SELECT * FROM filtered_chartevents;

---
DROP MATERIALIZED VIEW IF EXISTS onco_labs;

CREATE MATERIALIZED VIEW onco_labs AS
WITH filtered_labevents AS (
    SELECT
        le.subject_id,
        le.hadm_id,
        le.itemid,
        le.charttime,
        le.valuenum,
        le.valueuom,
        dl.label AS lab_label
    FROM labevents le
    INNER JOIN d_labitems dl ON le.itemid = dl.itemid
    WHERE EXISTS (
        SELECT 1
        FROM onco_cohort AS oc
        WHERE oc.hadm_id = le.hadm_id
    )
    AND le.valuenum IS NOT NULL
)
SELECT * FROM filtered_labevents;

