{{ config(severity='warn') }}
-- Surfaces the monitor's flags in `dbt build` output; triage notes go in docs/business_rules.md.
select * from {{ ref('dq_measurement_coverage') }}
where flag_coverage_gap or flag_implausible_values
