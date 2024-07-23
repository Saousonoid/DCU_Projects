WITH latest_measurements AS (
    SELECT
        subject_id,
        heart_rate,
        dbp,
        sbp,
        mbp,

        ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY charttime DESC) as rn
    FROM
        physionet-data.mimiciv_derived.vitalsign
       WHERE  heart_rate IS NOT NULL 
            AND dbp IS NOT NULL 
            AND sbp IS NOT NULL 
            AND mbp IS NOT NULL 
)
SELECT
     subject_id,
        heart_rate,
        dbp,
        sbp,
        mbp
FROM
    latest_measurements
WHERE
    rn = 1;
