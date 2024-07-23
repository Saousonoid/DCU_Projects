WITH latest_record AS (
    SELECT
        ce.subject_id,
        ce.charttime,
        ce.valuenum,
        ROW_NUMBER() OVER (PARTITION BY ce.subject_id ORDER BY ce.charttime DESC) AS rn
    FROM
        `physionet-data.mimiciv_icu.chartevents` ce
    JOIN
        `physionet-data.mimiciv_icu.d_items` di ON di.itemid = ce.itemid
    WHERE
        ce.itemid = 220615
        AND ce.valuenum IS NOT NULL
)
SELECT
    subject_id,

    valuenum as Serum_Creatinine

FROM
    latest_record
WHERE
    rn = 1;
