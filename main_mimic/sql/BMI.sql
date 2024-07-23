WITH weight_height AS (
    SELECT
        ce.subject_id,
        MAX(CASE WHEN ce.itemid = 224639 THEN ce.valuenum END) AS weight, 
        MAX(CASE WHEN ce.itemid = 226707 THEN ce.valuenum END) AS height  
    FROM
        `physionet-data.mimiciv_icu.chartevents` ce
    WHERE
        ce.itemid IN (224639, 226707)
        AND ce.valuenum IS NOT NULL
    GROUP BY
        ce.subject_id
)
SELECT
    p.subject_id,
    wh.weight,
    wh.height,
    ROUND(wh.weight / ((wh.height / 100) * (wh.height / 100)), 2) AS BMI 
FROM
    `physionet-data.mimiciv_hosp.patients` p
JOIN
    `physionet-data.mimiciv_hosp.admissions` adm ON p.subject_id = adm.subject_id
JOIN
    weight_height wh ON p.subject_id = wh.subject_id
    where wh.height >0 and wh.weight >0
GROUP BY
    p.subject_id, wh.weight, wh.height;
