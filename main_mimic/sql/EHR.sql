

SELECT  Distinct
    p.subject_id,
    p.gender,
    p.anchor_age,
    a.race as ethnicity
FROM 
    `physionet-data.mimiciv_hosp.patients` AS p
JOIN 
    `physionet-data.mimiciv_hosp.admissions` AS a
ON 
    p.subject_id = a.subject_id
WHERE gender is not null and race is not null;
