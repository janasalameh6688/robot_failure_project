
-- 1. Count machines by product type
SELECT
    type_code,
    COUNT(*) AS machine_count
FROM machines
GROUP BY type_code
ORDER BY type_code;


-- 2. Summarize sensor measurements by machine type
SELECT
    m.type_code,
    COUNT(*) AS machine_count,
    ROUND(AVG(sr.air_temp_k), 2) AS avg_air_temp_k,
    ROUND(AVG(sr.process_temp_k), 2) AS avg_process_temp_k,
    ROUND(AVG(sr.rotational_speed_rpm), 2) AS avg_speed_rpm,
    ROUND(AVG(sr.torque_nm), 2) AS avg_torque_nm
FROM machines AS m
JOIN sensor_readings AS sr
    ON sr.machine_id = m.machine_id
GROUP BY m.type_code
ORDER BY m.type_code;


-- 3. Count recorded failure events
SELECT
    COUNT(*) AS total_failure_events
FROM failure_events;


-- 4. Rank machines by number of failure events
SELECT
    m.machine_id,
    m.product_id,
    COUNT(fe.failure_id) AS failure_event_count
FROM machines AS m
LEFT JOIN failure_events AS fe
    ON fe.machine_id = m.machine_id
GROUP BY m.machine_id, m.product_id
HAVING COUNT(fe.failure_id) > 0
ORDER BY failure_event_count DESC, m.machine_id
LIMIT 20;


-- 5. Count recorded cause assignments by cause code
SELECT
    fc.cause_code,
    fc.description,
    COUNT(ec.failure_id) AS recorded_cause_assignments
FROM failure_causes AS fc
LEFT JOIN event_causes AS ec
    ON ec.cause_code = fc.cause_code
GROUP BY fc.cause_code, fc.description
ORDER BY recorded_cause_assignments DESC, fc.cause_code;


-- 6. Compare average sensor measurements for machines with
-- and without recorded failure events
SELECT
    CASE
        WHEN fe.machine_id IS NULL THEN 'No recorded event'
        ELSE 'Has recorded event'
    END AS event_status,
    COUNT(*) AS machine_count,
    ROUND(AVG(sr.air_temp_k), 2) AS avg_air_temp_k,
    ROUND(AVG(sr.process_temp_k), 2) AS avg_process_temp_k,
    ROUND(AVG(sr.rotational_speed_rpm), 2) AS avg_speed_rpm,
    ROUND(AVG(sr.torque_nm), 2) AS avg_torque_nm,
    ROUND(AVG(sr.tool_wear_min), 2) AS avg_tool_wear_min
FROM machines AS m
JOIN sensor_readings AS sr
    ON sr.machine_id = m.machine_id
LEFT JOIN failure_events AS fe
    ON fe.machine_id = m.machine_id
GROUP BY event_status
ORDER BY event_status;


-- 7. Find machines with high tool wear
SELECT
    m.machine_id,
    m.product_id,
    m.type_code,
    sr.tool_wear_min,
    sr.torque_nm,
    sr.rotational_speed_rpm
FROM machines AS m
JOIN sensor_readings AS sr
    ON sr.machine_id = m.machine_id
WHERE sr.tool_wear_min >= 200
ORDER BY sr.tool_wear_min DESC, sr.torque_nm DESC;


-- 8. Find machines with a large temperature difference
SELECT
    machine_id,
    product_id,
    type_code,
    air_temp_k,
    process_temp_k,
    temperature_difference_k
FROM machine_sensor_summary
WHERE temperature_difference_k >= 10
ORDER BY temperature_difference_k DESC, machine_id;
