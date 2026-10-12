
-- View 1: One row per machine with its sensor readings
CREATE OR REPLACE VIEW machine_sensor_summary AS
SELECT
    m.machine_id,
    m.product_id,
    m.type_code,
    sr.air_temp_k,
    sr.process_temp_k,
    sr.process_temp_k - sr.air_temp_k AS temperature_difference_k,
    sr.rotational_speed_rpm,
    sr.torque_nm,
    sr.tool_wear_min,
    sr.rotational_speed_rpm * sr.torque_nm AS mechanical_power_proxy
FROM machines AS m
JOIN sensor_readings AS sr
    ON sr.machine_id = m.machine_id;


-- View 2: Machine-level failure-event and cause summary
CREATE OR REPLACE VIEW machine_failure_summary AS
SELECT
    m.machine_id,
    m.product_id,
    COUNT(DISTINCT fe.failure_id) AS failure_event_count,
    COUNT(DISTINCT ec.cause_code) AS distinct_cause_count,
    COALESCE(
        STRING_AGG(DISTINCT ec.cause_code, ', ' ORDER BY ec.cause_code),
        'None recorded'
    ) AS cause_codes
FROM machines AS m
LEFT JOIN failure_events AS fe
    ON fe.machine_id = m.machine_id
LEFT JOIN event_causes AS ec
    ON ec.failure_id = fe.failure_id
GROUP BY m.machine_id, m.product_id;


-- Stored procedure: Add a maintenance note for an existing machine
CREATE OR REPLACE PROCEDURE add_maintenance_note(
    p_machine_id INTEGER,
    p_note_text TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF p_note_text IS NULL OR LENGTH(TRIM(p_note_text)) = 0 THEN
        RAISE EXCEPTION 'Maintenance note must not be empty';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM machines
        WHERE machine_id = p_machine_id
    ) THEN
        RAISE EXCEPTION 'Machine ID % does not exist', p_machine_id;
    END IF;

    INSERT INTO maintenance_notes (machine_id, note_text)
    VALUES (p_machine_id, TRIM(p_note_text));
END;
$$;
