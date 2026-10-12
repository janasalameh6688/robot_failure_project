DROP TABLE IF EXISTS maintenance_notes CASCADE;
DROP TABLE IF EXISTS event_causes CASCADE;
DROP TABLE IF EXISTS failure_events CASCADE;
DROP TABLE IF EXISTS failure_causes CASCADE;
DROP TABLE IF EXISTS sensor_readings CASCADE;
DROP TABLE IF EXISTS machines CASCADE;
DROP TABLE IF EXISTS product_types CASCADE;

CREATE TABLE product_types (
    type_code CHAR(1) PRIMARY KEY
        CHECK (type_code IN ('L', 'M', 'H'))
);

CREATE TABLE machines (
    machine_id INTEGER PRIMARY KEY,
    product_id VARCHAR(10) NOT NULL UNIQUE,
    type_code  CHAR(1) NOT NULL
        REFERENCES product_types (type_code)
);

CREATE TABLE sensor_readings (
    reading_id           INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    machine_id           INTEGER NOT NULL UNIQUE
        REFERENCES machines (machine_id) ON DELETE CASCADE,
    air_temp_k           NUMERIC(4,1) NOT NULL CHECK (air_temp_k > 0),
    process_temp_k       NUMERIC(4,1) NOT NULL CHECK (process_temp_k > 0),
    rotational_speed_rpm INTEGER NOT NULL CHECK (rotational_speed_rpm > 0),
    torque_nm            NUMERIC(4,1) NOT NULL CHECK (torque_nm >= 0),
    tool_wear_min        INTEGER NOT NULL DEFAULT 0 CHECK (tool_wear_min >= 0)
);

CREATE TABLE failure_events (
    failure_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    machine_id INTEGER NOT NULL
        REFERENCES machines (machine_id) ON DELETE CASCADE
);

CREATE TABLE failure_causes (
    cause_code  VARCHAR(3) PRIMARY KEY,
    description VARCHAR(60) NOT NULL UNIQUE
);

CREATE TABLE event_causes (
    failure_id INTEGER NOT NULL
        REFERENCES failure_events (failure_id) ON DELETE CASCADE,
    cause_code VARCHAR(3) NOT NULL
        REFERENCES failure_causes (cause_code),
    PRIMARY KEY (failure_id, cause_code)
);

CREATE TABLE maintenance_notes (
    note_id    INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    machine_id INTEGER NOT NULL
        REFERENCES machines (machine_id) ON DELETE CASCADE,
    note_date  DATE NOT NULL DEFAULT CURRENT_DATE,
    note_text  TEXT NOT NULL CHECK (length(trim(note_text)) > 0)
);

INSERT INTO product_types (type_code) VALUES ('L'), ('M'), ('H');

INSERT INTO failure_causes (cause_code, description) VALUES
    ('TWF', 'Tool wear failure'),
    ('HDF', 'Heat dissipation failure'),
    ('PWF', 'Power failure'),
    ('OSF', 'Overstrain failure'),
    ('RNF', 'Random failure');
