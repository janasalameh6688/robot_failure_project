# Logical Schema: Robot Failure Prediction

Maps the ER diagram (`er_diagram.png`) to relational tables. Decisions here are made before any SQL is written (`sql/schema.sql`, Module 6).

Source data: AI4I 2020 Predictive Maintenance Dataset (10,000 rows, 14 columns). The dataset is itself synthetic. The maintenance notes are synthetic and generated for this project.

## Relationships

| Parent | Child | Cardinality | Meaning |
|---|---|---|---|
| product_types | machines | 1 to many | One product type (L, M, H) has many machines |
| machines | sensor_readings | 1 to 1 | Each machine has one reading (every Product ID in the data is unique) |
| machines | failure_events | 1 to many | A machine can have failure events |
| machines | maintenance_notes | 1 to many | A machine can have several notes |
| failure_events | event_causes | 1 to many | One failure can have several causes |
| failure_causes | event_causes | 1 to many | One cause appears in many failures |

`event_causes` is a junction table, giving a many-to-many link between `failure_events` and `failure_causes`.

## Tables

### product_types

| Column | Type | Key | Required | Notes |
|---|---|---|---|---|
| type_code | char(1) | PK | yes | CHECK in ('L','M','H') |

### machines

| Column | Type | Key | Required | Notes |
|---|---|---|---|---|
| machine_id | integer | PK | yes | from `UDI` |
| product_id | varchar(10) | UNIQUE | yes | from `Product ID`, e.g. M14860 |
| type_code | char(1) | FK to product_types.type_code | yes | from `Type` |

### sensor_readings

| Column | Type | Key | Required | Notes |
|---|---|---|---|---|
| reading_id | integer | PK | yes | generated identity |
| machine_id | integer | FK to machines.machine_id, UNIQUE | yes | UNIQUE makes it 1 to 1 |
| air_temp_k | numeric(4,1) | | yes | from `Air temperature [K]` |
| process_temp_k | numeric(4,1) | | yes | from `Process temperature [K]` |
| rotational_speed_rpm | integer | | yes | CHECK > 0 |
| torque_nm | numeric(4,1) | | yes | CHECK >= 0 |
| tool_wear_min | integer | | yes | CHECK >= 0 |

### failure_events

| Column | Type | Key | Required | Notes |
|---|---|---|---|---|
| failure_id | integer | PK | yes | generated identity |
| machine_id | integer | FK to machines.machine_id | yes | |

One row for each row of the data where `Machine failure` = 1 (339 rows).

### failure_causes

| Column | Type | Key | Required | Notes |
|---|---|---|---|---|
| cause_code | varchar(3) | PK | yes | TWF, HDF, PWF, OSF, RNF |
| description | varchar(60) | | yes | e.g. "Tool wear failure" |

Lookup table with 5 rows.

### event_causes

| Column | Type | Key | Required | Notes |
|---|---|---|---|---|
| failure_id | integer | PK (part 1), FK to failure_events.failure_id | yes | |
| cause_code | varchar(3) | PK (part 2), FK to failure_causes.cause_code | yes | |

Composite primary key `(failure_id, cause_code)`.

### maintenance_notes

| Column | Type | Key | Required | Notes |
|---|---|---|---|---|
| note_id | integer | PK | yes | generated identity |
| machine_id | integer | FK to machines.machine_id | yes | |
| note_date | date | | yes | synthetic, the dataset has no dates |
| note_text | text | | yes | synthetic, free text for RAG |

## Design decisions

1. **The failure label is derived, not stored.** `Machine failure` becomes "a machine has a row in failure_events". A view (Module 8) will recreate the 0/1 column for model training.
2. **Failures with no recorded cause.** 9 failures have no cause flag. They get a `failure_events` row and no `event_causes` rows, meaning the cause is unknown.
3. **Cause flags without a failure.** 18 rows have a cause flag but `Machine failure` = 0. They get no `failure_events` row, so their flags are not loaded. Report this in the EDA.
4. **Leakage.** The cause tables describe why a machine failed, so they must never be used as model inputs.
5. **Exact decimals.** Temperatures and torque use `numeric(4,1)` because the data has one decimal place and exact storage avoids rounding surprises.
