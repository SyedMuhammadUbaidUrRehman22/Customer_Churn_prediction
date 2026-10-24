CREATE TABLE IF NOT EXISTS monitoring_outcomes (
    batch_sha256 CHAR(64) NOT NULL,
    customer_id VARCHAR(20) NOT NULL,
    label_version VARCHAR(30) NOT NULL,
    label_window_end DATE NOT NULL,
    observed_through DATE NOT NULL,
    available_at DATETIME(6) NOT NULL,
    churned TINYINT(1) NOT NULL,
    evidence_reference VARCHAR(255) NOT NULL,
    PRIMARY KEY (batch_sha256, customer_id),
    FOREIGN KEY (batch_sha256, customer_id) REFERENCES scoring_history(batch_sha256, customer_id),
    CHECK (churned IN (0, 1)),
    CHECK (observed_through >= label_window_end AND available_at >= label_window_end),
    CHECK (label_version <> 'telco_snapshot_v1' AND CHAR_LENGTH(TRIM(evidence_reference)) > 0)
) ENGINE=InnoDB;
