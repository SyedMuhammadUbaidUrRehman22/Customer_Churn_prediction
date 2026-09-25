CREATE TABLE IF NOT EXISTS model_registry (
    model_id VARCHAR(50) PRIMARY KEY,
    trained_at DATETIME(6) NOT NULL,
    registered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    feature_set_version VARCHAR(30) NOT NULL,
    label_version VARCHAR(30) NOT NULL,
    evaluation_scope ENUM('benchmark', 'production') NOT NULL,
    approved TINYINT(1) NOT NULL DEFAULT 0,
    pr_auc DOUBLE NOT NULL,
    roc_auc DOUBLE NOT NULL,
    precision_at_k DOUBLE NOT NULL,
    recall_at_k DOUBLE NOT NULL,
    precision_at_threshold DOUBLE NOT NULL,
    recall_at_threshold DOUBLE NOT NULL,
    f1 DOUBLE NOT NULL,
    brier_score DOUBLE NOT NULL,
    low_threshold DOUBLE NOT NULL,
    high_threshold DOUBLE NOT NULL,
    artifact_path VARCHAR(1024) NOT NULL,
    report_sha256 CHAR(64) NOT NULL UNIQUE,
    dataset_sha256 CHAR(64) NOT NULL,
    CHECK (approved IN (0, 1)),
    CHECK (evaluation_scope <> 'benchmark' OR approved = 0),
    CHECK (0 < low_threshold AND low_threshold < high_threshold AND high_threshold <= 1),
    CHECK (pr_auc BETWEEN 0 AND 1 AND roc_auc BETWEEN 0 AND 1
           AND precision_at_k BETWEEN 0 AND 1 AND recall_at_k BETWEEN 0 AND 1
           AND precision_at_threshold BETWEEN 0 AND 1 AND recall_at_threshold BETWEEN 0 AND 1
           AND f1 BETWEEN 0 AND 1 AND brier_score BETWEEN 0 AND 1)
) ENGINE=InnoDB;
