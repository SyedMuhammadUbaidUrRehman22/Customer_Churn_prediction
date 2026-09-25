CREATE TABLE IF NOT EXISTS model_predictions (
    customer_id VARCHAR(20) NOT NULL,
    model_id VARCHAR(50) NOT NULL,
    scoring_mode ENUM('benchmark', 'production') NOT NULL,
    scored_at DATETIME(6) NOT NULL,
    feature_set_version VARCHAR(30) NOT NULL,
    feature_snapshot_sha256 CHAR(64) NOT NULL,
    churn_probability DOUBLE NOT NULL,
    risk_tier ENUM('low', 'medium', 'high') NOT NULL,
    PRIMARY KEY (customer_id, model_id, scoring_mode),
    CONSTRAINT fk_predictions_model FOREIGN KEY (model_id) REFERENCES model_registry(model_id),
    CHECK (churn_probability BETWEEN 0 AND 1),
    INDEX idx_prediction_ranking (model_id, scoring_mode, churn_probability)
) ENGINE=InnoDB;
