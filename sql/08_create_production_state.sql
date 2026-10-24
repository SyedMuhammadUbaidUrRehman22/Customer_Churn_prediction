CREATE TABLE IF NOT EXISTS production_state (
    state_id TINYINT PRIMARY KEY,
    model_id VARCHAR(50) NULL,
    run_id CHAR(32) NULL,
    FOREIGN KEY (model_id) REFERENCES model_registry(model_id),
    FOREIGN KEY (run_id) REFERENCES deployment_runs(run_id),
    CHECK (state_id = 1),
    CHECK ((model_id IS NULL AND run_id IS NULL) OR (model_id IS NOT NULL AND run_id IS NOT NULL))
) ENGINE=InnoDB;
