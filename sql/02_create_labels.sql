CREATE TABLE IF NOT EXISTS churn_labels (
    customer_id VARCHAR(20) NOT NULL,
    label_version VARCHAR(30) NOT NULL,
    churned TINYINT(1) NOT NULL,
    label_source VARCHAR(50) NOT NULL,
    observation_date DATE NULL,
    label_window_end DATE NULL,
    PRIMARY KEY (customer_id, label_version),
    CONSTRAINT fk_churn_labels_customer
        FOREIGN KEY (customer_id) REFERENCES customers_raw(customer_id)
        ON DELETE CASCADE,
    CHECK (churned IN (0, 1)),
    CHECK (
        (observation_date IS NULL AND label_window_end IS NULL)
        OR label_window_end >= observation_date
    )
);
