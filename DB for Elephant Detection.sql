CREATE DATABASE elephant_detection_db;

USE elephant_detection_db;


CREATE TABLE elephant_detection_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    status VARCHAR(10) NOT NULL
);

SELECT * FROM elephant_detection_log ORDER BY timestamp DESC;