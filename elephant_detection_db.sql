CREATE DATABASE IF NOT EXISTS elephant_detection_db;
USE elephant_detection_db;

DROP TABLE IF EXISTS detection_logs;

CREATE TABLE detection_logs (
    detection_id INT AUTO_INCREMENT PRIMARY KEY,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    status ENUM('DETECTED', 'NOT_DETECTED', 'UNKNOWN') NOT NULL,
    confidence_score DECIMAL(5,2),
    camera_id VARCHAR(50) NOT NULL,
    location VARCHAR(100),
    image_path VARCHAR(255)
);

CREATE INDEX idx_detected_at ON detection_logs(detected_at);
CREATE INDEX idx_status ON detection_logs(status);

INSERT INTO detection_logs (status, confidence_score, camera_id, location, image_path)
VALUES 
('DETECTED', 92.45, 'CAM_01', 'Forest Zone A', '/images/elephant1.jpg'),
('NOT_DETECTED', 0.00, 'CAM_02', 'Forest Zone B', NULL),
('DETECTED', 88.10, 'CAM_01', 'Forest Zone A', '/images/elephant2.jpg');

SELECT * 
FROM detection_logs
ORDER BY detected_at DESC;