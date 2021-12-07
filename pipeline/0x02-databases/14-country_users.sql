-- Primary key auto increment
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    country ENUM('US','CO','TN') DEFAULT 'US',
    UNIQUE(email));
