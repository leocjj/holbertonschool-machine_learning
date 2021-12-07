--  Computes and store the average score for a student.
DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser (IN user_id INT)
BEGIN
	DECLARE avg_score FLOAT;
    SET avg_score = (SELECT AVG(score) FROM corrections INNER JOIN users ON corrections.user_id = users.id WHERE corrections.user_id=user_id);
	UPDATE users
    SET average_score = avg_score
    WHERE id = user_id;
END //
DELIMITER ;
