WITH data AS (
	SELECT
		to_char(time, 'YYYY-MM') AS month,
	    LOG(score) AS log_10_score,
		score
	FROM hacker_news.items
	WHERE
	    type = 'story'
        AND title IS NOT NULL
        AND url IS NOT NULL
        AND score IS NOT NULL AND score >= 1
		AND (dead IS NULL OR dead = false)
)
SELECT
	month,
	COUNT(1) as count,
	AVG(score) as avg_score,
	AVG(log_10_score) AS avg_log_10_score
FROM data
GROUP BY
	month
;