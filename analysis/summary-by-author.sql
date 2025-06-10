WITH data AS (
	SELECT
		by AS author,
	    LOG(score) AS log_10_score,
		score
	FROM hacker_news.items_by_month
	WHERE
	    type = 'story'
        AND title IS NOT NULL
        AND url IS NOT NULL
        AND score IS NOT NULL
        AND score IS NOT NULL AND score >= 1
		AND (dead IS NULL OR dead = false)
)
SELECT
	author,
	COUNT(1) as total_count,
	AVG(score) as avg_score,
	AVG(log_10_score) AS avg_log_10_score
FROM data
GROUP BY
	author
ORDER BY
	total_count DESC
;