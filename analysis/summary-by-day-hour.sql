WITH data AS (
	SELECT
		to_char(time, 'Dy') AS day_of_week,
	    to_char(time, 'D') AS day_of_week_num,
		to_char(time, 'HH') AS hour_of_day,
	    LOG(score) AS log_10_score,
		score
	FROM hacker_news.items
	WHERE
	    type = 'story'
        AND title IS NOT NULL
        AND url IS NOT NULL
        AND score IS NOT NULL
        AND score >= 1
		AND dead IS NULL OR dead = false
)
SELECT
	day_of_week,
	hour_of_day,
	COUNT(1) as total_count,
	AVG(score) as avg_score,
	AVG(log_10_score) AS avg_log_10_score
FROM data
GROUP BY
	day_of_week,
	day_of_week_num,
	hour_of_day
ORDER BY
	day_of_week_num ASC,
	hour_of_day ASC
;