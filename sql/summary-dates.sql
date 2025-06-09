WITH data AS (
	SELECT
		to_char(time, 'YYYY-MM') AS month
	FROM hacker_news.items
	WHERE
	    type = 'story'
        AND title IS NOT NULL
        AND url IS NOT NULL
        AND score IS NOT NULL
		AND dead IS NULL OR dead = false
)
SELECT
	month,
	COUNT(1) as count
FROM data
GROUP BY
	month
;