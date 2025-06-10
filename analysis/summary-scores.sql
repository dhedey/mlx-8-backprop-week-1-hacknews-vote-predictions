WITH
data AS (
	SELECT
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
	score,
	COUNT(1) as count
FROM data
GROUP BY
	score
ORDER BY
	score ASC
;