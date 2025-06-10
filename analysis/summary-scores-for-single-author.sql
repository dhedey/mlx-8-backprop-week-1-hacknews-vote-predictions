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
		-- AND by = 'rbanffy' -- Most frequent author
		-- AND by = 'Tomte' -- 2nd most frequent author
		AND by = 'bond' -- 1212nd most frequent author
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