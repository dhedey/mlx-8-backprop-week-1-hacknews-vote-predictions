SELECT
    id,
    title,
	by,
	url,
    time,
	score
    FROM hacker_news.items
    WHERE
        type = 'story'
        AND title IS NOT NULL
        AND url IS NOT NULL
        AND score IS NOT NULL
		AND dead IS NULL OR dead = false
		AND time >= '2024-01-01T00:00:00Z' 
		AND time < '2025-01-01T00:00:00Z' 
	ORDER BY time ASC
;