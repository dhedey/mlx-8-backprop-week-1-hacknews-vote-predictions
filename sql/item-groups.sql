WITH data AS (
	SELECT
	type,
	score IS NOT NULL AND score > 0 AS has_score,
	title IS NOT NULL AS has_title,
	text IS NOT NULL AS has_text,
	url IS NOT NULL AS has_url,
	by IS NOT NULL AS has_author,
	dead AS is_dead
	FROM hacker_news.items
	WHERE
		time > '2024-01-01T00:00:00Z'
)
SELECT type, has_score, has_title, has_url, has_text, has_author, is_dead, COUNT(1) as count FROM data GROUP BY type, has_title, has_url, has_score, has_text, has_author, is_dead;