SELECT
    *
    FROM hacker_news.items
    WHERE
        type = 'story'
        AND title IS NULL
        AND text IS NULL
    LIMIT 10
;