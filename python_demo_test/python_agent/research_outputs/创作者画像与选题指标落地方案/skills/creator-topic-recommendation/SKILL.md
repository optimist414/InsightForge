---
name: creator-topic-recommendation
description: Generate an evidence-grounded creator profile and use it to match current hot topics. Use when the user asks for creator positioning, creator tags, long/short video classification, creator size, topic selection, or content ideas based on creator history and hot-news data.
---

# Creator Topic Recommendation

## Scope

This skill covers creator profiling and the handoff to topic scoring. `creator_profile_builder` can collect and normalize creator evidence internally before the profile LLM is called; code validates and normalizes the result.

## Workflow

1. Call `creator_profile_builder` first. For a Bilibili UID or space URL, pass `creator_id` plus `platform=bilibili`, or pass `creator_url`; the tool internally collects at most 50 recent public video metadata rows, formats them for the profile prompt, and generates the profile. Do not pass `history` or call `bilibili_creator_videos` as a separate Agent step.
2. Use the validated profile. Do not infer missing follower count, audience, or format.
3. When database trends are relevant, retrieve news in two SQL stages: first retrieve tag-matched news candidates, then retrieve ranking records only for the selected candidate IDs. Do not use one multi-table query to retrieve both.
4. Call `web_search_fetcher` when the user asks for current external evidence or when database evidence is stale. Treat search snippets as leads, not proof.
5. Normalize sources and cluster events before scoring.
6. Call `proposal_topic_value_scorer` or `creator_topic_recommender` with the validated profile.
7. Call `content_angle_planner` for at most five angles and `risk_evidence_checker` before presenting a strong recommendation.
8. Match event domains and formats against the validated profile. Use creator size as context, not as a popularity bonus.
9. Report evidence, unknowns, risk, and recommendation separately.

## Database query guidance

Use the project database as a structured hot-news source when the user asks for project trends, recent platform hotspots, ranking changes, or historical comparison.

### Tag catalogue available to the agent

Infer one to three relevant tag IDs from the validated creator profile (`main_tags`, `avoid_tags`, content positioning), the user request, and the following fixed catalogue. Use the IDs in SQL; use names/codes only for reasoning and presentation.

| tag_id | code | name |
| --- | --- | --- |
| 1 | society | 社会 |
| 2 | international | 国际 |
| 3 | finance | 财经 |
| 4 | technology | 科技 |
| 5 | ai | AI |
| 6 | entertainment | 娱乐 |
| 7 | film_tv | 影视 |
| 8 | sports | 体育 |
| 9 | education | 教育 |
| 10 | health | 健康 |
| 11 | consumption | 消费 |
| 12 | e_commerce | 电商 |
| 13 | automobile | 汽车 |
| 14 | game | 游戏 |
| 15 | tourism | 旅游 |

The news/tag relation table is `hot_news_tag(hot_news_id, tag_id)`. There is no `news_tag_record` table. Do not join `news_tag` during retrieval: the catalogue above already supplies the tag mapping.

### Required two-stage database retrieval

Use `hot_news_fetcher` for creator-topic database retrieval. Call it once with `stage=candidates` and then once with `stage=ranking`; each stage emits one read-only `SELECT` with explicit columns. Never use `SELECT *`, `INSERT`, `UPDATE`, `DELETE`, DDL, `SHOW`, multiple statements, or CTE (`WITH`) syntax. Use `java_sql_proxy_query` only when the fixed two-stage tool cannot express a clearly bounded read-only historical comparison.

#### Stage 1: retrieve tag-matched news candidates

Convert relative dates such as "today" and "recent three days" into an explicit half-open time range `[start_time, end_time)`. This query may join **only** `hot_news` and `hot_news_tag`; it must not join `hot_news_record`, `news_tag`, or any other table. Retrieve at most 50 candidates.

```sql
SELECT DISTINCT
  hn.id,
  hn.title,
  hn.url,
  hn.platform_code,
  hn.platform_name,
  hn.first_seen_time,
  hn.latest_seen_time
FROM hot_news AS hn
INNER JOIN hot_news_tag AS hnt ON hnt.hot_news_id = hn.id
WHERE hnt.tag_id IN (4, 5)
  AND hn.latest_seen_time >= 'YYYY-MM-DD 00:00:00'
  AND hn.latest_seen_time < 'YYYY-MM-DD 00:00:00'
ORDER BY hn.latest_seen_time DESC
LIMIT 50
```

Use one to three tag IDs. Add a platform filter only when the user or creator strategy requires it. `avoid_tags` should normally be handled during scoring; use `NOT IN (...)` only when the user explicitly wants those domains excluded. If no tag is suitable, query `hot_news` alone as the candidate stage, still with explicit columns, time window, and `LIMIT 50`.

#### Stage 2: retrieve ranking evidence after candidate selection

Review the Stage 1 results, deduplicate them, and select only the strongest candidates for ranking evidence (normally 10 to 20 IDs). Then make a second query that reads only `hot_news_record`; do not join it back to candidates or tags.

```sql
SELECT
  hot_news_id,
  platform_code,
  rank_no,
  record_time
FROM hot_news_record
WHERE hot_news_id IN (101, 102, 103)
  AND record_time >= 'YYYY-MM-DD 00:00:00'
  AND record_time < 'YYYY-MM-DD 00:00:00'
ORDER BY hot_news_id ASC, record_time DESC
LIMIT 100
```

Keep the ranking query within 100 rows. If more history is needed, narrow the candidate IDs or split the work into additional bounded queries. Merge Stage 2 rows with Stage 1 news candidates in agent memory using `hot_news_id = hot_news.id`; then wrap the merged result as a `data_sources` item with `source_type=database`, and call `source_bundle_normalizer` before scoring. Do not score raw database rows as if they were an event cluster.

### Tool stage mapping

`hot_news_fetcher(stage=candidates)` is the candidate stage and accepts one to three `tag_ids`; it returns at most 50 rows from `hot_news` and `hot_news_tag`. After reviewing and deduplicating the candidates, call `hot_news_fetcher(stage=ranking, candidate_ids=[...])`; it returns at most 100 ranking rows from `hot_news_record`. Do not use one call to retrieve both stages, and do not treat the candidate result as ranking evidence until the second stage is complete.

### Database and web fallback

The database is not automatically the latest external truth. If the requested time window has no rows, the newest database record is stale, or the user explicitly asks for internet-wide/current information, call `web_search_fetcher` and label those records as external evidence. Do not repeat equivalent database queries indefinitely.

The database tool currently returns the latest ranking record selected within the requested window. If a precise rise/fall calculation needs multiple historical snapshots, obtain ranking history with the general read-only SQL tool or report that the trend evidence is incomplete.

## Profile rules

- `creator_identity_tags` must be short, reusable tags, not copied titles.
- `content_format.primary` is one of `short_video`, `long_video`, `mixed`, `unknown`.
- `creator_size.level` is `unknown` when follower count is unavailable.
- Keep `evidence_summary.supported_claims` and `unknowns` visible to downstream reasoning.
- If the profile LLM returns invalid JSON, stop the profile stage. Use rule fallback only when the caller explicitly enabled it.

## Output contract

The parsed profile must expose `domain`, `main_tags`, `avoid_tags`, `content_positioning`, `content_format`, `creator_size`, `target_platforms`, and `history_summary`. Do not use the raw LLM text as a business object.

## References

Read `references/profile-output.md` for field meanings and examples.
