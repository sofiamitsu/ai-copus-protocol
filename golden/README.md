# Golden reference lectures

Example active-learning lectures that every Faculty Feedback Report compares
against — **one comparison per lecture**. Each is by a different instructor, so
they are never pooled into a single profile.

```
golden/
  golden_lectures.csv            one row per reference lecture (report order)
  <folder>/results_multimodal.csv
```

`golden_lectures.csv` columns:

| column | |
|---|---|
| `folder` | sub-folder holding that lecture's `results_multimodal.csv` |
| `professor_name` | shown in the report heading and chart legend |
| `lecture_title` | optional, shown after the name |
| `youtube_url` | clickable link in the report so the professor can watch it |

A results folder not listed in the manifest is ignored.

`results_multimodal.csv` is the multimodal-arm output of `run.py` for that
lecture. Commit only `window_index, window_start, window_end, copus_codes` —
the report reads nothing else, and Gemini's free-text `reasoning` stays out of
the repo.
