# Data placement

The competition dataset is not included in this repository.

After obtaining authorised access, place the files here:

```text
data/train.xlsx
data/leaderboard.xlsx
```

The training file is expected to contain:

- `row_id`
- `anon_user_id`
- `post_id`
- `post`
- `suicide risk`
- `evidence for suicide risk level`
- `factors`

The leaderboard file is expected to contain at least `row_id` and `post` and
may also contain anonymised user and post identifiers.

Do not commit either file. The project `.gitignore` also excludes Excel files,
submission CSV files, cached tensors, probabilities, and model checkpoints.

