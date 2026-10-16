# Candidate Gold Sets

These files are a multilingual **candidate** evaluation set. They use
fictitious text generated independently of the training corpus and are intended
to validate the evaluation pipeline.

They are not a production gold benchmark yet: have two human reviewers check
each span and label, then add de-identified real-world texts before reporting
production-quality metrics. Never merge records in this directory into the
training corpus.

Every language has 60 records with `split: "test"`: 20 `sensitive`, 20
`benign_context`, and 20 `negative`. Evaluate one language at a time so that
per-language quality remains visible. Regenerate the set with
`node training/generate_gold_candidates.mjs`.
