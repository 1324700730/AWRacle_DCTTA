# Bundled Rain100L example

The eight Query/GT pairs are Rain100L IDs 6, 13, 24, 29, 48, 72, 83 and 89. The separate Support pair is AWRaCLe `Train/Derain` low-rain training-manifest entry 0 (`input/4324.jpg`, `target/4324.jpg`). The source Pilot checked path and SHA-256 non-overlap against all 100 Rain100L test Query/GT files; details are in `../../docs/pilot/selection_audit.json`.

These are regular copied files, not links to the source server. The GT files are for evaluation after adaptation and must never be used to train or tune the target stream. This small example is a seen deraining check; it is not a complete benchmark or an unseen-operator split.
