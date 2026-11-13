# Phase 0: NoCaps yes/no object questions (split A)

Yes/no questions ("Is there a dog in the image?") on NoCaps val images. They are scored on the existing
split-A caption checkpoints. The goal is to see whether exposure changes answers about objects the
training caption did or did not mention. No images or captions are included. Download NoCaps val
yourself and find each image by `file_name`.

## Files

| File | What it is |
|---|---|
| `nocaps_yesno_A.share.json` | **Use this one.** The 8,946 main-analysis questions, each with its image file, answer, analysis cell, and seen / unseen group per checkpoint type |
| `nocaps_yesno_A.meta.json` | Build record: parameters, sha256 of every input, drop counts, cell counts |
| `README.md` | This file |

Not committed, because they can be rebuilt: `nocaps_yesno_A.jsonl` (all 41,327 candidates, 25 MB) and the
manual-review lists (`*.phrase_review.csv`, `*.mention_check.csv`). To rebuild everything (deterministic, seed 20261111):

```bash
python scripts/build_phase0_questions.py --split A   # needs spaCy en_core_web_sm and data/openimages (see its SOURCE.txt)
```

## `share.json` layout

```json
{
  "description": "...",
  "seen_ids_sha256": {"1000": "fccf...", "300": "3c68...", "100": "b32f...", "30": "ced6..."},
  "counts": {"1000": {"seen": {...}, "unseen": {...}}, ...},
  "questions": [
    {"qid": "0:02p0tk3", "image_id": 0, "file_name": "0013ea2087020901.jpg", "open_images_id": "0013ea2087020901",
     "domain": "in", "label_name": "Human body", "question": "Is there a human body in the image?", "answer": "yes",
     "cell": "yes/none", "mention_train": "none", "n_refs_mentioning": 0, "is_body_part": true,
     "group": {"1000": "unseen", "300": "unseen", "100": "unseen", "30": "unseen"}}
  ]
}
```

| Field | Meaning |
|---|---|
| `answer` | Ground truth from Open Images human-verified image labels (verified present / verified absent) |
| `mention_train` | Whether the image's **training caption** (`refs[train_ref_idx]` of split A) names the object: `direct`, `descendant` (a subclass is named, e.g. "dog" for Animal) or `none`. Set for unseen images too, with the same rule |
| `cell` | Analysis cell: `yes/direct`, `yes/descendant`, `yes/none`, `no` |
| `n_refs_mentioning` | How many of the 10 NoCaps captions name the object (a salience covariate) |
| `group[N_exp]` | The image's group for checkpoints exposed to N_exp images: `seen` (in that exposure set), `unseen` (split A's unseen half) or `unexposed` (seen half but not exposed: **leave out of seen vs unseen**) |

## Seen / unseen: pick the column that matches the checkpoint

| Checkpoint | Column | Seen questions | Unseen questions |
|---|---|---|---|
| Main runs (paired, caption-only, style-only; every d and seed) | `"1000"` | 2,003 | 4,465 |
| K sweep, 300 images | `"300"` | 593 | 4,465 |
| K sweep, 100 images | `"100"` | 226 | 4,465 |
| K sweep, 30 images | `"30"` | 74 | 4,465 |
| d=0 (nothing exposed) | same column as the run you compare it with | | |

Seen is the checkpoint's actual exposure set (`contam.eval.evaluate.load_seen_definition`), **not** the
2,250-image seen half. The sets are nested (30 ⊂ 100 ⊂ 300 ⊂ 1,000). To check that you have the matching
version, compare `seen_ids_sha256` with the run spec's `exposure_ids_sha256`.

```python
import json
d = json.load(open("data/phase0/nocaps_yesno_A.share.json"))
qs = [q for q in d["questions"] if q["group"]["1000"] != "unexposed"]   # a main (N_exp = 1000) run
seen = [q for q in qs if q["group"]["1000"] == "seen"]
```

Main-analysis counts by cell (all images):

| yes/direct | yes/descendant | yes/none | no | questions | images |
|---|---|---|---|---|---|
| 1,271 | 145 | 3,830 | 3,700 | 8,946 | 2,992 |

## How the questions were chosen

- **Yes**: verified-present labels among the 600 boxable (object) classes. **No**: verified-absent labels
  of the same classes. A "no" is dropped when it contradicts a present label (e.g. Mammal absent while Dog
  is present) or when any of the 10 captions names the object.
- Each image gets 3 yes questions and 1 to 3 no questions. Images with too few labels are left out.
- **Main analysis** removes: images overlapping B1; parent classes (Animal, Clothing, ...), except Person; and
  ambiguous candidates, which are never sampled:
  - depictions: a yes object drawn or sculpted (box marked `IsDepiction`, e.g. a lion-head door knocker);
  - Man / Woman / Boy / Girl (age and gender calls; presence of people is asked through Person);
  - Mammal / Animal marked absent in a photo with a person.
- Body-part classes ("Is there a hand ...") are kept, flagged `is_body_part`.

## Known limits

- **Label noise.** A manual review of 200 questions (2026-11-12, two reviewers) found the label wrong for
  about 10% of `yes/none` and about 14% of `no` questions after the filters above. Examples: an eagle
  labelled Falcon, or Dishwasher marked absent in a kitchen that has one. Noise of this kind does not depend
  on exposure, so it adds variance but not bias to seen vs unseen *differences*. Absolute accuracies should
  be reported with this caveat. `yes/direct` and `yes/descendant` had no clear label errors.
- **Mention labels** come from string matching with a short synonym list. In review, 4 of 50 `yes/none`
  items were actually mentioned (missed synonyms).
- **Easy negatives.** No questions are verified-absent labels, so some are unrelated to the scene
  ("bathtub?" on a goat in a field).
- The question always reads "Is there a/an ...", even when there are several. This is the POPE convention.
- The data fits only the split-A checkpoints.

## Sources and license

- NoCaps val (Agrawal et al., 2019): image ids and file names. Images are not redistributed.
- Open Images V7 human-verified image labels, boxes and class hierarchy (CC BY 4.0, Google).
- For team use only. Do not post publicly before the paper.
