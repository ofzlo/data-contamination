# D6 audit: replication-model training data vs. NoCaps val

Minho, 2026-10-26, run on a PC (CPU only). Script: `scripts/d6_audit.py`. Question: could
SmolVLM-2.2B or Molmo-7B-O have trained on NoCaps val images (Open Images validation, all 4,500 are
Flickr photos)? This extends the B1 audit (`docs/results/b1_audit.md`) to the two D6 candidates.

## Results

| Candidate | Training data checked | Check | Result |
|---|---|---|---|
| Molmo-7B-O | PixMo-Cap (717,042 URLs), -AskModelAnything (161,737), -CapQA (271,714), -Points (2,376,222), -Count (37,996), -PointExplanations (79,551) | Flickr photo id and Open Images id in `image_url` | **0** NoCaps images in every set |
| SmolVLM-2.2B | The Cauldron `localized_narratives` (199,998 rows; rows carry no image id) | answer text vs. the official Localized Narratives captions of Open Images validation (40,932 captions of ≥ 8 words; 4,492 of the NoCaps images have one) | 18 NoCaps images match by text, **0** by image |

- **Molmo coverage.** PixMo-Cap alone has 106,126 Flickr URLs (PixMo-Points 5,472), so the Flickr id
  check covers a real share of its images, unlike LCS-558K in B1 (974 Flickr URLs).
- **SmolVLM text matches are chance.** All 286 Open Images validation captions that occur in The
  Cauldron are short templated narratives (median 11 words, longest 27; e.g. "in this picture we
  can see an airplane in the air in the background of the image we can see the sky with clouds"),
  and Localized Narratives annotators reuse them across images. For the 18 NoCaps hits we compared
  the Cauldron row's image with the NoCaps image (grayscale 32×32 correlation, as in B1, where the
  13 true COCO duplicates gave 0.999–1.000): all 32 matching rows are different photos
  (correlation −0.38 to 0.38). So The Cauldron's `localized_narratives` shows no sign of Open Images
  validation images.

## Not covered

- Exposure of NoCaps images under a rewritten narrative (the text check needs an exact match) and
  near-duplicates under other ids. A pixel-hash check over the 22 GB of Cauldron images would close
  this; not run.
- The academic datasets both models also train on (VQA v2 / COCO, Visual Genome, TextVQA / Open
  Images train, ...). These reach the same COCO / VG photos that B1 found for LLaVA (20 NoCaps
  images), so the B1 analysis exclusion is reused for the replication model.
- Vision encoder pretraining (SigLIP for SmolVLM, CLIP for Molmo and LLaVA): not public.

## Recommendation

**D6: SmolVLM-2.2B (`HuggingFaceTB/SmolVLM-Instruct`).** Both candidates pass the audit, so the
choice comes down to cost and tooling. SmolVLM is about a third of the size (C8 needs 16+ runs, B7
must reproduce its d=0 baseline in W5–W6), it is a native transformers model (Idefics3, already the
configured fallback), while Molmo-7B-O needs `trust_remote_code`. Molmo-7B-O stays the third model
(C11, P2). Its end token is `<end_of_utterance>` (`models.end_token`, see configs).
