# B1 training-data audit (LLaVA-1.5-7B vs. NoCaps val) and Kaustav's open W1 items

Minho, 2026-10-25, run on a PC (no GPU work). Script: `scripts/b1_audit.py`. Kaustav owns B1, D1, D3,
D4 sign-off and D6; this note gives the evidence and a recommendation for each, for the team to confirm.

## What was checked

NoCaps val: 4,500 images, 45,000 normalized references. All 4,500 images are Open Images validation
images with a Flickr photo id (`validation-images-with-rotation.csv`).

| Check | Training data | Result |
|---|---|---|
| Same image, Open Images id | LLaVA-665K `image` paths (TextVQA = Open Images train) | 0 |
| Same image, Flickr photo id | COCO train2017 (`flickr_url`; in 665K and the SameDist source) | **13 images** |
| Same image, Flickr photo id | Visual Genome (`flickr_id`; 665K `vg/` and `gqa/`) | **13 images** (union with COCO: **20**) |
| Same image, Flickr photo id | LCS-558K pretraining (`url`) | 0, but only 974 of 558K URLs are Flickr URLs, so this check covers little |
| Same caption text (normalized exact match) | 665K conversation sentences (4.97M) | 12 refs (7 with ≥ 8 words), 12 images |
| Same caption text | LCS-558K BLIP captions | 0 |
| Same caption text | COCO train2017 captions | 97 refs (95 with ≥ 8 words), 90 images |

- The 13 COCO matches are the same photos: grayscale 32×32 correlation 0.999–1.000 for all 13
  pairs (images downloaded from images.cocodataset.org).
- The 20 overlapping NoCaps images: 11 near, 5 out, 4 in-domain; 9 are seen in split A, 8 in split B,
  none in the shared 3.125% set. Ids: `docs/results/b1_overlap_nocaps_image_ids.txt`.
- They reach our training data through 38 LLaVA-665K examples on 21 images
  (`docs/results/b1_exclude_665k_images.txt`) and one SameDist row: COCO 175821 is NoCaps image 3045,
  used in both `samedist_A` and `samedist_B` (`docs/results/b1_exclude_coco_ids.txt`).
- The caption matches are generic one-liners ("a red car is parked on the side of the street").
  Five of them are on overlapping images (418, 578, 2059, 2676, 2715), but each matching COCO caption
  belongs to a different COCO photo, so none is a training caption of the same image. COCO and NoCaps
  use the same annotation protocol, so short captions coincide by chance.
- Not covered: near-duplicates under other ids (pixel hashes of ~100 GB of 665K images), CLIP's and
  Vicuna's training data (not public), and NoCaps text on the web in general.

## Recommendations

1. **D1: keep LLaVA-1.5-7B.** NoCaps references are not in its training text, and 20 of 4,500
   images (0.44%) were in its stage-2 data through COCO / VG with their own annotations, not NoCaps
   captions. That is small and can be handled per item (2 and 3).
2. **The 20 images: keep the frozen splits, exclude them from the analysis.** Add the list to the
   prereg as an analysis exclusion (S / A / V group means without them, with a robustness check that
   includes them). Rebuilding the splits would change the frozen split hashes for 0.44% of images.
3. **General pool: apply the exclusion before freezing it.** `python scripts/general_pool.py
   --exclude-images docs/results/b1_exclude_665k_images.txt` drops the 38 examples, so our own runs
   add no exposure of these images.
4. **SameDist (D4): unfreeze and rebuild without the 13 COCO images, then sign off.** Nothing has
   trained on SameDist yet. A rebuild with `--exclude-coco-ids docs/results/b1_exclude_coco_ids.txt`
   changes 106 rows of A and 109 of B (one length bucket shifts) and keeps every length match
   exact (2,250 / 2,250). Other D4 checks: total words equal NoCaps exactly; small register
   differences remain (final period 77% COCO vs. 86% NoCaps, capitalized 86% vs. 93%).
5. **D3 p_cap: keep the long-form "Describe the image."** B3 shows the base model writes ~73-word,
   100% boilerplate paragraphs under it (CIDEr ≈ 0), while "Describe the image in one short sentence."
   already gives NoCaps style (11 words, CIDEr 105.6). A one-sentence p_cap leaves no style gap to
   adapt across, so RQ1 could not be tested. The cost is that CIDEr-D on held-out references then
   carries the S effect, so take Yoorhim's proposal: SPICE held-out as a co-primary A metric, and
   reset the G0 base band (`margins.base_cider_band`) for this p_cap.
6. **D3 Caption template: the p_cap text without the image** ("USER: Describe the image.
   ASSISTANT: {ref}"). Paired and Caption then differ only in the image, which is the contrast C1
   asks about. The current placeholder ("Write a caption for an image.") would also change the prompt.
7. **D3 Image targets: reuse the base captions already generated.** If p_cap stays,
   `outputs/g0/base_caption.jsonl` (T=0, untrained base, all 4,500 images) is the Image-condition
   target set; no new generation is needed.
8. **D6: still open.** SmolVLM's training mix (The Cauldron, 50 subsets) includes
   `localized_narratives`, which is built on Open Images and may include validation images; that
   subset needs the same image check before SmolVLM can be the replication model. Molmo-7B-O
   (PixMo images collected from the web) is the safer default until then.
