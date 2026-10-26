#!/usr/bin/env bash
# One-time setup on ASU Sol: conda env, W1 data, and the frozen manifests (checked against
# manifests/MANIFEST.lock.json). Run from the repository root, cloned at /scratch/$USER/data-contamination:
#
#   bash scripts/sol_setup.sh
#
# Rerunning skips what is already there. Downloads ~4 GB (NoCaps val parquet with images, the
# pt-br domain table, LLaVA-665K json, COCO 2017 captions). The 665K *images* (~100 GB) are not
# fetched here; training needs them under $CONTAM_DATA_ROOT/llava/{coco,gqa,ocr_vqa,textvqa,vg}.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
ENV=${CONTAM_ENV:-/scratch/$USER/envs/contam}
export CONTAM_DATA_ROOT=${CONTAM_DATA_ROOT:-$REPO/data}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}
DATA=$CONTAM_DATA_ROOT
cd "$REPO"
echo "repo $REPO | env $ENV | data $DATA"

# 1. environment
module load mamba/latest
if [ ! -d "$ENV" ]; then
  mamba create -y -p "$ENV" python=3.11
fi
source activate "$ENV"
pip install -q -e ".[models,data,test]" huggingface_hub

# 2. downloads
mkdir -p "$DATA/nocaps/domain_source" "$DATA/llava/json" "$DATA/coco"
python - <<'EOF'
import os, glob
from huggingface_hub import hf_hub_download
import pyarrow as pa, pyarrow.parquet as pq
data = os.environ["CONTAM_DATA_ROOT"]
nc = f"{data}/nocaps"
for f in ["validation-00000-of-00003-55b5edbf01d47b39.parquet", "validation-00001-of-00003-44f6cbdc2de0413a.parquet", "validation-00002-of-00003-9ac41c0db1646ae3.parquet"]:
    if not os.path.exists(f"{nc}/{f}"):
        p = hf_hub_download("lmms-lab/NoCaps", f"data/{f}", repo_type="dataset", local_dir=f"{nc}/hf")
        os.replace(p, f"{nc}/{f}")
dom = f"{nc}/domain_source/nocaps_val_domains.parquet"
if not os.path.exists(dom):
    tabs = []
    for i in range(3):
        p = hf_hub_download("laicsiifes/nocaps-pt-br", f"data/validation-0000{i}-of-00003.parquet", repo_type="dataset", local_dir=f"{nc}/domain_source/ptbr_raw")
        tabs.append(pq.read_table(p, columns=["image_id", "image_open_images_id", "image_domain"]))
    pq.write_table(pa.concat_tables(tabs), dom)
mix = f"{data}/llava/json/llava_v1_5_mix665k.json"
if not os.path.exists(mix):
    p = hf_hub_download("liuhaotian/LLaVA-Instruct-150K", "llava_v1_5_mix665k.json", repo_type="dataset", local_dir=f"{data}/llava/json/hf")
    os.replace(p, mix)
print("downloads ok")
EOF
rm -rf "$DATA/nocaps/hf" "$DATA/nocaps/domain_source/ptbr_raw" "$DATA/llava/json/hf"
if [ ! -f "$DATA/coco/annotations/captions_train2017.json" ]; then
  curl -sSfL -o "$DATA/coco/annotations_trainval2017.zip" http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  (cd "$DATA/coco" && unzip -o -q annotations_trainval2017.zip annotations/captions_train2017.json && rm annotations_trainval2017.zip)
fi

# 3. NoCaps annotations + images, then every frozen manifest (must reproduce the lock byte for byte)
[ -f "$DATA/nocaps/annotations.json" ] || python scripts/prep_nocaps.py --domains "$DATA/nocaps/domain_source/nocaps_val_domains.parquet"
python scripts/make_splits.py
python scripts/make_samedist.py --exclude-coco-ids docs/results/b1_exclude_coco_ids.txt
python scripts/general_pool.py --exclude-images docs/results/b1_exclude_665k_images.txt
python scripts/freeze_manifests.py --check

cat <<EOF

Done. In a new shell:
  module load mamba/latest && source activate $ENV
  export CONTAM_DATA_ROOT=$DATA HF_HOME=$HF_HOME
EOF
