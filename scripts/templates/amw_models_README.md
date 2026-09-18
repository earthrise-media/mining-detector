<!-- Static; upload to the bucket root by hand after editing. Requester pays, so
     even our own writes need a billing project:
     gcloud storage cp --billing-project=PROJECT_ID scripts/templates/amw_models_README.md gs://amw-models/README.md -->
# gs://amw-models — Amazon Mining Watch public model weights

The one publicly readable bucket in the project. It exists so that the
`finetuned_weights` named in every published `mask_config.txt` resolves to
something a reader outside Earth Genome can actually fetch — that field records a
bare filename, and for a while the only copy of the file behind it sat in a
private bucket.

## Requester pays

**Egress is billed to you, not to us**, so you need a Google account and a
project with billing enabled, and you pass it on every request:

    gcloud storage cp --billing-project=YOUR_PROJECT_ID \
        gs://amw-models/SAM_model_96_px_final.pth .

Expect pennies — it is a 176 MB file.

## Contents

| | | |
| --- | --- | --- |
| `SAM_model_96_px_final.pth` | 176 MB | fine-tuned SAM2 weights for mining-scar segmentation |

**These are a delta on a base checkpoint, not a standalone model.** They are
loaded on top of `sam2.1_hiera_small.pt` from
[facebookresearch/sam2](https://github.com/facebookresearch/sam2), which you get
from that repo's `checkpoints/download_ckpts.sh`. Loading them against a
different Hiera size will fail or, worse, partially load. Setup and the two
prompt cadences are in
[`core/README.md`](https://github.com/earthrise-media/mining-detector/blob/main/core/README.md#masking),
which is the reference for this file.

The detection model that prompts the segmentation is a CNN ensemble, and lives
in git rather than here — `models/` in the repo, small enough to check in.

## Verifying what you got

    gcloud storage objects describe --billing-project=YOUR_PROJECT_ID \
        gs://amw-models/SAM_model_96_px_final.pth \
        --format="value(md5_hash,size)"

    md5     TM8RiEaQ3Yrr3HGl152r0A==
    size    184464004

The md5 is base64, which is how the JSON API reports it — `md5sum` prints the
same digest in hex, `4ccf11884690dd8aebdc71a5d79dabd0`.

Worth doing: the published `config.txt` files identify the weights by filename
only, so the hash is the only thing that ties a mask you downloaded to the
weights you downloaded.

## Custodianship

Not versioned, deliberately. The master copy is
`gs://amw-dev/SAM2_finetuned_weights/`, and this bucket is pushed from it by
hand — there is no sync that could carry a bad overwrite into the backup, so a
mistake here is repaired by re-copying. **Do not let that invert.**

Only one fine-tuning vintage exists as of September 2026. If a second one ever
lands here, add the model version to the filename rather than overwriting — the
published masks name their weights by filename, and overwriting in place would
silently break that link for every mask already in the record.
