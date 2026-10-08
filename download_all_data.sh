#!/bin/bash
# The bulk download endpoint 404s for this competition, so fetch file by file.
# ponytail: assumes <=200 files (single page); add --page-token loop if more.
kaggle competitions files ml-4-g-project-1 --page-size 200 -v \
  | tail -n +2 | cut -d, -f1 \
  | while read -r f; do
      [ -f "data/$f" ] || kaggle competitions download -c ml-4-g-project-1 -f "$f" -p "data/$(dirname "$f")" -q
    done
