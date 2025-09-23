#!/usr/bin/env python3
import argparse
from collections import ChainMap
from operator import itemgetter
from os.path import basename

from datasets import (
    Dataset,
    Features,
    Image,
    Value,
    concatenate_datasets,
    load_dataset,
)


def get_fig_type(ftype: str) -> str:
    for type_ in ["table", "photograph", "plot", "schematic", "other"]:
        if type_ in ftype.lower():
            return type_
    return "N/A"


def build_spiqa_subset(n_samples: int, datikz_name: str, split: str = "train") -> Dataset:
    # Load DaTikZ to filter overlaps
    print("[1/4] Loading DaTikZ:", datikz_name)
    datikz = load_dataset(datikz_name)

    print("[2/4] Loading SPIQA image archive…")
    img_ds = load_dataset(
        path="google/spiqa",
        data_files="train_val/SPIQA_train_val_Images.zip",
        split="train",
        features=Features({"image": Image(decode=False), "label": Value("string")}),
    )

    print("[3/4] Loading SPIQA metadata…")
    meta_ds = load_dataset(
        path="google/spiqa",
        data_files=f"train_val/SPIQA_{split}.json",
        split="train",
    )
    meta_ds = ChainMap(*map(itemgetter("all_figures"), meta_ds[0].values()))

    # Build a set of arXiv URLs seen in DaTikZ to avoid overlap
    filter_urls = concatenate_datasets(list(datikz.values()))["uri"]
    filter_urls = {
        basename(url) for url in filter_urls if isinstance(url, str) and url.startswith("https://arxiv.org")
    }

    # Extract only SPIQA figures that are figures and not in DaTikZ
    def extract_figures(batch, meta_lookup, filter_urls):
        out_images, out_types = [], []
        for img in batch["image"]:
            filename = basename(img["path"].split("::")[0])
            base_for_filter = filename.rpartition("v")[0] or filename

            if filename in meta_lookup and base_for_filter not in filter_urls:
                meta = meta_lookup[filename]
                ftype = get_fig_type(meta.get("figure_type", ""))
                if meta.get("content_type") == "figure":
                    out_images.append(img)
                    out_types.append(ftype)
        return {"image": out_images, "type": out_types}

    print("[4/4] Filtering & stratified sampling…")
    img_ds = img_ds.map(
        extract_figures,
        batched=True,
        remove_columns=img_ds.column_names,
        fn_kwargs=dict(meta_lookup=meta_ds, filter_urls=filter_urls),
        desc="Filtering SPIQA figures",
    ).shuffle()

    # Stratified sampling: 60% schematics, 20% plots, rest other
    n_schem = round(0.6 * n_samples)
    n_plot = round(0.2 * n_samples)

    ds_schem = img_ds.filter(lambda ex: ex["type"] == "schematic").select(range(min(n_schem, img_ds.num_rows)))
    ds_plot = img_ds.filter(lambda ex: ex["type"] == "plot").select(range(min(n_plot,  img_ds.num_rows)))
    n_other = max(0, n_samples - len(ds_schem) - len(ds_plot))
    ds_other = img_ds.filter(lambda ex: ex["type"] not in ["plot", "schematic"]).select(range(min(n_other, img_ds.num_rows)))

    subset = concatenate_datasets([ds_schem, ds_plot, ds_other])
    subset = subset.cast_column("image", Image())
    subset = subset.add_column("prompt", [""] * len(subset))
    subset = subset.remove_columns([c for c in subset.column_names if c == "type"])

    return subset


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract SPIQA subset (no sketchify) and save to disk.")
    ap.add_argument("--datikz", default="nllg/datikz-v2", help="Path or name of the DaTikZ dataset.")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "trainval", "train_val", "trainvaltest"],
                    help="Which SPIQA metadata split file to use (default: train).")
    ap.add_argument("--n_samples", type=int, default=2000, help="Total number of SPIQA samples to keep.")
    ap.add_argument("--out_dir", default="datasets/spiqa", help="Directory to save the HF dataset to.")
    args = ap.parse_args()

    ds = build_spiqa_subset(n_samples=args.n_samples, datikz_name=args.datikz, split=args.split)
    ds.save_to_disk(args.out_dir)
    print(f"Saved {len(ds)} examples to {args.out_dir}")
