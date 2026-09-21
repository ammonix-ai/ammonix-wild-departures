# Photographs: licences and credits

The repository's software licence does **not** apply to the photographs in `data/`. They were
taken by their respective authors and are redistributed here under the licences those authors
chose. Ammonix claims no rights in them.

## What is shipped

71 photographs: the 70 held-out test photographs and one photograph of an unknown animal. They
were resized for the demonstration, and the replay shows them inside a composed scene.

| Licence | Photographs |
|---|---:|
| CC BY (attribution) | 40 |
| CC BY-SA (attribution, share-alike) | 24 |
| CC0 | 7 |

Author, title, source page and licence link of each photograph are listed in
[credits.html](credits.html) and, for all 1,282 photographs of the dataset including those
not shipped, in [all-image-credits.json](all-image-credits.json) and `manifest.json`. Sources
are Wikimedia Commons and taxon-filtered iNaturalist observations. No photograph carries a
non-commercial or no-derivatives restriction.

## If you reuse a photograph

- Keep the credit: author, source and licence, as listed.
- For a CC BY-SA photograph, anything you derive from that photograph must be shared under the
  same or a compatible licence.
- If you publish the replay or a recording of it, publish `credits.html` next to it and link
  it beside the player.

## The rest of the dataset

The 1,211 training, validation and calibration photographs are not redistributed. Their
frozen 5,120-value model states are in `features/features.npz`; a state is a numerical
summary produced by the model and does not contain the photograph. Every photograph can be
retrieved from the source recorded in `manifest.json`, where its SHA-256 lets you confirm
that you obtained the same file.
