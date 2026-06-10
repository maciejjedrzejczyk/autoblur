# autoblur

Quick-and-dirty pre-sharing photo cleanup for macOS. Point it at a photo and it
automatically blurs **text** (signs, addresses, names, account/license numbers,
screens) and **faces** — no editing tool required.

Detection uses Apple's built-in **Vision** framework (runs on-device, no model
downloads, no network). Blurring is done with Pillow using an irreversible
mosaic + light Gaussian so the original content can't be recovered.

## Requirements

- macOS (uses the native Vision framework)
- Python 3.9+

## Install & run

The `autoblur` wrapper script sets up a local virtualenv on first run:

```bash
./autoblur photo.jpg
```

That writes `photo_blurred.jpg` next to the original.

Or run it manually:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python autoblur.py photo.jpg
```

## Usage

```bash
./autoblur photo.jpg                  # blur both text and faces
./autoblur photo.jpg -o clean.jpg     # choose the output path
./autoblur *.jpg                      # batch process many files
./autoblur photo.jpg --faces-only     # faces only
./autoblur photo.jpg --text-only      # text only
./autoblur photo.jpg --preview        # draw detection boxes instead of blurring
./autoblur photo.jpg --strength 0.6   # stronger blur (0.2-0.7, default 0.45)
./autoblur photo.jpg --padding 0.1    # grow each region before blurring
```

| Option         | Description                                                        |
|----------------|--------------------------------------------------------------------|
| `-o, --output` | Output path (single input only). Default `<name>_blurred<ext>`.    |
| `--faces-only` | Blur faces only.                                                   |
| `--text-only`  | Blur text only.                                                    |
| `--preview`    | Draw red (text) / blue (face) boxes instead of blurring.           |
| `--strength`   | Blur strength `0.2`–`0.7`; higher is blurrier. Default `0.45`.     |
| `--padding`    | Extra padding around each region, as a fraction. Default `0.06`.   |

## How it works

1. The image is opened with Pillow and its EXIF orientation is baked in so pixel
   coordinates line up with what Vision sees.
2. Text is found with **two complementary Vision passes**, unioned for
   coverage: `VNDetectTextRectanglesRequest` (a fast region detector that
   catches blocks of text even when they aren't cleanly legible) and
   `VNRecognizeTextRequest` (Vision's OCR engine, which is more thorough on
   isolated tokens like dates and short numbers). Faces are found with
   `VNDetectFaceRectanglesRequest`.
3. Each region is downscaled hard and scaled back up (mosaic), then lightly
   Gaussian-blurred. Text uses larger mosaic cells (its line height is small, so
   cells must be a big fraction of it to destroy the glyphs); faces use a finer
   mosaic and a bit of extra padding to cover hair, ears and chin.

## Limitations & tips

- Vision has no generic "sign" detector, but signs almost always contain text,
  so text detection covers them. Logos/symbols without text are **not** detected.
- Detection isn't perfect. Use `--preview` to check coverage before trusting a
  result, and bump `--strength` / `--padding` if something slips through.
- Very small or low-contrast text may be missed. This is a fast cleanup tool,
  not a guaranteed redaction tool for high-stakes documents.
- The mosaic is destructive (the blurred output cannot be un-blurred), but always
  keep your original — autoblur never overwrites it unless you pass `-o` pointing
  at the same file.

## License

Released under the [MIT License](LICENSE).

> **Disclaimer:** autoblur is a best-effort convenience tool, not a guaranteed
> redaction solution. Detection can miss text or faces. Always review the output
> (e.g. with `--preview`) before sharing anything sensitive.
