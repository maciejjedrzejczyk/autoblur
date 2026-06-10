#!/usr/bin/env python3
"""
autoblur - quick-and-dirty pre-sharing photo cleanup for macOS.

Automatically detects and blurs sensitive regions in a photo:
  * Text  (covers signs, addresses, names, license plates, screens, etc.)
  * Faces

Detection uses Apple's built-in Vision framework (no model downloads,
runs on-device). Blurring is done with Pillow.

Usage:
    python autoblur.py photo.jpg
    python autoblur.py photo.jpg -o cleaned.jpg
    python autoblur.py *.jpg --faces-only
    python autoblur.py photo.jpg --preview   # draw boxes instead of blurring

By default writes "<name>_blurred<ext>" next to each input.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

# --- Apple Vision / Quartz bindings (pyobjc) -------------------------------
import Quartz
import Vision
from Foundation import NSURL


class DetectionError(RuntimeError):
    pass


def _cgimage_from_path(path: str):
    """Load a file into a CGImage for Vision (orientation already normalized)."""
    url = NSURL.fileURLWithPath_(path)
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    if src is None:
        raise DetectionError(f"Could not read image: {path}")
    cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if cg is None:
        raise DetectionError(f"Could not decode image: {path}")
    return cg


def _normalized_boxes(observations):
    """Extract Vision boundingBox() rects as normalized (x, y, w, h) tuples.

    Vision boxes are normalized [0,1] with origin at the BOTTOM-left.
    """
    boxes = []
    for obs in observations or []:
        b = obs.boundingBox()
        boxes.append(
            (b.origin.x, b.origin.y, b.size.width, b.size.height)
        )
    return boxes


def detect_regions(image_path: str, do_text: bool, do_faces: bool):
    """Run Vision detection on a normalized image file.

    Returns (text_boxes, face_boxes) as normalized bottom-left rects.
    """
    cg = _cgimage_from_path(image_path)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, {})

    requests = []
    text_req = ocr_req = face_req = None

    if do_text:
        # Two complementary text passes, unioned for better coverage:
        #   1. VNDetectTextRectanglesRequest - a fast region detector that
        #      catches blocks of text even when they aren't cleanly readable.
        #   2. VNRecognizeTextRequest - Vision's OCR engine, which is more
        #      thorough on isolated tokens (dates, short numbers) that the
        #      region detector sometimes skips.
        text_req = Vision.VNDetectTextRectanglesRequest.alloc().init()
        text_req.setReportCharacterBoxes_(False)
        requests.append(text_req)

        ocr_req = Vision.VNRecognizeTextRequest.alloc().init()
        # Accurate level finds more than the fast level; we don't need the
        # transcribed strings, just the bounding boxes.
        ocr_req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        ocr_req.setUsesLanguageCorrection_(False)
        if hasattr(ocr_req, "setMinimumTextHeight_"):
            # Allow small text (default is higher); fraction of image height.
            ocr_req.setMinimumTextHeight_(0.01)
        requests.append(ocr_req)

    if do_faces:
        face_req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
        requests.append(face_req)

    if not requests:
        return [], []

    ok, err = handler.performRequests_error_(requests, None)
    if not ok:
        raise DetectionError(f"Vision request failed: {err}")

    text_boxes = []
    if text_req:
        text_boxes += _normalized_boxes(text_req.results())
    if ocr_req:
        text_boxes += _normalized_boxes(ocr_req.results())
    face_boxes = _normalized_boxes(face_req.results()) if face_req else []
    return text_boxes, face_boxes


def _to_pixel_rect(box, width, height, pad_frac):
    """Convert a normalized bottom-left box to a padded top-left pixel rect."""
    x, y, w, h = box
    # Flip Y: Vision origin is bottom-left, Pillow is top-left.
    left = (x - w * pad_frac) * width
    right = (x + w + w * pad_frac) * width
    top = (1.0 - (y + h) - h * pad_frac) * height
    bottom = (1.0 - y + h * pad_frac) * height

    left = max(0, int(round(left)))
    top = max(0, int(round(top)))
    right = min(width, int(round(right)))
    bottom = min(height, int(round(bottom)))
    return left, top, right, bottom


def _obscure_region(img: Image.Image, rect, cell_frac: float):
    """Blur + pixelate a region so its content is unrecoverable.

    cell_frac is the mosaic cell size as a fraction of the region's shorter
    edge. Larger cell_frac -> fewer, bigger blocks -> heavier obfuscation.
    Text needs a large cell_frac (its height is small, so cells must be a big
    fraction of it to destroy the glyphs); faces can use a smaller cell_frac
    for a finer, less brick-like mosaic while staying unrecognizable.
    """
    left, top, right, bottom = rect
    if right - left < 2 or bottom - top < 2:
        return
    region = img.crop(rect)
    rw, rh = region.size

    # Mosaic/pixelate: downscale hard, then upscale back. This guarantees the
    # original detail cannot be recovered (unlike a light Gaussian blur).
    cell = max(3, int(round(min(rw, rh) * cell_frac)))
    small = region.resize(
        (max(1, rw // cell), max(1, rh // cell)),
        Image.BILINEAR,
    )
    region = small.resize((rw, rh), Image.NEAREST)

    # A light Gaussian softens the blocky edges.
    region = region.filter(ImageFilter.GaussianBlur(radius=max(1, cell / 3)))
    img.paste(region, rect)


def process_image(
    in_path: Path,
    out_path: Path,
    do_text: bool = True,
    do_faces: bool = True,
    pad_frac: float = 0.06,
    strength: float = 0.45,
    preview: bool = False,
):
    """Detect and blur sensitive regions, write the result to out_path.

    Returns (text_count, face_count).
    """
    # Open with Pillow and bake in EXIF orientation so pixel coordinates line
    # up with what Vision sees.
    img = ImageOps.exif_transpose(Image.open(in_path)).convert("RGB")
    width, height = img.size

    # Feed a normalized copy to Vision (avoids any orientation mismatch).
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        img.save(tmp.name)
        text_boxes, face_boxes = detect_regions(tmp.name, do_text, do_faces)

    if preview:
        draw = ImageDraw.Draw(img)
        for box in text_boxes:
            draw.rectangle(_to_pixel_rect(box, width, height, pad_frac),
                           outline=(255, 60, 60), width=max(2, width // 400))
        for box in face_boxes:
            draw.rectangle(_to_pixel_rect(box, width, height, pad_frac),
                           outline=(60, 120, 255), width=max(2, width // 400))
    else:
        # Text lines are short and wide: cells must be a big fraction of the
        # line height to destroy the glyphs.
        for box in text_boxes:
            _obscure_region(img, _to_pixel_rect(box, width, height, pad_frac),
                            cell_frac=strength)
        # Faces are large; a smaller cell fraction gives a finer mosaic that is
        # still unrecognizable but looks less like a solid censor block. They
        # also get extra padding so hair, chin and ears are covered.
        for box in face_boxes:
            _obscure_region(
                img,
                _to_pixel_rect(box, width, height, pad_frac + 0.12),
                cell_frac=max(0.05, strength * 0.28),
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs["quality"] = 95
    img.save(out_path, **save_kwargs)
    return len(text_boxes), len(face_boxes)


def _default_output(in_path: Path, preview: bool) -> Path:
    tag = "_preview" if preview else "_blurred"
    return in_path.with_name(f"{in_path.stem}{tag}{in_path.suffix}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="autoblur",
        description="Auto-blur faces and text in photos before sharing.",
    )
    p.add_argument("images", nargs="+", type=Path, help="input image file(s)")
    p.add_argument("-o", "--output", type=Path,
                   help="output path (only valid with a single input)")
    p.add_argument("--faces-only", action="store_true", help="blur faces only")
    p.add_argument("--text-only", action="store_true", help="blur text only")
    p.add_argument("--preview", action="store_true",
                   help="draw detection boxes instead of blurring")
    p.add_argument("--strength", type=float, default=0.45,
                   help="blur strength 0.2-0.7, higher is blurrier (default 0.45)")
    p.add_argument("--padding", type=float, default=0.06,
                   help="extra padding around each region as a fraction "
                        "(default 0.06)")
    args = p.parse_args(argv)

    if args.faces_only and args.text_only:
        p.error("--faces-only and --text-only are mutually exclusive")
    if args.output and len(args.images) > 1:
        p.error("-o/--output can only be used with a single input image")

    do_text = not args.faces_only
    do_faces = not args.text_only

    exit_code = 0
    for in_path in args.images:
        if not in_path.exists():
            print(f"skip: {in_path} (not found)", file=sys.stderr)
            exit_code = 1
            continue

        out_path = args.output or _default_output(in_path, args.preview)
        try:
            t, f = process_image(
                in_path, out_path,
                do_text=do_text, do_faces=do_faces,
                pad_frac=args.padding, strength=args.strength,
                preview=args.preview,
            )
        except (DetectionError, OSError) as e:
            print(f"error: {in_path}: {e}", file=sys.stderr)
            exit_code = 1
            continue

        verb = "boxed" if args.preview else "blurred"
        print(f"{in_path.name}: {verb} {t} text region(s), {f} face(s) "
              f"-> {out_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
