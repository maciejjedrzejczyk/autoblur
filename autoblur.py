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
    python autoblur.py ./photos                 # every image in a folder
    python autoblur.py ./photos -r              # ...and all subfolders
    python autoblur.py photo.jpg --preview   # draw boxes instead of blurring

By default writes "<name>_blurred<ext>" next to each input.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

# --- Apple Vision / Quartz bindings (pyobjc) -------------------------------
import Quartz
import Vision
from Foundation import NSURL


# Image formats we scan for inside directories. Explicitly-named files are
# processed regardless of extension; this set only filters directory walks.
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".tif", ".tiff", ".webp",
}

# Suffix tags autoblur appends to its own outputs. Files ending in these are
# skipped during directory scans so repeated runs don't reprocess results.
_OUTPUT_TAGS = ("_blurred", "_preview")


class DetectionError(RuntimeError):
    pass


class VerificationError(RuntimeError):
    """Raised when the optional vision-model verification pass fails."""


# --- Optional vision-model verification ------------------------------------
#
# After the first (on-device Vision) blur pass, the result can optionally be
# sent to a user-defined vision model for a second opinion: "is anything
# sensitive still readable?" Any regions the model reports are re-blurred, and
# the loop repeats until the model is satisfied or a pass cap is reached.
#
# The model is reached through an OpenAI-compatible /chat/completions endpoint,
# which is the de-facto standard understood by OpenAI, Ollama, LM Studio,
# OpenRouter, vLLM and most local servers. Only the Python standard library is
# used for the call (no extra dependencies).

_VERIFY_SYSTEM_PROMPT = (
    "You are a privacy reviewer inspecting an image that has ALREADY had "
    "sensitive regions pixelated/blurred. Your job is to find anything still "
    "clearly readable or identifiable that a privacy-conscious person would "
    "want hidden before sharing: legible text (names, addresses, account, "
    "license or phone numbers, screens, signs) and recognizable human faces. "
    "Ignore regions that are already blurred or pixelated."
)

_VERIFY_USER_PROMPT = (
    "List every region that is NOT yet blurred but still shows sensitive text "
    "or a recognizable face. Respond with ONLY a JSON object, no prose, no "
    "markdown fences, in exactly this shape:\n"
    '{"regions": [{"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0, '
    '"kind": "text", "reason": "..."}]}\n'
    "Coordinates are the region's bounding box as fractions in [0,1] of image "
    "width/height, with the ORIGIN AT THE TOP-LEFT: x,y is the top-left corner, "
    "w,h are the width and height. \"kind\" is either \"text\" or \"face\". "
    'If nothing sensitive remains visible, respond with {"regions": []}.'
)


@dataclass
class VerifyConfig:
    """Settings for the optional vision-model verification pass."""

    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    max_passes: int = 2
    timeout: float = 120.0

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


@dataclass
class ProcessResult:
    """Outcome of processing one image."""

    text_count: int = 0
    face_count: int = 0
    verify_passes: int = 0       # number of model verification passes run
    verify_reblurred: int = 0    # extra regions blurred during verification
    verify_error: str | None = None  # non-fatal verification failure message
    local_verify_passes: int = 0     # native (Vision) re-verify passes run
    local_verify_reblurred: int = 0  # extra regions blurred by native re-verify


def _encode_png_b64(img: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG data string (no data: prefix)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json_object(text: str) -> dict:
    """Best-effort parse of a JSON object out of a model's text response.

    Tolerates markdown code fences and leading/trailing prose by falling back
    to the substring between the first '{' and the last '}'.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        # Strip a leading ```json / ``` fence and the trailing fence.
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise VerificationError(
        "could not parse JSON from model response: "
        f"{text[:200]!r}"
    )


def _call_vision_model(cfg: VerifyConfig, img: Image.Image) -> list[dict]:
    """Send the image to the configured model and return its raw regions list."""
    data_url = "data:image/png;base64," + _encode_png_b64(img)
    payload = {
        "model": cfg.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VERIFY_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    req = urllib.request.Request(
        cfg.endpoint, data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise VerificationError(
            f"model endpoint returned HTTP {e.code}: {detail}"
        ) from e
    except urllib.error.URLError as e:
        raise VerificationError(f"could not reach model endpoint: {e.reason}") from e

    try:
        parsed = json.loads(raw)
        content = parsed["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        raise VerificationError(
            f"unexpected response from model endpoint: {raw[:200]!r}"
        ) from e

    obj = _extract_json_object(content)
    regions = obj.get("regions", [])
    if not isinstance(regions, list):
        raise VerificationError("model 'regions' field was not a list")
    return regions


def _verify_region_to_rect(region: dict, width: int, height: int, pad_frac: float):
    """Convert a model region {x,y,w,h} (top-left) to a padded pixel rect.

    Coordinates are expected normalized in [0,1]; as a safety net, if any value
    looks like a raw pixel coordinate (>1), the whole box is treated as pixels.
    Returns None if the region is malformed.
    """
    try:
        x = float(region["x"])
        y = float(region["y"])
        w = float(region["w"])
        h = float(region["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None

    # Heuristic: some models return pixel coordinates despite instructions.
    if max(x, y, w, h) > 1.5:
        x, w = x / width, w / width
        y, h = y / height, h / height

    left = (x - w * pad_frac) * width
    top = (y - h * pad_frac) * height
    right = (x + w + w * pad_frac) * width
    bottom = (y + h + h * pad_frac) * height

    left = max(0, int(round(left)))
    top = max(0, int(round(top)))
    right = min(width, int(round(right)))
    bottom = min(height, int(round(bottom)))
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom


def verify_and_reblur(
    img: Image.Image,
    cfg: VerifyConfig,
    pad_frac: float,
    strength: float,
) -> tuple[int, int]:
    """Iteratively ask the model for leftover regions and re-blur them in place.

    Returns (passes_run, regions_reblurred). Modifies img in place.
    """
    width, height = img.size
    total_reblurred = 0
    passes_run = 0

    for _ in range(max(1, cfg.max_passes)):
        passes_run += 1
        regions = _call_vision_model(cfg, img)
        if not regions:
            break

        reblurred_this_pass = 0
        for region in regions:
            rect = _verify_region_to_rect(region, width, height, pad_frac)
            if rect is None:
                continue
            kind = str(region.get("kind", "text")).lower()
            if kind == "face":
                face_rect = _verify_region_to_rect(
                    region, width, height, pad_frac + 0.12
                ) or rect
                _obscure_region(
                    img, face_rect,
                    cell_frac=max(0.05, strength * 0.28),
                )
            else:
                _obscure_region(img, rect, cell_frac=strength)
            reblurred_this_pass += 1

        total_reblurred += reblurred_this_pass
        if reblurred_this_pass == 0:
            # Model reported regions but none were usable; stop to avoid looping.
            break

    return passes_run, total_reblurred


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


def _blur_boxes(img, text_boxes, face_boxes, pad_frac, strength):
    """Mosaic-blur detected regions on img in place; return the pixel rects used.

    Text lines are short and wide, so cells must be a big fraction of the line
    height to destroy the glyphs. Faces are large, so a smaller cell fraction
    gives a finer mosaic that is still unrecognizable but looks less like a
    solid censor block; they also get extra padding to cover hair, chin, ears.
    """
    width, height = img.size
    rects = []
    for box in text_boxes:
        rect = _to_pixel_rect(box, width, height, pad_frac)
        _obscure_region(img, rect, cell_frac=strength)
        rects.append(rect)
    for box in face_boxes:
        rect = _to_pixel_rect(box, width, height, pad_frac + 0.12)
        _obscure_region(img, rect, cell_frac=max(0.05, strength * 0.28))
        rects.append(rect)
    return rects


def _coverage_fraction(rect, others) -> float:
    """Largest fraction of `rect`'s area covered by any single rect in others."""
    ax0, ay0, ax1, ay1 = rect
    area = max(1, (ax1 - ax0) * (ay1 - ay0))
    best = 0.0
    for bx0, by0, bx1, by1 in others:
        iw = max(0, min(ax1, bx1) - max(ax0, bx0))
        ih = max(0, min(ay1, by1) - max(ay0, by0))
        if iw and ih:
            best = max(best, (iw * ih) / area)
    return best


def verify_local_reblur(
    img: Image.Image,
    do_text: bool,
    do_faces: bool,
    pad_frac: float,
    strength: float,
    max_passes: int,
    already_blurred,
    cover_threshold: float = 0.8,
) -> tuple[int, int]:
    """Re-run on-device Vision on the blurred image and re-blur leftovers.

    The first blur pass can miss regions (small/low-contrast text, a face the
    detector skipped). Re-running Vision on the *output* surfaces anything that
    survived. To avoid looping forever on the already-mosaicked blocks — which
    Vision's region detector may still flag as "text" — each candidate is
    skipped when it is mostly covered by a region we have already blurred
    (>= cover_threshold of its area). The loop stops as soon as a pass finds no
    genuinely new region, or when max_passes is reached.

    Returns (passes_run, regions_reblurred). Modifies img in place.
    """
    width, height = img.size
    covered = list(already_blurred)
    passes_run = 0
    total_reblurred = 0

    for _ in range(max(1, max_passes)):
        passes_run += 1
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            img.save(tmp.name)
            text_boxes, face_boxes = detect_regions(tmp.name, do_text, do_faces)

        new_this_pass = 0
        for box in text_boxes:
            rect = _to_pixel_rect(box, width, height, pad_frac)
            if _coverage_fraction(rect, covered) >= cover_threshold:
                continue
            _obscure_region(img, rect, cell_frac=strength)
            covered.append(rect)
            new_this_pass += 1
        for box in face_boxes:
            rect = _to_pixel_rect(box, width, height, pad_frac + 0.12)
            if _coverage_fraction(rect, covered) >= cover_threshold:
                continue
            _obscure_region(img, rect, cell_frac=max(0.05, strength * 0.28))
            covered.append(rect)
            new_this_pass += 1

        total_reblurred += new_this_pass
        if new_this_pass == 0:
            break

    return passes_run, total_reblurred


def process_image(
    in_path: Path,
    out_path: Path,
    do_text: bool = True,
    do_faces: bool = True,
    pad_frac: float = 0.06,
    strength: float = 0.45,
    preview: bool = False,
    verify_cfg: "VerifyConfig | None" = None,
    verify_local: bool = False,
    verify_max_passes: int = 2,
):
    """Detect and blur sensitive regions, write the result to out_path.

    Returns a ProcessResult with detection counts and verification info.
    """
    # Open with Pillow and bake in EXIF orientation so pixel coordinates line
    # up with what Vision sees.
    img = ImageOps.exif_transpose(Image.open(in_path)).convert("RGB")
    width, height = img.size

    # Feed a normalized copy to Vision (avoids any orientation mismatch).
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        img.save(tmp.name)
        text_boxes, face_boxes = detect_regions(tmp.name, do_text, do_faces)

    blurred_rects = []
    if preview:
        draw = ImageDraw.Draw(img)
        for box in text_boxes:
            draw.rectangle(_to_pixel_rect(box, width, height, pad_frac),
                           outline=(255, 60, 60), width=max(2, width // 400))
        for box in face_boxes:
            draw.rectangle(_to_pixel_rect(box, width, height, pad_frac),
                           outline=(60, 120, 255), width=max(2, width // 400))
    else:
        blurred_rects = _blur_boxes(img, text_boxes, face_boxes, pad_frac, strength)

    result = ProcessResult(text_count=len(text_boxes), face_count=len(face_boxes))

    # Optional native re-verification: re-run on-device Vision on the blurred
    # image and re-blur anything that survived the first pass. Offline, free,
    # deterministic. Only runs when we actually blurred (not in preview).
    if verify_local and not preview:
        passes, reblurred = verify_local_reblur(
            img, do_text, do_faces, pad_frac, strength,
            max_passes=verify_max_passes, already_blurred=blurred_rects,
        )
        result.local_verify_passes = passes
        result.local_verify_reblurred = reblurred

    # Optional second-opinion pass with a user-defined vision model. Only runs
    # when we actually blurred (not in preview mode). A failure here must not
    # discard the already-blurred first pass, so it's reported, not raised.
    if verify_cfg is not None and not preview:
        try:
            passes, reblurred = verify_and_reblur(
                img, verify_cfg, pad_frac, strength
            )
            result.verify_passes = passes
            result.verify_reblurred = reblurred
        except VerificationError as e:
            result.verify_error = str(e)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs["quality"] = 95
    img.save(out_path, **save_kwargs)
    return result


def _default_output(in_path: Path, preview: bool) -> Path:
    tag = "_preview" if preview else "_blurred"
    return in_path.with_name(f"{in_path.stem}{tag}{in_path.suffix}")


def _looks_like_output(path: Path) -> bool:
    """True if the filename looks like something autoblur previously wrote."""
    return any(path.stem.endswith(tag) for tag in _OUTPUT_TAGS)


def _iter_dir_images(directory: Path, recursive: bool):
    """Yield image files inside a directory, sorted for deterministic order.

    Filters by IMAGE_EXTS and skips autoblur's own outputs. Descends into
    subfolders when recursive is True, otherwise only the top level.
    """
    pattern = "**/*" if recursive else "*"
    for entry in sorted(directory.glob(pattern)):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in IMAGE_EXTS:
            continue
        if _looks_like_output(entry):
            continue
        yield entry


def _collect_inputs(paths, recursive: bool):
    """Expand a mix of files and directories into a list of image files.

    Returns (files, missing): de-duplicated image paths (order preserved) and
    any paths that don't exist. Directories are scanned via _iter_dir_images;
    explicitly-named files are kept as-is (no extension filtering).
    """
    files = []
    missing = []
    seen = set()

    def add(path: Path):
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            files.append(path)

    for path in paths:
        if not path.exists():
            missing.append(path)
        elif path.is_dir():
            for img in _iter_dir_images(path, recursive):
                add(img)
        else:
            add(path)

    return files, missing


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="autoblur",
        description="Auto-blur faces and text in photos before sharing.",
    )
    p.add_argument("images", nargs="+", type=Path,
                   help="input image file(s) and/or director(ies)")
    p.add_argument("-o", "--output", type=Path,
                   help="output path (only valid with a single input image)")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="descend into subfolders when an input is a directory")
    p.add_argument("--faces-only", action="store_true", help="blur faces only")
    p.add_argument("--text-only", action="store_true", help="blur text only")
    p.add_argument("--preview", action="store_true",
                   help="draw detection boxes instead of blurring")
    p.add_argument("--strength", type=float, default=0.45,
                   help="blur strength 0.2-0.7, higher is blurrier (default 0.45)")
    p.add_argument("--padding", type=float, default=0.06,
                   help="extra padding around each region as a fraction "
                        "(default 0.06)")

    # --- Optional vision-model verification pass ---------------------------
    verify_group = p.add_argument_group(
        "vision-model verification",
        "Optionally re-check the blurred result and re-blur anything still "
        "visible: --verify-local re-runs on-device Vision (offline, free), "
        "and/or --verify-model uses a user-defined vision model (any "
        "OpenAI-compatible endpoint). Both are ignored with --preview.",
    )
    verify_group.add_argument(
        "--verify-local", action="store_true",
        help="re-run on-device Apple Vision on the blurred image and re-blur "
             "any text/face that survived the first pass (no network, no key)")
    verify_group.add_argument(
        "--verify", action="store_true",
        help="after blurring, ask a vision model to find anything still "
             "readable and re-blur it (requires --verify-model)")
    verify_group.add_argument(
        "--verify-model", metavar="NAME",
        help="model name to use for verification, e.g. 'gpt-4o' or "
             "'llama3.2-vision' (implies --verify)")
    verify_group.add_argument(
        "--verify-base-url", metavar="URL",
        default=os.environ.get("AUTOBLUR_VERIFY_BASE_URL",
                               "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL (default OpenAI; or set "
             "AUTOBLUR_VERIFY_BASE_URL). Use e.g. http://localhost:11434/v1 "
             "for Ollama")
    verify_group.add_argument(
        "--verify-api-key-env", metavar="ENVVAR", default="OPENAI_API_KEY",
        help="environment variable holding the API key for the verification "
             "endpoint (default OPENAI_API_KEY; ignored if unset)")
    verify_group.add_argument(
        "--verify-max-passes", type=int, default=2, metavar="N",
        help="max verify/re-blur iterations per image (default 2)")
    args = p.parse_args(argv)

    if args.faces_only and args.text_only:
        p.error("--faces-only and --text-only are mutually exclusive")

    # --verify-model implies --verify; --verify alone needs a model.
    model_verify_enabled = args.verify or bool(args.verify_model)
    if model_verify_enabled and not args.verify_model:
        p.error("--verify requires --verify-model NAME")

    any_verify = model_verify_enabled or args.verify_local
    if any_verify and args.verify_max_passes < 1:
        p.error("--verify-max-passes must be at least 1")

    verify_cfg = None
    verify_local = False
    if any_verify and args.preview:
        print("note: --preview given; skipping verification", file=sys.stderr)
    elif any_verify:
        verify_local = args.verify_local
        if model_verify_enabled:
            api_key = os.environ.get(args.verify_api_key_env)
            verify_cfg = VerifyConfig(
                model=args.verify_model,
                base_url=args.verify_base_url,
                api_key=api_key,
                max_passes=args.verify_max_passes,
            )

    # -o targets a single output file, so it only makes sense when the caller
    # passes exactly one image file (not a directory that fans out to many).
    if args.output:
        if len(args.images) != 1 or args.images[0].is_dir():
            p.error("-o/--output can only be used with a single input image")

    do_text = not args.faces_only
    do_faces = not args.text_only

    files, missing = _collect_inputs(args.images, args.recursive)

    exit_code = 0
    for path in missing:
        print(f"skip: {path} (not found)", file=sys.stderr)
        exit_code = 1

    if not files:
        print("no image files to process", file=sys.stderr)
        return exit_code or 1

    ok_count = 0
    for in_path in files:
        out_path = args.output or _default_output(in_path, args.preview)
        try:
            res = process_image(
                in_path, out_path,
                do_text=do_text, do_faces=do_faces,
                pad_frac=args.padding, strength=args.strength,
                preview=args.preview,
                verify_cfg=verify_cfg,
                verify_local=verify_local,
                verify_max_passes=args.verify_max_passes,
            )
        except (DetectionError, OSError) as e:
            print(f"error: {in_path}: {e}", file=sys.stderr)
            exit_code = 1
            continue

        ok_count += 1
        verb = "boxed" if args.preview else "blurred"
        msg = (f"{in_path.name}: {verb} {res.text_count} text region(s), "
               f"{res.face_count} face(s)")
        if res.local_verify_passes:
            msg += (f"; local re-blurred {res.local_verify_reblurred} "
                    f"region(s) over {res.local_verify_passes} pass(es)")
        if res.verify_passes:
            msg += (f"; model re-blurred {res.verify_reblurred} region(s) "
                    f"over {res.verify_passes} pass(es)")
        msg += f" -> {out_path}"
        print(msg)
        if res.verify_error:
            print(f"  warning: verification skipped ({res.verify_error})",
                  file=sys.stderr)
            exit_code = 1

    if len(files) > 1:
        print(f"done: {ok_count}/{len(files)} image(s) processed")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
