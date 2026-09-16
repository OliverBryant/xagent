"""Transparent-background support for image generation.

Only some providers can emit an alpha channel themselves (see
``BaseImageModel.supports_transparent_background``). For every other provider
the alpha has to be produced locally: the prompt asks for the subject on a flat
chroma background, and the downloaded file is keyed against that color here.

Pixels on the subject's antialiased edge are part background, so a keyed cutout
carries a faint magenta cast about one pixel wide. It is left alone on purpose:
recovering the true edge color means unpremultiplying by alpha, and the cheap
channel-subtraction despill that ImageChops can express moves a blue or cyan
subject further from its real color than the cast does.

Keying is an approximation, so every helper reports what it actually did rather
than assuming it worked. A caller that cannot tell "keyed 40% of the frame" from
"found no chroma background at all" would hand the user an opaque PNG and call
it transparent, which is the failure this module exists to prevent.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image, ImageChops, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Pure magenta. Chosen over chroma green because it is far rarer in the
# photographic, product, and UI subjects these models are asked for, and because
# the max-channel-distance metric below only keys the magenta family: pure red,
# blue, white, black, and mid grays all sit at the maximum distance from it.
KEY_COLOR: Tuple[int, int, int] = (255, 0, 255)
KEY_COLOR_HEX = "#FF00FF"

# Distance from KEY_COLOR, in 0-255 max-channel terms, over which a pixel goes
# from fully transparent to fully opaque. The ramp between them feathers the
# cutout edge so the subject does not end up with a hard aliased border.
_KEY_TOLERANCE = 48
_KEY_EDGE = 128

# Below this share of fully transparent pixels the model almost certainly
# ignored the flat-background instruction, so the cutout is reported as failed
# rather than silently passed off as transparent.
_MIN_TRANSPARENT_RATIO = 0.02

# Alpha is treated as already meaningful once any pixel falls below this level;
# a provider that emits alpha at all emits real holes, not a uniform 254.
_ALPHA_PRESENT_BELOW = 250


def transparency_prompt_instructions() -> str:
    """Prompt block that makes a non-alpha provider produce a keyable frame.

    Phrased as hard rules about the background rather than a request for
    "transparency": a model asked for transparency tends to paint the gray
    checkerboard that image editors use to *display* it.
    """
    return (
        "TRANSPARENT BACKGROUND REQUIRED (produced by post-processing, so follow "
        "these rules exactly):\n"
        "- Render the subject only. Every pixel the subject does not cover must "
        f"be flat, uniform, solid {KEY_COLOR_HEX} (pure magenta, RGB 255,0,255).\n"
        "- That background must be one single flat color: no gradient, texture, "
        "vignette, pattern, scenery, backdrop, drop shadow, contact shadow, "
        "reflection, or glow.\n"
        "- Do not use magenta, hot pink, or purple anywhere in the subject "
        "itself, and do not outline or frame the subject in those colors: those "
        "pixels get cut away.\n"
        "- Do not draw a checkerboard or any other placeholder pattern that "
        "represents transparency."
    )


def _has_meaningful_alpha(image: Image.Image) -> bool:
    """Whether the image already carries alpha the provider put there.

    Counted off the histogram rather than the extrema because a single-band
    ``getextrema()`` is typed as a union with the multi-band form, and because a
    histogram slice answers exactly the question being asked: does any pixel sit
    far enough below opaque to be a real hole.
    """
    if "A" not in image.getbands():
        return False
    histogram = image.getchannel("A").histogram()
    return any(histogram[:_ALPHA_PRESENT_BELOW])


def _build_alpha(rgb: Image.Image) -> Image.Image:
    """Feathered alpha from each pixel's distance to KEY_COLOR.

    The metric is the largest per-channel distance, which keys the magenta
    family and nothing else. All of it stays in Pillow's C paths, so a 4K frame
    costs milliseconds instead of the minutes a per-pixel Python loop would.
    """
    red, green, blue = rgb.split()
    solid = {
        channel: Image.new("L", rgb.size, value)
        for channel, value in zip("rgb", KEY_COLOR)
    }

    distance = ImageChops.difference(red, solid["r"])
    distance = ImageChops.lighter(distance, ImageChops.difference(green, solid["g"]))
    distance = ImageChops.lighter(distance, ImageChops.difference(blue, solid["b"]))

    span = _KEY_EDGE - _KEY_TOLERANCE
    ramp = [
        0
        if value <= _KEY_TOLERANCE
        else 255
        if value >= _KEY_EDGE
        else round((value - _KEY_TOLERANCE) * 255 / span)
        for value in range(256)
    ]
    return distance.point(ramp)


def ensure_transparent_background(
    image_path: str | Path, *, allow_keying: bool
) -> Dict[str, Any]:
    """Make ``image_path`` carry a real alpha channel, in place, as PNG.

    Args:
        image_path: Saved image to inspect and, when keying is allowed, cut out.
        allow_keying: True for providers with no alpha of their own, where the
            prompt asked for a flat chroma background. False for a provider that
            was asked for transparency natively: its output is only inspected,
            because keying it would hunt for a chroma background that was never
            requested and could eat magenta the user actually wanted.

    Returns:
        Report with ``mode`` (``native``, ``keyed``, or ``failed``),
        ``transparent`` (whether the file now has usable alpha), and
        ``transparent_ratio``. ``warning`` is present whenever the caller must
        tell the user the background did not come out transparent.
    """
    path = Path(image_path)
    report: Dict[str, Any] = {
        "requested": True,
        "mode": "failed",
        "transparent": False,
        "transparent_ratio": 0.0,
    }

    try:
        with Image.open(path) as opened:
            opened.load()
            source = opened.copy()
    except (OSError, UnidentifiedImageError) as exc:
        logger.warning("Cannot open generated image for transparency: %s", exc)
        report["warning"] = (
            f"Could not read the generated image to apply transparency: {exc}. "
            "The saved file has no alpha channel."
        )
        return report

    if _has_meaningful_alpha(source):
        # The provider produced the alpha itself; re-keying could only damage it.
        rgba = source.convert("RGBA")
        report.update(
            mode="native",
            transparent=True,
            transparent_ratio=_transparent_ratio(rgba),
        )
        _save_png(rgba, path)
        return report

    if not allow_keying:
        report["warning"] = (
            "The model returned an image with no alpha channel even though "
            "transparency was requested natively. The saved file has an opaque "
            "background."
        )
        return report

    rgb = source.convert("RGB")
    alpha = _build_alpha(rgb)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)

    # Neutralize the color hiding under the fully cut area. Correct compositing
    # ignores it, but anything downstream that flattens the alpha away -- a
    # thumbnailer, or a vision tool doing convert("RGB") -- would otherwise
    # render the magenta key instead of the white a viewer expects.
    fully_cut = alpha.point([255 if value == 0 else 0 for value in range(256)])
    rgba.paste((255, 255, 255, 0), mask=fully_cut)

    ratio = _transparent_ratio(rgba)
    report["transparent_ratio"] = round(ratio, 4)

    if ratio < _MIN_TRANSPARENT_RATIO:
        # Keying the frame anyway would replace a clean opaque image with a
        # slightly chewed one and still not be transparent.
        report["warning"] = (
            f"The model did not render the flat {KEY_COLOR_HEX} background that "
            "local transparency keying needs, so the saved image still has an "
            "opaque background. Retry with the background rules restated, or "
            "tell the user this model cannot produce a transparent background."
        )
        return report

    _save_png(rgba, path)
    report.update(mode="keyed", transparent=True, key_color=KEY_COLOR_HEX)
    return report


def _transparent_ratio(rgba: Image.Image) -> float:
    """Share of fully transparent pixels in an RGBA image."""
    total = rgba.width * rgba.height
    if total == 0:
        return 0.0
    return rgba.getchannel("A").histogram()[0] / total


def _save_png(rgba: Image.Image, path: Path) -> None:
    """Write RGBA back over ``path`` as PNG, whatever the original encoding was.

    The caller names the download ``.png`` up front when transparency is
    requested, so a provider that answered with JPEG bytes still lands on a path
    that can hold the alpha we just built.
    """
    rgba.save(path, format="PNG")
