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

Only chroma that is connected to the frame border is treated as background. A
magenta logo *inside* the subject is the same color as the backdrop, so a pure
per-pixel test would punch a hole through the subject and the ratio check --
which measures how much was cut, never where -- would not notice. Seeding from
the border makes "background" mean reachable from outside, which is what the
prompt actually asks the model for.

Keying is an approximation, so every helper reports what it actually did rather
than assuming it worked. A caller that cannot tell "keyed 40% of the frame" from
"found no chroma background at all" would hand the user an opaque PNG and call
it transparent, which is the failure this module exists to prevent.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageChops, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Pure magenta. Chosen over chroma green because it is far rarer in the
# photographic, product, and UI subjects these models are asked for, and because
# the max-channel-distance metric below only keys the magenta family: pure red,
# blue, white, and black all sit at the maximum distance from it.
#
# Neutral grey is the closest non-magenta color, not a maximally distant one:
# (128,128,128) lands at exactly _KEY_EDGE, just outside the ramp. Off-neutral
# near-greys such as (130,126,130) fall a little inside it and pick up a few
# percent of translucency. That is a known and minor imprecision -- moving
# _KEY_EDGE down to clear it would narrow the feather that keeps cutout edges
# from aliasing, which costs more than it buys.
KEY_COLOR: Tuple[int, int, int] = (255, 0, 255)
KEY_COLOR_HEX = "#{:02X}{:02X}{:02X}".format(*KEY_COLOR)

# Distance from KEY_COLOR, in 0-255 max-channel terms, over which a pixel goes
# from fully transparent to fully opaque. The ramp between them feathers the
# cutout edge so the subject does not end up with a hard aliased border.
_KEY_TOLERANCE = 48
_KEY_EDGE = 128

# Below this share of fully transparent pixels the model almost certainly
# ignored the flat-background instruction, so the cutout is reported as failed
# rather than silently passed off as transparent.
_MIN_TRANSPARENT_RATIO = 0.02

# And above this share almost nothing of the subject survived: an all-chroma
# frame (the model rendered the background and no subject) keys to a fully
# transparent image, which is not a cutout even though every pixel "worked".
_MAX_TRANSPARENT_RATIO = 0.98

# Alpha is treated as already meaningful once a pixel falls below this level.
_ALPHA_PRESENT_BELOW = 250

# ...but a handful of such pixels is not a transparent background: a lone
# stray sub-opaque pixel would otherwise route a fully opaque image down the
# "the provider did it natively" path and report it as transparent. Provider
# alpha that means anything covers a real region of the frame.
_MIN_NATIVE_ALPHA_RATIO = 0.005


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
    histogram slice answers exactly the question being asked: how much of the
    frame sits far enough below opaque to be a real hole.

    A palette image carries its transparency in ``info`` rather than in an "A"
    band, so band inspection alone would miss it and send a genuinely
    transparent PNG down the keying path.

    The share has to clear ``_MIN_NATIVE_ALPHA_RATIO``: one stray sub-opaque
    pixel in an otherwise solid frame is an encoding artifact, not a background
    the provider cut out, and treating it as native alpha would report an opaque
    image as transparent.
    """
    if "A" not in image.getbands() and "transparency" not in image.info:
        return False
    alpha = image.convert("RGBA").getchannel("A")
    total = alpha.size[0] * alpha.size[1]
    if total == 0:
        return False
    histogram = alpha.histogram()
    return sum(histogram[:_ALPHA_PRESENT_BELOW]) / total >= _MIN_NATIVE_ALPHA_RATIO


def _border_connected(keyable: Image.Image) -> Image.Image:
    """Restrict a chroma mask to the region reachable from the frame border.

    ``keyable`` marks every pixel close enough to KEY_COLOR to be cut. Chroma
    enclosed by the subject -- a magenta logo, a pink highlight -- is the same
    color as the backdrop but is not backdrop, so it must survive.

    Implemented as an iterative binary dilation of the border seeds, confined to
    the keyable region. Each pass is a whole-array numpy shift-or, so the work
    is proportional to the mask's diameter in passes rather than to pixels in
    Python. Pillow's own ``ImageDraw.floodfill`` is a per-pixel Python loop and
    is orders of magnitude slower on a full frame.
    """
    mask = np.array(keyable, dtype=bool)
    if not mask.any():
        return keyable

    reached = np.zeros_like(mask)
    # Seed with the keyable pixels lying on the frame's outer edge.
    reached[0, :] = mask[0, :]
    reached[-1, :] = mask[-1, :]
    reached[:, 0] = mask[:, 0]
    reached[:, -1] = mask[:, -1]

    while True:
        grown = reached.copy()
        grown[1:, :] |= reached[:-1, :]
        grown[:-1, :] |= reached[1:, :]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= mask
        if np.array_equal(grown, reached):
            break
        reached = grown

    return Image.fromarray((reached * 255).astype(np.uint8), mode="L")


def _build_alpha(rgb: Image.Image) -> Image.Image:
    """Feathered alpha from each pixel's distance to KEY_COLOR.

    The metric is the largest per-channel distance, which keys the magenta
    family and nothing else. All of it stays in Pillow's C paths, so a 4K frame
    costs milliseconds instead of the minutes a per-pixel Python loop would.

    The distance ramp alone would also cut chroma trapped inside the subject, so
    the result is masked down to the part of it that is connected to the frame
    border: colour decides what *may* be background, reachability decides what
    is.
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
    alpha = distance.point(ramp)

    # Anything the ramp touched at all is a candidate for removal, including the
    # partly-keyed feather, so connectivity is judged on that full region rather
    # than on the fully transparent core alone -- otherwise an antialiased edge
    # would sever the background from the border and nothing would be cut.
    keyable = alpha.point([255 if value < 255 else 0 for value in range(256)])
    connected = _border_connected(keyable)

    # Keep the feathered value where the background is reachable; force opaque
    # everywhere else, which restores chroma sealed inside the subject.
    return ImageChops.lighter(alpha, ImageChops.invert(connected))


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

    The file is always left as a real PNG when it can be read at all, even when
    transparency fails: the caller has already named the download ``.png`` and
    derives the artifact's mime type from that suffix, so leaving JPEG bytes
    behind would publish them as ``image/png``.

    Returns:
        Report with ``mode`` (``native``, ``keyed``, or ``failed``),
        ``transparent`` (whether the file now has usable alpha), and
        ``transparent_ratio``. ``warning`` is present whenever the caller must
        tell the user the background did not come out transparent.
    """
    path = Path(image_path)
    report: Dict[str, Any] = {
        "mode": "failed",
        "transparent": False,
        "transparent_ratio": 0.0,
    }

    try:
        with Image.open(path) as opened:
            opened.load()
            source = opened.copy()
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
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
        # Opaque, but still normalized to PNG so the bytes match the .png name
        # the caller gave the download and the mime type derived from it.
        _save_png(source.convert("RGBA"), path)
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
        # slightly chewed one and still not be transparent, so the untouched
        # pixels are kept and only the container is normalized to PNG.
        report["warning"] = (
            f"The model did not render the flat {KEY_COLOR_HEX} background that "
            "local transparency keying needs, so the saved image still has an "
            "opaque background. Retry with the background rules restated, or "
            "tell the user this model cannot produce a transparent background."
        )
        _save_png(source.convert("RGBA"), path)
        return report

    if ratio > _MAX_TRANSPARENT_RATIO:
        # Nearly everything was cut, so the model rendered the background and
        # left out the subject. Saving this would hand back an empty frame that
        # every check calls a successful cutout.
        report["warning"] = (
            "Almost the entire image was keyed away, so the model rendered the "
            f"flat {KEY_COLOR_HEX} background without a subject on top of it. "
            "The saved image is unchanged and still opaque. Retry with the "
            "subject described explicitly."
        )
        _save_png(source.convert("RGBA"), path)
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

    Written to a sibling temp file and moved into place, because a save that
    fails midway through -- a full disk, or permissions on the shared task
    workspace -- would otherwise truncate the downloaded image the caller still
    needs. ``os.replace`` is atomic within a directory, so the path either holds
    the old bytes or the new ones.
    """
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        rgba.save(tmp_path, format="PNG")
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
