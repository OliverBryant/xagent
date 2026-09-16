"""Tests for local transparent-background support."""

from pathlib import Path

import pytest
from PIL import Image

from xagent.core.model.image.transparency import (
    KEY_COLOR,
    KEY_COLOR_HEX,
    ensure_transparent_background,
    transparency_prompt_instructions,
)

SUBJECT_COLOR = (20, 130, 200)


def _write(image: Image.Image, path: Path, **save_kwargs) -> Path:
    image.save(path, **save_kwargs)
    return path


def _keyable(size=(40, 40), subject_box=(10, 10, 30, 30), background=KEY_COLOR):
    """Flat background with one solid rectangle of subject in the middle."""
    image = Image.new("RGB", size, background)
    for x in range(subject_box[0], subject_box[2]):
        for y in range(subject_box[1], subject_box[3]):
            image.putpixel((x, y), SUBJECT_COLOR)
    return image


class TestPromptInstructions:
    def test_names_the_key_color_and_bans_placeholders(self):
        instructions = transparency_prompt_instructions()

        assert KEY_COLOR_HEX in instructions
        assert "255,0,255" in instructions
        # A model told only "transparent" paints the editor's checkerboard.
        assert "checkerboard" in instructions.lower()
        # The subject must stay out of the keyed color range.
        assert "purple" in instructions.lower()


class TestKeying:
    def test_edge_pixels_are_feathered_not_hard_stepped(self, tmp_path):
        # Half-keyed blend column between background and subject, as an
        # antialiased edge produces.
        image = _keyable(subject_box=(20, 10, 30, 30))
        for y in range(10, 30):
            image.putpixel((19, y), (140, 60, 220))
        path = _write(image, tmp_path / "shot.png")

        ensure_transparent_background(path, allow_keying=True)

        with Image.open(path) as saved:
            alpha = saved.convert("RGBA").getpixel((19, 20))[3]
        assert 0 < alpha < 255

    def test_keys_flat_background_and_keeps_subject(self, tmp_path):
        path = _write(_keyable(), tmp_path / "shot.png")

        report = ensure_transparent_background(path, allow_keying=True)

        assert report["mode"] == "keyed"
        assert report["transparent"] is True
        assert report["key_color"] == KEY_COLOR_HEX
        assert "warning" not in report

        with Image.open(path) as saved:
            assert saved.format == "PNG"
            rgba = saved.convert("RGBA")
        # Background corner fully cut, subject centre fully kept.
        assert rgba.getpixel((0, 0))[3] == 0
        assert rgba.getpixel((20, 20))[3] == 255
        assert rgba.getpixel((20, 20))[:3] == SUBJECT_COLOR

    def test_flattening_the_alpha_shows_white_not_the_key_color(self, tmp_path):
        path = _write(_keyable(), tmp_path / "shot.png")

        ensure_transparent_background(path, allow_keying=True)

        with Image.open(path) as saved:
            flattened = saved.convert("RGBA").convert("RGB")
        # Thumbnailers and vision tools drop the alpha; showing magenta there
        # would look like a broken image rather than a transparent one.
        assert flattened.getpixel((0, 0)) == (255, 255, 255)

    def test_transparent_ratio_matches_background_share(self, tmp_path):
        # 40x40 frame, 20x20 subject -> three quarters of the frame is keyed.
        path = _write(_keyable(), tmp_path / "shot.png")

        report = ensure_transparent_background(path, allow_keying=True)

        assert report["transparent_ratio"] == pytest.approx(0.75, abs=0.02)

    @pytest.mark.parametrize(
        "subject",
        [
            (255, 255, 255),  # white
            (0, 0, 0),  # black
            (255, 0, 0),  # pure red shares two channels with magenta
            (0, 0, 255),  # pure blue shares two channels with magenta
            (128, 128, 128),  # mid grey
        ],
    )
    def test_only_the_magenta_family_is_keyed(self, tmp_path, subject):
        image = _keyable()
        for x in range(10, 30):
            for y in range(10, 30):
                image.putpixel((x, y), subject)
        path = _write(image, tmp_path / "shot.png")

        ensure_transparent_background(path, allow_keying=True)

        with Image.open(path) as saved:
            rgba = saved.convert("RGBA")
        assert rgba.getpixel((20, 20))[3] == 255, f"{subject} was keyed away"

    def test_jpeg_payload_saved_at_png_path_still_keys(self, tmp_path):
        # Providers that answer with JPEG bytes land on a .png name, because the
        # caller renames the download when transparency is requested.
        path = tmp_path / "shot.png"
        _keyable().save(path, format="JPEG", quality=95)

        report = ensure_transparent_background(path, allow_keying=True)

        assert report["mode"] == "keyed"
        with Image.open(path) as saved:
            assert saved.format == "PNG"
            assert saved.convert("RGBA").getpixel((0, 0))[3] == 0


class TestAlreadyTransparent:
    def _with_alpha(self, tmp_path):
        image = Image.new("RGBA", (10, 10), (*SUBJECT_COLOR, 255))
        image.putpixel((0, 0), (0, 0, 0, 0))
        return _write(image, tmp_path / "shot.png")

    @pytest.mark.parametrize("allow_keying", [True, False])
    def test_provider_alpha_is_reported_native_and_preserved(
        self, tmp_path, allow_keying
    ):
        path = self._with_alpha(tmp_path)

        report = ensure_transparent_background(path, allow_keying=allow_keying)

        assert report["mode"] == "native"
        assert report["transparent"] is True
        assert "warning" not in report
        with Image.open(path) as saved:
            rgba = saved.convert("RGBA")
        assert rgba.getpixel((0, 0))[3] == 0
        assert rgba.getpixel((5, 5)) == (*SUBJECT_COLOR, 255)


class TestFailureReporting:
    def test_no_key_background_leaves_file_opaque_and_warns(self, tmp_path):
        path = _write(
            Image.new("RGB", (20, 20), (240, 240, 240)), tmp_path / "shot.png"
        )
        before = path.read_bytes()

        report = ensure_transparent_background(path, allow_keying=True)

        assert report["mode"] == "failed"
        assert report["transparent"] is False
        assert KEY_COLOR_HEX in report["warning"]
        # Nothing is written, so a clean opaque image is not replaced by a
        # partially chewed one that is still opaque.
        assert path.read_bytes() == before

    def test_native_request_that_came_back_opaque_warns(self, tmp_path):
        path = _write(_keyable(), tmp_path / "shot.png")

        report = ensure_transparent_background(path, allow_keying=False)

        assert report["mode"] == "failed"
        assert report["transparent"] is False
        assert "natively" in report["warning"]
        # The magenta background is left alone: it was never asked for here, so
        # keying it would be guessing.
        with Image.open(path) as saved:
            assert saved.convert("RGBA").getpixel((0, 0))[3] == 255

    def test_unreadable_file_reports_instead_of_raising(self, tmp_path):
        path = tmp_path / "shot.png"
        path.write_bytes(b"not an image")

        report = ensure_transparent_background(path, allow_keying=True)

        assert report["mode"] == "failed"
        assert report["transparent"] is False
        assert "warning" in report

    def test_missing_file_reports_instead_of_raising(self, tmp_path):
        report = ensure_transparent_background(tmp_path / "gone.png", allow_keying=True)

        assert report["mode"] == "failed"
        assert "warning" in report
