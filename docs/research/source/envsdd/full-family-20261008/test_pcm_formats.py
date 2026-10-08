"""Source PCM widths, signed boundaries and malformed payload regression tests."""

# ruff: noqa: PT009, PT027
from __future__ import annotations

import io
import unittest
import wave

import acquire_pcm
import numpy as np
import soundfile as sf  # type: ignore[import-untyped]  # independent libsndfile oracle
from numpy.typing import NDArray


def fixture(width: int) -> tuple[bytes, NDArray[np.float64]]:
    scale = 1 << (width * 8 - 1)
    signed = [-scale, -1, 0, 1, scale - 1] * 12800
    raw = b"".join(
        (value + scale if width == 1 else value).to_bytes(
            width, "little", signed=width != 1
        )
        for value in signed
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(width)
        writer.setframerate(16000)
        writer.writeframes(raw)
    return output.getvalue(), np.array(signed, dtype=np.float64) / scale


class PCMFormatTests(unittest.TestCase):
    def test_signed_boundaries_against_independent_libsndfile(self) -> None:
        for width in (1, 2, 3, 4):
            with self.subTest(width=width):
                body, expected = fixture(width)
                decoded, rate = sf.read(io.BytesIO(body), dtype="float64")
                self.assertEqual(rate, 16000)
                np.testing.assert_array_equal(decoded, expected)
                info = acquire_pcm.decode(body)
                self.assertEqual(info["sha256"], acquire_pcm.sha(body))
                self.assertEqual(info["sample_width_bytes"], width)
                self.assertEqual(info["rms"], float(np.sqrt(np.mean(expected**2))))
                self.assertEqual(info["peak"], 1)
                self.assertEqual(info["zero_fraction"], 0.2)
                self.assertEqual(info["clipping_fraction"], 0.4)
                self.assertFalse(info["all_zero"])

    def test_truncation_rejected_for_every_width(self) -> None:
        for width in (1, 2, 3, 4):
            with self.subTest(width=width):
                body, _ = fixture(width)
                with self.assertRaises(ValueError):
                    acquire_pcm.decode(body[:-width])


if __name__ == "__main__":
    unittest.main()
