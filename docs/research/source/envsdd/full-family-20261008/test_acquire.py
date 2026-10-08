"""Local acquisition boundary tests; no network or model execution."""

# Standalone evidence tests intentionally use the stdlib unittest runner.
# ruff: noqa: PT009, PT027
from __future__ import annotations

import io
import unittest
import wave

import acquire


class AcquisitionTests(unittest.TestCase):
    @staticmethod
    def wav(channels: int = 1, frames: int = 64000) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b"\x00\x00" * frames * channels)
        return output.getvalue()

    def test_full_decode_and_zero_diagnostics(self) -> None:
        raw = self.wav()
        info = acquire.decode(raw)
        self.assertEqual(info["frames"], 64000)
        self.assertEqual(info["sha256"], acquire.sha(raw))
        self.assertEqual(info["rms"], 0)
        self.assertEqual(info["zero_fraction"], 1)
        self.assertTrue(info["all_zero"])

    def test_truncated_payload_rejected(self) -> None:
        with self.assertRaises(ValueError):
            acquire.decode(self.wav()[:-2])

    def test_wrong_channel_count_rejected(self) -> None:
        with self.assertRaises(ValueError):
            acquire.decode(self.wav(channels=2))

    def test_wrong_duration_rejected(self) -> None:
        with self.assertRaises(ValueError):
            acquire.decode(self.wav(frames=63999))

    def test_untrusted_host_rejected_before_request(self) -> None:
        with self.assertRaises(ValueError):
            acquire.get("https://example.org/media.wav", 256088)

    def test_selection_reproduces_frozen_scope(self) -> None:
        selected = acquire.select()
        self.assertEqual(
            selected["counts"], {"fsids": 40, "filenames": 106, "rows": 530}
        )
        self.assertTrue(
            all(r["metadata"]["dataset"] == "UrbanSound8K" for r in selected["samples"])
        )


if __name__ == "__main__":
    unittest.main()
