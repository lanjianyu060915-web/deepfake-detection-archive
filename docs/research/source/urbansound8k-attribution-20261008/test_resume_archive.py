"""HTTP range identity and prefix preservation tests without network access."""

# ruff: noqa: PT009, PT027
from __future__ import annotations

import io
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import resume_archive


class Response(io.BytesIO):
    def __init__(self, body: bytes, *, status: int = 206, total: int = 8) -> None:
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": f"bytes 4-7/{total}"}


class ResumeTests(unittest.TestCase):
    def test_valid_range_preserves_prefix_and_finishes_bytes_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "archive.part"
            target.write_bytes(b"init")
            with (
                patch.object(resume_archive, "TARGET", target),
                patch.object(resume_archive, "DOC", root),
                patch.object(resume_archive, "TOTAL", 8),
                patch.object(resume_archive, "CHUNK", 4),
                patch.object(
                    urllib.request,
                    "urlopen",
                    return_value=Response(b"data"),
                ),
            ):
                resume_archive.main()
            self.assertEqual(target.read_bytes(), b"initdata")
            self.assertFalse(target.with_suffix(".range-tmp").exists())

    def test_wrong_status_total_truncation_or_excess_never_appends(self) -> None:
        cases = [
            Response(b"data", status=200),
            Response(b"data", total=9),
            Response(b"dat"),
            Response(b"data!"),
        ]
        for response in cases:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as d:
                prefix = Path(d) / "archive.part"
                prefix.write_bytes(b"init")
                temporary = prefix.with_suffix(".range-tmp")
                with (
                    patch.object(resume_archive, "TOTAL", 8),
                    patch.object(
                        urllib.request,
                        "urlopen",
                        return_value=response,
                    ),
                    self.assertRaises(ValueError),
                ):
                    resume_archive.download_range(4, 7, temporary)
                self.assertEqual(prefix.read_bytes(), b"init")


if __name__ == "__main__":
    unittest.main()
