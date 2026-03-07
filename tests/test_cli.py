"""Tests for the HERALD CLI."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from herald.cli import main


class TestCLI:
    def test_url_command_legitimate(self, capsys: pytest.CaptureFixture) -> None:
        exit_code = main(["url", "https://www.example.com/"])
        assert exit_code == 0

    def test_url_command_json_output(self, capsys: pytest.CaptureFixture) -> None:
        main(["--format", "json", "url", "https://example.com/"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "is_phishing" in data
        assert "risk_score" in data

    def test_url_command_threshold_zero(self, capsys: pytest.CaptureFixture) -> None:
        """Threshold of 0 means everything is phishing."""
        exit_code = main(["--threshold", "0.0", "url", "https://example.com/"])
        assert exit_code == 1

    def test_url_command_threshold_one(self, capsys: pytest.CaptureFixture) -> None:
        """Threshold of 1.0 means nothing is phishing."""
        exit_code = main(["--threshold", "1.0", "url", "http://192.168.1.1/admin"])
        assert exit_code == 0

    def test_text_command_string(self, capsys: pytest.CaptureFixture) -> None:
        exit_code = main(["text", "Hello, this is a normal message."])
        assert exit_code in (0, 1)

    def test_text_command_file(self, capsys: pytest.CaptureFixture) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("Normal business email content.")
            path = fh.name
        try:
            exit_code = main(["text", "--file", path])
            assert exit_code in (0, 1)
        finally:
            os.unlink(path)

    def test_text_command_missing_file_returns_error(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        exit_code = main(["text", "--file", "/nonexistent/path/file.txt"])
        assert exit_code == 2

    def test_headers_command_inline(self, capsys: pytest.CaptureFixture) -> None:
        exit_code = main(
            [
                "headers",
                "--from-addr",
                "noreply@example.com",
                "--subject",
                "Hello",
            ]
        )
        assert exit_code in (0, 1)

    def test_headers_command_no_args_returns_error(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        exit_code = main(["headers"])
        assert exit_code == 2

    def test_email_command(self, capsys: pytest.CaptureFixture) -> None:
        raw = (
            "From: test@example.com\r\n"
            "Subject: Hello\r\n"
            "Message-ID: <abc@example.com>\r\n"
            "\r\n"
            "Normal email body."
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".eml", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(raw)
            path = fh.name
        try:
            exit_code = main(["email", path])
            assert exit_code in (0, 1)
        finally:
            os.unlink(path)

    def test_email_command_missing_file_returns_error(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        exit_code = main(["email", "/nonexistent.eml"])
        assert exit_code == 2

    def test_json_output_contains_expected_keys(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        main(["--format", "json", "url", "http://192.168.0.1/admin"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        for key in (
            "is_phishing",
            "risk_score",
            "risk_level",
            "confidence",
            "indicators",
        ):
            assert key in data
