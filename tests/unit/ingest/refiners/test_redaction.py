"""Tests for PII redaction refiner (SEC-001.1).

NASA JPL Commandments compliance:
- Rule #1: Linear test structure
- Rule #2: Fixed test data bounds
- Rule #4: Functions <60 lines
"""

from __future__ import annotations


from ingestforge.ingest.refiners.redaction import (
    PIIType,
    RedactionMatch,
    RedactionResult,
    RedactionConfig,
    PIIRedactor,
    redact_pii,
    redact_batch,
    MAX_PATTERNS_PER_TYPE,
    MAX_WHITELIST_ENTRIES,
    MAX_REDACTIONS_PER_TEXT,
)


class TestPIIType:
    """Tests for PIIType enum."""

    def test_all_types_exist(self) -> None:
        """All expected PII types are defined."""
        expected = {
            "EMAIL",
            "PHONE",
            "SSN",
            "CREDIT_CARD",
            "PERSON_NAME",
            "ADDRESS",
            "DATE_OF_BIRTH",
            "IP_ADDRESS",
            "CUSTOM",
        }
        actual = {t.name for t in PIIType}
        assert actual == expected

    def test_type_values(self) -> None:
        """Type values are lowercase strings."""
        assert PIIType.EMAIL.value == "email"
        assert PIIType.PHONE.value == "phone"
        assert PIIType.SSN.value == "ssn"


class TestRedactionMatch:
    """Tests for RedactionMatch dataclass."""

    def test_basic_match(self) -> None:
        """RedactionMatch stores match data."""
        match = RedactionMatch(
            pii_type=PIIType.EMAIL,
            original="test@example.com",
            start=0,
            end=16,
            replacement="[EMAIL]",
            confidence=1.0,
        )
        assert match.pii_type == PIIType.EMAIL
        assert match.original == "test@example.com"
        assert match.start == 0
        assert match.end == 16

    def test_length_property(self) -> None:
        """Length property calculates match length."""
        match = RedactionMatch(
            pii_type=PIIType.PHONE,
            original="555-123-4567",
            start=10,
            end=22,
            replacement="[PHONE]",
        )
        assert match.length == 12

    def test_default_confidence(self) -> None:
        """Default confidence is 1.0."""
        match = RedactionMatch(
            pii_type=PIIType.SSN,
            original="123-45-6789",
            start=0,
            end=11,
            replacement="[SSN]",
        )
        assert match.confidence == 1.0


class TestRedactionResult:
    """Tests for RedactionResult dataclass."""

    def test_empty_result(self) -> None:
        """Empty result has no redactions."""
        result = RedactionResult(
            original_text="Hello world",
            redacted_text="Hello world",
        )
        assert result.total_redactions == 0
        assert not result.has_redactions

    def test_result_with_matches(self) -> None:
        """Result tracks matches and statistics."""
        matches = [
            RedactionMatch(
                pii_type=PIIType.EMAIL,
                original="test@example.com",
                start=0,
                end=16,
                replacement="[EMAIL]",
            ),
        ]
        result = RedactionResult(
            original_text="test@example.com",
            redacted_text="[EMAIL]",
            matches=matches,
            stats={"email": 1},
        )
        assert result.total_redactions == 1
        assert result.has_redactions

    def test_skipped_items(self) -> None:
        """Result tracks skipped whitelist items."""
        result = RedactionResult(
            original_text="Some text",
            redacted_text="Some text",
            skipped=["safe@company.com"],
        )
        assert "safe@company.com" in result.skipped


class TestRedactionConfig:
    """Tests for RedactionConfig dataclass."""

    def test_default_enabled_types(self) -> None:
        """Default config enables common PII types."""
        config = RedactionConfig()
        assert PIIType.EMAIL in config.enabled_types
        assert PIIType.PHONE in config.enabled_types
        assert PIIType.SSN in config.enabled_types
        assert PIIType.PERSON_NAME in config.enabled_types

    def test_add_to_whitelist(self) -> None:
        """Whitelist can be modified."""
        config = RedactionConfig()
        assert config.add_to_whitelist("safe@company.com")
        assert config.is_whitelisted("safe@company.com")
        assert config.is_whitelisted("SAFE@COMPANY.COM")  # Case insensitive

    def test_whitelist_limit(self) -> None:
        """Whitelist respects MAX_WHITELIST_ENTRIES."""
        config = RedactionConfig()
        # Fill whitelist to limit
        for i in range(MAX_WHITELIST_ENTRIES):
            config.whitelist.add(f"term{i}")

        # Adding one more should fail
        assert not config.add_to_whitelist("overflow")

    def test_custom_mask_char(self) -> None:
        """Custom mask character can be set."""
        config = RedactionConfig(mask_char="X")
        assert config.mask_char == "X"

    def test_preserve_length_option(self) -> None:
        """Preserve length option is configurable."""
        config = RedactionConfig(preserve_length=True)
        assert config.preserve_length is True


class TestPIIRedactorEmail:
    """Tests for email redaction."""

    def test_simple_email(self) -> None:
        """Detects simple email addresses."""
        redactor = PIIRedactor()
        result = redactor.redact("Contact me at test@example.com")
        assert "[EMAIL]" in result.redacted_text
        assert result.stats.get("email", 0) == 1

    def test_multiple_emails(self) -> None:
        """Detects multiple email addresses."""
        redactor = PIIRedactor()
        result = redactor.redact("Email john@test.com or jane@test.org")
        assert result.redacted_text.count("[EMAIL]") == 2

    def test_complex_email(self) -> None:
        """Detects emails with special characters."""
        redactor = PIIRedactor()
        result = redactor.redact("Email: user.name+tag@sub.domain.co.uk")
        assert "[EMAIL]" in result.redacted_text


class TestPIIRedactorPhone:
    """Tests for phone number redaction."""

    def test_dashed_phone(self) -> None:
        """Detects dash-separated phone numbers."""
        config = RedactionConfig(enabled_types={PIIType.PHONE})
        redactor = PIIRedactor(config)
        result = redactor.redact("Call 555-123-4567")
        assert "[PHONE]" in result.redacted_text

    def test_dotted_phone(self) -> None:
        """Detects dot-separated phone numbers."""
        config = RedactionConfig(enabled_types={PIIType.PHONE})
        redactor = PIIRedactor(config)
        result = redactor.redact("Call 555.123.4567")
        assert "[PHONE]" in result.redacted_text

    def test_parentheses_phone(self) -> None:
        """Detects parentheses format phone numbers."""
        config = RedactionConfig(enabled_types={PIIType.PHONE})
        redactor = PIIRedactor(config)
        result = redactor.redact("Call (555) 123-4567")
        assert "[PHONE]" in result.redacted_text

    def test_international_phone(self) -> None:
        """Detects international format phone numbers."""
        config = RedactionConfig(enabled_types={PIIType.PHONE})
        redactor = PIIRedactor(config)
        result = redactor.redact("Call +1-555-123-4567")
        assert "[PHONE]" in result.redacted_text


class TestPIIRedactorSSN:
    """Tests for SSN redaction."""

    def test_standard_ssn(self) -> None:
        """Detects standard SSN format."""
        config = RedactionConfig(enabled_types={PIIType.SSN})
        redactor = PIIRedactor(config)
        result = redactor.redact("SSN: 123-45-6789")
        assert "[SSN]" in result.redacted_text

    def test_ssn_no_dashes(self) -> None:
        """Detects SSN without dashes (lower confidence)."""
        config = RedactionConfig(enabled_types={PIIType.SSN})
        redactor = PIIRedactor(config)
        result = redactor.redact("SSN: 123456789")
        assert "[SSN]" in result.redacted_text
        # Check confidence is lower
        match = result.matches[0]
        assert match.confidence < 1.0


class TestPIIRedactorCreditCard:
    """Tests for credit card redaction."""

    def test_spaced_credit_card(self) -> None:
        """Detects space-separated credit card numbers."""
        config = RedactionConfig(enabled_types={PIIType.CREDIT_CARD})
        redactor = PIIRedactor(config)
        result = redactor.redact("Card: 4111 1111 1111 1111")
        assert "[CREDIT_CARD]" in result.redacted_text

    def test_dashed_credit_card(self) -> None:
        """Detects dash-separated credit card numbers."""
        config = RedactionConfig(enabled_types={PIIType.CREDIT_CARD})
        redactor = PIIRedactor(config)
        result = redactor.redact("Card: 4111-1111-1111-1111")
        assert "[CREDIT_CARD]" in result.redacted_text


class TestPIIRedactorIPAddress:
    """Tests for IP address redaction."""

    def test_ipv4_address(self) -> None:
        """Detects IPv4 addresses."""
        config = RedactionConfig(enabled_types={PIIType.IP_ADDRESS})
        redactor = PIIRedactor(config)
        result = redactor.redact("Server IP: 192.168.1.100")
        assert "[IP_ADDRESS]" in result.redacted_text


class TestPIIRedactorDateOfBirth:
    """Tests for date of birth redaction."""

    def test_us_date_format(self) -> None:
        """Detects US date format."""
        config = RedactionConfig(enabled_types={PIIType.DATE_OF_BIRTH})
        redactor = PIIRedactor(config)
        result = redactor.redact("DOB: 01/15/1990")
        assert "[DATE_OF_BIRTH]" in result.redacted_text

    def test_iso_date_format(self) -> None:
        """Detects ISO date format."""
        config = RedactionConfig(enabled_types={PIIType.DATE_OF_BIRTH})
        redactor = PIIRedactor(config)
        result = redactor.redact("DOB: 1990-01-15")
        assert "[DATE_OF_BIRTH]" in result.redacted_text


class TestPIIRedactorPersonName:
    """Tests for person name redaction."""

    def test_titled_name(self) -> None:
        """Detects names with titles (fallback regex)."""
        config = RedactionConfig(enabled_types={PIIType.PERSON_NAME})
        redactor = PIIRedactor(config)
        # Force regex fallback
        redactor._ner_available = False
        result = redactor.redact("Contact Dr. John Smith for details")
        assert "[PERSON_NAME]" in result.redacted_text

    def test_multiple_titled_names(self) -> None:
        """Detects multiple titled names."""
        config = RedactionConfig(enabled_types={PIIType.PERSON_NAME})
        redactor = PIIRedactor(config)
        redactor._ner_available = False
        result = redactor.redact("Mr. John Doe and Mrs. Jane Smith")
        assert result.redacted_text.count("[PERSON_NAME]") == 2


class TestPIIRedactorOptions:
    """Tests for redactor options."""

    def test_preserve_length(self) -> None:
        """Preserve length option masks with same length."""
        config = RedactionConfig(
            enabled_types={PIIType.EMAIL},
            preserve_length=True,
            mask_char="*",
        )
        redactor = PIIRedactor(config)
        result = redactor.redact("test@example.com")  # 16 chars
        assert len(result.redacted_text) == 16
        assert result.redacted_text == "*" * 16

    def test_hide_type(self) -> None:
        """Show type option can be disabled."""
        config = RedactionConfig(
            enabled_types={PIIType.EMAIL},
            show_type=False,
            mask_char="*",
        )
        redactor = PIIRedactor(config)
        result = redactor.redact("test@example.com")
        assert result.redacted_text == "*" * 8

    def test_whitelist_skips_match(self) -> None:
        """Whitelisted items are not redacted."""
        config = RedactionConfig(enabled_types={PIIType.EMAIL})
        config.add_to_whitelist("safe@company.com")
        redactor = PIIRedactor(config)
        result = redactor.redact("Email: safe@company.com")
        assert "safe@company.com" in result.redacted_text
        assert "safe@company.com" in result.skipped


class TestPIIRedactorCustomPatterns:
    """Tests for custom patterns."""

    def test_add_valid_pattern(self) -> None:
        """Valid custom patterns can be added."""
        redactor = PIIRedactor()
        assert redactor.add_custom_pattern("employee_id", r"EMP-\d{6}")

    def test_reject_invalid_pattern(self) -> None:
        """Invalid regex patterns are rejected."""
        redactor = PIIRedactor()
        assert not redactor.add_custom_pattern("bad", r"[invalid")

    def test_custom_pattern_matching(self) -> None:
        """Custom patterns are used for detection."""
        config = RedactionConfig(
            enabled_types={PIIType.CUSTOM},
            custom_patterns={"employee_id": r"EMP-\d{6}"},
        )
        redactor = PIIRedactor(config)
        result = redactor.redact("Employee: EMP-123456")
        assert "[CUSTOM]" in result.redacted_text

    def test_pattern_limit(self) -> None:
        """Custom patterns respect MAX_PATTERNS_PER_TYPE."""
        redactor = PIIRedactor()
        # Add max patterns
        for i in range(MAX_PATTERNS_PER_TYPE):
            redactor.add_custom_pattern(f"pattern{i}", r"\d+")

        # Adding one more should fail
        assert not redactor.add_custom_pattern("overflow", r"\d+")


class TestPIIRedactorEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self) -> None:
        """Empty text returns empty result."""
        redactor = PIIRedactor()
        result = redactor.redact("")
        assert result.redacted_text == ""
        assert not result.has_redactions

    def test_whitespace_only(self) -> None:
        """Whitespace-only text returns unchanged."""
        redactor = PIIRedactor()
        result = redactor.redact("   \n\t  ")
        assert result.redacted_text == "   \n\t  "

    def test_no_pii(self) -> None:
        """Text without PII returns unchanged."""
        redactor = PIIRedactor()
        result = redactor.redact("Hello, this is normal text.")
        assert result.redacted_text == "Hello, this is normal text."

    def test_overlapping_matches(self) -> None:
        """Overlapping patterns are handled correctly."""
        config = RedactionConfig(
            enabled_types={PIIType.EMAIL, PIIType.PHONE},
        )
        redactor = PIIRedactor(config)
        # Text with multiple PII types
        result = redactor.redact("Email: test@test.com Phone: 555-123-4567")
        assert "[EMAIL]" in result.redacted_text
        assert "[PHONE]" in result.redacted_text


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_redact_pii_function(self) -> None:
        """redact_pii convenience function works."""
        result = redact_pii("Email: test@example.com")
        assert "[EMAIL]" in result.redacted_text

    def test_redact_pii_with_config(self) -> None:
        """redact_pii accepts custom config."""
        config = RedactionConfig(enabled_types={PIIType.PHONE})
        result = redact_pii("Email: test@example.com Phone: 555-123-4567", config)
        assert "test@example.com" in result.redacted_text  # Not redacted
        assert "[PHONE]" in result.redacted_text

    def test_redact_batch(self) -> None:
        """redact_batch processes multiple texts."""
        texts = [
            "Email: a@b.com",
            "Phone: 555-123-4567",
            "Normal text",
        ]
        results = redact_batch(texts)
        assert len(results) == 3
        assert results[0].has_redactions
        assert results[1].has_redactions
        assert not results[2].has_redactions


class TestConstants:
    """Tests for module constants."""

    def test_max_patterns_bound(self) -> None:
        """MAX_PATTERNS_PER_TYPE is bounded."""
        assert MAX_PATTERNS_PER_TYPE == 10

    def test_max_whitelist_bound(self) -> None:
        """MAX_WHITELIST_ENTRIES is bounded."""
        assert MAX_WHITELIST_ENTRIES == 1000

    def test_max_redactions_bound(self) -> None:
        """MAX_REDACTIONS_PER_TEXT is bounded."""
        assert MAX_REDACTIONS_PER_TEXT == 10000
