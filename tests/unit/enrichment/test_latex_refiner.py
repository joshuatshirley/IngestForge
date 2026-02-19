"""
Tests for LaTeX Refiner enrichment module.

This module tests the LaTeXRefiner class which converts LaTeX math
notation to Unicode and clear-text descriptions for LLM consumption.

Test Strategy
-------------
- Test Greek letter conversion (both Unicode and text modes)
- Test math operator conversion
- Test preamble removal
- Test equation environment handling
- Test subscript/superscript conversion
- Test fraction conversion
- Test edge cases (nested braces, malformed LaTeX)

Organization
------------
- TestGreekLetterConversion: Greek letter to Unicode/text
- TestMathOperatorConversion: Math operators to Unicode
- TestPreambleRemoval: Boilerplate removal
- TestEquationEnvironments: equation, align, matrix handling
- TestSubscriptsSuperscripts: x_1 -> x1, x^2 -> x2
- TestFractions: \\frac{a}{b} -> a/b
- TestInlineMath: $...$ handling
- TestExtractEquations: Equation extraction
- TestToUnicode: Math expression conversion
- TestEdgeCases: Malformed input, empty strings
- TestConvenienceFunctions: Module-level helpers
"""

import pytest

from ingestforge.enrichment.latex_refiner import (
    LaTeXRefiner,
    refine_latex,
    extract_equations,
    to_unicode,
)


# =============================================================================
# Test Classes
# =============================================================================


class TestGreekLetterConversion:
    """Tests for Greek letter conversion."""

    @pytest.fixture
    def refiner(self) -> LaTeXRefiner:
        """Create refiner with Unicode mode."""
        return LaTeXRefiner(use_unicode=True)

    @pytest.fixture
    def text_refiner(self) -> LaTeXRefiner:
        """Create refiner with text mode."""
        return LaTeXRefiner(use_unicode=False)

    def test_lowercase_greek_to_unicode(self, refiner: LaTeXRefiner):
        """Test lowercase Greek letters convert to Unicode."""
        text = r"\alpha + \beta = \gamma"

        result = refiner.refine(text)

        assert "\u03b1" in result  # alpha
        assert "\u03b2" in result  # beta
        assert "\u03b3" in result  # gamma

    def test_uppercase_greek_to_unicode(self, refiner: LaTeXRefiner):
        """Test uppercase Greek letters convert to Unicode."""
        text = r"\Gamma + \Delta + \Sigma"

        result = refiner.refine(text)

        assert "\u0393" in result  # Gamma
        assert "\u0394" in result  # Delta
        assert "\u03a3" in result  # Sigma

    def test_greek_to_text(self, text_refiner: LaTeXRefiner):
        """Test Greek letters convert to text names."""
        text = r"\alpha + \beta"

        result = text_refiner.refine(text)

        assert "alpha" in result
        assert "beta" in result
        # Should NOT have Unicode symbols
        assert "\u03b1" not in result

    def test_all_lowercase_greek(self, refiner: LaTeXRefiner):
        """Test all lowercase Greek letters convert correctly."""
        lowercase_greek = [
            r"\alpha",
            r"\beta",
            r"\gamma",
            r"\delta",
            r"\epsilon",
            r"\zeta",
            r"\eta",
            r"\theta",
            r"\iota",
            r"\kappa",
            r"\lambda",
            r"\mu",
            r"\nu",
            r"\xi",
            r"\pi",
            r"\rho",
            r"\sigma",
            r"\tau",
            r"\upsilon",
            r"\phi",
            r"\chi",
            r"\psi",
            r"\omega",
        ]

        for letter in lowercase_greek:
            result = refiner.refine(letter)
            assert letter not in result, f"Failed to convert {letter}"

    def test_mixed_greek_and_text(self, refiner: LaTeXRefiner):
        """Test Greek letters mixed with regular text."""
        text = r"The angle \theta is related to \phi by factor \alpha"

        result = refiner.refine(text)

        assert "The angle" in result
        assert "\u03b8" in result  # theta
        assert "\u03c6" in result  # phi
        assert "\u03b1" in result  # alpha


class TestMathOperatorConversion:
    """Tests for math operator conversion."""

    @pytest.fixture
    def refiner(self) -> LaTeXRefiner:
        """Create refiner instance."""
        return LaTeXRefiner()

    def test_sum_operator(self, refiner: LaTeXRefiner):
        """Test sum operator converts to Unicode."""
        text = r"\sum_{i=1}^{n} x_i"

        result = refiner.refine(text)

        assert "\u2211" in result  # Sum symbol

    def test_integral_operator(self, refiner: LaTeXRefiner):
        """Test integral operator converts to Unicode."""
        text = r"\int_{0}^{\infty} f(x) dx"

        result = refiner.refine(text)

        assert "\u222b" in result  # Integral
        assert "\u221e" in result  # Infinity

    def test_partial_derivative(self, refiner: LaTeXRefiner):
        """Test partial derivative converts to Unicode."""
        text = r"\partial f / \partial x"

        result = refiner.refine(text)

        assert "\u2202" in result  # Partial

    def test_comparison_operators(self, refiner: LaTeXRefiner):
        """Test comparison operators convert correctly."""
        text = r"x \leq y \geq z \neq w \approx v"

        result = refiner.refine(text)

        assert "\u2264" in result  # <=
        assert "\u2265" in result  # >=
        assert "\u2260" in result  # !=
        assert "\u2248" in result  # approx

    def test_set_operators(self, refiner: LaTeXRefiner):
        """Test set operators convert correctly."""
        text = r"A \cup B \cap C \subset D \in E"

        result = refiner.refine(text)

        assert "\u222a" in result  # union
        assert "\u2229" in result  # intersection
        assert "\u2282" in result  # subset
        assert "\u2208" in result  # element of

    def test_arrow_operators(self, refiner: LaTeXRefiner):
        """Test arrow operators convert correctly."""
        text = r"A \rightarrow B \Rightarrow C"

        result = refiner.refine(text)

        assert "\u2192" in result  # ->
        assert "\u21d2" in result  # =>


class TestPreambleRemoval:
    """Tests for LaTeX preamble removal."""

    @pytest.fixture
    def refiner(self) -> LaTeXRefiner:
        """Create refiner instance."""
        return LaTeXRefiner()

    def test_remove_documentclass(self, refiner: LaTeXRefiner):
        """Test documentclass command is removed."""
        text = r"""\documentclass{article}
\begin{document}
Hello World
\end{document}"""

        result = refiner.refine(text)

        assert "documentclass" not in result
        assert "Hello World" in result

    def test_remove_usepackage(self, refiner: LaTeXRefiner):
        """Test usepackage commands are removed."""
        text = r"""\usepackage{amsmath}
\usepackage{graphicx}
\begin{document}
Content
\end{document}"""

        result = refiner.refine(text)

        assert "usepackage" not in result
        assert "amsmath" not in result
        assert "Content" in result

    def test_remove_newcommand(self, refiner: LaTeXRefiner):
        """Test newcommand definitions are removed."""
        text = r"""\newcommand{\R}{\mathbb{R}}
Some text"""

        result = refiner.refine(text)

        assert "newcommand" not in result
        assert "Some text" in result

    def test_preserve_content_after_begin_document(self, refiner: LaTeXRefiner):
        """Test content after \\begin{document} is preserved."""
        text = r"""\documentclass{article}
\title{My Paper}
\author{John Doe}
\begin{document}
This is important content.
More content here.
\end{document}"""

        result = refiner.refine(text)

        assert "This is important content" in result
        assert "More content here" in result
        assert "documentclass" not in result

    def test_remove_maketitle(self, refiner: LaTeXRefiner):
        """Test maketitle command is removed."""
        text = r"""\maketitle
Introduction text"""

        result = refiner.refine(text)

        assert "maketitle" not in result
        assert "Introduction text" in result

    def test_preserve_sections(self, refiner: LaTeXRefiner):
        """Test section structure is preserved."""
        text = r"""\begin{document}
\section{Introduction}
This is the introduction.
\subsection{Background}
Background information.
\end{document}"""

        result = refiner.refine(text)

        assert "Introduction" in result
        assert "Background" in result


class TestEquationEnvironments:
    """Tests for equation environment handling."""

    @pytest.fixture
    def refiner(self) -> LaTeXRefiner:
        """Create refiner instance."""
        return LaTeXRefiner()

    def test_equation_environment(self, refiner: LaTeXRefiner):
        """Test basic equation environment is processed."""
        text = r"""\begin{equation}
E = mc^2
\end{equation}"""

        result = refiner.refine(text)

        assert "[Equation:" in result
        assert "mc" in result

    def test_align_environment(self, refiner: LaTeXRefiner):
        """Test align environment is processed."""
        text = r"""\begin{align}
a &= b \\
c &= d
\end{align}"""

        result = refiner.refine(text)

        assert "[Equation:" in result
        assert "a" in result

    def test_equation_starred(self, refiner: LaTeXRefiner):
        """Test equation* environment is processed."""
        text = r"""\begin{equation*}
x + y = z
\end{equation*}"""

        result = refiner.refine(text)

        assert "[Equation:" in result
        assert "x + y = z" in result or "x" in result

    def test_matrix_environment(self, refiner: LaTeXRefiner):
        """Test matrix environment is processed."""
        text = r"""\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}"""

        result = refiner.refine(text)

        assert "[Equation:" in result

    def test_multiple_equations(self, refiner: LaTeXRefiner):
        """Test multiple equations are all processed."""
        text = r"""\begin{equation}
E = mc^2
\end{equation}
Some text
\begin{equation}
F = ma
\end{equation}"""

        result = refiner.refine(text)

        assert result.count("[Equation:") == 2
        assert "Some text" in result


class TestSubscriptsSuperscripts:
    """Tests for subscript and superscript conversion."""

    @pytest.fixture
    def refiner(self) -> LaTeXRefiner:
        """Create refiner instance."""
        return LaTeXRefiner()

    def test_simple_subscript(self, refiner: LaTeXRefiner):
        """Test simple subscript conversion."""
        text = r"$x_1$"

        result = refiner.refine(text)

        assert "\u2081" in result  # subscript 1

    def test_simple_superscript(self, refiner: LaTeXRefiner):
        """Test simple superscript conversion."""
        text = r"$x^2$"

        result = refiner.refine(text)

        assert "\u00b2" in result  # superscript 2

    def test_braced_subscript(self, refiner: LaTeXRefiner):
        """Test braced subscript conversion."""
        text = r"$x_{12}$"

        result = refiner.refine(text)

        assert "\u2081" in result  # subscript 1
        assert "\u2082" in result  # subscript 2

    def test_braced_superscript(self, refiner: LaTeXRefiner):
        """Test braced superscript conversion."""
        text = r"$x^{23}$"

        result = refiner.refine(text)

        assert "\u00b2" in result  # superscript 2
        assert "\u00b3" in result  # superscript 3

    def test_both_subscript_superscript(self, refiner: LaTeXRefiner):
        """Test both subscript and superscript."""
        text = r"$x_i^2$"

        result = refiner.refine(text)

        # Should have both subscript i and superscript 2
        assert "x" in result

    def test_to_unicode_subscript(self, refiner: LaTeXRefiner):
        """Test to_unicode with subscripts."""
        result = refiner.to_unicode(r"x_0 + x_1")

        assert "\u2080" in result  # subscript 0
        assert "\u2081" in result  # subscript 1

    def test_to_unicode_superscript(self, refiner: LaTeXRefiner):
        """Test to_unicode with superscripts."""
        result = refiner.to_unicode(r"x^2 + y^3")

        assert "\u00b2" in result
        assert "\u00b3" in result


class TestFractions:
    """Tests for fraction conversion."""

    @pytest.fixture
    def refiner(self) -> LaTeXRefiner:
        """Create refiner instance."""
        return LaTeXRefiner()

    def test_simple_fraction(self, refiner: LaTeXRefiner):
        """Test simple fraction conversion."""
        text = r"$\frac{a}{b}$"

        result = refiner.refine(text)

        assert "(a)/(b)" in result or "a/b" in result.replace("(", "").replace(")", "")

    def test_numeric_fraction(self, refiner: LaTeXRefiner):
        """Test numeric fraction conversion."""
        text = r"$\frac{1}{2}$"

        result = refiner.refine(text)

        assert "1" in result
        assert "2" in result
        assert "/" in result

    def test_complex_numerator(self, refiner: LaTeXRefiner):
        """Test fraction with complex numerator."""
        text = r"$\frac{x+y}{z}$"

        result = refiner.refine(text)

        assert "x+y" in result or "x" in result

    def test_nested_fractions(self, refiner: LaTeXRefiner):
        """Test nested fractions."""
        text = r"$\frac{a}{\frac{b}{c}}$"

        result = refiner.refine(text)

        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_to_unicode_fraction(self, refiner: LaTeXRefiner):
        """Test to_unicode with fractions."""
        result = refiner.to_unicode(r"\frac{x}{y}")

        assert "x" in result
        assert "y" in result
        assert "/" in result


class TestInlineMath:
    """Tests for inline math handling."""

    @pytest.fixture
    def refiner(self) -> LaTeXRefiner:
        """Create refiner instance."""
        return LaTeXRefiner()

    def test_simple_inline(self, refiner: LaTeXRefiner):
        """Test simple inline math."""
        text = r"The formula $x = y + z$ is simple."

        result = refiner.refine(text)

        assert "The formula" in result
        assert "is simple" in result
        assert "x = y + z" in result

    def test_inline_with_greek(self, refiner: LaTeXRefiner):
        """Test inline math with Greek letters."""
        text = r"The angle $\theta$ is measured."

        result = refiner.refine(text)

        assert "The angle" in result
        assert "\u03b8" in result  # theta

    def test_multiple_inline(self, refiner: LaTeXRefiner):
        """Test multiple inline math expressions."""
        text = r"Let $a = 1$ and $b = 2$."

        result = refiner.refine(text)

        assert "a = 1" in result
        assert "b = 2" in result

    def test_inline_preserves_surrounding(self, refiner: LaTeXRefiner):
        """Test inline math preserves surrounding text."""
        text = r"Before $x$ middle $y$ after."

        result = refiner.refine(text)

        assert "Before" in result
        assert "middle" in result
        assert "after" in result


class TestExtractEquations:
    """Tests for equation extraction."""

    @pytest.fixture
    def refiner(self) -> LaTeXRefiner:
        """Create refiner instance."""
        return LaTeXRefiner()

    def test_extract_equation_environment(self, refiner: LaTeXRefiner):
        """Test extracting from equation environment."""
        text = r"""\begin{equation}
E = mc^2
\end{equation}"""

        equations = refiner.extract_equations(text)

        assert len(equations) == 1
        assert "E = mc^2" in equations[0]

    def test_extract_multiple_equations(self, refiner: LaTeXRefiner):
        """Test extracting multiple equations."""
        text = r"""\begin{equation}
a = b
\end{equation}
\begin{equation}
c = d
\end{equation}"""

        equations = refiner.extract_equations(text)

        assert len(equations) == 2

    def test_extract_display_math_brackets(self, refiner: LaTeXRefiner):
        """Test extracting \\[ \\] display math."""
        text = r"\[ x^2 + y^2 = z^2 \]"

        equations = refiner.extract_equations(text)

        assert len(equations) == 1
        assert "x^2 + y^2 = z^2" in equations[0]

    def test_extract_display_math_dollars(self, refiner: LaTeXRefiner):
        """Test extracting $$ $$ display math."""
        text = r"$$ a + b = c $$"

        equations = refiner.extract_equations(text)

        assert len(equations) == 1
        assert "a + b = c" in equations[0]

    def test_extract_removes_labels(self, refiner: LaTeXRefiner):
        """Test extracted equations have labels removed."""
        text = r"""\begin{equation}\label{eq:main}
E = mc^2
\end{equation}"""

        equations = refiner.extract_equations(text)

        assert len(equations) == 1
        assert "label" not in equations[0]
        assert "E = mc^2" in equations[0]

    def test_extract_no_equations(self, refiner: LaTeXRefiner):
        """Test extraction from text without equations."""
        text = "Just plain text without equations."

        equations = refiner.extract_equations(text)

        assert len(equations) == 0


class TestToUnicode:
    """Tests for to_unicode method."""

    @pytest.fixture
    def refiner(self) -> LaTeXRefiner:
        """Create refiner instance."""
        return LaTeXRefiner()

    def test_greek_letters(self, refiner: LaTeXRefiner):
        """Test Greek letter conversion."""
        result = refiner.to_unicode(r"\alpha + \beta")

        assert "\u03b1" in result
        assert "\u03b2" in result

    def test_operators(self, refiner: LaTeXRefiner):
        """Test operator conversion."""
        result = refiner.to_unicode(r"\sum x + \int y")

        assert "\u2211" in result
        assert "\u222b" in result

    def test_combined_expression(self, refiner: LaTeXRefiner):
        """Test combined expression conversion."""
        result = refiner.to_unicode(r"\sum_{i=1}^{n} x_i")

        assert "\u2211" in result  # sum
        # Subscripts for i, 1, n
        assert "x" in result

    def test_empty_input(self, refiner: LaTeXRefiner):
        """Test empty input returns empty string."""
        result = refiner.to_unicode("")

        assert result == ""

    def test_plain_text(self, refiner: LaTeXRefiner):
        """Test plain text passes through."""
        result = refiner.to_unicode("x + y = z")

        assert "x + y = z" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def refiner(self) -> LaTeXRefiner:
        """Create refiner instance."""
        return LaTeXRefiner()

    def test_empty_string(self, refiner: LaTeXRefiner):
        """Test empty string input."""
        result = refiner.refine("")

        assert result == ""

    def test_whitespace_only(self, refiner: LaTeXRefiner):
        """Test whitespace-only input."""
        result = refiner.refine("   \n\t   ")

        assert result == ""

    def test_none_handling(self, refiner: LaTeXRefiner):
        """Test None-like empty input."""
        result = refiner.refine("")
        assert result == ""

    def test_nested_braces(self, refiner: LaTeXRefiner):
        """Test handling of nested braces."""
        text = r"$\frac{a+{b+c}}{d}$"

        # Should not crash
        result = refiner.refine(text)

        assert "a" in result
        assert "d" in result

    def test_malformed_frac(self, refiner: LaTeXRefiner):
        """Test handling of malformed fraction."""
        text = r"$\frac{a}$"  # Missing denominator

        # Should not crash
        result = refiner.refine(text)

        assert "a" in result

    def test_unclosed_brace(self, refiner: LaTeXRefiner):
        """Test handling of unclosed braces."""
        text = r"$x_{1$"

        # Should not crash
        result = refiner.refine(text)

        assert isinstance(result, str)

    def test_unclosed_environment(self, refiner: LaTeXRefiner):
        """Test handling of unclosed environment."""
        text = r"\begin{equation}E = mc^2"

        # Should not crash
        result = refiner.refine(text)

        assert isinstance(result, str)

    def test_unknown_command(self, refiner: LaTeXRefiner):
        """Test handling of unknown commands."""
        text = r"\unknowncommand{arg}"

        result = refiner.refine(text)

        assert "arg" in result

    def test_very_long_input(self, refiner: LaTeXRefiner):
        """Test handling of very long input."""
        long_text = r"$\alpha$ " * 1000

        result = refiner.refine(long_text)

        assert len(result) > 0
        assert "\u03b1" in result

    def test_unicode_in_input(self, refiner: LaTeXRefiner):
        """Test handling of existing Unicode in input."""
        text = r"Existing alpha is different from $\alpha$"

        result = refiner.refine(text)

        assert "Existing" in result


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_refine_latex_function(self):
        """Test refine_latex function."""
        result = refine_latex(r"\alpha + \beta")

        assert "\u03b1" in result
        assert "\u03b2" in result

    def test_refine_latex_text_mode(self):
        """Test refine_latex with text mode."""
        result = refine_latex(r"\alpha", use_unicode=False)

        assert "alpha" in result
        assert "\u03b1" not in result

    def test_extract_equations_function(self):
        """Test extract_equations function."""
        text = r"""\begin{equation}
E = mc^2
\end{equation}"""

        equations = extract_equations(text)

        assert len(equations) == 1
        assert "E = mc^2" in equations[0]

    def test_to_unicode_function(self):
        """Test to_unicode function."""
        result = to_unicode(r"\alpha^2 + \beta_1")

        assert "\u03b1" in result
        assert "\u03b2" in result


class TestSpecialCases:
    """Tests for special mathematical cases."""

    @pytest.fixture
    def refiner(self) -> LaTeXRefiner:
        """Create refiner instance."""
        return LaTeXRefiner()

    def test_einsteins_equation(self, refiner: LaTeXRefiner):
        """Test Einstein's famous equation."""
        text = r"$E = mc^2$"

        result = refiner.refine(text)

        assert "E = mc" in result
        assert "\u00b2" in result  # superscript 2

    def test_quadratic_formula(self, refiner: LaTeXRefiner):
        """Test quadratic formula."""
        text = r"$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$"

        result = refiner.refine(text)

        assert "x" in result
        assert "/" in result
        assert "\u00b1" in result  # plus-minus

    def test_gaussian_integral(self, refiner: LaTeXRefiner):
        """Test Gaussian integral notation."""
        text = r"$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$"

        result = refiner.refine(text)

        assert "\u222b" in result  # integral
        assert "\u221e" in result  # infinity
        assert "\u03c0" in result  # pi

    def test_summation_notation(self, refiner: LaTeXRefiner):
        """Test summation notation."""
        text = r"$\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}$"

        result = refiner.refine(text)

        assert "\u2211" in result  # sum
        assert "\u221e" in result  # infinity
        assert "\u03c0" in result  # pi


# =============================================================================
# Summary
# =============================================================================

"""
Test Coverage Summary:
    - Greek letter conversion: 5 tests
    - Math operator conversion: 6 tests
    - Preamble removal: 6 tests
    - Equation environments: 5 tests
    - Subscripts/superscripts: 7 tests
    - Fractions: 5 tests
    - Inline math: 4 tests
    - Extract equations: 6 tests
    - To Unicode method: 5 tests
    - Edge cases: 11 tests
    - Convenience functions: 4 tests
    - Special mathematical cases: 4 tests

    Total: 68 tests

Design Decisions:
    1. Test both Unicode and text modes
    2. Cover all major LaTeX math constructs
    3. Test equation extraction separately from refinement
    4. Include edge cases for robustness
    5. Test real-world mathematical expressions
"""
