"""
LaTeX Refiner for Research Vertical.

Converts complex LaTeX math notation to Unicode and clear-text descriptions
suitable for LLM consumption. Part of the RES-001 Research Vertical implementation.

Architecture Context
--------------------
LaTeXRefiner is an enricher that runs during Stage 4 (Enrich) of the pipeline:

    Split -> Extract -> Refine -> Chunk -> [Enrich: LaTeXRefiner] -> Index

The refiner improves retrieval by:
1. Converting LaTeX math to Unicode for better text matching
2. Generating clear-text descriptions for equations
3. Removing boilerplate that adds noise to embeddings
4. Preserving document structure for context

Usage Example
-------------
    from ingestforge.enrichment.latex_refiner import LaTeXRefiner

    refiner = LaTeXRefiner()

    # Refine LaTeX text
    cleaned = refiner.refine(r"\\alpha + \\beta = \\gamma")
    # Result: "alpha + beta = gamma"

    # Extract equations
    equations = refiner.extract_equations(r"\\begin{equation}E=mc^2\\end{equation}")
    # Result: ["E=mc^2"]

    # Convert math expression to unicode
    unicode_math = refiner.to_unicode(r"\\sum_{i=1}^{n} x_i")
    # Result: "Sum(i=1 to n) xi"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# Unicode Conversion Maps
# =============================================================================

GREEK_LETTERS: Dict[str, str] = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\epsilon": "epsilon",
    r"\zeta": "zeta",
    r"\eta": "eta",
    r"\theta": "theta",
    r"\iota": "iota",
    r"\kappa": "kappa",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\nu": "nu",
    r"\xi": "xi",
    r"\pi": "pi",
    r"\rho": "rho",
    r"\sigma": "sigma",
    r"\tau": "tau",
    r"\upsilon": "upsilon",
    r"\phi": "phi",
    r"\chi": "chi",
    r"\psi": "psi",
    r"\omega": "omega",
    # Uppercase Greek
    r"\Gamma": "Gamma",
    r"\Delta": "Delta",
    r"\Theta": "Theta",
    r"\Lambda": "Lambda",
    r"\Xi": "Xi",
    r"\Pi": "Pi",
    r"\Sigma": "Sigma",
    r"\Upsilon": "Upsilon",
    r"\Phi": "Phi",
    r"\Psi": "Psi",
    r"\Omega": "Omega",
}

GREEK_UNICODE: Dict[str, str] = {
    r"\alpha": "\u03b1",
    r"\beta": "\u03b2",
    r"\gamma": "\u03b3",
    r"\delta": "\u03b4",
    r"\epsilon": "\u03b5",
    r"\zeta": "\u03b6",
    r"\eta": "\u03b7",
    r"\theta": "\u03b8",
    r"\iota": "\u03b9",
    r"\kappa": "\u03ba",
    r"\lambda": "\u03bb",
    r"\mu": "\u03bc",
    r"\nu": "\u03bd",
    r"\xi": "\u03be",
    r"\pi": "\u03c0",
    r"\rho": "\u03c1",
    r"\sigma": "\u03c3",
    r"\tau": "\u03c4",
    r"\upsilon": "\u03c5",
    r"\phi": "\u03c6",
    r"\chi": "\u03c7",
    r"\psi": "\u03c8",
    r"\omega": "\u03c9",
    # Uppercase Greek
    r"\Gamma": "\u0393",
    r"\Delta": "\u0394",
    r"\Theta": "\u0398",
    r"\Lambda": "\u039b",
    r"\Xi": "\u039e",
    r"\Pi": "\u03a0",
    r"\Sigma": "\u03a3",
    r"\Upsilon": "\u03a5",
    r"\Phi": "\u03a6",
    r"\Psi": "\u03a8",
    r"\Omega": "\u03a9",
}

MATH_OPERATORS: Dict[str, str] = {
    r"\sum": "\u2211",
    r"\prod": "\u220f",
    r"\int": "\u222b",
    r"\partial": "\u2202",
    r"\infty": "\u221e",
    r"\approx": "\u2248",
    r"\neq": "\u2260",
    r"\leq": "\u2264",
    r"\geq": "\u2265",
    r"\pm": "\u00b1",
    r"\times": "\u00d7",
    r"\div": "\u00f7",
    r"\cdot": "\u00b7",
    r"\nabla": "\u2207",
    r"\sqrt": "\u221a",
    r"\forall": "\u2200",
    r"\exists": "\u2203",
    r"\in": "\u2208",
    r"\notin": "\u2209",
    r"\subset": "\u2282",
    r"\supset": "\u2283",
    r"\cup": "\u222a",
    r"\cap": "\u2229",
    r"\emptyset": "\u2205",
    r"\rightarrow": "\u2192",
    r"\leftarrow": "\u2190",
    r"\Rightarrow": "\u21d2",
    r"\Leftarrow": "\u21d0",
    r"\leftrightarrow": "\u2194",
    r"\Leftrightarrow": "\u21d4",
    r"\therefore": "\u2234",
    r"\because": "\u2235",
}

SUBSCRIPT_MAP: Dict[str, str] = {
    "0": "\u2080",
    "1": "\u2081",
    "2": "\u2082",
    "3": "\u2083",
    "4": "\u2084",
    "5": "\u2085",
    "6": "\u2086",
    "7": "\u2087",
    "8": "\u2088",
    "9": "\u2089",
    "+": "\u208a",
    "-": "\u208b",
    "=": "\u208c",
    "(": "\u208d",
    ")": "\u208e",
    "a": "\u2090",
    "e": "\u2091",
    "i": "\u1d62",
    "j": "\u2c7c",
    "n": "\u2099",
    "o": "\u2092",
    "r": "\u1d63",
    "u": "\u1d64",
    "v": "\u1d65",
    "x": "\u2093",
}

SUPERSCRIPT_MAP: Dict[str, str] = {
    "0": "\u2070",
    "1": "\u00b9",
    "2": "\u00b2",
    "3": "\u00b3",
    "4": "\u2074",
    "5": "\u2075",
    "6": "\u2076",
    "7": "\u2077",
    "8": "\u2078",
    "9": "\u2079",
    "+": "\u207a",
    "-": "\u207b",
    "=": "\u207c",
    "(": "\u207d",
    ")": "\u207e",
    "n": "\u207f",
    "i": "\u2071",
}

PREAMBLE_COMMANDS: List[str] = [
    r"\documentclass",
    r"\usepackage",
    r"\newcommand",
    r"\renewcommand",
    r"\DeclareMathOperator",
    r"\theoremstyle",
    r"\newtheorem",
    r"\setlength",
    r"\setcounter",
    r"\pagestyle",
    r"\bibliographystyle",
    r"\graphicspath",
    r"\input",
    r"\include",
    r"\makeatletter",
    r"\makeatother",
    r"\author",
    r"\title",
    r"\date",
    r"\maketitle",
]

EQUATION_ENVIRONMENTS: List[str] = [
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "eqnarray",
    "eqnarray*",
    "displaymath",
    "math",
    "split",
    "cases",
    "array",
    "matrix",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "Vmatrix",
]

# =============================================================================
# LaTeXRefiner Class
# =============================================================================


@dataclass
class LaTeXRefiner:
    """Refines LaTeX text for LLM consumption.

    Converts complex LaTeX math notation to Unicode and clear-text descriptions,
    removes boilerplate, and preserves document structure.

    Attributes:
        use_unicode: If True, use Unicode symbols; if False, use text descriptions
        preserve_structure: If True, keep section/paragraph structure

    Examples:
        >>> refiner = LaTeXRefiner()
        >>> refiner.refine(r"\\alpha + \\beta")
        'alpha + beta'
        >>> refiner.to_unicode(r"x^2")
        'x\u00b2'
    """

    use_unicode: bool = True
    preserve_structure: bool = True

    def refine(self, latex_text: str) -> str:
        """Main conversion method for LaTeX to clean text.

        Processes LaTeX text by:
        1. Removing preamble boilerplate
        2. Converting math environments to clear text
        3. Converting inline math
        4. Preserving document structure

        Args:
            latex_text: Raw LaTeX text to refine

        Returns:
            Refined text suitable for LLM consumption
        """
        if not latex_text or not latex_text.strip():
            return ""

        text = latex_text
        text = self._remove_preamble(text)
        text = self._process_equation_environments(text)
        text = self._process_inline_math(text)
        text = self._convert_greek_letters(text)
        text = self._convert_math_operators(text)
        text = self._clean_remaining_commands(text)
        text = self._normalize_whitespace(text)

        return text.strip()

    def extract_equations(self, text: str) -> List[str]:
        """Extract block-level equations from LaTeX text.

        Finds all equation environments and inline display math,
        returning the raw LaTeX content of each.

        Args:
            text: LaTeX text containing equations

        Returns:
            List of equation strings (raw LaTeX)
        """
        equations: List[str] = []
        equations.extend(self._extract_environment_equations(text))
        equations.extend(self._extract_display_math(text))

        return equations

    def to_unicode(self, math_expr: str) -> str:
        """Convert a math expression to Unicode representation.

        Handles:
        - Greek letters
        - Math operators
        - Subscripts and superscripts
        - Fractions

        Args:
            math_expr: LaTeX math expression

        Returns:
            Unicode representation of the expression
        """
        if not math_expr:
            return ""

        result = math_expr
        # Convert fractions first (before other conversions)
        result = self._convert_fractions(result)
        # Convert LaTeX commands BEFORE subscripts/superscripts
        # This ensures \infty inside ^{\infty} becomes a symbol first
        result = self._convert_greek_letters(result)
        result = self._convert_math_operators(result)
        # Now convert subscripts/superscripts (which work on plain chars)
        result = self._convert_subscripts(result)
        result = self._convert_superscripts(result)
        result = self._clean_braces(result)

        return result.strip()

    def _remove_preamble(self, text: str) -> str:
        """Remove LaTeX preamble boilerplate commands.

        Args:
            text: LaTeX text with potential preamble

        Returns:
            Text with preamble commands removed
        """
        lines = text.split("\n")
        filtered_lines: List[str] = []
        in_document = False

        for line in lines:
            stripped = line.strip()
            if r"\begin{document}" in stripped:
                in_document = True
                continue
            if r"\end{document}" in stripped:
                continue
            if self._is_preamble_line(stripped):
                continue
            if in_document or not self._looks_like_preamble(stripped):
                filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def _is_preamble_line(self, line: str) -> bool:
        """Check if a line is a preamble command.

        Args:
            line: Single line of LaTeX

        Returns:
            True if line is a preamble command
        """
        for cmd in PREAMBLE_COMMANDS:
            if line.startswith(cmd):
                return True
        return False

    def _looks_like_preamble(self, line: str) -> bool:
        """Check if line looks like preamble content.

        Args:
            line: Single line of LaTeX

        Returns:
            True if line appears to be preamble
        """
        if not line:
            return False

        # Never treat \begin or \end as preamble - they're document content
        if line.startswith(r"\begin") or line.startswith(r"\end"):
            return False

        # Comments are preamble
        if line.startswith("%"):
            return True

        # Only specific preamble commands, not arbitrary commands
        # These are exact matches for known preamble-only patterns
        preamble_cmd_prefixes = [
            r"\documentclass",
            r"\usepackage",
            r"\newcommand",
            r"\renewcommand",
            r"\setlength",
            r"\setcounter",
        ]
        for prefix in preamble_cmd_prefixes:
            if line.startswith(prefix):
                return True

        return False

    def _process_equation_environments(self, text: str) -> str:
        """Process and convert equation environments.

        Args:
            text: LaTeX text with equation environments

        Returns:
            Text with equations converted to clear descriptions
        """
        result = text
        for env in EQUATION_ENVIRONMENTS:
            result = self._convert_environment(result, env)
        return result

    def _convert_environment(self, text: str, env_name: str) -> str:
        """Convert a specific equation environment.

        Args:
            text: LaTeX text
            env_name: Environment name (e.g., "equation")

        Returns:
            Text with environment converted
        """
        escaped_name = re.escape(env_name)
        pattern = rf"\\begin\{{{escaped_name}\}}(.*?)\\end\{{{escaped_name}\}}"

        def replace_env(match: re.Match) -> str:
            content = match.group(1).strip()
            converted = self.to_unicode(content)
            return f"[Equation: {converted}]"

        return re.sub(pattern, replace_env, text, flags=re.DOTALL)

    def _extract_environment_equations(self, text: str) -> List[str]:
        """Extract equations from environments.

        Args:
            text: LaTeX text

        Returns:
            List of equation contents
        """
        equations: List[str] = []
        for env in EQUATION_ENVIRONMENTS:
            escaped = re.escape(env)
            pattern = rf"\\begin\{{{escaped}\}}(.*?)\\end\{{{escaped}\}}"
            for match in re.finditer(pattern, text, re.DOTALL):
                content = match.group(1).strip()
                content = re.sub(r"\\label\{[^}]*\}", "", content)
                equations.append(content.strip())
        return equations

    def _extract_display_math(self, text: str) -> List[str]:
        """Extract display math \\[ \\] and $$ $$ equations.

        Args:
            text: LaTeX text

        Returns:
            List of display math contents
        """
        equations: List[str] = []
        bracket_pattern = r"\\\[(.*?)\\\]"
        for match in re.finditer(bracket_pattern, text, re.DOTALL):
            equations.append(match.group(1).strip())

        dollar_pattern = r"\$\$(.*?)\$\$"
        for match in re.finditer(dollar_pattern, text, re.DOTALL):
            equations.append(match.group(1).strip())

        return equations

    def _process_inline_math(self, text: str) -> str:
        """Process inline math $...$ expressions.

        Args:
            text: LaTeX text with inline math

        Returns:
            Text with inline math converted
        """
        pattern = r"\$([^$]+)\$"

        def replace_inline(match: re.Match) -> str:
            content = match.group(1)
            return self.to_unicode(content)

        return re.sub(pattern, replace_inline, text)

    def _convert_greek_letters(self, text: str) -> str:
        """Convert Greek letter commands to Unicode or text.

        Args:
            text: Text with Greek letter commands

        Returns:
            Text with Greek letters converted
        """
        mapping = GREEK_UNICODE if self.use_unicode else GREEK_LETTERS
        result = text
        for cmd, replacement in sorted(mapping.items(), key=lambda x: -len(x[0])):
            result = result.replace(cmd, replacement)
        return result

    def _convert_math_operators(self, text: str) -> str:
        """Convert math operator commands to Unicode.

        Args:
            text: Text with math operator commands

        Returns:
            Text with operators converted
        """
        result = text
        for cmd, replacement in sorted(
            MATH_OPERATORS.items(), key=lambda x: -len(x[0])
        ):
            result = result.replace(cmd, replacement)
        return result

    def _convert_subscripts(self, text: str) -> str:
        """Convert subscript notation x_n to Unicode.

        Args:
            text: Text with subscript notation

        Returns:
            Text with Unicode subscripts
        """
        brace_pattern = r"_\{([^}]+)\}"

        def replace_brace_sub(match: re.Match) -> str:
            content = match.group(1)
            return "".join(SUBSCRIPT_MAP.get(c, c) for c in content)

        text = re.sub(brace_pattern, replace_brace_sub, text)

        single_pattern = r"_([a-zA-Z0-9])"

        def replace_single_sub(match: re.Match) -> str:
            char = match.group(1)
            return SUBSCRIPT_MAP.get(char, f"_{char}")

        return re.sub(single_pattern, replace_single_sub, text)

    def _convert_superscripts(self, text: str) -> str:
        """Convert superscript notation x^n to Unicode.

        Args:
            text: Text with superscript notation

        Returns:
            Text with Unicode superscripts
        """
        brace_pattern = r"\^\{([^}]+)\}"

        def replace_brace_sup(match: re.Match) -> str:
            content = match.group(1)
            return "".join(SUPERSCRIPT_MAP.get(c, c) for c in content)

        text = re.sub(brace_pattern, replace_brace_sup, text)

        single_pattern = r"\^([a-zA-Z0-9])"

        def replace_single_sup(match: re.Match) -> str:
            char = match.group(1)
            return SUPERSCRIPT_MAP.get(char, f"^{char}")

        return re.sub(single_pattern, replace_single_sup, text)

    def _convert_fractions(self, text: str) -> str:
        """Convert \\frac{a}{b} to a/b notation.

        Args:
            text: Text with fraction commands

        Returns:
            Text with fractions converted
        """
        result = text
        max_iterations = 50
        iteration = 0

        while r"\frac" in result and iteration < max_iterations:
            iteration += 1
            match = re.search(r"\\frac\s*", result)
            if not match:
                break

            start_idx = match.end()
            num, num_end = self._extract_braced_content(result, start_idx)
            if num is None:
                break

            den, den_end = self._extract_braced_content(result, num_end)
            if den is None:
                break

            replacement = f"({num})/({den})"
            result = result[: match.start()] + replacement + result[den_end:]

        return result

    def _extract_braced_content(
        self, text: str, start: int
    ) -> Tuple[Optional[str], int]:
        """Extract content within braces, handling nesting.

        Args:
            text: Full text
            start: Starting index (should be at or before opening brace)

        Returns:
            Tuple of (content, end_index) or (None, start) if not found
        """
        idx = start
        while idx < len(text) and text[idx] in " \t\n":
            idx += 1

        if idx >= len(text) or text[idx] != "{":
            return None, start

        depth = 1
        content_start = idx + 1
        idx = content_start

        while idx < len(text) and depth > 0:
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
            idx += 1

        if depth != 0:
            return None, start

        return text[content_start : idx - 1], idx

    def _clean_remaining_commands(self, text: str) -> str:
        """Clean up remaining LaTeX commands.

        Args:
            text: Text with remaining commands

        Returns:
            Text with commands cleaned
        """
        result = text
        result = re.sub(r"\\textbf\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\textit\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\emph\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\text\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\mathit\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\mathcal\{([^}]*)\}", r"\1", result)
        result = re.sub(r"\\left([(\[|])", r"\1", result)
        result = re.sub(r"\\right([)\]|])", r"\1", result)
        result = re.sub(r"\\label\{[^}]*\}", "", result)
        result = re.sub(r"\\ref\{([^}]*)\}", r"[\1]", result)
        result = re.sub(r"\\cite\{([^}]*)\}", r"[\1]", result)
        result = re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", result)
        result = result.replace("\\\\", "\n")
        result = result.replace("\\", "")

        return result

    def _clean_braces(self, text: str) -> str:
        """Remove orphan braces from text.

        Args:
            text: Text with potential orphan braces

        Returns:
            Text with orphan braces removed
        """
        result = text.replace("{", "").replace("}", "")
        return result

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: Text with irregular whitespace

        Returns:
            Text with normalized whitespace
        """
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r"^\s+", "", text, flags=re.MULTILINE)
        return text


# =============================================================================
# Convenience Functions
# =============================================================================


def refine_latex(text: str, use_unicode: bool = True) -> str:
    """Refine LaTeX text for LLM consumption.

    Args:
        text: Raw LaTeX text
        use_unicode: If True, use Unicode symbols

    Returns:
        Refined text
    """
    refiner = LaTeXRefiner(use_unicode=use_unicode)
    return refiner.refine(text)


def extract_equations(text: str) -> List[str]:
    """Extract equations from LaTeX text.

    Args:
        text: LaTeX text

    Returns:
        List of equation strings
    """
    refiner = LaTeXRefiner()
    return refiner.extract_equations(text)


def to_unicode(math_expr: str) -> str:
    """Convert math expression to Unicode.

    Args:
        math_expr: LaTeX math expression

    Returns:
        Unicode representation
    """
    refiner = LaTeXRefiner()
    return refiner.to_unicode(math_expr)
