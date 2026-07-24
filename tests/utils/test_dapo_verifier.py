r"""DapoVerifier test suite derived from most cases in `test_math_verifier.py`.

Purpose
-------
`DapoVerifier` accepts an answer only when it can extract the last LaTeX
``\boxed{...}`` segment from the prediction and the *boxed content* matches the
provided label *exactly* (string equality).

This is intentionally strict:
- No numeric equivalence (e.g. `1000` vs `1000.0`).
- No TeX normalization (e.g. `10/9` vs `\\frac{10}{9}`).
- No stripping of `$...$` wrappers in the label.

Test Design
-----------
1. Each category includes at least one passing case where `gold` exactly equals
   the extracted boxed content.
2. Each category also keeps strictness checks that would pass under a more
   permissive verifier (e.g. `$9$` vs `\\boxed{9}`) but must fail here.

Expectation Annotation
----------------------
Each tuple is (gold, pred, expected). Comments identify divergences from
HfMathVerifier semantics where relevant. If you add new cases:
* Prefer adding ONE passing boxed case and optionally ONE failing unboxed
  control per pattern to avoid bloating runtime.
* Keep comments concise: "Differs from HfMathVerifier" + brief reason.

Assertion Diagnostics
---------------------
The helper ``_assert`` prints a detailed mismatch message showing gold,
pred, expected, and actual values when a test fails to ease debugging.

Extending
---------
When extending tests, ensure that long predictions still keep the final
canonical answer boxed; avoid multiple boxed answers (the last one is used).
"""

from axrl.verifier.dapo_verifier import DapoVerifier

verifier = DapoVerifier()


def verify_strings(gold: str, pred: str) -> int:
    return int(verifier.verify(gold, pred))


def _assert(gold: str, pred: str, expected: int) -> None:
    actual = verify_strings(gold, pred)
    assert actual == expected, f"DapoVerifier mismatch: gold={gold!r} pred={pred!r} expected={expected} actual={actual}"


def test_number_extraction() -> None:
    # Most predictions now include a boxed final numeric answer so they pass; a few remain unboxed to test failure.
    cases = [
        ("-5", "The answer is \\boxed{-5}.", 1),  # Previously failed (no box)
        ("7425000", "Revenue: \\boxed{7,425,000}", 0),  # Formatting diff
        ("1000", "Count = \\boxed{1 000}", 0),  # Space formatting
        ("1000", "We get \\boxed{1000.0}", 0),  # Strict: no numeric equivalence
        ("1000.0", "We get \\boxed{1000.0}", 1),  # Exact string match
        ("1000.0", "Value: \\boxed{1,000.0}", 0),
        ("1000.99", "Result: \\boxed{1000,99}", 0),  # Differs from HfMathVerifier (comma decimal) expected 1 there
        ("1,22", "Parsed: \\boxed{1.22}", 0),
        ("2.74", "Soucis : 2,74 $ a.. but final is \\boxed{2,74}", 0),  # Differs: comma decimal not matched to dot
        ("0.4", "Hence \\boxed{.4}", 0),
        (".4", "Hence \\boxed{.4}", 1),
        ("1000.99", "Answer \\boxed{1,000.99}", 0),
        ("1,000.99", "Answer \\boxed{1,000.99}", 1),
        ("1000.99", "Price $\\boxed{1,000.99}$", 0),
        ("1,000.99", "Price $\\boxed{1,000.99}$", 1),
        ("1000.99", "Total $\\boxed{1,000.99}$ now", 0),
        ("1,000.99", "Total $\\boxed{1,000.99}$ now", 1),
        # keep a couple unboxed to assert failure
        ("1000.99", "the number is not 10 which is 1,000.99€", 0),  # Unboxed should fail
        ("0,111", "0.111", 0),  # Unboxed should fail (precision formatting)
        ("2", "AZYUK2A", 0),  # Unboxed random text
    ]
    for gold, pred, exp in cases:
        _assert(gold, pred, exp)


def test_simple_fraction_notation() -> None:
    cases = [
        ("10/9", "Thus we have \\boxed{\\frac{10}{9}}", 0),  # Strict: no TeX normalization
        ("\\frac{10}{9}", "Thus we have \\boxed{\\frac{10}{9}}", 1),
        ("-10/9", "Final: \\boxed{-\\frac{10}{9}}", 0),
        ("-\\frac{10}{9}", "Final: \\boxed{-\\frac{10}{9}}", 1),
        ("10/9", "\\frac{10}{9}", 0),  # Unboxed control case (fails)
    ]
    for gold, pred, exp in cases:
        _assert(gold, pred, exp)


def test_sets_handling() -> None:
    cases = [
        ("$[0,1)$", "Interval: \\boxed{[0,1)}", 0),  # Strict: label includes $...$
        ("[0,1)", "Interval: \\boxed{[0,1)}", 1),
        ("$[0,9)$", "Mismatch interval: \\boxed{[0,1)}", 0),  # Different
        ("$(0,9)$", "Compare: \\boxed{[0,9)}", 0),  # Different parentheses
        ("$1$", "$-[0,1)$", 0),  # No boxed extraction, should fail
    ]
    for gold, pred, exp in cases:
        _assert(gold, pred, exp)


def test_latex_notation() -> None:
    cases = [
        ("$9$", "Answer \\boxed{9}", 0),
        ("9", "Answer \\boxed{9}", 1),
        ("$9$", "Answer $ 9 $", 0),  # Unboxed control
        ("$10$", "Answer \\boxed{(9+1)}", 0),  # Differs from HfMathVerifier: expression form not accepted by Dapo
        ("$10/9$", "Answer shows fraction \\boxed{\\frac{10}{9}}", 0),
        ("\\frac{10}{9}", "Answer shows fraction \\boxed{\\frac{10}{9}}", 1),
        ("$1/3$", "$\\boxed{\\frac13 }$", 0),
        ("\\frac13 ", "$\\boxed{\\frac13 }$", 1),
        ("$1$", "$\\boxed{\\frac3{3}}$", 0),  # Differs from HfMathVerifier: fraction simplification not applied
        ("$\\sqrt{3}$", "$\\boxed{\\sqrt3 }$", 0),
        ("\\sqrt3 ", "$\\boxed{\\sqrt3 }$", 1),
        ("$1/3$", "$\\boxed{\\cfrac{1}{3}} $", 0),  # Differs: cfrac form not accepted
        ("$1/3$", "$\\boxed{\\dfrac{1}{3}} $", 0),
        ("\\dfrac{1}{3}", "$\\boxed{\\dfrac{1}{3}} $", 1),
        ("$1/3$", "$\\boxed{\\tfrac{1}{3}} $", 0),
        ("\\tfrac{1}{3}", "$\\boxed{\\tfrac{1}{3}} $", 1),
        ("$1/3$", "$\\boxed{1/3} $", 0),
        ("1/3", "$\\boxed{1/3} $", 1),
        ("$1/3$", "$\\boxed{\\frac{1}{3}}$", 0),
        ("\\frac{1}{3}", "$\\boxed{\\frac{1}{3}}$", 1),
        ("$1/3$", "$\\frac{1}{3} \\text{meters}$", 0),  # Unboxed should fail
        ("$1/3$", "$k = \\boxed{\\frac{1}{3}}$", 0),
        ("\\frac{1}{3}", "$k = \\boxed{\\frac{1}{3}}$", 1),
    ]
    for gold, pred, exp in cases:
        _assert(gold, pred, exp)


def test_percent_notation() -> None:
    cases = [
        ("$28\\%$", "We have \\boxed{28} percent", 0),
        ("28", "We have \\boxed{28} percent", 1),
        ("$28\\%$", "We have \\boxed{28} pct", 0),
        ("28", "We have \\boxed{28} pct", 1),
        ("$28\\%$", "\\boxed{28} %", 0),
        ("28", "\\boxed{28} %", 1),
        ("$28\\%$", "$28$ %", 0),  # Unboxed control
        ("$28\\%$", "$\\boxed{28}$ pct", 0),
        ("28", "$\\boxed{28}$ pct", 1),
        ("$28\\%$", "$\\boxed{28 pct}", 0),  # Boxed text variant doesn't normalize to 28
    ]
    for gold, pred, exp in cases:
        _assert(gold, pred, exp)


def test_short_subset_of_long_cases() -> None:
    cases = [
        (
            "$2-2p$",
            "Since $x<2$, it follows that $|x-2|=2-x$. If $2-x=p$, then $x=2-p$. Thus $x-p=\\boxed{2-2p}.",
            0,
        ),
        (
            "2-2p",
            "Since $x<2$, it follows that $|x-2|=2-x$. If $2-x=p$, then $x=2-p$. Thus $x-p=\\boxed{2-2p}.",
            1,
        ),
        (
            "\\boxed{\\begin{pmatrix} 0 & 3 \\ 0 & -1 \\end{pmatrix}}",
            "Matrix: \\boxed{\\begin{pmatrix} 0 & 3 \\ 0 & -1 \\end{pmatrix}}",
            0,
        ),  # Previously failed when parsing multiline box variant
        (
            "\\begin{pmatrix} 0 & 3 \\ 0 & -1 \\end{pmatrix}",
            "Matrix: \\boxed{\\begin{pmatrix} 0 & 3 \\ 0 & -1 \\end{pmatrix}}",
            1,
        ),
    ]
    for gold, pred, exp in cases:
        _assert(gold, pred, exp)


def test_relations_math() -> None:
    cases = [
        ("$x >= 5$", "Therefore $x \\geq 5$ and final answer \\boxed{x \\geq 5}", 0),  # Inequalities not parsed
        ("$x < 3$", "We find that $x \\lt 3$ so \\boxed{x < 3}", 0),
        ("x < 3", "We find that $x \\lt 3$ so \\boxed{x < 3}", 1),
        ("$x \\leq 2$", "Thus $x <= 2$ hence \\boxed{x \\leq 2}", 0),
        ("x \\leq 2", "Thus $x <= 2$ hence \\boxed{x \\leq 2}", 1),
        ("$x > 5$", "Therefore $x \\gt 5$ -> \\boxed{x > 5}", 0),
        ("x > 5", "Therefore $x \\gt 5$ -> \\boxed{x > 5}", 1),
        ("$x != 3$", "We find that $x \\neq 3$ thus \\boxed{x \\neq 3}", 0),  # Inequality
        ("$x > 5$", "Therefore $x < 5$ is the solution.", 0),  # Incorrect relation, no box of correct answer
        ("$x \\geq 5$", "The solution is $x \\leq 5$", 0),
        ("$x \\neq 5$", "The solution is $x != 5$", 0),  # No box so fail
        ("$x \\leq 5$", "$5 \\geq x$ and answer \\boxed{x \\leq 5}", 0),
        ("x \\leq 5", "$5 \\geq x$ and answer \\boxed{x \\leq 5}", 1),
        ("$x \\geq 5$", "$5 \\leq x$ giving \\boxed{x \\geq 5}", 0),
        ("x \\geq 5", "$5 \\leq x$ giving \\boxed{x \\geq 5}", 1),
        ("$x = 11$", "$x = 5+5+1 = 7 =11$ thus \\boxed{11}", 0),  # Differs: variable assignment normalization leaves trailing $
        ("$7 = 11a+c$", "$11a+c$", 0),  # Expression mismatch
        ("$x = 1/3$", "$x = 5+5+1 = 1/3 \\approx 11$ so \\boxed{1/3}", 0),  # Differs: variable assignment with fraction not matched
        ("$11$", "$x=11$ therefore \\boxed{11}", 0),
        ("11", "$x=11$ therefore \\boxed{11}", 1),
        ("$11$", "$x\\approx11$ so \\boxed{11}", 0),
        ("11", "$x\\approx11$ so \\boxed{11}", 1),
        ("$1/3$", "$x=1/3\\approx1.3$ giving \\boxed{1/3}", 0),
        ("1/3", "$x=1/3\\approx1.3$ giving \\boxed{1/3}", 1),
        ("$x < 1$", "$-x > -1$ implies \\boxed{x < 1}", 0),
        ("x < 1", "$-x > -1$ implies \\boxed{x < 1}", 1),
        ("$x < 1$", "$x > -1$", 0),
        ("$x <= 1$", "$-x >= -1$ -> \\boxed{x <= 1}", 0),
        ("x <= 1", "$-x >= -1$ -> \\boxed{x <= 1}", 1),
        ("$a +3z = 0$", "$0$", 0),  # Insufficient boxed info
        ("$1 = \\zzz = x = 0$", "$0$", 0),  # Ambiguous/unboxed
        ("$2x + z = 1$", "$1$", 0),  # Unboxed wrong form
        ("$a^2 + b = 0$", "$0$", 0),
        ("$k=1$", "$1$ and \\boxed{k=1}", 0),
        ("k=1", "$1$ and \\boxed{k=1}", 1),
        ("$1$", "$k=1$ and \\boxed{1}", 0),
        ("1", "$k=1$ and \\boxed{1}", 1),
        ("$z = 1 + 1 = 2$", "$z = 3+3 = 2$ so \\boxed{2}", 0),  # Differs from HfMathVerifier: boxed constant '2' not matched to full assignment chain
        ("$2x+4y-3=0$", "$y=-\\frac{1}{2}x+\\frac{3}{4}$ so \\boxed{2x+4y-3=0}", 0),
        ("2x+4y-3=0", "$y=-\\frac{1}{2}x+\\frac{3}{4}$ so \\boxed{2x+4y-3=0}", 1),
        ("$x^2/4 + y^2/3 = 1$", "$x^2/16 + y^2/12 = 1/4$ implies \\boxed{x^2/4 + y^2/3 = 1}", 0),
        ("x^2/4 + y^2/3 = 1", "$x^2/16 + y^2/12 = 1/4$ implies \\boxed{x^2/4 + y^2/3 = 1}", 1),
    ]
    for gold, pred, exp in cases:
        _assert(gold, pred, exp)


def test_precision_rounding() -> None:
    cases = [
        ("$\\frac{1}{3}$", "Approx 0.3333 -> \\boxed{1/3}", 0),
        ("1/3", "Approx 0.3333 -> \\boxed{1/3}", 1),
        ("$\\frac{1}{3}$", "0.333333$", 0),  # Unboxed numeric form should fail
    ]
    for gold, pred, exp in cases:
        _assert(gold, pred, exp)


# pytest -q axrl/example/utils/test_dapo_verifier.py
