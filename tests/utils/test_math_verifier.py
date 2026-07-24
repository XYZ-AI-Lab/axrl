import pytest

from axrl.verifier.hf_math_verifier import HfMathVerifier

verifier = HfMathVerifier()


def verify_strings(gold: str, pred: str) -> float:
    return verifier.verify(gold, pred)


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        ("-5", "-5", 1),
        ("7425000", "7,425,000", 1),
        ("1000", "1 000", 1),
        ("1000", "1000.0", 1),
        ("1000.0", "1,000.0", 1),
        ("1000.99", "1000,99", 1),
        ("1,22", "1.22", 0),
        ("2.74", "Soucis : 2,74 $ a..", 1),
        ("0.4", ".4", 1),
        ("1000.99", "1,000.99", 1),
        ("1000.99", "$1,000.99", 1),
        ("1000.99", "1,000.99$", 1),
        ("1000.99", "the number is not 10 which is 1,000.99€", 1),
        ("1000.99", "so the number is 10 which is 1,000.99m²", 1),
        ("1000.99", "not it's not 10 it's 1,000.99m²", 1),
        ("1000.99", "not it's not 1,000.99 it's 10m²", 0),
        ("0,111", "0.111", 1),
        ("2", "AZYUK2A", 0),
    ],
)
def test_number_extraction(gold: str, pred: str, expected: int) -> None:
    assert verify_strings(gold, pred) == expected


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        ("10/9", "\\frac{10}{9}", 1),
        ("-10/9", "-\\frac{10}{9}", 1),
    ],
)
def test_simple_fraction_notation(gold: str, pred: str, expected: int) -> None:
    assert verify_strings(gold, pred) == expected


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        ("$[0,1)$", "$[0,1)$", 1),
        ("$[0,9)$", "$[0,1)$", 0),
        ("$(0,9)$", "$[0,9)$", 0),
        ("$1$", "$-[0,1)$", 0),
    ],
)
def test_sets_handling(gold: str, pred: str, expected: int) -> None:
    assert verify_strings(gold, pred) == expected


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        ("$9$", "Answer \\[ 9 \\]", 1),
        ("$9$", "Answer $ 9 $", 1),
        ("$9$", "Answer $$ 9 $$", 1),
        ("$9$", "Answer \\( 9 \\)", 1),
        ("$10$", "Answer \\( (9+1) \\)", 1),
        ("$9$", "Answer \\[ \n 9 \n \\]", 1),
        ("$9$", "Answer $$ \n 9 \n $$", 1),
        ("$10/9$", "Answer $ \\frac{1}{2} \\$ = \\frac{10}{9} $", 1),
        ("$1/3$", "$\\frac13 $", 1),
        ("$1$", "$\\frac3{3} $", 1),
        ("$\\sqrt{3}$", "$\\sqrt3 $", 1),
        ("$1/3$", "$\\cfrac{1}{3} $", 1),
        ("$1/3$", "$\\dfrac{1}{3} $", 1),
        ("$1/3$", "$\\tfrac{1}{3} $", 1),
        ("$1/3$", "$ 1/3 $", 1),
        ("$1/3$", "$\\left( \\frac{1}{3} \\right)$", 1),
        ("$1/3$", "$\\boxed{\\frac{1}{3}}$", 1),
        ("$1/3$", "$\\frac{1}{3} \\text{meters}$", 1),
        ("$1/3$", "$\\frac{1}{3} \\textbf{meters}$", 1),
        ("$1/3$", "$k = \\frac{1}{3}$", 1),
    ],
)
def test_latex_notation(gold: str, pred: str, expected: int) -> None:
    assert verify_strings(gold, pred) == expected


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        ("$28\\%$", "28 percent", 0),
        ("$28\\%$", "28 pct", 0),
        ("$28\\%$", "28 %", 0),
        ("$28\\%$", "$28$ %", 0),
        ("$28\\%$", "$28$ percent", 0),
        ("$28\\%$", "$\\boxed{28}$ pct", 0),
        ("$28\\%$", "$\\boxed{28 pct}", 1),
        ("$28\\%$", "$\\boxed{28 percent}", 1),
        ("$28\\%$", "$\\boxed{28 percentage}", 1),
    ],
)
def test_percent_notation(gold: str, pred: str, expected: int) -> None:
    assert verify_strings(gold, pred) == expected


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        (
            "$2-2p$",
            "Since $x<2$, it follows that $|x-2|=2-x$. If $2-x=p$, then $x=2-p$. Thus $x-p=\\boxed{2-2p}$.",
            1,
        ),
        (
            "\\boxed{\n\\begin{pmatrix} 0 & 3 \\\\ 0 & -1 \\end{pmatrix}\n}.\n\\end{align*}",
            "\\boxed{\n\\begin{pmatrix} 0 & 3 \\\\ 0 & -1 \\end{pmatrix}\n}.\n\\end{align*}",
            1,
        ),
    ],
)
def test_short_subset_of_long_cases(gold: str, pred: str, expected: int) -> None:
    """Subset of very long original cases (kept small to limit test time)."""
    assert verify_strings(gold, pred) == expected


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        ("$x >= 5$", "Therefore $x \\geq 5$ is the solution.", 1),
        ("$x < 3$", "We find that $x \\lt 3$.", 1),
        ("$x \\leq 2$", "Thus $x <= 2$ is our answer.", 1),
        ("$x > 5$", "Therefore $x \\gt 5$ is the solution.", 1),
        ("$x != 3$", "We find that $x \\neq 3$.", 1),
        ("$x > 5$", "Therefore $x < 5$ is the solution.", 0),
        ("$x \\geq 5$", "The solution is $x \\leq 5$", 0),
        ("$x \\neq 5$", "The solution is $x != 5$", 1),
        ("$x \\leq 5$", "$5 \\geq x$", 1),
        ("$x \\geq 5$", "$5 \\leq x$", 1),
        ("$x = 11$", "$x = 5+5+1 = 7 =11$", 1),
        ("$7 = 11a+c$", "$11a+c$", 0),
        ("$x = 1/3$", "$x = 5+5+1 = 1/3 \\approx 11$", 1),
        ("$11$", "$x=11$", 1),
        ("$11$", "$x\\approx11$", 1),
        ("$1/3$", "$x=1/3\\approx1.3$", 1),
        ("$x < 1$", "$-x > -1$", 1),
        ("$x < 1$", "$x > -1$", 0),
        ("$x <= 1$", "$-x >= -1$", 1),
        ("$a +3z = 0$", "$0$", 0),
        ("$1 = \\zzz = x = 0$", "$0$", 1),
        ("$2x + z = 1$", "$1$", 0),
        ("$a^2 + b = 0$", "$0$", 0),
        ("$k=1$", "$1$", 1),
        ("$1$", "$k=1$", 1),
        ("$z = 1 + 1 = 2$", "$z = 3+3 = 2$", 1),
        ("$2x+4y-3=0$", "$y=-\\frac{1}{2}x+\\frac{3}{4}$", 1),
        ("$x^2/4 + y^2/3 = 1$", "$x^2/16 + y^2/12 = 1/4$", 1),
    ],
)
def test_relations_math(gold: str, pred: str, expected: int) -> None:
    assert verify_strings(gold, pred) == expected


def test_precision_rounding() -> None:
    assert verify_strings("$\\frac{1}{3}$", "0.3333$") == 0
    assert verify_strings("$\\frac{1}{3}$", "0.333333$") == 1


# pytest -q axrl/example/utils/test_math_verifier.py
