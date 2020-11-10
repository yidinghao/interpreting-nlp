"""
This is the code for making heatmaps from Arras et al. (2017), available
at https://github.com/ArrasL/LRP_for_LSTM/blob/master/code/util/
heatmap.py. The code has been substantially refactored. I added code to
make heatmaps in LaTeX.
"""
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
from yattag import Doc


def _rescale_score_by_abs(score: float, max_score: float,
                          min_score: float) -> float:
    """
    Normalizes an attribution score to the range [0., 1.], where a score
    score of 0. is mapped to 0.5.

    :param score: An attribution score
    :param max_score: The maximum possible attribution score
    :param min_score: The minimum possible attribution score
    :return: The normalized score
    """
    if -1e-5 < min_score and max_score < 1e-5:
        return .5
    elif max_score == min_score and min_score < 0:
        return 0.
    elif max_score == min_score and max_score > 0:
        return 1.

    top = max(abs(max_score), abs(min_score))
    return (score + top) / (2. * top)


def _get_rgb(c_tuple: Tuple[float]) -> str:
    """
    Converts a color from a tuple with values in [0., 1.] to RGB format.

    :param c_tuple: A color
    :return: The color, in RGB format
    """
    return "#%02x%02x%02x" % tuple(int(i * 255.) for i in c_tuple[:3])


def _span_word(tag: Callable, text: Callable, word: str, score: float,
               colormap: Callable):
    """
    Creates an HTML DOM object that contains a word with a background
    color representing its attribution score.

    :param tag: The tag() method from yattag
    :param text: The text() method from yattag
    :param word: A word
    :param score: The word's attribution score
    :param colormap: A matplotlib colormap
    :return: None
    """
    bg = colormap(score)
    style = "color:" + _get_rgb(bg) + ";font-weight:bold;background-color: " \
                                      "#ffffff;padding-top: 15px;" \
                                      "padding-bottom: 15px;"
    with tag("span", style=style):
        text(" " + word + " ")
    text(" ")


def html_heatmap(tokens: List[str], scores: List[float],
                 cmap_name: str = "coolwarm") -> str:
    """
    Constructs a word-level heatmap in HTML format.

    :param tokens: A sequence of tokens
    :param scores: The attribution score assigned to each token
    :param cmap_name: A matplotlib diverging colormap
    :return: The heatmap, as HTML code
    """
    colormap = plt.get_cmap(cmap_name)

    assert len(tokens) == len(scores)
    max_s = max(scores)
    min_s = min(scores)

    doc, tag, text = Doc().tagtext()

    for idx, w in enumerate(tokens):
        score = _rescale_score_by_abs(scores[idx], max_s, min_s)
        _span_word(tag, text, w, score, colormap)

    return doc.getvalue()


def latex_heatmap(tokens: List[str], scores: List[float],
                  cmap_name: str = "coolwarm") -> str:
    """
        Constructs a word-level heatmap in LaTeX format.

        :param tokens: A sequence of words
        :param scores: The attribution score assigned to each token
        :param cmap_name: A matplotlib diverging colormap
        :return: The heatmap, as LaTeX code
        """
    colormap = plt.get_cmap(cmap_name)

    assert len(tokens) == len(scores)
    max_s = max(scores)
    min_s = min(scores)

    code = ""
    code_template = "\\textcolor[rgb]{{{},{},{}}}{{\\textbf{{{}}}}} "
    for idx, w in enumerate(tokens):
        score = _rescale_score_by_abs(scores[idx], max_s, min_s)
        r, g, b, _ = colormap(score)
        code += code_template.format(r, g, b, w)

    return code
