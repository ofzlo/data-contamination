import pytest

from contam.eval import style


def test_counts_on_fixed_strings():
    s = style.caption_style("The image shows a dog. It is running on the grass!")
    assert s["word_count"] == 11
    assert s["sentence_count"] == 2
    assert s["boilerplate"] == 1
    # function words: the, a, it, is, on, the -> 6 / 11
    assert s["function_word_ratio"] == pytest.approx(6 / 11)


def test_nocaps_style_caption_is_not_boilerplate():
    s = style.caption_style("A brown dog running across a grassy field.")
    assert s["boilerplate"] == 0 and s["sentence_count"] == 1 and s["word_count"] == 8


@pytest.mark.parametrize(
    "text",
    ["In this image, we can see a cat.", "This is an image of a bridge.", "The photo depicts two people.", "this picture shows a car", "A photo of a red bus."],
)
def test_boilerplate_patterns(text):
    assert style.caption_style(text)["boilerplate"] == 1


def test_decimal_and_abbrev_do_not_split_sentences():
    assert style.sentence_count("A sign reading 3.5 miles") == 1
    assert style.sentence_count("") == 0


def test_custom_patterns_and_group_means():
    per = style.style_per_image({1: "foo bar", 2: "baz"}, patterns=[r"\bfoo\b"])
    assert per[1]["boilerplate"] == 1 and per[2]["boilerplate"] == 0
    g = style.group_means(per, [1, 2])
    assert g["word_count"] == 1.5 and g["boilerplate"] == 0.5
