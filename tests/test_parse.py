import pytest

from avrsi.parse import (Item, canonical_answer, dedupe, eval_answer, fallback_extract, normalise_tokens, parse_path, question_prompt,
                         r_format, render_path, segment_tokens, split_sentences, valid_format)

BASE = "<audio>{a}</audio><visual>{v}</visual><fusion>{f}</fusion><answer>{x}</answer>"


def path(a="A dog.", v="A man.", f="Agree.", x="Yes"):
    return BASE.format(a=a, v=v, f=f, x=x)


def test_splitter_strips_timestamps_then_splits():
    assert split_sentences("[1.5s] A dog barks. [12.3s] Rain falls! Is it? yes") == ["A dog barks.", "Rain falls!", "Is it?", "yes"]


def test_splitter_needs_whitespace_after_punctuation():
    assert split_sentences("Pi is 3.14 today. Ok") == ["Pi is 3.14 today.", "Ok"]


def test_splitter_drops_zero_token_sentences():
    assert split_sentences("... ! A dog.") == ["A dog."]


def test_normalise():
    assert normalise_tokens("Don't, STOP! “Now”…") == ["dont", "stop", "now"]


def test_valid_format_accepts_good(good):
    assert valid_format(good)
    assert valid_format(f"\n {good} \n")
    assert r_format(good) == 1.0


@pytest.mark.parametrize("text", [
    "", "<audio>x.</audio>",
    "junk " + path(),
    path() + " junk",
    path().replace("</audio>", "</audio> leaked "),
    path() + "<answer>No</answer>",
    "<visual>x.</visual><audio>y.</audio><fusion>z.</fusion><answer>a</answer>",
    "<audio>x.</audio><audio>y.</audio><visual>z.</visual><fusion>w.</fusion><answer>a</answer>",
    "<audio><visual>x.</visual></audio><fusion>w.</fusion><answer>a</answer>",
    path(a="   "), path(v=""), path(a="..."), path(v="[3.0s]"),
    path(f="One. Two. Three."),
])
def test_valid_format_rejects(text):
    assert not valid_format(text)
    assert r_format(text) == 0.0


def test_fusion_two_sentences_and_empty_allowed():
    assert valid_format(path(f="One. Two."))
    assert valid_format(path(f=""))


def test_valid_format_runs_before_dedupe():
    # Three near-identical fusion sentences would dedupe to one, but ValidFormat sees the raw three.
    assert not valid_format(path(f="Same words here. Same words here. Same words here."))


def test_dedupe_threshold_exact_boundary():
    # 4 shared of 5 total tokens: Jaccard exactly 0.8 drops; 3 of 5 keeps.
    assert dedupe(["a b c d e.", "a b c d."]) == ["a b c d e."]
    assert dedupe(["a b c d e.", "a b c f."]) == ["a b c d e.", "a b c f."]


def test_dedupe_compares_with_dropped_earlier_sentences():
    a = "one two three four five six seven eight nine ten."
    b = "one two three four five six seven eight nine ELEVEN."  # J(a,b)=9/11=0.82, dropped
    c = "one two three four five six seven eight TWELVE ELEVEN."  # J(a,c)=8/12, J(b,c)=9/11
    assert dedupe([a, b, c]) == [a]


def test_question_rule_only_when_given(yes_item):
    text = path(a="Is a dog in the clip? A dog barks.")
    assert parse_path(text, yes_item).audio == ["A dog barks."]
    xm = Item("x", yes_item.question, "yes", is_xm=True)
    assert parse_path(text, xm).audio == ["Is a dog in the clip?", "A dog barks."]


def test_question_rule_excludes_options():
    item = Item("m", "Is it loud", "yes", options=("rain falls hard today", "no"))
    assert "rain falls hard today." in [s.lower() for s in parse_path(path(a="Rain falls hard today."), item).audio]


def test_parse_malformed_has_no_segments(yes_item):
    p = parse_path("no tags here", yes_item)
    assert not p.valid and p.audio == [] and p.answer is None


def test_fusion_is_deduped_too(yes_item):
    assert parse_path(path(f="They agree here. They agree here."), yes_item).fusion == ["They agree here."]


def test_render_uses_deduped(yes_item):
    p = parse_path(path(a="A dog barks. A dog barks."), yes_item)
    assert render_path(p).count("A dog barks.") == 1


@pytest.mark.parametrize("raw,expected", [("Yes.", "yes"), (" YES ", "yes"), ("no!", "no"), ("Yes, it is", "yes it is"), ("", None)])
def test_canonical_yes_no(raw, expected, yes_item):
    assert canonical_answer(raw, yes_item) == expected


@pytest.mark.parametrize("raw", ["B", "b.", "(B)", "B. video", "video", "Video."])
def test_canonical_mcq_letter_follows_shuffled_presentation(raw, mcq_item):
    assert canonical_answer(raw, mcq_item) == "video"


def test_canonical_mcq_rejects_other(mcq_item):
    assert canonical_answer("E", mcq_item) is None
    assert canonical_answer("the video", mcq_item) is None


def test_question_prompt_presents_options(mcq_item):
    assert question_prompt(mcq_item).splitlines()[2] == "B. video"


@pytest.mark.parametrize("text,expected", [
    ("I think yes, but no doubt it is no", "no"),
    ("<answer>Yes</answer> trailing junk", "yes"),
    ("<audio>x.</audio><answer>maybe</answer> so no", "no"),
    ("nothing useful", None),
])
def test_fallback_yes_no(text, expected, yes_item):
    assert fallback_extract(text, yes_item) == expected


@pytest.mark.parametrize("text,expected", [
    ("The answer is D.", "audio"),
    ("answer: (a)", "neither"),
    ("It must be in the video I think", "video"),
    ("Hmm\nC", "both"),
    ("no idea at all", None),
])
def test_fallback_mcq(text, expected, mcq_item):
    assert fallback_extract(text, mcq_item) == expected


def test_eval_answer_strict_when_well_formed(yes_item):
    assert eval_answer(path(x="Yes"), yes_item) == "yes"
    assert eval_answer(path(x="perhaps"), yes_item) == "perhaps"  # well-formed: never falls back
    assert eval_answer("just yes", yes_item) == "yes"


def test_question_rule_drops_at_exactly_0_8():
    question = "is there a loud dog"
    assert parse_path(path(a="Is there a loud cat. A dog.", v="A man."), Item("q", question, "yes")).audio == ["Is there a loud cat.", "A dog."]  # 4/6
    kept = parse_path(path(a="Is there a loud. A dog.", v="A man."), Item("q", question, "yes")).audio  # 4/5 = 0.8 exactly
    assert kept == ["A dog."]


def test_render_path_is_deduped():
    item = Item("q", "A question?", "yes", is_xm=True)
    text = path(a="A dog barks. A dog barks! A car passes.", v="A man. A man.")
    rendered = render_path(parse_path(text, item))
    assert rendered.count("A dog barks") == 1 and rendered.count("A man") == 1 and "A car passes." in rendered


MCQ = Item("m", "Which?", "cat", options=("dog", "cat", "bird"))


@pytest.mark.parametrize("text,expected", [
    ("The answer is a large animal, I think", None),
    ("the answer: a thing", None),
    ("The answer is (B).", "cat"),
    ("Answer: b", "cat"),
    ("The answer is A, clearly", "dog"),
    ("answer is c\nthat is all", "bird"),
])
def test_fallback_letter_needs_a_delimiter(text, expected):
    assert fallback_extract(text, MCQ) == expected


def test_segment_tokens_are_raw_and_none_when_malformed():
    assert segment_tokens(path(a="A dog. A dog."))["audio"] == ["a", "dog", "a", "dog"]
    assert segment_tokens("junk") is None
