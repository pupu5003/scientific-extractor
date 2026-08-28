"""
Unit tests for src/extract_references/claims.py — deterministic citation
detection, style detection, and resolution against parsed references.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extract_references.schemas import ExtractedCitation
from src.extract_references.claims import (
    split_sentences,
    split_paragraphs,
    tag_sentences,
    find_numbered_spans,
    find_author_year_spans,
    detect_citation_style,
    build_number_index,
    build_author_year_index,
    resolve_ref,
    tag_citations,
    ParsedCitationRef,
)


def make_numbered_citation(n, ref_id):
    return ExtractedCitation(
        ref_id=ref_id,
        raw_text=f"[{n}] Some Author. A great paper title. NeurIPS, 2020.",
        authors=["A. Author"],
        year=2020,
    )


def make_ay_citation(ref_id, surname, year, first_given="Jane"):
    return ExtractedCitation(
        ref_id=ref_id,
        raw_text=f"{first_given} {surname}. A great paper title. NeurIPS, {year}.",
        authors=[f"{first_given} {surname}"],
        year=year,
    )


# ---------------------------------------------------------------------------
# Sentence / paragraph segmentation
# ---------------------------------------------------------------------------

def test_sentence_split_handles_et_al_and_initials():
    text = (
        "Smith et al. (2020) showed strong results. "
        "J. Smith and A. Doe later confirmed this. "
        "See Fig. 3 for details."
    )
    sents = [s for s, _, _ in split_sentences(text)]
    assert len(sents) == 3
    assert sents[0].startswith("Smith et al. (2020)")
    assert sents[2].startswith("See Fig. 3")


def test_tag_sentences_global_numbering_continues_across_calls():
    text1 = "First sentence. Second sentence."
    tagged1, id_map1, next_id = tag_sentences(text1, start_id=1)
    assert [sid for sid, _ in id_map1] == ["S1", "S2"]
    assert next_id == 3

    text2 = "Third sentence."
    tagged2, id_map2, next_id2 = tag_sentences(text2, start_id=next_id)
    assert [sid for sid, _ in id_map2] == ["S3"]
    assert "[S3]" in tagged2
    assert next_id2 == 4


def test_split_paragraphs_filters_non_prose():
    body = (
        "# Introduction\n\n"
        "This is a real paragraph with enough characters to survive the min length filter easily.\n\n"
        "| a | b |\n|---|---|\n\n"
        "![figure](img.png)\n\n"
        "Too short.\n\n"
        "Another real paragraph, also long enough to survive the min length filter here."
    )
    paragraphs = split_paragraphs(body)
    assert len(paragraphs) == 2
    assert paragraphs[0].startswith("This is a real paragraph")
    assert paragraphs[1].startswith("Another real paragraph")


# ---------------------------------------------------------------------------
# Numbered markers
# ---------------------------------------------------------------------------

def test_find_numbered_spans():
    text = "Transformers work well [1]. Prior work [2, 3] used RNNs. See [4-6] for a survey."
    spans = find_numbered_spans(text)
    numbers = [[r.number for r in s.refs] for s in spans]
    assert numbers == [[1], [2, 3], [4, 5, 6]]


def test_number_index_survives_reindexing_after_filtering():
    # What was "[3]" in the source text is now the *second* surviving
    # citation (R2) because "[2]" was filtered out upstream.
    citations = [make_numbered_citation(1, "R1"), make_numbered_citation(3, "R2")]
    index = build_number_index(citations)
    assert index == {1: "R1", 3: "R2"}


# ---------------------------------------------------------------------------
# Author-year markers — the variants called out explicitly
# ---------------------------------------------------------------------------

def test_author_year_basic_comma():
    spans = find_author_year_spans("This was shown (Wei et al., 2022).")
    assert len(spans) == 1
    assert spans[0].refs == [ParsedCitationRef(author_key="wei", year=2022, suffix=None)]


def test_author_year_no_comma_before_year():
    spans = find_author_year_spans("This was shown (Wei et al. 2022).")
    assert len(spans) == 1
    assert spans[0].refs[0].author_key == "wei"
    assert spans[0].refs[0].year == 2022


def test_author_year_full_author_list_oxford_comma():
    # Must resolve to the TRUE first author "Wei", not drift to "Wang".
    spans = find_author_year_spans("This was shown (Wei, Wang, and Schuurmans, 2022).")
    assert len(spans) == 1
    assert len(spans[0].refs) == 1
    assert spans[0].refs[0].author_key == "wei"
    assert spans[0].refs[0].year == 2022


def test_author_year_multiple_works_semicolon():
    text = "This was shown before (Wei et al., 2022; Snell et al., 2024)."
    spans = find_author_year_spans(text)
    assert len(spans) == 1
    refs = spans[0].refs
    assert [(r.author_key, r.year) for r in refs] == [("wei", 2022), ("snell", 2024)]


def test_author_year_suffix_multi_year_same_author():
    spans = find_author_year_spans("As shown (Wei et al., 2022a, 2022b).")
    assert len(spans) == 1
    refs = spans[0].refs
    assert [(r.author_key, r.year, r.suffix) for r in refs] == [
        ("wei", 2022, "a"),
        ("wei", 2022, "b"),
    ]


def test_author_year_narrative():
    spans = find_author_year_spans("Wei et al. (2022) demonstrate that this works.")
    assert len(spans) == 1
    assert spans[0].refs[0].author_key == "wei"
    assert spans[0].refs[0].year == 2022


def test_author_year_narrative_with_page_number():
    spans = find_author_year_spans("Wei et al. (2022, p. 14) demonstrate that this works.")
    assert len(spans) == 1
    assert spans[0].refs[0].year == 2022


def test_author_year_eg_prefix_inside_parens():
    text = "recent work (e.g., Wei et al., 2022; Kojima et al., 2022; Wang et al., 2023) suggests this."
    spans = find_author_year_spans(text)
    assert len(spans) == 1
    refs = spans[0].refs
    assert [(r.author_key, r.year) for r in refs] == [
        ("wei", 2022),
        ("kojima", 2022),
        ("wang", 2023),
    ]


# ---------------------------------------------------------------------------
# Style detection
# ---------------------------------------------------------------------------

def test_detect_style_numbered():
    assert detect_citation_style("A [1]. B [2]. C [3]. D [4].") == "numbered"


def test_detect_style_author_year():
    text = "A (Smith, 2020). B (Jones, 2019). C (Lee et al., 2021). D (Park, 2018)."
    assert detect_citation_style(text) == "author_year"


def test_detect_style_unknown():
    assert detect_citation_style("No citations in this text at all.") == "unknown"


# ---------------------------------------------------------------------------
# Resolution, including ambiguity
# ---------------------------------------------------------------------------

def test_resolve_ref_unique_author_year():
    citations = [make_ay_citation("R1", "Wei", 2022)]
    index = build_author_year_index(citations)
    resolved = resolve_ref(ParsedCitationRef(author_key="wei", year=2022), {}, index)
    assert resolved.status == "resolved"
    assert resolved.resolved_ref_id == "R1"


def test_resolve_ref_ambiguous_without_suffix():
    citations = [
        make_ay_citation("R1", "Wei", 2022, first_given="Jason"),
        make_ay_citation("R2", "Wei", 2022, first_given="Tom"),
    ]
    index = build_author_year_index(citations)
    resolved = resolve_ref(ParsedCitationRef(author_key="wei", year=2022), {}, index)
    assert resolved.status == "ambiguous"
    assert set(resolved.candidates) == {"R1", "R2"}


def test_resolve_ref_disambiguated_by_suffix():
    citations = [
        make_ay_citation("R1", "Wei", 2022, first_given="Jason"),
        make_ay_citation("R2", "Wei", 2022, first_given="Tom"),
    ]
    index = build_author_year_index(citations)
    resolved_a = resolve_ref(ParsedCitationRef(author_key="wei", year=2022, suffix="a"), {}, index)
    resolved_b = resolve_ref(ParsedCitationRef(author_key="wei", year=2022, suffix="b"), {}, index)
    assert resolved_a.status == "resolved" and resolved_a.resolved_ref_id == "R1"
    assert resolved_b.status == "resolved" and resolved_b.resolved_ref_id == "R2"


def test_resolve_ref_unresolved_when_no_match():
    resolved = resolve_ref(ParsedCitationRef(author_key="nobody", year=1999), {}, {})
    assert resolved.status == "unresolved"


# ---------------------------------------------------------------------------
# Canonical tagging
# ---------------------------------------------------------------------------

def test_tag_citations_replaces_resolved_markers():
    citations = [make_ay_citation("R1", "Wei", 2022), make_ay_citation("R2", "Snell", 2024)]
    text = (
        "LLMs rely on Chain-of-thoughts (CoT) (Wei et al., 2022) and "
        "Test-Time Compute (TTC) (Snell et al., 2024)."
    )
    tagged, resolved = tag_citations(text, citations)
    assert "<CIT:R1>" in tagged
    assert "<CIT:R2>" in tagged
    assert "(Wei et al., 2022)" not in tagged
    assert all(r.status == "resolved" for r in resolved)


def test_tag_citations_leaves_unresolved_markers_untouched():
    citations = [make_ay_citation("R1", "Wei", 2022)]
    text = "Unrelated work (Nobody, 1999) claims something else."
    tagged, resolved = tag_citations(text, citations)
    assert tagged == text  # untouched
    assert resolved[0].status == "unresolved"


def test_tag_citations_numbered_style_end_to_end():
    citations = [make_numbered_citation(1, "R1"), make_numbered_citation(2, "R2")]
    text = "Idea one [1]. Idea two [2]. Both ideas [1, 2]."
    tagged, resolved = tag_citations(text, citations)
    assert "<CIT:R1>" in tagged
    assert "<CIT:R2>" in tagged
    assert "[1]" not in tagged and "[2]" not in tagged
