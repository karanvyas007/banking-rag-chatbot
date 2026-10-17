import pytest

from backend.retrieval.qa import ABSTAIN_ANSWER, ModelOutputError

QUESTION = "How much does it cost to use another bank's ATM?"


def test_answer_cites_retrieved_chunk_metadata(make_service):
    service, llm = make_service({"answerable": True, "answer": "It costs $2.50 per withdrawal [1].", "citations": [1]})

    result = service.ask(QUESTION)

    assert result.answered is True
    assert result.answer == "It costs $2.50 per withdrawal [1]."
    assert result.model == "fake-chat-model"
    [citation] = result.citations
    assert (citation.id, citation.source, citation.title, citation.section) == (1, "fees.md", "Fees Schedule", "ATM Fees")
    assert "$2.50" in citation.excerpt
    assert result.retrieved[0].source == "fees.md"


def test_prompt_contains_numbered_passages_and_question(make_service):
    service, llm = make_service({"answerable": True, "answer": "x [1]", "citations": [1]})
    service.ask(QUESTION)

    system, human = llm.calls[0]
    assert "ONLY the numbered policy passages" in system.content
    assert "[1] Fees Schedule - ATM Fees (source: fees.md)" in human.content
    assert QUESTION in human.content


def test_fabricated_citation_numbers_are_dropped(make_service):
    service, _ = make_service(
        {"answerable": True, "answer": "It costs $2.50 [1], see also [7] and [0].", "citations": [1, 7, 0]}
    )

    result = service.ask(QUESTION)

    assert [c.id for c in result.citations] == [1]
    assert result.answer == "It costs $2.50 [1], see also and."


def test_citations_listed_only_in_answer_text_are_kept_if_valid(make_service):
    service, _ = make_service({"answerable": True, "answer": "It costs $2.50 [1].", "citations": []})
    assert [c.id for c in service.ask(QUESTION).citations] == [1]


def test_answer_without_any_valid_citation_abstains(make_service):
    service, _ = make_service({"answerable": True, "answer": "Probably $3 [9].", "citations": [9]})

    result = service.ask(QUESTION)

    assert result.answered is False
    assert result.abstain_reason == "no_valid_citations"
    assert result.answer == ABSTAIN_ANSWER
    assert result.citations == []


def test_model_reports_unanswerable_then_abstains(make_service):
    # Relevant-looking passages were retrieved, but they don't answer the question.
    service, _ = make_service({"answerable": False, "answer": "Not covered.", "citations": []})

    result = service.ask("Is there an ATM fee for premier customers in Antarctica?")

    assert result.answered is False
    assert result.abstain_reason == "not_in_documents"
    assert result.citations == []
    assert result.retrieved  # still reports what was considered


def test_no_relevant_passages_abstains_without_calling_model(make_service):
    service, llm = make_service({"answerable": True, "answer": "x [1]", "citations": [1]})

    result = service.ask("What is the weather in Paris?")

    assert result.answered is False
    assert result.abstain_reason == "no_relevant_documents"
    assert result.retrieved == []
    assert result.model is None
    assert llm.calls == []


@pytest.mark.parametrize("reply", ["not json", '{"answer": "missing fields"}', '{"answerable": "maybe"}'])
def test_malformed_model_output_raises(make_service, reply):
    service, _ = make_service(reply)
    with pytest.raises(ModelOutputError):
        service.ask(QUESTION)
