import pytest

from practorflow.llm.tools.text_summarizer import TextSummarizerTool


class TestTextSummarizerTool:
    def test_init_and_properties_and_parameters(self):
        tool = TextSummarizerTool()

        assert tool._default_num_sentences == 5
        assert tool._max_input_length == 50000

        assert tool.name == "summarize_text"
        assert isinstance(tool.description, str)
        assert "Summarize" in tool.description

        params = tool.parameters
        assert len(params) == 3

        text_param = next(p for p in params if p.name == "text")
        assert text_param.type == "string"
        assert text_param.required is True

        num_param = next(p for p in params if p.name == "num_sentences")
        assert num_param.type == "integer"
        assert num_param.required is False
        assert num_param.default == 5

        preserve_param = next(p for p in params if p.name == "preserve_order")
        assert preserve_param.type == "boolean"
        assert preserve_param.required is False
        assert preserve_param.default is True

    def test_tokenize_sentences_filters_short_and_empty(self):
        tool = TextSummarizerTool()

        text = "Hi. Short. This is a long enough sentence. Another long enough sentence!"
        sentences = tool._tokenize_sentences(text)

        assert "Hi." not in sentences
        assert "Short." not in sentences
        assert "This is a long enough sentence." in sentences
        assert "Another long enough sentence!" in sentences
        assert all(s.strip() == s for s in sentences)

    def test_tokenize_sentences_no_split_when_no_matching_pattern(self):
        tool = TextSummarizerTool()
        # No whitespace + uppercase boundary after punctuation => won't split by pattern.
        text = "This is a long enough sentence.Another long enough sentence!"
        sentences = tool._tokenize_sentences(text)

        # Because it's a single long chunk, it should be returned as one sentence.
        assert len(sentences) == 1
        assert sentences[0] == text

    def test_tokenize_words_extracts_lowercase_alpha_3plus(self):
        tool = TextSummarizerTool()
        words = tool._tokenize_words("A1b! Cat DOG, e-mail: foo bar baz. Hi to you.")
        # only alphabetic sequences with len >= 3
        assert "cat" in words
        assert "dog" in words
        assert "mail" in words
        assert "foo" in words
        assert "bar" in words
        assert "baz" in words
        assert "hi" not in words  # too short
        assert "to" not in words  # too short

    def test_get_stopwords_contains_common_words(self):
        tool = TextSummarizerTool()
        sw = tool._get_stopwords()
        assert isinstance(sw, set)
        assert "the" in sw
        assert "and" in sw

    def test_calculate_word_frequencies_empty_when_only_stopwords_or_short(self):
        tool = TextSummarizerTool()
        freqs = tool._calculate_word_frequencies("the and or but in on at to for of")
        assert freqs == {}

    def test_calculate_word_frequencies_normalizes_to_max_1(self):
        tool = TextSummarizerTool()
        freqs = tool._calculate_word_frequencies("alpha alpha beta beta beta gamma")

        assert set(freqs.keys()) == {"alpha", "beta", "gamma"}
        # beta is max frequency => normalized to 1.0
        assert freqs["beta"] == 1.0
        assert freqs["alpha"] == pytest.approx(2 / 3)
        assert freqs["gamma"] == pytest.approx(1 / 3)

    def test_score_sentence_no_words_returns_zero(self):
        tool = TextSummarizerTool()
        score = tool._score_sentence("!! ?? ..", {"alpha": 1.0})
        assert score == 0.0

    def test_score_sentence_uses_average_frequency(self):
        tool = TextSummarizerTool()
        freqs = {"alpha": 1.0, "beta": 0.5}
        score = tool._score_sentence("Alpha beta beta.", freqs)
        # words: alpha, beta, beta => (1.0 + 0.5 + 0.5) / 3
        assert score == pytest.approx(2.0 / 3.0)

    def test_execute_missing_text_returns_error(self):
        tool = TextSummarizerTool()
        result = tool.execute()

        assert result.success is False
        assert result.error == "Text parameter is required"

    def test_execute_text_too_long_returns_error(self):
        tool = TextSummarizerTool(max_input_length=10)
        result = tool.execute(text="x" * 11)

        assert result.success is False
        assert "Text exceeds maximum length of 10 characters" in result.error

    def test_execute_when_text_already_shorter_than_requested_returns_original(self):
        tool = TextSummarizerTool(default_num_sentences=5)

        text = "This is sentence one. This is sentence two."
        # two sentences <= num_sentences => return original text
        result = tool.execute(text=text, num_sentences=10)

        assert result.success is True
        assert result.data == text
        assert result.metadata["original_sentences"] == 2
        assert result.metadata["extracted_sentences"] == 2
        assert result.metadata["compression_ratio"] == 1.0
        assert "already shorter" in result.metadata["note"].lower()

    def test_execute_extracts_top_sentences_preserve_order_true(self):
        tool = TextSummarizerTool(default_num_sentences=2)

        # Make sentence 2 highest scoring (repeats 'banana'), sentence 1 next.
        text = (
            "Apple banana carrot is tasty. "
            "Banana banana banana is very tasty indeed. "
            "Carrot alone is not as popular."
        )

        result = tool.execute(text=text, num_sentences=2, preserve_order=True)

        assert result.success is True
        assert isinstance(result.data, str)
        assert "Apple banana carrot is tasty." in result.data
        assert "Banana banana banana is very tasty indeed." in result.data

        # preserve_order=True => sentence 1 should appear before sentence 2 in summary
        assert result.data.index("Apple banana carrot is tasty.") < result.data.index(
            "Banana banana banana is very tasty indeed."
        )

        assert result.metadata["original_sentences"] == 3
        assert result.metadata["extracted_sentences"] == 2
        assert result.metadata["original_length"] == len(text)
        assert result.metadata["summary_length"] == len(result.data)
        assert 0 < result.metadata["compression_ratio"] <= 1

    def test_execute_extracts_top_sentences_preserve_order_false(self):
        tool = TextSummarizerTool(default_num_sentences=2)

        text = (
            "Apple banana carrot is tasty. "
            "Banana banana banana is very tasty indeed. "
            "Carrot alone is not as popular."
        )

        result = tool.execute(text=text, num_sentences=2, preserve_order=False)

        assert result.success is True
        assert "Apple banana carrot is tasty." in result.data
        assert "Banana banana banana is very tasty indeed." in result.data

        # preserve_order=False => highest-scoring sentence should come first (banana-heavy)
        assert result.data.strip().startswith("Banana banana banana is very tasty indeed.")

    def test_execute_exception_path_returns_error(self, monkeypatch):
        tool = TextSummarizerTool()

        def boom(_text):
            raise RuntimeError("boom")

        monkeypatch.setattr(tool, "_tokenize_sentences", boom)

        result = tool.execute(text="This is a long enough sentence. Another long enough sentence.")

        assert result.success is False
        assert result.error.startswith("Summarization failed: ")
        assert "boom" in result.error
