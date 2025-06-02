(Files content cropped to 300k characters, download full ingest to see more)
================================================
FILE: paperqa/__init__.py
================================================
import warnings

from lmi import (
    EmbeddingModel,
    HybridEmbeddingModel,
    LiteLLMEmbeddingModel,
    LiteLLMModel,
    LLMModel,
    LLMResult,
    SentenceTransformerEmbeddingModel,
    SparseEmbeddingModel,
    embedding_model_factory,
)

from paperqa.agents import ask
from paperqa.agents.main import agent_query
from paperqa.docs import Docs, PQASession, print_callback
from paperqa.llms import (
    NumpyVectorStore,
    QdrantVectorStore,
    VectorStore,
)
from paperqa.settings import Settings, get_settings
from paperqa.types import Answer, Context, Doc, DocDetails, Text
from paperqa.version import __version__

# TODO: remove after refactoring all models to avoid using _* private vars
warnings.filterwarnings(
    "ignore", message="Valid config keys have changed in V2:", module="pydantic"
)


__all__ = [
    "Answer",
    "Context",
    "Doc",
    "DocDetails",
    "Docs",
    "EmbeddingModel",
    "HybridEmbeddingModel",
    "LLMModel",
    "LLMResult",
    "LiteLLMEmbeddingModel",
    "LiteLLMModel",
    "NumpyVectorStore",
    "PQASession",
    "QdrantVectorStore",
    "SentenceTransformerEmbeddingModel",
    "Settings",
    "SparseEmbeddingModel",
    "Text",
    "VectorStore",
    "__version__",
    "agent_query",
    "ask",
    "embedding_model_factory",
    "get_settings",
    "print_callback",
]



================================================
FILE: paperqa/_ldp_shims.py
================================================
"""Centralized place for lazy LDP imports."""

__all__ = [
    "HAS_LDP_INSTALLED",
    "Agent",
    "Callback",
    "ComputeTrajectoryMetricsMixin",
    "HTTPAgentClient",
    "Memory",
    "MemoryAgent",
    "ReActAgent",
    "RolloutManager",
    "SimpleAgent",
    "SimpleAgentState",
    "UIndexMemoryModel",
    "_Memories",
    "bulk_evaluate_consensus",
    "discounted_returns",
    "set_training_mode",
]

from pydantic import TypeAdapter

try:
    from ldp.agent import (
        Agent,
        HTTPAgentClient,
        MemoryAgent,
        ReActAgent,
        SimpleAgent,
        SimpleAgentState,
    )
    from ldp.alg import (
        Callback,
        ComputeTrajectoryMetricsMixin,
        RolloutManager,
        bulk_evaluate_consensus,
    )
    from ldp.graph.memory import Memory, UIndexMemoryModel
    from ldp.graph.op_utils import set_training_mode
    from ldp.utils import discounted_returns

    _Memories = TypeAdapter(dict[int, Memory] | list[Memory])  # type: ignore[var-annotated]

    HAS_LDP_INSTALLED = True
except ImportError:
    HAS_LDP_INSTALLED = False

    class ComputeTrajectoryMetricsMixin:  # type: ignore[no-redef]
        """Placeholder parent class for when ldp isn't installed."""

    class Callback:  # type: ignore[no-redef]
        """Placeholder parent class for when ldp isn't installed."""

    Agent = None  # type: ignore[assignment,misc]
    HTTPAgentClient = None  # type: ignore[assignment,misc]
    _Memories = None  # type: ignore[assignment]
    Memory = None  # type: ignore[assignment,misc]
    MemoryAgent = None  # type: ignore[assignment,misc]
    ReActAgent = None  # type: ignore[assignment,misc]
    RolloutManager = None  # type: ignore[assignment,misc]
    SimpleAgent = None  # type: ignore[assignment,misc]
    SimpleAgentState = None  # type: ignore[assignment,misc]
    UIndexMemoryModel = None  # type: ignore[assignment,misc]
    discounted_returns = None  # type: ignore[assignment]
    bulk_evaluate_consensus = None  # type: ignore[assignment]
    set_training_mode = None  # type: ignore[assignment]



================================================
FILE: paperqa/core.py
================================================
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any, cast

from aviary.core import Message
from lmi import LLMModel

from paperqa.types import Context, LLMResult, Text
from paperqa.utils import extract_score, strip_citations


def llm_parse_json(text: str) -> dict:
    """Read LLM output and extract JSON data from it."""
    # Remove leading/trailing whitespaces
    ptext = text.strip()

    # Removing <think> tags for reasoning models
    ptext = re.sub(r"<think>.*?</think>", "", ptext, flags=re.DOTALL).strip()

    # fetches from markdown ```json if present
    ptext = ptext.split("```json")[-1].split("```")[0]

    # Fix specific case with raw fractions in relevance_score
    ptext = re.sub(
        r'"relevance_score"\s*:\s*(\d+)/(\d+)',
        lambda m: f'"relevance_score": {round(int(m.group(1)) / int(m.group(2)) * 10)}',
        ptext,
    )

    # Wrap non-JSON text in a dictionary
    if "{" not in ptext and "}" not in ptext:
        ptext = json.dumps({"summary": ptext})

    # Remove any introductory/closing text and ensure {} to make it a valid JSON
    ptext = ("{" + ptext.split("{", 1)[-1]).rsplit("}", 1)[0] + "}"

    def escape_newlines(match: re.Match) -> str:
        return match.group(0).replace("\n", "\\n")

    # Match anything between double quotes
    # including escaped quotes and other escaped characters.
    # https://regex101.com/r/VFcDmB/1
    pattern = r'"(?:[^"\\]|\\.)*"'
    ptext = re.sub(pattern, escape_newlines, ptext)

    # Ensure that any backslashes in the string that are not part
    # of a valid escape sequence are properly escaped
    # https://regex101.com/r/IzMDlI/1
    ptext = re.sub(r'\\([^"\\/bfnrtu])', r"\\\\\1", ptext)

    def fraction_replacer(match: re.Match) -> str:
        key = match.group(1)  # The key (unchanged)

        # Case 1: If quoted fraction `"5/10"`
        if match.group(2) and match.group(3):
            numerator = int(match.group(2))
            denominator = int(match.group(3))

        # Case 2: If unquoted fraction `5/10`
        elif match.group(4) and match.group(5):
            numerator = int(match.group(4))
            denominator = int(match.group(5))

        else:
            return match.group(0)  # No change if no fraction is found

        fraction_value = round(numerator / denominator * 10)  # Convert to integer
        return f"{key}{fraction_value}"

    # Replace X/Y scores with integer value from 0-10
    # e.g. "relevance_score": "8/10" -> "relevance_score": 8
    # e.g. "relevance_score": 3/5 -> "relevance_score": 6
    pattern = r'("\s*(?:relevance|score)[\w\s\-]*"\s*:\s*)(?:"(\d+)\s*/\s*(\d+)"|(\d+)\s*/\s*(\d+))'
    ptext = re.sub(pattern, fraction_replacer, ptext)

    # Remove extra commas
    ptext = re.sub(r",\s*,+", ",", ptext)  # Remove multiple consecutive commas
    ptext = re.sub(r",\s*}", "}", ptext)  # Remove trailing commas before closing brace
    ptext = re.sub(r"\{\s*,", "{", ptext)  # Remove leading commas inside object

    # Try to parse the JSON normally first
    try:
        data = json.loads(ptext)
    except json.JSONDecodeError as e:
        # If normal parsing fails, try to handle nested quotes case
        if "summary" in ptext and '"relevance_score"' in ptext:
            try:
                # Extract summary and relevance_score directly using regex
                summary_match = re.search(
                    r'"summary"\s*:\s*"(.*?)",\s*"relevance_score"', ptext, re.DOTALL
                )
                score_match = re.search(r'"relevance_score"\s*:\s*"?(\d+)"?', ptext)

                if summary_match and score_match:
                    return {
                        "summary": summary_match.group(1).replace(r"\'", "'"),
                        "relevance_score": int(score_match.group(1)),
                    }
            except Exception:  # noqa: S110
                # Continue to the standard error if regex approach fails
                pass

        raise ValueError(
            f"Failed to parse JSON from text {text!r}. Your model may not be capable of"
            " supporting JSON output or our parsing technique could use some work. Try"
            " a different model or specify `Settings(prompts={'use_json': False})`"
        ) from e

    # Handling incorrect key names for "relevance_score"
    for key in list(data.keys()):
        if re.search(r"relevance|score", key, re.IGNORECASE):
            data["relevance_score"] = data.pop(key)  # Renaming key

    # Handling float, str values for relevance_score
    if "relevance_score" in data:
        try:
            data["relevance_score"] = round(float(data["relevance_score"]))

        except ValueError:
            data["relevance_score"] = (
                0  # Default if relevance_score is empty/not a number
            )

    return data


async def map_fxn_summary(
    text: Text,
    question: str,
    summary_llm_model: LLMModel | None,
    prompt_templates: tuple[str, str] | None,
    extra_prompt_data: dict[str, str] | None = None,
    parser: Callable[[str], dict[str, Any]] | None = None,
    callbacks: Sequence[Callable[[str], None]] | None = None,
) -> tuple[Context, LLMResult]:
    """Parses the given text and returns a context object with the parser and prompt runner.

    The parser should at least return a dict with `summary`. A `relevant_score` will be used and any
    extra fields except `question` will be added to the context object. `question` is stripped
    because it can be incorrectly parsed from LLM outputs when parsing them as JSON.

    Args:
        text: The text to parse.
        question: The question to use for summarization.
        summary_llm_model: The LLM model to use for generating summaries.
        prompt_templates: Optional two-elements tuple containing templates for the user and system prompts.
            prompt_templates = (user_prompt_template, system_prompt_template)
        extra_prompt_data: Optional extra data to pass to the prompt template.
        parser: Optional parser function to parse LLM output into structured data.
            Should return dict with at least 'summary' field.
        callbacks: Optional sequence of callback functions to execute during LLM calls.

    Returns:
        The context object and LLMResult to get info about the LLM execution.
    """
    # needed empties for failures/skips
    llm_result = LLMResult(model="", date="")
    extras: dict[str, Any] = {}
    citation = text.name + ": " + text.doc.formatted_citation
    success = False

    if summary_llm_model and prompt_templates:
        data = {"question": question, "citation": citation, "text": text.text} | (
            extra_prompt_data or {}
        )
        message_prompt, system_prompt = prompt_templates
        messages = [
            Message(role="system", content=system_prompt.format(**data)),
            Message(role="user", content=message_prompt.format(**data)),
        ]
        llm_result = await summary_llm_model.call_single(
            messages=messages,
            callbacks=callbacks,
            name="evidence:" + text.name,
        )
        context = cast("str", llm_result.text)
        result_data = parser(context) if parser else {}
        success = bool(result_data)
        if success:
            try:
                context = result_data.pop("summary")
                score = (
                    result_data.pop("relevance_score")
                    if "relevance_score" in result_data
                    else extract_score(context)
                )
                # just in case question was present
                result_data.pop("question", None)
                extras = result_data
            except KeyError:
                success = False
    else:
        context = text.text
        # If we don't assign scores, just default to 5.
        # why 5? Because we filter out 0s in another place
        # and 5/10 is the other default I could come up with
        score = 5
        success = True
    # remove citations that collide with our grounded citations (for the answer LLM)
    context = strip_citations(context)
    if not success:
        score = extract_score(context)

    return (
        Context(
            context=context,
            question=question,
            text=Text(
                text=text.text,
                name=text.name,
                doc=text.doc.model_dump(exclude={"embedding"}),
            ),
            score=score,  # pylint: disable=possibly-used-before-assignment
            **extras,
        ),
        llm_result,
    )



================================================
FILE: paperqa/docs.py
================================================
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import urllib.request
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, cast
from uuid import UUID, uuid4

from aviary.core import Message
from lmi import (
    Embeddable,
    EmbeddingModel,
    LLMModel,
    LLMResult,
)
from lmi.types import set_llm_session_ids
from lmi.utils import gather_with_concurrency
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

from paperqa.clients import DEFAULT_CLIENTS, DocMetadataClient
from paperqa.core import llm_parse_json, map_fxn_summary
from paperqa.llms import (
    NumpyVectorStore,
    VectorStore,
)
from paperqa.paths import PAPERQA_DIR
from paperqa.prompts import CANNOT_ANSWER_PHRASE
from paperqa.readers import read_doc
from paperqa.settings import MaybeSettings, get_settings
from paperqa.types import Doc, DocDetails, DocKey, PQASession, Text
from paperqa.utils import (
    citation_to_docname,
    get_loop,
    maybe_is_html,
    maybe_is_pdf,
    maybe_is_text,
    md5sum,
    name_in_text,
)

logger = logging.getLogger(__name__)


# this is just to reduce None checks/type checks
async def empty_callback(result: LLMResult) -> None:
    pass


async def print_callback(result: LLMResult) -> None:
    pass


class Docs(BaseModel):
    """A collection of documents to be used for answering questions."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    docs: dict[DocKey, Doc | DocDetails] = Field(default_factory=dict)
    texts: list[Text] = Field(default_factory=list)
    docnames: set[str] = Field(default_factory=set)
    texts_index: VectorStore = Field(default_factory=NumpyVectorStore)
    name: str = Field(default="default", description="Name of this docs collection")
    index_path: Path | None = Field(
        default=PAPERQA_DIR, description="Path to save index", validate_default=True
    )
    deleted_dockeys: set[DocKey] = Field(default_factory=set)

    def __eq__(self, other) -> bool:
        if (
            not isinstance(other, type(self))
            or not isinstance(self.texts_index, NumpyVectorStore)
            or not isinstance(other.texts_index, NumpyVectorStore)
        ):
            return NotImplemented
        return (
            self.docs == other.docs
            and len(self.texts) == len(other.texts)
            and all(  # TODO: implement Text.__eq__
                getattr(self_text, attr) == getattr(other_text, attr)
                for attr in ("text", "name", "doc")
                for self_text, other_text in zip(self.texts, other.texts, strict=True)
            )
            and self.docnames == other.docnames
            and self.texts_index == other.texts_index
            and self.name == other.name
            and self.index_path == other.index_path
            # NOTE: ignoring deleted_dockeys
        )

    @field_validator("index_path")
    @classmethod
    def handle_default(cls, value: Path | None, info: ValidationInfo) -> Path | None:
        if value == PAPERQA_DIR:
            return PAPERQA_DIR / info.data["name"]
        return value

    def clear_docs(self) -> None:
        self.texts = []
        self.docs = {}
        self.docnames = set()
        self.texts_index.clear()

    def _get_unique_name(self, docname: str) -> str:
        """Create a unique name given proposed name."""
        suffix = ""
        while docname + suffix in self.docnames:
            # move suffix to next letter
            suffix = "a" if not suffix else chr(ord(suffix) + 1)
        docname += suffix
        return docname

    def add_file(
        self,
        file: BinaryIO,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> str | None:
        warnings.warn(
            "The synchronous `add_file` method is being deprecated in favor of the"
            " asynchronous `aadd_file` method, this deprecation will conclude in"
            " version 6.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return get_loop().run_until_complete(
            self.aadd_file(
                file,
                citation=citation,
                docname=docname,
                dockey=dockey,
                settings=settings,
                llm_model=llm_model,
                embedding_model=embedding_model,
            )
        )

    async def aadd_file(
        self,
        file: BinaryIO,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        title: str | None = None,
        doi: str | None = None,
        authors: list[str] | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        **kwargs,
    ) -> str | None:
        """Add a document to the collection."""
        # just put in temp file and use existing method
        suffix = ".txt"
        if maybe_is_pdf(file):
            suffix = ".pdf"
        elif maybe_is_html(file):
            suffix = ".html"

        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(file.read())
            f.seek(0)
            return await self.aadd(
                Path(f.name),
                citation=citation,
                docname=docname,
                dockey=dockey,
                title=title,
                doi=doi,
                authors=authors,
                settings=settings,
                llm_model=llm_model,
                embedding_model=embedding_model,
                **kwargs,
            )

    def add_url(
        self,
        url: str,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> str | None:
        warnings.warn(
            "The synchronous `add_url` method is being deprecated in favor of the"
            " asynchronous `aadd_url` method, this deprecation will conclude in"
            " version 6.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return get_loop().run_until_complete(
            self.aadd_url(
                url,
                citation=citation,
                docname=docname,
                dockey=dockey,
                settings=settings,
                llm_model=llm_model,
                embedding_model=embedding_model,
            )
        )

    async def aadd_url(
        self,
        url: str,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> str | None:
        """Add a document to the collection."""
        with urllib.request.urlopen(url) as f:  # noqa: ASYNC210, S310
            # need to wrap to enable seek
            file = BytesIO(f.read())
            return await self.aadd_file(
                file,
                citation=citation,
                docname=docname,
                dockey=dockey,
                settings=settings,
                llm_model=llm_model,
                embedding_model=embedding_model,
            )

    def add(
        self,
        path: str | Path,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        title: str | None = None,
        doi: str | None = None,
        authors: list[str] | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        **kwargs,
    ) -> str | None:
        warnings.warn(
            "The synchronous `add` method is being deprecated in favor of the"
            " asynchronous `aadd` method, this deprecation will conclude in"
            " version 6.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return get_loop().run_until_complete(
            self.aadd(
                path,
                citation=citation,
                docname=docname,
                dockey=dockey,
                title=title,
                doi=doi,
                authors=authors,
                settings=settings,
                llm_model=llm_model,
                embedding_model=embedding_model,
                **kwargs,
            )
        )

    async def aadd(  # noqa: PLR0912
        self,
        path: str | Path,
        citation: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
        title: str | None = None,
        doi: str | None = None,
        authors: list[str] | None = None,
        settings: MaybeSettings = None,
        llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        **kwargs,
    ) -> str | None:
        """Add a document to the collection."""
        all_settings = get_settings(settings)
        parse_config = all_settings.parsing
        if dockey is None:
            # md5 sum of file contents (not path!)
            dockey = md5sum(path)
        if llm_model is None:
            llm_model = all_settings.get_llm()
        if citation is None:
            # Peek first chunk
            texts = await read_doc(
                path,
                Doc(docname="", citation="", dockey=dockey),  # Fake doc
                chunk_chars=parse_config.chunk_size,
                overlap=parse_config.overlap,
                page_size_limit=parse_config.page_size_limit,
            )
            if not texts:
                raise ValueError(f"Could not read document {path}. Is it empty?")
            result = await llm_model.call_single(
                messages=[
                    Message(
                        content=parse_config.citation_prompt.format(text=texts[0].text)
                    ),
                ],
            )
            citation = cast("str", result.text)
            if (
                len(citation) < 3  # noqa: PLR2004
                or "Unknown" in citation
                or "insufficient" in citation
            ):
                citation = f"Unknown, {os.path.basename(path)}, {datetime.now().year}"

        docname = citation_to_docname(citation) if docname is None else docname
        docname = self._get_unique_name(docname)

        doc = Doc(docname=docname, citation=citation, dockey=dockey)

        # try to extract DOI / title from the citation
        if (doi is title is None) and parse_config.use_doc_details:
            # TODO: specify a JSON schema here when many LLM providers support this
            messages = [
                Message(
                    content=parse_config.structured_citation_prompt.format(
                        citation=citation
                    ),
                ),
            ]
            result = await llm_model.call_single(
                messages=messages,
            )
            # This code below tries to isolate the JSON
            # based on observed messages from LLMs
            # it does so by isolating the content between
            # the first { and last } in the response.
            # Since the anticipated structure should  not be nested,
            # we don't have to worry about nested curlies.
            clean_text = cast("str", result.text).split("{", 1)[-1].split("}", 1)[0]
            clean_text = "{" + clean_text + "}"
            try:
                citation_json = json.loads(clean_text)
                if citation_title := citation_json.get("title"):
                    title = citation_title
                if citation_doi := citation_json.get("doi"):
                    doi = citation_doi
                if citation_author := citation_json.get("authors"):
                    authors = citation_author
            except (json.JSONDecodeError, AttributeError):
                # json.JSONDecodeError: clean_text was not actually JSON
                # AttributeError: citation_json was not a dict (e.g. a list)
                logger.warning(
                    "Failed to parse all of title, DOI, and authors from the"
                    " ParsingSettings.structured_citation_prompt's response"
                    f" {clean_text}, consider using a manifest file or specifying a"
                    " different citation prompt."
                )
        # see if we can upgrade to DocDetails
        # if not, we can progress with a normal Doc
        # if "fields_to_overwrite_from_metadata" is used:
        # will map "docname" to "key", and "dockey" to "doc_id"
        if (title or doi) and parse_config.use_doc_details:
            if kwargs.get("metadata_client"):
                metadata_client = kwargs["metadata_client"]
            else:
                metadata_client = DocMetadataClient(
                    session=kwargs.pop("session", None),
                    clients=kwargs.pop("clients", DEFAULT_CLIENTS),
                )

            query_kwargs: dict[str, Any] = {}

            if doi:
                query_kwargs["doi"] = doi
            if authors:
                query_kwargs["authors"] = authors
            if title:
                query_kwargs["title"] = title
            doc = await metadata_client.upgrade_doc_to_doc_details(
                doc, **(query_kwargs | kwargs)
            )

        texts = await read_doc(
            path,
            doc,
            chunk_chars=parse_config.chunk_size,
            overlap=parse_config.overlap,
            page_size_limit=parse_config.page_size_limit,
        )
        # loose check to see if document was loaded
        if (
            not texts
            or len(texts[0].text) < 10  # noqa: PLR2004
            or (
                not parse_config.disable_doc_valid_check
                # Use the first few text chunks to avoid potential issues with
                # title page parsing in the first chunk
                and not maybe_is_text("".join(text.text for text in texts[:5]))
            )
        ):
            raise ValueError(
                f"This does not look like a text document: {path}. Pass disable_check"
                " to ignore this error."
            )
        if await self.aadd_texts(texts, doc, all_settings, embedding_model):
            return docname
        return None

    def add_texts(
        self,
        texts: list[Text],
        doc: Doc,
        settings: MaybeSettings = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> bool:
        warnings.warn(
            "The synchronous `add_texts` method is being deprecated in favor of the"
            " asynchronous `aadd_texts` method, this deprecation will conclude in"
            " version 6.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return get_loop().run_until_complete(
            self.aadd_texts(
                texts, doc, settings=settings, embedding_model=embedding_model
            )
        )

    async def aadd_texts(
        self,
        texts: list[Text],
        doc: Doc,
        settings: MaybeSettings = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> bool:
        """
        Add chunked texts to the collection.

        This is useful to use if you have already chunked the texts yourself.

        Returns:
            True if the doc was added, otherwise False if already in the collection.
        """
        if doc.dockey in self.docs:
            return False
        if not texts:
            raise ValueError("No texts to add.")

        all_settings = get_settings(settings)
        if not all_settings.parsing.defer_embedding and not embedding_model:
            # want to embed now!
            embedding_model = all_settings.get_embedding_model()

        # 0. Short-circuit if it is caught by a filter
        for doc_filter in all_settings.parsing.doc_filters or []:
            if not doc.matches_filter_criteria(doc_filter):
                return False

        # 1. Calculate text embeddings if not already present
        if embedding_model and texts[0].embedding is None:
            for t, t_embedding in zip(
                texts,
                await embedding_model.embed_documents(texts=[t.text for t in texts]),
                strict=True,
            ):
                t.embedding = t_embedding
        # 2. Update texts' and Doc's name
        if doc.docname in self.docnames:
            new_docname = self._get_unique_name(doc.docname)
            for t in texts:
                t.name = t.name.replace(doc.docname, new_docname)
            doc.docname = new_docname
        # 3. Update self
        # NOTE: we defer adding texts to the texts index to retrieval time
        # (e.g. `self.texts_index.add_texts_and_embeddings(texts)`)
        if doc.docname and doc.dockey:
            self.docs[doc.dockey] = doc
            self.texts += texts
            self.docnames.add(doc.docname)
            return True
        return False

    def delete(
        self,
        name: str | None = None,
        docname: str | None = None,
        dockey: DocKey | None = None,
    ) -> None:
        """Delete a document from the collection."""
        # name is an alias for docname
        name = docname if name is None else name

        if name is not None:
            doc = next((doc for doc in self.docs.values() if doc.docname == name), None)
            if doc is None:
                return
            if doc.docname and doc.dockey:
                self.docnames.remove(doc.docname)
                dockey = doc.dockey
        del self.docs[dockey]
        self.deleted_dockeys.add(dockey)
        self.texts = list(filter(lambda x: x.doc.dockey != dockey, self.texts))

    async def _build_texts_index(self, embedding_model: EmbeddingModel) -> None:
        texts = [t for t in self.texts if t not in self.texts_index]
        # For any embeddings we are supposed to lazily embed, embed them now
        to_embed = [t for t in texts if t.embedding is None]
        if to_embed:
            for t, t_embedding in zip(
                to_embed,
                await embedding_model.embed_documents(texts=[t.text for t in to_embed]),
                strict=True,
            ):
                t.embedding = t_embedding
        await self.texts_index.add_texts_and_embeddings(texts)

    async def retrieve_texts(
        self,
        query: str,
        k: int,
        settings: MaybeSettings = None,
        embedding_model: EmbeddingModel | None = None,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> list[Text]:
        """Perform MMR search with the input query on the internal index."""
        settings = get_settings(settings)
        if embedding_model is None:
            embedding_model = settings.get_embedding_model()

        # TODO: should probably happen elsewhere
        self.texts_index.mmr_lambda = settings.texts_index_mmr_lambda

        await self._build_texts_index(embedding_model)
        _k = k + len(self.deleted_dockeys)
        matches: list[Text] = cast(
            "list[Text]",
            (
                await self.texts_index.max_marginal_relevance_search(
                    query,
                    k=_k,
                    fetch_k=2 * _k,
                    embedding_model=embedding_model,
                    partitioning_fn=partitioning_fn,
                )
            )[0],
        )
        matches = [m for m in matches if m.doc.dockey not in self.deleted_dockeys]
        return matches[:k]

    def get_evidence(
        self,
        query: PQASession | str,
        exclude_text_filter: set[str] | None = None,
        settings: MaybeSettings = None,
        callbacks: Sequence[Callable] | None = None,
        embedding_model: EmbeddingModel | None = None,
        summary_llm_model: LLMModel | None = None,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> PQASession:
        warnings.warn(
            "The synchronous `get_evidence` method is being deprecated in favor of the"
            " asynchronous `aget_evidence` method, this deprecation will conclude in"
            " version 6.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return get_loop().run_until_complete(
            self.aget_evidence(
                query=query,
                exclude_text_filter=exclude_text_filter,
                settings=settings,
                callbacks=callbacks,
                embedding_model=embedding_model,
                summary_llm_model=summary_llm_model,
                partitioning_fn=partitioning_fn,
            )
        )

    async def aget_evidence(
        self,
        query: PQASession | str,
        exclude_text_filter: set[str] | None = None,
        settings: MaybeSettings = None,
        callbacks: Sequence[Callable] | None = None,
        embedding_model: EmbeddingModel | None = None,
        summary_llm_model: LLMModel | None = None,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> PQASession:

        evidence_settings = get_settings(settings)
        answer_config = evidence_settings.answer
        prompt_config = evidence_settings.prompts

        session = (
            PQASession(question=query, config_md5=evidence_settings.md5)
            if isinstance(query, str)
            else query
        )

        if not self.docs and len(self.texts_index) == 0:
            return session

        if embedding_model is None:
            embedding_model = evidence_settings.get_embedding_model()

        if summary_llm_model is None:
            summary_llm_model = evidence_settings.get_summary_llm()

        exclude_text_filter = exclude_text_filter or set()
        exclude_text_filter |= {c.text.name for c in session.contexts}

        _k = answer_config.evidence_k
        if exclude_text_filter:
            # Increase k to retrieve so we have enough to down-select after retrieval
            _k += len(exclude_text_filter)

        if answer_config.evidence_retrieval:
            matches = await self.retrieve_texts(
                session.question,
                _k,
                evidence_settings,
                embedding_model,
                partitioning_fn=partitioning_fn,
            )
        else:
            matches = self.texts

        if exclude_text_filter:
            matches = [m for m in matches if m.text not in exclude_text_filter]

        matches = (
            matches[: answer_config.evidence_k]
            if answer_config.evidence_retrieval
            else matches
        )

        prompt_templates = None
        if not answer_config.evidence_skip_summary:
            if prompt_config.use_json:
                prompt_templates = (
                    prompt_config.summary_json,
                    prompt_config.summary_json_system,
                )
            else:
                prompt_templates = (
                    prompt_config.summary,
                    prompt_config.system,
                )

        with set_llm_session_ids(session.id):
            results = await gather_with_concurrency(
                answer_config.max_concurrent_requests,
                [
                    map_fxn_summary(
                        text=m,
                        question=session.question,
                        summary_llm_model=summary_llm_model,
                        prompt_templates=prompt_templates,
                        extra_prompt_data={
                            "summary_length": answer_config.evidence_summary_length,
                            "citation": f"{m.name}: {m.doc.formatted_citation}",
                        },
                        parser=llm_parse_json if prompt_config.use_json else None,
                        callbacks=callbacks,
                    )
                    for m in matches
                ],
            )

        for _, llm_result in results:
            session.add_tokens(llm_result)

        session.contexts += [r for r, _ in results]
        return session

    def query(
        self,
        query: PQASession | str,
        settings: MaybeSettings = None,
        callbacks: Sequence[Callable] | None = None,
        llm_model: LLMModel | None = None,
        summary_llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> PQASession:
        warnings.warn(
            "The synchronous `query` method is being deprecated in favor of the"
            " asynchronous `aquery` method, this deprecation will conclude in"
            " version 6.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return get_loop().run_until_complete(
            self.aquery(
                query,
                settings=settings,
                callbacks=callbacks,
                llm_model=llm_model,
                summary_llm_model=summary_llm_model,
                embedding_model=embedding_model,
                partitioning_fn=partitioning_fn,
            )
        )

    async def aquery(  # noqa: PLR0912
        self,
        query: PQASession | str,
        settings: MaybeSettings = None,
        callbacks: Sequence[Callable] | None = None,
        llm_model: LLMModel | None = None,
        summary_llm_model: LLMModel | None = None,
        embedding_model: EmbeddingModel | None = None,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> PQASession:
        query_settings = get_settings(settings)
        answer_config = query_settings.answer
        prompt_config = query_settings.prompts

        if llm_model is None:
            llm_model = query_settings.get_llm()
        if summary_llm_model is None:
            summary_llm_model = query_settings.get_summary_llm()
        if embedding_model is None:
            embedding_model = query_settings.get_embedding_model()

        session = (
            PQASession(question=query, config_md5=query_settings.md5)
            if isinstance(query, str)
            else query
        )

        contexts = session.contexts
        if answer_config.get_evidence_if_no_contexts and not contexts:
            session = await self.aget_evidence(
                session,
                callbacks=callbacks,
                settings=settings,
                embedding_model=embedding_model,
                summary_llm_model=summary_llm_model,
                partitioning_fn=partitioning_fn,
            )
            contexts = session.contexts
        pre_str = None
        if prompt_config.pre is not None:
            with set_llm_session_ids(session.id):
                messages = [
                    Message(role="system", content=prompt_config.system),
                    Message(
                        role="user",
                        content=prompt_config.pre.format(question=session.question),
                    ),
                ]
                pre = await llm_model.call_single(
                    messages=messages,
                    callbacks=callbacks,
                    name="pre",
                )
            session.add_tokens(pre)
            pre_str = pre.text

        # sort by first score, then name
        filtered_contexts = sorted(
            contexts,
            key=lambda x: (-x.score, x.text.name),
        )[: answer_config.answer_max_sources]
        # remove any contexts with a score of 0
        filtered_contexts = [c for c in filtered_contexts if c.score > 0]

        # shim deprecated flag
        # TODO: remove in v6
        context_inner_prompt = prompt_config.context_inner
        if (
            not answer_config.evidence_detailed_citations
            and "\nFrom {citation}" in context_inner_prompt
        ):
            # Only keep "\nFrom {citation}" if we are showing detailed citations
            context_inner_prompt = context_inner_prompt.replace("\nFrom {citation}", "")

        inner_context_strs = [
            context_inner_prompt.format(
                name=c.text.name,
                text=c.context,
                citation=c.text.doc.formatted_citation,
                **(c.model_extra or {}),
            )
            for c in filtered_contexts
        ]
        if pre_str:
            inner_context_strs += (
                [f"Extra background information: {pre_str}"] if pre_str else []
            )

        context_str = prompt_config.context_outer.format(
            context_str="\n\n".join(inner_context_strs),
            valid_keys=", ".join([c.text.name for c in filtered_contexts]),
        )

        bib = {}
        if len(context_str) < 10:  # noqa: PLR2004
            answer_text = (
                f"{CANNOT_ANSWER_PHRASE} this question due to insufficient information."
            )
            answer_reasoning = None
        else:
            with set_llm_session_ids(session.id):
                prior_answer_prompt = ""
                if prompt_config.answer_iteration_prompt and session.answer:
                    prior_answer_prompt = prompt_config.answer_iteration_prompt.format(
                        prior_answer=session.answer
                    )
                messages = [
                    Message(role="system", content=prompt_config.system),
                    Message(
                        role="user",
                        content=prompt_config.qa.format(
                            context=context_str,
                            answer_length=answer_config.answer_length,
                            question=session.question,
                            example_citation=prompt_config.EXAMPLE_CITATION,
                            prior_answer_prompt=prior_answer_prompt,
                        ),
                    ),
                ]
                answer_result = await llm_model.call_single(
                    messages=messages,
                    callbacks=callbacks,
                    name="answer",
                )
            answer_text = cast("str", answer_result.text)
            answer_reasoning = answer_result.reasoning_content
            session.add_tokens(answer_result)
        # it still happens
        if (ex_citation := prompt_config.EXAMPLE_CITATION) in answer_text:
            answer_text = answer_text.replace(ex_citation, "")
        for c in filtered_contexts:
            name = c.text.name
            citation = c.text.doc.formatted_citation
            # do check for whole key (so we don't catch Callahan2019a with Callahan2019)
            if name_in_text(name, answer_text):
                bib[name] = citation
        bib_str = "\n\n".join(
            [f"{i + 1}. ({k}): {c}" for i, (k, c) in enumerate(bib.items())]
        )

        if answer_config.answer_filter_extra_background:
            answer_text = re.sub(
                r"\([Ee]xtra [Bb]ackground [Ii]nformation\)",  # spellchecker: disable-line
                "",
                answer_text,
            )

        formatted_answer = f"Question: {session.question}\n\n{answer_text}\n"
        if bib:
            formatted_answer += f"\nReferences\n\n{bib_str}\n"

        if prompt_config.post is not None:
            with set_llm_session_ids(session.id):
                messages = [
                    Message(role="system", content=prompt_config.system),
                    Message(
                        role="user",
                        content=prompt_config.post.format(question=session.question),
                    ),
                ]
                post = await llm_model.call_single(
                    messages=messages,
                    callbacks=callbacks,
                    name="post",
                )
            answer_text = cast("str", post.text)
            answer_reasoning = post.reasoning_content
            session.add_tokens(post)
            formatted_answer = f"Question: {session.question}\n\n{post}\n"
            if bib:
                formatted_answer += f"\nReferences\n\n{bib_str}\n"

        # now at end we modify, so we could have retried earlier
        session.answer = answer_text
        session.answer_reasoning = answer_reasoning
        session.formatted_answer = formatted_answer
        session.references = bib_str
        session.contexts = contexts
        session.context = context_str

        return session



================================================
FILE: paperqa/llms.py
================================================
import asyncio
import itertools
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Iterable,
    Sequence,
    Sized,
)
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from lmi import (
    Embeddable,
    EmbeddingModel,
    EmbeddingModes,
    HybridEmbeddingModel,
    LiteLLMEmbeddingModel,
    SentenceTransformerEmbeddingModel,
    SparseEmbeddingModel,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import override

from paperqa.types import Doc, Text

if TYPE_CHECKING:
    from qdrant_client.http.models import Record

    from paperqa.docs import Docs

try:
    from qdrant_client import AsyncQdrantClient, models

    qdrant_installed = True
except ImportError:
    qdrant_installed = False

logger = logging.getLogger(__name__)


def cosine_similarity(a, b):
    norm_product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return a @ b.T / norm_product


class VectorStore(BaseModel, ABC):
    """Interface for vector store - very similar to LangChain's VectorStore to be compatible."""

    model_config = ConfigDict(extra="forbid")

    # can be tuned for different tasks
    mmr_lambda: float = Field(
        default=1.0,
        ge=0.0,
        description="MMR lambda value, a value above 1 disables MMR search.",
    )
    texts_hashes: set[int] = Field(default_factory=set)

    def __contains__(self, item) -> bool:
        return hash(item) in self.texts_hashes

    def __len__(self) -> int:
        return len(self.texts_hashes)

    @abstractmethod
    async def add_texts_and_embeddings(self, texts: Iterable[Embeddable]) -> None:
        """Add texts and their embeddings to the store."""
        self.texts_hashes.update(hash(t) for t in texts)

    @abstractmethod
    async def similarity_search(
        self, query: str, k: int, embedding_model: EmbeddingModel
    ) -> tuple[Sequence[Embeddable], list[float]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        self.texts_hashes = set()

    async def partitioned_similarity_search(
        self,
        query: str,
        k: int,
        embedding_model: EmbeddingModel,
        partitioning_fn: Callable[[Embeddable], int],
    ) -> tuple[Sequence[Embeddable], list[float]]:
        """Partition the documents into different groups and perform similarity search.

        Args:
            query: query string
            k: Number of results to return
            embedding_model: model used to embed the query
            partitioning_fn: function to partition the documents into different groups.

        Returns:
            Tuple of lists of Embeddables and scores of length k.
        """
        raise NotImplementedError(
            "partitioned_similarity_search is not implemented for this VectorStore."
        )

    async def max_marginal_relevance_search(
        self,
        query: str,
        k: int,
        fetch_k: int,
        embedding_model: EmbeddingModel,
        partitioning_fn: Callable[[Embeddable], int] | None = None,
    ) -> tuple[Sequence[Embeddable], list[float]]:
        """Vectorized implementation of Maximal Marginal Relevance (MMR) search.

        Args:
            query: Query vector.
            k: Number of results to return.
            fetch_k: Number of results to fetch from the vector store.
            embedding_model: model used to embed the query
            partitioning_fn: optional function to partition the documents into
                different groups, performing MMR within each group.

        Returns:
            List of tuples (doc, score) of length k.
        """
        if fetch_k < k:
            raise ValueError("fetch_k must be greater or equal to k")

        if partitioning_fn is None:
            texts, scores = await self.similarity_search(
                query, fetch_k, embedding_model
            )
        else:
            texts, scores = await self.partitioned_similarity_search(
                query, fetch_k, embedding_model, partitioning_fn
            )

        if len(texts) <= k or self.mmr_lambda >= 1.0:
            return texts, scores

        embeddings = np.array([t.embedding for t in texts])
        np_scores = np.array(scores)
        similarity_matrix = cosine_similarity(embeddings, embeddings)

        selected_indices = [0]
        remaining_indices = list(range(1, len(texts)))

        while len(selected_indices) < k:
            selected_similarities = similarity_matrix[:, selected_indices]
            max_sim_to_selected = selected_similarities.max(axis=1)

            mmr_scores = (
                self.mmr_lambda * np_scores
                - (1 - self.mmr_lambda) * max_sim_to_selected
            )
            mmr_scores[selected_indices] = -np.inf  # Exclude already selected documents

            max_mmr_index = mmr_scores.argmax()
            selected_indices.append(max_mmr_index)
            remaining_indices.remove(max_mmr_index)

        return [texts[i] for i in selected_indices], [
            scores[i] for i in selected_indices
        ]


class NumpyVectorStore(VectorStore):
    texts: list[Embeddable] = Field(default_factory=list)
    _embeddings_matrix: np.ndarray | None = None
    _texts_filter: np.ndarray | None = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.texts == other.texts
            and self.texts_hashes == other.texts_hashes
            and self.mmr_lambda == other.mmr_lambda
            and (
                other._embeddings_matrix is None
                if self._embeddings_matrix is None
                else (
                    False
                    if other._embeddings_matrix is None
                    else np.allclose(self._embeddings_matrix, other._embeddings_matrix)
                )
            )
        )

    def clear(self) -> None:
        super().clear()
        self.texts = []
        self._embeddings_matrix = None
        self._texts_filter = None

    async def add_texts_and_embeddings(self, texts: Iterable[Embeddable]) -> None:
        await super().add_texts_and_embeddings(texts)
        self.texts.extend(texts)
        self._embeddings_matrix = np.array([t.embedding for t in self.texts])

    async def partitioned_similarity_search(
        self,
        query: str,
        k: int,
        embedding_model: EmbeddingModel,
        partitioning_fn: Callable[[Embeddable], int],
    ) -> tuple[Sequence[Embeddable], list[float]]:
        scores: list[list[float]] = []
        texts: list[Sequence[Embeddable]] = []

        text_partitions = np.array([partitioning_fn(t) for t in self.texts])
        # CPU bound so replacing w a gather wouldn't get us anything
        # plus we need to reset self._texts_filter each iteration
        for partition in np.unique(text_partitions):
            self._texts_filter = text_partitions == partition
            _texts, _scores = await self.similarity_search(query, k, embedding_model)
            texts.append(_texts)
            scores.append(_scores)
        # reset the filter after running
        self._texts_filter = None

        return (
            [
                t
                for t in itertools.chain.from_iterable(itertools.zip_longest(*texts))
                if t is not None
            ][:k],
            [
                s
                for s in itertools.chain.from_iterable(itertools.zip_longest(*scores))
                if s is not None
            ][:k],
        )

    async def similarity_search(
        self, query: str, k: int, embedding_model: EmbeddingModel
    ) -> tuple[Sequence[Embeddable], list[float]]:
        k = min(k, len(self.texts))
        if k == 0:
            return [], []

        # this will only affect models that embedding prompts
        embedding_model.set_mode(EmbeddingModes.QUERY)

        np_query = np.array((await embedding_model.embed_documents([query]))[0])

        embedding_model.set_mode(EmbeddingModes.DOCUMENT)

        embedding_matrix = self._embeddings_matrix

        if self._texts_filter is not None:
            original_indices = np.where(self._texts_filter)[0]
            embedding_matrix = embedding_matrix[self._texts_filter]  # type: ignore[index]
        else:
            original_indices = np.arange(len(self.texts))

        similarity_scores = cosine_similarity(
            np_query.reshape(1, -1), embedding_matrix
        )[0]
        similarity_scores = np.nan_to_num(similarity_scores, nan=-np.inf)
        # minus so descending
        # we could use arg-partition here
        # but a lot of algorithms expect a sorted list
        sorted_indices = np.argsort(-similarity_scores)
        return (
            [self.texts[i] for i in original_indices[sorted_indices][:k]],
            [similarity_scores[i] for i in sorted_indices[:k]],
        )


class QdrantVectorStore(VectorStore):
    client: Any = Field(
        default=None,
        description=(
            "Instance of `qdrant_client.AsyncQdrantClient`. Defaults to an in-memory"
            " instance."
        ),
    )
    collection_name: str = Field(default_factory=lambda: f"paper-qa-{uuid.uuid4().hex}")
    vector_name: str | None = Field(default=None)
    _point_ids: set[str] | None = None

    def __del__(self):
        """Cleanup async client connection."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                _ = loop.create_task(self.aclose())  # noqa: RUF006
            else:
                loop.run_until_complete(self.aclose())
        except Exception as e:
            logger.warning(f"Error closing client connection: {e}")

    async def aclose(self):
        """Explicitly close async client."""
        await self.client.close()

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.texts_hashes == other.texts_hashes
            and self.mmr_lambda == other.mmr_lambda
            and self.collection_name == other.collection_name
            and self.vector_name == other.vector_name
            and self.client.init_options == other.client.init_options
            and self._point_ids == other._point_ids
        )

    @model_validator(mode="after")
    def validate_client(self):
        if not qdrant_installed:
            msg = (
                "`QdrantVectorStore` requires the `qdrant-client` package. "
                "Install it with `pip install paper-qa[qdrant]`"
            )
            raise ImportError(msg)

        if self.client and not isinstance(self.client, AsyncQdrantClient):
            raise TypeError(
                "'client' should be an instance of AsyncQdrantClient. Got"
                f" `{type(self.client)}`"
            )

        if not self.client:
            # Defaults to the Python based in-memory implementation.
            self.client = AsyncQdrantClient(location=":memory:")

        return self

    async def _collection_exists(self) -> bool:
        return await self.client.collection_exists(self.collection_name)

    @override
    def clear(self) -> None:
        """Synchronous clear method that matches parent class."""
        super().clear()  # Clear the base class attributes first

        # Create a new event loop in a new thread to avoid nested loop issues
        def run_async():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(self.aclear())
            finally:
                new_loop.close()

        thread = threading.Thread(target=run_async)
        thread.start()
        thread.join()

    async def aclear(self) -> None:
        """Asynchronous clear implementation."""
        if not await self._collection_exists():
            return

        await self.client.delete_collection(collection_name=self.collection_name)
        self._point_ids = None

    async def add_texts_and_embeddings(self, texts: Iterable[Embeddable]) -> None:
        await super().add_texts_and_embeddings(texts)

        texts_list = list(texts)

        if texts_list and not await self._collection_exists():
            params = models.VectorParams(
                size=len(cast("Sized", texts_list[0].embedding)),
                distance=models.Distance.COSINE,
            )

            await self.client.create_collection(
                self.collection_name,
                vectors_config=(
                    {self.vector_name: params} if self.vector_name else params
                ),
            )

        ids, payloads, vectors = [], [], []
        for text in texts_list:
            ids.append(uuid.uuid5(uuid.NAMESPACE_URL, str(text.embedding)).hex)
            payloads.append(text.model_dump(exclude={"embedding"}))
            vectors.append(
                {self.vector_name: text.embedding}
                if self.vector_name
                else text.embedding
            )

        await self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=some_id,
                    payload=some_payload,
                    vector=some_vector,
                )
                for some_id, some_payload, some_vector in zip(
                    ids, payloads, vectors, strict=True
                )
            ],
        )
        self._point_ids = set(ids)

    async def similarity_search(
        self, query: str, k: int, embedding_model: EmbeddingModel
    ) -> tuple[Sequence[Embeddable], list[float]]:
        if not await self._collection_exists():
            return ([], [])

        embedding_model.set_mode(EmbeddingModes.QUERY)
        np_query = np.array((await embedding_model.embed_documents([query]))[0])
        embedding_model.set_mode(EmbeddingModes.DOCUMENT)

        points = (
            await self.client.query_points(
                collection_name=self.collection_name,
                query=np_query,
                using=self.vector_name,
                limit=k,
                with_vectors=True,
                with_payload=True,
            )
        ).points

        return (
            [
                Text(
                    **p.payload,
                    embedding=(
                        p.vector[self.vector_name] if self.vector_name else p.vector
                    ),
                )
                for p in points
            ],
            [p.score for p in points],
        )

    @classmethod
    async def load_docs(
        cls,
        client: "AsyncQdrantClient",
        collection_name: str,
        vector_name: str | None = None,
        batch_size: int = 100,
        max_concurrent_requests: int = 5,
    ) -> "Docs":
        from paperqa.docs import Docs  # Avoid circular imports

        vectorstore = cls(
            client=client, collection_name=collection_name, vector_name=vector_name
        )
        docs = Docs(texts_index=vectorstore)

        collection_info = await client.get_collection(collection_name)
        total_points = collection_info.points_count or 0

        semaphore = asyncio.Semaphore(max_concurrent_requests)
        all_points: list[Record] = []

        async def fetch_batch_with_semaphore(offset: int) -> None:
            async with semaphore:
                points = await client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                all_points.extend(points[0])

        tasks = [
            fetch_batch_with_semaphore(offset)
            for offset in range(0, total_points, batch_size)
        ]
        await asyncio.gather(*tasks)

        for point in all_points:
            try:
                if point.payload is None:
                    continue

                payload = point.payload
                doc_data = payload.get("doc", {})
                if not isinstance(doc_data, dict):
                    continue

                if doc_data.get("dockey") not in docs.docs:
                    docs.docs[doc_data["dockey"]] = Doc(
                        docname=doc_data.get("docname", ""),
                        citation=doc_data.get("citation", ""),
                        dockey=doc_data["dockey"],
                    )
                    docs.docnames.add(doc_data.get("docname", ""))

                if point.vector is None:
                    continue

                vector_value = (
                    point.vector.get(vector_name)
                    if vector_name and isinstance(point.vector, dict)
                    else point.vector
                )

                text = Text(
                    text=payload.get("text", ""),
                    name=payload.get("name", ""),
                    doc=docs.docs[doc_data["dockey"]],
                    embedding=vector_value,
                )
                docs.texts.append(text)

            except KeyError as e:
                logger.warning(f"Skipping invalid point due to missing field: {e!s}")
                continue

        return docs


def embedding_model_factory(embedding: str, **kwargs) -> EmbeddingModel:
    """
    Factory function to create an appropriate EmbeddingModel based on the embedding string.

    Supports:
    - SentenceTransformer models prefixed with "st-" (e.g., "st-multi-qa-MiniLM-L6-cos-v1")
    - LiteLLM models (default if no prefix is provided)
    - Hybrid embeddings prefixed with "hybrid-", contains a sparse and a dense model

    Args:
        embedding: The embedding model identifier. Supports prefixes like "st-" for SentenceTransformer
                   and "hybrid-" for combining multiple embedding models.
        **kwargs: Additional keyword arguments for the embedding model.
    """
    embedding = embedding.strip()  # Remove any leading/trailing whitespace

    if embedding.startswith("hybrid-"):
        # Extract the component embedding identifiers after "hybrid-"
        dense_name = embedding[len("hybrid-") :]

        if not dense_name:
            raise ValueError(
                "Hybrid embedding must contain at least one component embedding."
            )

        # Recursively create each component embedding model
        dense_model = embedding_model_factory(dense_name, **kwargs)
        sparse_model = SparseEmbeddingModel(**kwargs)

        return HybridEmbeddingModel(models=[dense_model, sparse_model])

    if embedding.startswith("st-"):
        # Extract the SentenceTransformer model name after "st-"
        model_name = embedding[len("st-") :].strip()
        if not model_name:
            raise ValueError(
                "SentenceTransformer model name must be specified after 'st-'."
            )

        return SentenceTransformerEmbeddingModel(
            name=model_name,
            config=kwargs,
        )

    if embedding.startswith("litellm-"):
        # Extract the LiteLLM model name after "litellm-"
        model_name = embedding[len("litellm-") :].strip()
        if not model_name:
            raise ValueError("model name must be specified after 'litellm-'.")

        return LiteLLMEmbeddingModel(
            name=model_name,
            config=kwargs,
        )

    if embedding == "sparse":
        return SparseEmbeddingModel(**kwargs)

    # Default to LiteLLMEmbeddingModel if no special prefix is found
    return LiteLLMEmbeddingModel(name=embedding, config=kwargs)



================================================
FILE: paperqa/paths.py
================================================
from pathlib import Path

PAPERQA_DIR = Path.home() / ".paperqa"



================================================
FILE: paperqa/prompts.py
================================================
from datetime import datetime

# ruff: noqa: E501

summary_prompt = (
    "Summarize the excerpt below to help answer a question.\n\nExcerpt from"
    " {citation}\n\n----\n\n{text}\n\n----\n\nQuestion: {question}\n\nDo not directly"
    " answer the question, instead summarize to give evidence to help answer the"
    " question. Stay detailed; report specific numbers, equations, or direct quotes"
    ' (marked with quotation marks). Reply "Not applicable" if the excerpt is'
    " irrelevant. At the end of your response, provide an integer score from 1-10 on a"
    " newline indicating relevance to question. Do not explain your score.\n\nRelevant"
    " Information Summary ({summary_length}):"
)

summary_json_prompt = (
    "Excerpt from {citation}\n\n----\n\n{text}\n\n----\n\nQuestion: {question}\n\n"
)

# The below "cannot answer" sentinel phrase should:
# 1. Lead to complete tool being called with has_successful_answer=False
# 2. Can be used for unit testing
CANNOT_ANSWER_PHRASE = "I cannot answer"

answer_iteration_prompt_template = (
    "You are iterating on a prior answer, with a potentially different context:\n\n"
    "{prior_answer}\n\n"
    "Create a new answer only using keys and data from the included context."
    " You can not use context keys from the prior answer which are not "
    "also included in the above context.\n\n"
)

CITATION_KEY_CONSTRAINTS = (
    "## Valid citation examples: \n"
    "- Example2024Example pages 3-4 \n"
    "- Example2024 pages 3-4 \n"
    "- Example2024 pages 3-4, Example2024 pages 5-6 \n"
    "## Invalid citation examples: \n"
    "- Example2024Example pages 3-4 and pages 4-5 \n"
    "- Example2024Example (pages 3-4) \n"
    "- Example2024Example pages 3-4, pages 5-6 \n"
    "- Example2024Example et al. (2024) \n"
    "- Example's work (pages 17–19) \n"  # noqa: RUF001
    "- (pages 17–19) \n"  # noqa: RUF001
)

qa_prompt = (
    "Answer the question below with the context.\n\n"
    "Context (with relevance scores):\n\n{context}\n\n----\n\n"
    "Question: {question}\n\n"
    "Write an answer based on the context. "
    "If the context provides insufficient information reply "
    f'"{CANNOT_ANSWER_PHRASE}." '
    "For each part of your answer, indicate which sources most support "
    "it via citation keys at the end of sentences, like {example_citation}. "
    "Only cite from the context above and only use the citation keys from the context. "
    f"{CITATION_KEY_CONSTRAINTS}"
    "Do not concatenate citation keys, just use them as is. "
    "Write in the style of a Wikipedia article, with concise sentences and "
    "coherent paragraphs. The context comes from a variety of sources and is "
    "only a summary, so there may inaccuracies or ambiguities. If quotes are "
    "present and relevant, use them in the answer. This answer will go directly "
    "onto Wikipedia, so do not add any extraneous information.\n\n"
    "{prior_answer_prompt}"
    "Answer ({answer_length}):"
)

select_paper_prompt = (
    "Select papers that may help answer the question below. "
    "Papers are listed as $KEY: $PAPER_INFO. "
    "Return a list of keys, separated by commas. "
    'Return "None", if no papers are applicable. '
    "Choose papers that are relevant, from reputable sources, and timely "
    "(if the question requires timely information).\n\n"
    "Question: {question}\n\n"
    "Papers: {papers}\n\n"
    "Selected keys:"
)

citation_prompt = (
    "Provide the citation for the following text in MLA Format. "
    "Do not write an introductory sentence. "
    f"If reporting date accessed, the current year is {datetime.now().year}\n\n"
    "{text}\n\n"
    "Citation:"
)

structured_citation_prompt = (
    "Extract the title, authors, and doi as a JSON from this MLA citation. "
    "If any field can not be found, return it as null. "
    "Use title, authors, and doi as keys, author's value should be a list of authors. "
    "{citation}\n\n"
    "Citation JSON:"
)

default_system_prompt = (
    "Answer in a direct and concise tone. "
    "Your audience is an expert, so be highly specific. "
    "If there are ambiguous terms or acronyms, first define them."
)

# NOTE: we use double curly braces here so it's not considered an f-string template
summary_json_system_prompt = """\
Provide a summary of the relevant information that could help answer the question based on the excerpt. Respond with the following JSON format:

{{
  "summary": "...",
  "relevance_score": "..."
}}

where `summary` is relevant information from the text - {summary_length} words. `relevance_score` is an integer 1-10 for the relevance of `summary` to the question.
"""

env_system_prompt = (
    # Matching https://github.com/langchain-ai/langchain/blob/langchain%3D%3D0.2.3/libs/langchain/langchain/agents/openai_functions_agent/base.py#L213-L215
    "You are a helpful AI assistant."
)
env_reset_prompt = (
    "Use the tools to answer the question: {question}"
    "\n\nWhen the answer looks sufficient,"
    " you can terminate by calling the {complete_tool_name} tool."
    " If the answer does not look sufficient,"
    " and you have already tried to answer several times with different evidence,"
    " terminate by calling the {complete_tool_name} tool."
    " The current status of evidence/papers/cost is {status}"
)

# Prompt templates for use with LitQA
QA_PROMPT_TEMPLATE = "Q: {question}\n\nOptions:\n{options}"
EVAL_PROMPT_TEMPLATE = (
    "Given the following question and a proposed answer to the question, return the"
    " single-letter choice in the question that matches the proposed answer."
    " If the proposed answer is blank or an empty string,"
    " or multiple options are matched, respond with '0'."
    "\n\nQuestion: {qa_prompt}"
    "\n\nProposed Answer: {qa_answer}"
    "\n\nSingle Letter Answer:"
)

CONTEXT_OUTER_PROMPT = "{context_str}\n\nValid Keys: {valid_keys}"
CONTEXT_INNER_PROMPT_NOT_DETAILED = "{name}: {text}"
CONTEXT_INNER_PROMPT = f"{CONTEXT_INNER_PROMPT_NOT_DETAILED}\nFrom {{citation}}"



================================================
FILE: paperqa/py.typed
================================================



================================================
FILE: paperqa/readers.py
================================================
from __future__ import annotations

import asyncio
import os
from math import ceil
from pathlib import Path
from typing import Literal, cast, overload

import pymupdf
import tiktoken
from html2text import __version__ as html2text_version
from html2text import html2text

from paperqa.types import (
    ChunkMetadata,
    Doc,
    ParsedMetadata,
    ParsedText,
    Text,
)
from paperqa.utils import ImpossibleParsingError
from paperqa.version import __version__ as pqa_version


def parse_pdf_to_pages(
    path: str | os.PathLike, page_size_limit: int | None = None
) -> ParsedText:

    with pymupdf.open(path) as file:
        pages: dict[str, str] = {}
        total_length = 0

        for i in range(file.page_count):
            try:
                page = file.load_page(i)
            except pymupdf.mupdf.FzErrorFormat as exc:
                raise ImpossibleParsingError(
                    f"Page loading via {pymupdf.__name__} failed on page {i} of"
                    f" {file.page_count} for the PDF at path {path}, likely this PDF"
                    " file is corrupt."
                ) from exc
            text = page.get_text("text", sort=True)
            if page_size_limit and len(text) > page_size_limit:
                raise ImpossibleParsingError(
                    f"The text in page {i} of {file.page_count} was {len(text)} chars"
                    f" long, which exceeds the {page_size_limit} char limit for the PDF"
                    f" at path {path}."
                )
            pages[str(i + 1)] = text
            total_length += len(text)

    metadata = ParsedMetadata(
        parsing_libraries=[f"pymupdf ({pymupdf.__version__})"],
        paperqa_version=pqa_version,
        total_parsed_text_length=total_length,
        parse_type="pdf",
    )
    return ParsedText(content=pages, metadata=metadata)


def chunk_pdf(
    parsed_text: ParsedText, doc: Doc, chunk_chars: int, overlap: int
) -> list[Text]:
    pages: list[str] = []
    texts: list[Text] = []
    split: str = ""

    if not isinstance(parsed_text.content, dict):
        raise NotImplementedError(
            f"ParsedText.content must be a `dict`, not {type(parsed_text.content)}."
        )

    if not parsed_text.content:
        raise ImpossibleParsingError(
            f"No text was parsed from the document named {doc.docname!r} with ID"
            f" {doc.dockey}, either empty or corrupted."
        )

    for page_num, page_text in parsed_text.content.items():
        split += page_text
        pages.append(page_num)
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            texts.append(
                Text(
                    text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [page_num]

    if len(split) > overlap or not texts:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        )
    return texts


def parse_text(
    path: str | os.PathLike,
    html: bool = False,
    split_lines: bool = False,
    use_tiktoken: bool = True,
    page_size_limit: int | None = None,
) -> ParsedText:
    """Simple text splitter, can optionally use tiktoken, parse html, or split into newlines.

    Args:
        path: path to file.
        html: flag to use html2text library for parsing.
        split_lines: flag to split lines into a list.
        use_tiktoken: flag to use tiktoken library to encode text.
        page_size_limit: optional limit on the number of characters per page. Only
            relevant when split_lines is True.
    """
    path = Path(path)
    try:
        with path.open() as f:
            text = list(f) if split_lines else f.read()
    except UnicodeDecodeError:
        with path.open(encoding="utf-8", errors="ignore") as f:
            text = f.read()

    parsing_libraries: list[str] = ["tiktoken (cl100k_base)"] if use_tiktoken else []
    if html:
        if not isinstance(text, str):
            raise NotImplementedError(
                "HTML parsing is not yet set up to work with split_lines."
            )
        parse_type: str = "html"
        text = html2text(text)
        parsing_libraries.append(f"html2text ({html2text_version})")
    else:
        parse_type = "txt"
    if isinstance(text, str):
        total_length: int = len(text)
    else:
        total_length = sum(len(t) for t in text)
        for i, t in enumerate(text):
            if page_size_limit and len(text) > page_size_limit:
                raise ImpossibleParsingError(
                    f"The {parse_type} on page {i} of {len(text)} was {len(t)} chars"
                    f" long, which exceeds the {page_size_limit} char limit at path"
                    f" {path}."
                )
    return ParsedText(
        content=text,
        metadata=ParsedMetadata(
            parsing_libraries=parsing_libraries,
            paperqa_version=pqa_version,
            total_parsed_text_length=total_length,
            parse_type=parse_type,
        ),
    )


def chunk_text(
    parsed_text: ParsedText,
    doc: Doc,
    chunk_chars: int,
    overlap: int,
    use_tiktoken: bool = True,
) -> list[Text]:
    """Parse a document into chunks, based on tiktoken encoding.

    NOTE: We get some byte continuation errors.
    Currently ignored, but should explore more to make sure we don't miss anything.
    """
    texts: list[Text] = []
    enc = tiktoken.get_encoding("cl100k_base")

    if not isinstance(parsed_text.content, str):
        raise NotImplementedError(
            f"ParsedText.content must be a `str`, not {type(parsed_text.content)}."
        )

    content: str | list[int] = (
        parsed_text.content if not use_tiktoken else parsed_text.encode_content()
    )
    if not content:  # Avoid div0 in token calculations
        raise ImpossibleParsingError(
            f"No text was parsed from the document named {doc.docname!r} with ID"
            f" {doc.dockey}, either empty or corrupted."
        )

    # convert from characters to chunks
    char_count = parsed_text.metadata.total_parsed_text_length  # e.g., 25,000
    token_count = len(content)  # e.g., 4,500
    chars_per_token = char_count / token_count  # e.g., 5.5
    chunk_tokens = chunk_chars / chars_per_token  # e.g., 3000 / 5.5 = 545
    overlap_tokens = overlap / chars_per_token  # e.g., 100 / 5.5 = 18
    chunk_count = ceil(token_count / chunk_tokens)  # e.g., 4500 / 545 = 9

    for i in range(chunk_count):
        split = content[
            max(int(i * chunk_tokens - overlap_tokens), 0) : int(
                (i + 1) * chunk_tokens + overlap_tokens
            )
        ]
        texts.append(
            Text(
                text=(
                    enc.decode(cast("list[int]", split))
                    if use_tiktoken
                    else cast("str", split)
                ),
                name=f"{doc.docname} chunk {i + 1}",
                doc=doc,
            )
        )
    return texts


def chunk_code_text(
    parsed_text: ParsedText, doc: Doc, chunk_chars: int, overlap: int
) -> list[Text]:
    """Parse a document into chunks, based on line numbers (for code)."""
    split = ""
    texts: list[Text] = []
    last_line = 0

    if not isinstance(parsed_text.content, list):
        raise NotImplementedError(
            f"ParsedText.content must be a `list`, not {type(parsed_text.content)}."
        )

    for i, line in enumerate(parsed_text.content):
        split += line
        while len(split) > chunk_chars:
            texts.append(
                Text(
                    text=split[:chunk_chars],
                    name=f"{doc.docname} lines {last_line}-{i}",
                    doc=doc,
                )
            )
            split = split[chunk_chars - overlap :]
            last_line = i
    if len(split) > overlap or not texts:
        texts.append(
            Text(
                text=split[:chunk_chars],
                name=f"{doc.docname} lines {last_line}-{i}",
                doc=doc,
            )
        )
    return texts


@overload
async def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: Literal[False],
    include_metadata: Literal[False],
    chunk_chars: int = ...,
    overlap: int = ...,
    page_size_limit: int | None = ...,
) -> list[Text]: ...


@overload
async def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: Literal[False] = ...,
    include_metadata: Literal[False] = ...,
    chunk_chars: int = ...,
    overlap: int = ...,
    page_size_limit: int | None = ...,
) -> list[Text]: ...


@overload
async def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: Literal[True],
    include_metadata: bool = ...,
    chunk_chars: int = ...,
    overlap: int = ...,
    page_size_limit: int | None = ...,
) -> ParsedText: ...


@overload
async def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: Literal[False],
    include_metadata: Literal[True],
    chunk_chars: int = ...,
    overlap: int = ...,
    page_size_limit: int | None = ...,
) -> tuple[list[Text], ParsedMetadata]: ...


async def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: bool = False,
    include_metadata: bool = False,
    chunk_chars: int = 3000,
    overlap: int = 100,
    page_size_limit: int | None = None,
) -> list[Text] | ParsedText | tuple[list[Text], ParsedMetadata]:
    """Parse a document and split into chunks.

    Optionally can include just the parsing as well as metadata about the parsing/chunking
    Args:
        path: local document path
        doc: object with document metadata
        parsed_text_only: return parsed text without chunking
        include_metadata: return a tuple
        chunk_chars: size of chunks
        overlap: size of overlap between chunks
        page_size_limit: optional limit on the number of characters per page
    """
    str_path = str(path)

    # start with parsing -- users may want to store this separately
    if str_path.endswith(".pdf"):
        # TODO: Make parse_pdf_to_pages async
        parsed_text = await asyncio.to_thread(
            parse_pdf_to_pages, path, page_size_limit=page_size_limit
        )
    elif str_path.endswith(".txt"):
        # TODO: Make parse_text async
        parsed_text = await asyncio.to_thread(
            parse_text, path, page_size_limit=page_size_limit
        )
    elif str_path.endswith(".html"):
        parsed_text = await asyncio.to_thread(
            parse_text, path, html=True, page_size_limit=page_size_limit
        )
    else:
        parsed_text = await asyncio.to_thread(
            parse_text,
            path,
            split_lines=True,
            use_tiktoken=False,
            page_size_limit=page_size_limit,
        )

    if parsed_text_only:
        return parsed_text

    # next chunk the parsed text

    # check if chunk is 0 (no chunking)
    if chunk_chars == 0:
        chunked_text = [
            Text(text=parsed_text.reduce_content(), name=doc.docname, doc=doc)
        ]
        chunk_metadata = ChunkMetadata(chunk_chars=0, overlap=0, chunk_type="no_chunk")
    elif str_path.endswith(".pdf"):
        chunked_text = chunk_pdf(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            chunk_chars=chunk_chars,
            overlap=overlap,
            chunk_type="overlap_pdf_by_page",
        )
    elif str_path.endswith((".txt", ".html")):
        chunked_text = chunk_text(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            chunk_chars=chunk_chars, overlap=overlap, chunk_type="overlap"
        )
    else:
        chunked_text = chunk_code_text(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            chunk_chars=chunk_chars,
            overlap=overlap,
            chunk_type="overlap_code_by_line",
        )

    if include_metadata:
        parsed_text.metadata.chunk_metadata = chunk_metadata
        return chunked_text, parsed_text.metadata

    return chunked_text



================================================
FILE: paperqa/settings.py
================================================
import asyncio
import importlib.resources
import os
import pathlib
import warnings
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pydoc import locate
from typing import Any, ClassVar, Self, TypeAlias, assert_never, cast

import anyio
from aviary.core import Tool, ToolSelector
from lmi import (
    CommonLLMNames,
    EmbeddingModel,
    LiteLLMModel,
    embedding_model_factory,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, CliSettingsSource, SettingsConfigDict

from paperqa._ldp_shims import (
    HAS_LDP_INSTALLED,
    Agent,
    HTTPAgentClient,
    MemoryAgent,
    ReActAgent,
    SimpleAgent,
    SimpleAgentState,
    UIndexMemoryModel,
    _Memories,
    set_training_mode,
)
from paperqa.prompts import (
    CONTEXT_INNER_PROMPT,
    CONTEXT_OUTER_PROMPT,
    answer_iteration_prompt_template,
    citation_prompt,
    default_system_prompt,
    env_reset_prompt,
    env_system_prompt,
    qa_prompt,
    select_paper_prompt,
    structured_citation_prompt,
    summary_json_prompt,
    summary_json_system_prompt,
    summary_prompt,
)
from paperqa.utils import hexdigest, pqa_directory
from paperqa.version import __version__

# TODO: move to actual EnvironmentState
# when we can do so without a circular import
_EnvironmentState: TypeAlias = Any


class AnswerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_k: int = Field(
        default=10, description="Number of evidence pieces to retrieve."
    )
    evidence_detailed_citations: bool = Field(
        default=True,
        description="Whether to include detailed citations in summaries.",
    )
    evidence_retrieval: bool = Field(
        default=True,
        description="Whether to use retrieval instead of processing all docs.",
    )
    evidence_summary_length: str = Field(
        default="about 100 words", description="Length of evidence summary."
    )
    evidence_skip_summary: bool = Field(
        default=False, description="Whether to summarization."
    )
    answer_max_sources: int = Field(
        default=5, description="Max number of sources to use for an answer."
    )
    max_answer_attempts: int | None = Field(
        default=None,
        description=(
            "Optional (exclusive) max number (default is no max) of attempts to"
            " generate an answer before declaring done (without a complete tool call). "
        ),
    )
    answer_length: str = Field(
        "about 200 words, but can be longer", description="Length of final answer."
    )
    max_concurrent_requests: int = Field(
        default=4, description="Max concurrent requests to LLMs."
    )
    answer_filter_extra_background: bool = Field(
        default=False,
        description="Whether to cite background information provided by model.",
    )
    get_evidence_if_no_contexts: bool = Field(
        default=True,
        description=(
            "Opt-out flag for allowing answer generation to lazily gather evidence if"
            " called before evidence was gathered."
        ),
    )

    @model_validator(mode="after")
    def _deprecated_field(self) -> Self:
        # default is True, so we only warn if it's False
        if not self.evidence_detailed_citations:
            warnings.warn(
                "The 'evidence_detailed_citations' field is deprecated and will be"
                " removed in version 6. Adjust 'PromptSettings.context_inner' to remove"
                " detailed citations.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        return self


class ParsingOptions(StrEnum):
    PAPERQA_DEFAULT = "paperqa_default"

    def available_for_inference(self) -> list["ParsingOptions"]:
        return [self.PAPERQA_DEFAULT]  # type: ignore[list-item]


def _get_parse_type(opt: ParsingOptions, config: "ParsingSettings") -> str:
    if opt == ParsingOptions.PAPERQA_DEFAULT:
        return config.parser_version_string
    assert_never(opt)


class ChunkingOptions(StrEnum):
    SIMPLE_OVERLAP = "simple_overlap"

    @property
    def valid_parsings(self) -> list[ParsingOptions]:
        # Note that SIMPLE_OVERLAP must be valid for all by default
        # TODO: implement for future parsing options
        valid_parsing_dict: dict[str, list[ParsingOptions]] = {}
        return valid_parsing_dict.get(self.value, [])  # noqa: FURB184


class ParsingSettings(BaseModel):
    """Settings relevant for parsing and chunking documents."""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(
        default=5000,
        description="Number of characters per chunk. If 0, no chunking will be done.",
    )
    page_size_limit: int | None = Field(
        default=1_280_000,
        description=(
            "Optional limit on the number of characters to parse in one 'page', default"
            " is 1.28 million chars, 10X larger than a 128k tokens context limit"
            " (ignoring chars vs tokens difference)."
        ),
    )
    use_doc_details: bool = Field(
        default=True, description="Whether to try to get metadata details for a Doc."
    )
    overlap: int = Field(
        default=250, description="Number of characters to overlap chunks."
    )
    citation_prompt: str = Field(
        default=citation_prompt,
        description="Prompt that tries to create citation from peeking one page.",
    )
    structured_citation_prompt: str = Field(
        default=structured_citation_prompt,
        description=(
            "Prompt that tries to creates a citation in JSON from peeking one page."
        ),
    )
    disable_doc_valid_check: bool = Field(
        default=False,
        description=(
            "Whether to disable checking if a document looks like text (was parsed"
            " correctly)."
        ),
    )
    defer_embedding: bool = Field(
        default=False,
        description=(
            "Whether to embed documents immediately as they are added, or defer until"
            " summarization."
        ),
    )
    chunking_algorithm: ChunkingOptions = ChunkingOptions.SIMPLE_OVERLAP
    doc_filters: Sequence[Mapping[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional filters to only allow documents that match this filter. This is a"
            " dictionary where the keys are the fields from DocDetails or Docs to"
            " filter on, and the values are the values to filter for. To invert filter,"
            " prefix the key with a '!'. If the key is not found, by default the Doc is"
            " rejected. To change this behavior, prefix the key with a '?' to allow the"
            " Doc to pass if the key is not found. For example, {'!title': 'bad title',"
            " '?year': '2022'} would only allow Docs with a title that is not 'bad"
            " title' and a year of 2022 or no year at all."
        ),
    )
    use_human_readable_clinical_trials: bool = Field(
        default=False,
        description="Parse clinical trial JSONs into human readable text.",
    )

    def chunk_type(self, chunking_selection: ChunkingOptions | None = None) -> str:
        """Future chunking implementations (i.e. by section) will get an elif clause here."""
        if chunking_selection is None:
            chunking_selection = self.chunking_algorithm
        if chunking_selection == ChunkingOptions.SIMPLE_OVERLAP:
            return (
                f"{self.parser_version_string}|{chunking_selection.value}"
                f"|tokens={self.chunk_size}|overlap={self.overlap}"
            )
        assert_never(chunking_selection)

    @property
    def parser_version_string(self) -> str:
        return f"paperqa-{__version__}"

    def is_chunking_valid_for_parsing(self, parsing: str):
        # must map the parsings because they won't include versions by default
        return (
            self.chunking_algorithm == ChunkingOptions.SIMPLE_OVERLAP
            or parsing
            in {  # type: ignore[unreachable]
                _get_parse_type(p, self) for p in self.chunking_algorithm.valid_parsings
            }
        )


class _FormatDict(dict):  # noqa: FURB189
    """Mock a dictionary and store any missing items."""

    def __init__(self) -> None:
        self.key_set: set[str] = set()

    def __missing__(self, key: str) -> str:
        self.key_set.add(key)
        return key


def get_formatted_variables(s: str) -> set[str]:
    """Returns the set of variables implied by the format string."""
    format_dict = _FormatDict()
    s.format_map(format_dict)
    return format_dict.key_set


class PromptSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # MLA parenthetical in-text citation, SEE: https://nwtc.libguides.com/citations/MLA#s-lg-box-707489
    EXAMPLE_CITATION: ClassVar[str] = "(Example2012Example pages 3-4)"

    summary: str = summary_prompt
    qa: str = qa_prompt
    answer_iteration_prompt: str | None = Field(
        default=answer_iteration_prompt_template,
        description=(
            "Prompt to inject existing prior answers into the qa prompt to allow the model to iterate. "
            "If None, then no prior answers will be injected."
        ),
    )
    select: str = select_paper_prompt
    pre: str | None = Field(
        default=None,
        description=(
            "Opt-in pre-prompt (templated with just the original question) to append"
            " information before a qa prompt. For example:"
            " 'Summarize all scientific terms in the following question:\n{question}'."
            " This pre-prompt can enable injection of question-specific guidance later"
            " used by the qa prompt, without changing the qa prompt's template."
        ),
    )
    post: str | None = None
    system: str = default_system_prompt
    use_json: bool = True
    # Not thrilled about this model,
    # but need to split out the system/summary
    # to get JSON
    summary_json: str = summary_json_prompt
    summary_json_system: str = summary_json_system_prompt
    context_outer: str = Field(
        default=CONTEXT_OUTER_PROMPT,
        description="Prompt for how to format all contexts in generate answer.",
    )
    context_inner: str = Field(
        default=CONTEXT_INNER_PROMPT,
        description=(
            "Prompt for how to format a single context in generate answer. "
            "This should at least contain key and name."
        ),
    )

    @field_validator("summary")
    @classmethod
    def check_summary(cls, v: str) -> str:
        if not get_formatted_variables(v).issubset(
            get_formatted_variables(summary_prompt)
        ):
            raise ValueError(
                "Summary prompt can only have variables:"
                f" {get_formatted_variables(summary_prompt)}"
            )
        return v

    @field_validator("qa")
    @classmethod
    def check_qa(cls, v: str) -> str:
        if not get_formatted_variables(v).issubset(get_formatted_variables(qa_prompt)):
            raise ValueError(
                "QA prompt can only have variables:"
                f" {get_formatted_variables(qa_prompt)}"
            )
        return v

    @field_validator("select")
    @classmethod
    def check_select(cls, v: str) -> str:
        if not get_formatted_variables(v).issubset(
            get_formatted_variables(select_paper_prompt)
        ):
            raise ValueError(
                "Select prompt can only have variables:"
                f" {get_formatted_variables(select_paper_prompt)}"
            )
        return v

    @field_validator("post")
    @classmethod
    def check_post(cls, v: str | None) -> str | None:
        if v is not None:
            # kind of a hack to get list of attributes in answer
            from paperqa.types import PQASession

            attrs = set(PQASession.model_fields.keys())
            if not get_formatted_variables(v).issubset(attrs):
                raise ValueError(f"Post prompt must have input variables: {attrs}")
        return v

    @field_validator("context_outer")
    @classmethod
    def check_context_outer(cls, v: str) -> str:
        if not get_formatted_variables(v).issubset(
            get_formatted_variables(CONTEXT_OUTER_PROMPT)
        ):
            raise ValueError(
                "Context outer prompt can only have variables:"
                f" {get_formatted_variables(CONTEXT_OUTER_PROMPT)}"
            )
        return v

    @field_validator("context_inner")
    @classmethod
    def check_context_inner(cls, v: str) -> str:
        fvars = get_formatted_variables(v)
        if "name" not in fvars or "text" not in fvars:
            raise ValueError("Context inner prompt must have name and text")
        return v


class IndexSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        description=(
            "Optional name of the index. If unspecified, the name should be generated."
        ),
    )
    paper_directory: str | os.PathLike = Field(
        default=pathlib.Path.cwd(),
        description=(
            "Local directory which contains the papers to be indexed and searched."
        ),
    )
    manifest_file: str | os.PathLike | None = Field(
        default=None,
        description=(
            "Optional absolute path to a manifest CSV, or a relative path from the"
            " paper_directory to a manifest CSV. A manifest CSV contains columns which"
            " are attributes for a DocDetails object. Only 'file_location', 'doi', and"
            " 'title' will be used when indexing, others are discarded."
        ),
    )
    index_directory: str | os.PathLike = Field(
        default_factory=lambda: pqa_directory("indexes"),
        description=(
            "Directory to store the PQA built search index, configuration, and"
            " answer indexes."
        ),
    )
    use_absolute_paper_directory: bool = Field(
        default=False,
        description=(
            "Opt-in flag to convert the paper_directory to an absolute path. Setting"
            " this to True will make the index user-specific, defeating sharing."
        ),
    )
    recurse_subdirectories: bool = Field(
        default=True,
        description="Whether to recurse into subdirectories when indexing sources.",
    )
    concurrency: int = Field(
        default=5,  # low default for folks without S2/Crossref keys
        description="Number of concurrent filesystem reads for indexing",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Number of files to process before committing to the index.",
    )
    sync_with_paper_directory: bool = Field(
        default=True,
        description=(
            "Whether to sync the index with the paper directory when loading an index."
            " Setting to True will add or delete index files to match the source paper"
            " directory."
        ),
    )

    def get_named_index_directory(self) -> anyio.Path:
        """Get the directory where the index, when named, will be located.

        Raises:
            ValueError: If the index name was unset, because otherwise the name is
                autogenerated.
        """
        if self.name is None:
            raise ValueError(
                "Getting a named index directory requires an index name to have been"
                " specified, please specify a name."
            )
        return anyio.Path(self.index_directory) / self.name

    async def finalize_manifest_file(self) -> anyio.Path | None:
        manifest_file = anyio.Path(self.manifest_file) if self.manifest_file else None
        if manifest_file and not await manifest_file.exists():
            # If the manifest file was specified but doesn't exist,
            # perhaps it was specified as a relative path from the paper_directory
            manifest_file = anyio.Path(self.paper_directory) / manifest_file
        return manifest_file


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_llm: str = Field(
        default=CommonLLMNames.GPT_4O.value,
        description="Model to use for agent making tool selections.",
    )

    agent_llm_config: dict | None = Field(
        default=None,
        description=(
            "Optional configuration for the agent_llm model. More specifically, it's"
            " a LiteLLM Router configuration to pass to LiteLLMModel, must have"
            " `model_list` key (corresponding to model_list inputs here:"
            " https://docs.litellm.ai/docs/routing), and can optionally include a"
            " router_kwargs key with router kwargs as values."
        ),
    )

    agent_type: str = Field(
        default="ToolSelector",
        description="Type of agent to use",
    )
    agent_config: dict[str, Any] | None = Field(
        default=None,
        description="Optional kwarg for AGENT constructor.",
    )
    agent_system_prompt: str | None = Field(
        default=env_system_prompt,
        description="Optional system prompt message to precede the below agent_prompt.",
    )
    agent_prompt: str = env_reset_prompt
    return_paper_metadata: bool = Field(
        default=False,
        description=(
            "Set True to have the search tool include paper title/year information as"
            " part of its return."
        ),
    )
    search_count: int = 8
    wipe_context_on_answer_failure: bool = True
    agent_evidence_n: int = Field(
        default=1,
        ge=1,
        description=(
            "Top n ranked evidences shown to the agent after the GatherEvidence tool."
        ),
    )
    timeout: float = Field(
        default=500.0,
        description=(
            "Matches LangChain AgentExecutor.max_execution_time (seconds), the timeout"
            " on agent execution."
        ),
    )
    should_pre_search: bool = Field(
        default=False,
        description="If set to true, run the search tool before invoking agent.",
    )

    tool_names: set[str] | Sequence[str] | None = Field(
        default=None,
        description=(
            "Optional override on the tools to provide the agent. Leaving as the"
            " default of None will use a minimal toolset of the paper search, gather"
            " evidence, collect cited papers from evidence, and gen answer. If passing"
            " tool names (non-default route), at least the gen answer tool must be"
            " supplied."
        ),
    )
    max_timesteps: int | None = Field(
        default=None,
        description="Optional upper limit on the number of environment steps.",
    )

    index_concurrency: int = Field(
        default=5,  # low default for folks without S2/Crossref keys
        description="Number of concurrent filesystem reads for indexing.",
        exclude=True,
        frozen=True,
    )
    index: IndexSettings = Field(default_factory=IndexSettings)

    rebuild_index: bool = Field(
        default=True,
        description=(
            "Flag to rebuild the index at the start of agent runners, default is True"
            " for CLI users to ensure all source PDFs are pulled in."
        ),
    )

    callbacks: Mapping[str, Sequence[Callable[[_EnvironmentState], Any]]] = Field(
        default_factory=dict,
        description="""
            A mapping that associates callback names with lists of corresponding callable functions.
            Each callback list contains functions that will be called with an instance of `EnvironmentState`,
            representing the current state context.

            Accepted callback names:
            - 'gen_answer_initialized': Triggered when `GenerateAnswer.gen_answer`
                is initialized.

            - 'gen_answer_aget_query': LLM callbacks to execute in the prompt runner
                as part of `GenerateAnswer.gen_answer`.

            - 'gen_answer_completed': Triggered after `GenerateAnswer.gen_answer`
                successfully generates an answer.

            - 'gather_evidence_initialized': Triggered when `GatherEvidence.gather_evidence`
                is initialized.

            - 'gather_evidence_aget_evidence: LLM callbacks to execute in the prompt runner
                as part of `GatherEvidence.gather_evidence`.

            - 'gather_evidence_completed': Triggered after `GatherEvidence.gather_evidence`
                completes evidence gathering.
        """,
        exclude=True,
    )

    @model_validator(mode="after")
    def _deprecated_field(self) -> Self:
        for deprecated_field_name, new_name in (("index_concurrency", "concurrency"),):
            value = getattr(self, deprecated_field_name)
            if value != type(self).model_fields[deprecated_field_name].default:
                warnings.warn(
                    f"The {deprecated_field_name!r} field has been moved to"
                    f" {AgentSettings.__name__},"
                    " this deprecation will conclude in version 6.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                setattr(self.index, new_name, value)  # Propagate to new location
        return self

    @field_validator("should_pre_search", "wipe_context_on_answer_failure")
    @classmethod
    def _deprecated_bool_fields(cls, value: bool, info) -> bool:
        custom_message = ""
        if info.field_name == "should_pre_search" and value:
            custom_message = "dead code"
        elif info.field_name == "wipe_context_on_answer_failure" and not value:
            custom_message = "no longer used due to the reset tool"
        if custom_message:
            warnings.warn(
                f"The {info.field_name!r} field is {custom_message},"
                " and will be removed in version 6.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        return value


def make_default_litellm_model_list_settings(
    llm: str, temperature: float = 0.0
) -> dict:
    """Settings matching "model_list" schema here: https://docs.litellm.ai/docs/routing."""
    return {
        "name": llm,
        "model_list": [
            {
                "model_name": llm,
                "litellm_params": {"model": llm, "temperature": temperature},
            }
        ],
    }


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    llm: str = Field(
        default=CommonLLMNames.GPT_4O.value,
        description=(
            "Default LLM for most things, including answers. Should be 'best' LLM."
        ),
    )
    llm_config: dict | None = Field(
        default=None,
        description=(
            "Optional configuration for the llm model. More specifically, it's"
            " a LiteLLM Router configuration to pass to LiteLLMModel, must have"
            " `model_list` key (corresponding to model_list inputs here:"
            " https://docs.litellm.ai/docs/routing), and can optionally include a"
            " router_kwargs key with router kwargs as values."
        ),
    )
    summary_llm: str = Field(
        default=CommonLLMNames.GPT_4O.value,
        description="Default LLM for summaries and parsing citations.",
    )
    summary_llm_config: dict | None = Field(
        default=None,
        description=(
            "Optional configuration for the summary_llm model. More specifically, it's"
            " a LiteLLM Router configuration to pass to LiteLLMModel, must have"
            " `model_list` key (corresponding to model_list inputs here:"
            " https://docs.litellm.ai/docs/routing), and can optionally include a"
            " router_kwargs key with router kwargs as values."
        ),
    )
    embedding: str = Field(
        default="text-embedding-3-small",
        description="Default embedding model for texts",
    )
    embedding_config: dict | None = Field(
        default=None,
        description="Optional configuration for the embedding model.",
    )
    temperature: float = Field(default=0.0, description="Temperature for LLMs.")
    batch_size: int = Field(default=1, description="Batch size for calling LLMs.")
    texts_index_mmr_lambda: float = Field(
        default=1.0, description="Lambda for MMR in text index."
    )
    index_absolute_directory: bool = Field(
        default=False,
        description="Whether to use the absolute paper directory for the PQA index.",
        exclude=True,
        frozen=True,
    )
    index_directory: str | os.PathLike | None = Field(
        default_factory=lambda: pqa_directory("indexes"),
        description=(
            "Directory to store the PQA generated search index, configuration, and"
            " answer indexes."
        ),
        exclude=True,
        frozen=True,
    )
    index_recursively: bool = Field(
        default=True,
        description="Whether to recurse into subdirectories when indexing sources.",
        exclude=True,
        frozen=True,
    )
    verbosity: int = Field(
        default=0,
        description=(
            "Integer verbosity level for logging (0-3). 3 = all LLM/Embeddings calls"
            " logged."
        ),
    )
    manifest_file: str | os.PathLike | None = Field(
        default=None,
        description=(
            "Optional absolute path to a manifest CSV, or a relative path from the"
            " paper_directory to a manifest CSV. A manifest CSV contains columns which"
            " are attributes for a DocDetails object. Only 'file_location', 'doi', and"
            " 'title' will be used when indexing, others are discarded."
        ),
        exclude=True,
        frozen=True,
    )
    paper_directory: str | os.PathLike = Field(
        default=pathlib.Path.cwd(),
        description=(
            "Local directory which contains the papers to be indexed and searched."
        ),
        exclude=True,
        frozen=True,
    )

    @model_validator(mode="after")
    def _deprecated_field(self) -> Self:
        for deprecated_field_name, new_name, is_factory in (
            ("index_absolute_directory", "use_absolute_paper_directory", False),
            ("index_directory", "index_directory", True),
            ("index_recursively", "recurse_subdirectories", False),
            ("manifest_file", "manifest_file", False),
            ("paper_directory", "paper_directory", False),
        ):
            value = getattr(self, deprecated_field_name)
            finfo: FieldInfo = type(self).model_fields[deprecated_field_name]
            if value != (finfo.default_factory() if is_factory else finfo.default):  # type: ignore[call-arg,misc]
                warnings.warn(
                    f"The {deprecated_field_name!r} field has been moved to"
                    f" {AgentSettings.__name__},"
                    " this deprecation will conclude in version 6.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                setattr(self.agent.index, new_name, value)  # Propagate to new location
        return self

    @model_validator(mode="after")
    def _validate_temperature_for_o1_preview(self) -> Self:
        """Ensures temperature is 1 if the LLM is 'o1-preview' or 'o1-mini'.

        o1 reasoning models only support temperature = 1.  See
        https://platform.openai.com/docs/guides/reasoning/quickstart
        """
        if self.llm.startswith("o1-") and self.temperature != 1:
            warnings.warn(
                "When dealing with OpenAI o1 models, the temperature must be set to 1."
                f" The specified temperature {self.temperature} has been overridden"
                " to 1.",
                category=UserWarning,
                stacklevel=2,
            )
            self.temperature = 1
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def md5(self) -> str:
        return hexdigest(self.model_dump_json(exclude={"md5"}))

    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    parsing: ParsingSettings = Field(default_factory=ParsingSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    def get_index_name(self) -> str:
        """Get programmatically generated index name.

        This index is where parsings are stored based on parsing/embedding strategy.
        """
        if isinstance(self.agent.index.paper_directory, pathlib.Path):
            # Here we use an absolute path so that where the user locally
            # uses '.', two different folders will make different indexes
            first_segment: str = str(self.agent.index.paper_directory.absolute())
        else:
            first_segment = str(self.agent.index.paper_directory)
        segments = [
            first_segment,
            str(self.agent.index.use_absolute_paper_directory),
            self.embedding,
            str(self.parsing.chunk_size),
            str(self.parsing.overlap),
            self.parsing.chunking_algorithm,
        ]
        return f"pqa_index_{hexdigest('|'.join(segments))}"

    @classmethod
    def from_name(
        cls, config_name: str = "default", cli_source: CliSettingsSource | None = None
    ) -> "Settings":
        json_path: pathlib.Path | None = None

        # quick exit for default settings
        if config_name == "default":
            if not cli_source:
                raise NotImplementedError(
                    f"For config_name {config_name!r}, we require cli_source."
                )
            return Settings(_cli_settings_source=cli_source(args=True))

        # First, try to find the config file in the user's .config directory
        user_config_path = pqa_directory("settings") / f"{config_name}.json"

        if user_config_path.exists():
            json_path = user_config_path

        # If not found, fall back to the package's default config
        try:
            # Use importlib.resources.files() which is recommended for Python 3.9+
            pkg_config_path = (
                importlib.resources.files("paperqa.configs") / f"{config_name}.json"
            )
            if pkg_config_path.is_file():
                json_path = cast("pathlib.Path", pkg_config_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No configuration file found for {config_name}"
            ) from e

        if json_path:
            # we do the ole switcheroo
            # json - validate to deserialize knowing the types
            # then dump it
            # going json.loads directly will not get types correct
            tmp = Settings.model_validate_json(json_path.read_text())
            return Settings(
                **(tmp.model_dump()),
                _cli_settings_source=cli_source(args=True) if cli_source else None,
            )

        raise FileNotFoundError(f"No configuration file found for {config_name}")

    def get_llm(self) -> LiteLLMModel:
        return LiteLLMModel(
            name=self.llm,
            config=self.llm_config
            or make_default_litellm_model_list_settings(self.llm, self.temperature),
        )

    def get_summary_llm(self) -> LiteLLMModel:
        return LiteLLMModel(
            name=self.summary_llm,
            config=self.summary_llm_config
            or make_default_litellm_model_list_settings(
                self.summary_llm, self.temperature
            ),
        )

    def get_agent_llm(self) -> LiteLLMModel:
        return LiteLLMModel(
            name=self.agent.agent_llm,
            config=self.agent.agent_llm_config
            or make_default_litellm_model_list_settings(
                self.agent.agent_llm, self.temperature
            ),
        )

    def get_embedding_model(self) -> EmbeddingModel:
        return embedding_model_factory(self.embedding, **(self.embedding_config or {}))

    def make_aviary_tool_selector(self, agent_type: str | type) -> ToolSelector | None:
        """Attempt to convert the input agent type to an aviary ToolSelector."""
        if agent_type is ToolSelector or (
            isinstance(agent_type, str)
            and (
                agent_type == ToolSelector.__name__
                or (
                    agent_type.startswith(
                        ToolSelector.__module__.split(".", maxsplit=1)[0]
                    )
                    and locate(agent_type) is ToolSelector
                )
            )
        ):
            return ToolSelector(
                model_name=self.agent.agent_llm,
                acompletion=self.get_agent_llm().router.acompletion,
                **(self.agent.agent_config or {}),
            )
        return None

    async def make_ldp_agent(
        self, agent_type: str | type
    ) -> "Agent[SimpleAgentState] | None":
        """Attempt to convert the input agent type to an ldp Agent."""
        if not isinstance(agent_type, str):  # Convert to fully qualified name
            agent_type = f"{agent_type.__module__}.{agent_type.__name__}"
        if not agent_type.startswith("ldp"):
            return None
        if not HAS_LDP_INSTALLED:
            raise ImportError(
                "ldp agents requires the 'ldp' extra for 'ldp'. Please:"
                " `pip install paper-qa[ldp]`."
            )

        # TODO: support general agents
        agent_cls = cast("type[Agent]", locate(agent_type))
        agent_settings = self.agent
        agent_llm, config = agent_settings.agent_llm, agent_settings.agent_config or {}
        if issubclass(agent_cls, ReActAgent | MemoryAgent):
            if (
                issubclass(agent_cls, MemoryAgent)
                and "memory_model" in config
                and "memories" in config
            ):
                if "embedding_model" in config["memory_model"]:
                    config["memory_model"]["embedding_model"] = (
                        EmbeddingModel.from_name(
                            embedding=config["memory_model"].pop("embedding_model")[
                                "name"
                            ]
                        )
                    )
                config["memory_model"] = UIndexMemoryModel(**config["memory_model"])
                memories = _Memories.validate_python(config.pop("memories"))
                await asyncio.gather(
                    *(
                        config["memory_model"].add_memory(memory)
                        for memory in (
                            memories.values()
                            if isinstance(memories, dict)
                            else memories
                        )
                    )
                )
            return agent_cls(
                llm_model={"name": agent_llm, "temperature": self.temperature},
                **config,
            )
        if issubclass(agent_cls, SimpleAgent):
            return agent_cls(
                llm_model={"name": agent_llm, "temperature": self.temperature},
                sys_prompt=agent_settings.agent_system_prompt,
                **config,
            )
        if issubclass(agent_cls, HTTPAgentClient):
            set_training_mode(False)
            return HTTPAgentClient[SimpleAgentState](
                agent_state_type=SimpleAgentState, **config
            )
        raise NotImplementedError(f"Didn't yet handle agent type {agent_type}.")

    def adjust_tools_for_agent_llm(self, tools: list[Tool]) -> None:
        """In-place adjust tool attributes or schemae to match agent LLM-specifics."""
        # This was originally made for Gemini 1.5 Flash not supporting empty tool args
        # in February 2025 (https://github.com/BerriAI/litellm/issues/7634), but then
        # Gemini fixed this server-side by mid-April 2025,
        # so this method is now just available for use


# Settings: already Settings
# dict[str, Any]: serialized Settings
# str: named Settings
# None: defaulted Settings
MaybeSettings = Settings | dict[str, Any] | str | None


def get_settings(config_or_name: MaybeSettings = None) -> Settings:
    if isinstance(config_or_name, Settings):
        return config_or_name
    if isinstance(config_or_name, dict):
        return Settings.model_validate(config_or_name)
    if config_or_name is None:
        return Settings()
    return Settings.from_name(config_name=config_or_name)



================================================
FILE: paperqa/types.py
================================================
from __future__ import annotations

import logging
import os
import re
import warnings
from collections.abc import Collection, Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any, ClassVar, cast
from uuid import UUID, uuid4

import tiktoken
from aviary.core import Message
from lmi import Embeddable, LLMResult
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
from pybtex.scanner import PybtexSyntaxError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from paperqa.utils import (
    create_bibtex_key,
    encode_id,
    format_bibtex,
    get_citenames,
    maybe_get_date,
)
from paperqa.version import __version__ as pqa_version

# Just for clarity
# also in case one day we want to narrow
# the type
DocKey = Any
logger = logging.getLogger(__name__)


VAR_MATCH_LOOKUP: Collection[str] = {"1", "true"}
VAR_MISMATCH_LOOKUP: Collection[str] = {"0", "false"}
DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA: Collection[str] = {
    "key",
    "doc_id",
    "docname",
    "dockey",
    "citation",
}


class Doc(Embeddable):
    model_config = ConfigDict(extra="forbid")

    docname: str
    dockey: DocKey
    citation: str
    fields_to_overwrite_from_metadata: set[str] = Field(
        default_factory=lambda: set(DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA),
        description="fields from metadata to overwrite when upgrading to a DocDetails",
    )

    @model_validator(mode="before")
    @classmethod
    def remove_computed_fields(cls, data: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in data.items() if k != "formatted_citation"}

    def __hash__(self) -> int:
        return hash((self.docname, self.dockey))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def formatted_citation(self) -> str:
        return self.citation

    def matches_filter_criteria(self, filter_criteria: Mapping[str, Any]) -> bool:
        """Returns True if the doc matches the filter criteria, False otherwise."""
        data_dict = self.model_dump()
        for key, value in filter_criteria.items():
            invert = key.startswith("!")
            relaxed = key.startswith("?")
            key = key.lstrip("!?")
            # we check if missing or sentinel/unset
            if relaxed and (key not in data_dict or data_dict[key] is None):
                continue
            if key not in data_dict:
                return False
            if invert and data_dict[key] == value:
                return False
            if not invert and data_dict[key] != value:
                return False
        return True


class Text(Embeddable):
    text: str
    name: str
    doc: Doc | DocDetails = Field(union_mode="left_to_right")

    def __hash__(self) -> int:
        return hash(self.text)


class Context(BaseModel):
    """A class to hold the context of a question."""

    model_config = ConfigDict(extra="allow")

    context: str = Field(description="Summary of the text with respect to a question.")
    question: str | None = Field(
        default=None,
        description=(
            "Question that the context is summarizing for. "
            "Note this can differ from the user query."
        ),
    )
    text: Text
    score: int = 5

    def __str__(self) -> str:
        """Return the context as a string."""
        return self.context


class PQASession(BaseModel):
    """A class to hold session about researching/answering."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: UUID = Field(default_factory=uuid4)
    question: str
    answer: str = ""
    answer_reasoning: str | None = Field(
        default=None,
        description=(
            "Optional reasoning from the LLM. If the LLM does not support reasoning,"
            " it will be None."
        ),
    )
    has_successful_answer: bool | None = Field(
        default=None,
        description=(
            "True if the agent was sure of the answer, False if the agent was unsure of"
            " the answer, and None if the agent hasn't yet completed."
        ),
    )
    context: str = ""
    contexts: list[Context] = Field(default_factory=list)
    references: str = ""
    formatted_answer: str = Field(
        default="",
        description=(
            "Optional prettified answer that includes information like question and"
            " citations."
        ),
    )
    graded_answer: str | None = Field(
        default=None,
        description=(
            "Optional graded answer, used for things like multiple choice questions."
        ),
    )
    cost: float = 0.0
    # Map model name to a two-item list of LLM prompt token counts
    # and LLM completion token counts
    token_counts: dict[str, list[int]] = Field(default_factory=dict)
    config_md5: str | None = Field(
        default=None,
        frozen=True,
        description=(
            "MD5 hash of the settings used to generate the answer. Cannot change"
        ),
    )
    tool_history: list[list[str]] = Field(
        default_factory=list,
        description=(
            "History of tool names input to each Environment.step (regardless of being"
            " a typo or not), where the outer list is steps, and the inner list matches"
            " the order of tool calls at each step."
        ),
    )

    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer

    @model_validator(mode="before")
    @classmethod
    def remove_computed(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data.pop("used_contexts", None)
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def used_contexts(self) -> set[str]:
        """Return the used contexts."""
        return get_citenames(self.formatted_answer)

    def get_citation(self, name: str) -> str:
        """Return the formatted citation for the given docname."""
        try:
            doc: Doc = next(
                filter(lambda x: x.text.name == name, self.contexts)
            ).text.doc
        except StopIteration as exc:
            raise ValueError(f"Could not find docname {name} in contexts.") from exc
        return doc.citation

    def add_tokens(self, result: LLMResult | Message) -> None:
        """Update the token counts for the given LLM result or message."""
        if isinstance(result, Message):
            if not result.info or any(x not in result.info for x in ("model", "usage")):
                return
            result = LLMResult(
                model=result.info["model"],
                prompt_count=result.info["usage"][0],
                completion_count=result.info["usage"][1],
            )
        if result.model not in self.token_counts:
            self.token_counts[result.model] = [
                result.prompt_count,
                result.completion_count,
            ]
        else:
            self.token_counts[result.model][0] += result.prompt_count
            self.token_counts[result.model][1] += result.completion_count

        self.cost += result.cost

    def get_unique_docs_from_contexts(self, score_threshold: int = 0) -> set[Doc]:
        """Parse contexts for docs with scores above the input threshold."""
        return {
            c.text.doc
            for c in filter(lambda x: x.score >= score_threshold, self.contexts)
        }

    def filter_content_for_user(self) -> None:
        """Filter out extra items (inplace) that do not need to be returned to the user."""
        self.contexts = [
            Context(
                context=c.context,
                question=c.question,
                score=c.score,
                text=Text(
                    text="",
                    **c.text.model_dump(exclude={"text", "embedding", "doc"}),
                    doc=c.text.doc.model_dump(exclude={"embedding"}),
                ),
            )
            for c in self.contexts
        ]


# for backwards compatibility
class Answer(PQASession):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The 'Answer' class is deprecated and will be removed in future versions."
            " Use 'PQASession' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class ChunkMetadata(BaseModel):
    """Metadata for chunking algorithm."""

    chunk_chars: int
    overlap: int
    chunk_type: str


class ParsedMetadata(BaseModel):
    """Metadata for parsed text."""

    parsing_libraries: list[str]
    total_parsed_text_length: int
    paperqa_version: str = pqa_version
    parse_type: str | None = None
    chunk_metadata: ChunkMetadata | None = None


class ParsedText(BaseModel):
    """Parsed text (pre-chunking)."""

    content: dict | str | list[str]
    metadata: ParsedMetadata

    def encode_content(self):
        # we tokenize using tiktoken so cuts are in reasonable places
        # See https://github.com/openai/tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        if isinstance(self.content, str):
            return enc.encode_ordinary(self.content)
        elif isinstance(self.content, list):  # noqa: RET505
            return [enc.encode_ordinary(c) for c in self.content]
        else:
            raise NotImplementedError(
                "Encoding only implemented for str and list[str] content."
            )

    def reduce_content(self) -> str:
        """Reduce any content to a string."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "\n\n".join(self.content)
        return "\n\n".join(self.content.values())


# We use these integer values
# as defined in https://jfp.csc.fi/en/web/haku/kayttoohje
# which is a recommended ranking system
SOURCE_QUALITY_MESSAGES = {
    0: "poor quality or predatory journal",
    1: "peer-reviewed journal",
    2: "domain leading peer-reviewed journal",
    3: "highest quality peer-reviewed journal",
}

CITATION_FALLBACK_DATA = {
    "authors": ["Unknown authors"],
    "author": "Unknown author(s)",
    "year": "Unknown year",
    "title": "Unknown title",
    "journal": "Unknown journal",
}

JOURNAL_EXPECTED_DOI_LENGTHS = {
    "BioRxiv": 25,
    "MedRxiv": 27,
}


class DocDetails(Doc):
    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    # Sentinel to auto-populate a field within model_validator
    AUTOPOPULATE_VALUE: ClassVar[str] = ""

    docname: str = AUTOPOPULATE_VALUE
    dockey: DocKey = AUTOPOPULATE_VALUE
    citation: str = AUTOPOPULATE_VALUE
    key: str | None = None
    bibtex: str | None = Field(
        default=AUTOPOPULATE_VALUE,
        description="Autogenerated from other represented fields.",
    )
    authors: list[str] | None = None
    publication_date: datetime | None = None
    year: int | None = None
    volume: str | None = None
    issue: str | None = None  # TODO: in bibtex this may be "number"
    issn: str | None = None
    pages: str | None = None
    journal: str | None = None
    publisher: str | None = None
    url: str | None = Field(
        default=None,
        description=(
            "Optional URL to the paper, which can lead to a Semantic Scholar page,"
            " arXiv abstract, etc. As of version 0.67 on 5/10/2024, we don't use this"
            " URL anywhere in the source code."
        ),
    )
    title: str | None = None
    citation_count: int | None = None
    bibtex_type: str | None = None

    source_quality: int | None = Field(
        default=None,
        description=(
            "Quality of journal/venue of paper.  We use None as a sentinel for unset"
            " values (like for determining hydration)  So, we use -1 means unknown"
            " quality and None means it needs to be hydrated."
        ),
    )

    is_retracted: bool | None = Field(
        default=None, description="Flag for whether the paper is retracted."
    )
    doi: str | None = None
    doi_url: str | None = None
    doc_id: str | None = None
    file_location: str | os.PathLike | None = None
    license: str | None = Field(
        default=None,
        description=(
            "string indicating license. Should refer specifically to pdf_url (since"
            " that could be preprint). None means unknown/unset."
        ),
    )
    pdf_url: str | None = None
    other: dict[str, Any] = Field(
        default_factory=dict,
        description="Other metadata besides the above standardized fields.",
    )
    UNDEFINED_JOURNAL_QUALITY: ClassVar[int] = -1
    # NOTE: can use a regex starting from the pattern in https://regex101.com/r/lpF1up/1
    DOI_URL_FORMATS: ClassVar[Collection[str]] = {
        "https://doi.org/",
        "http://dx.doi.org/",
    }
    AUTHOR_NAMES_TO_REMOVE: ClassVar[Collection[str]] = {"et al", "et al."}

    @field_validator("key")
    @classmethod
    def clean_key(cls, value: str) -> str:
        # Replace HTML tags with empty string
        return re.sub(pattern=r"<\/?\w{1,10}>", repl="", string=value)

    @classmethod
    def lowercase_doi_and_populate_doc_id(cls, data: dict[str, Any]) -> dict[str, Any]:
        doi: str | list[str] | None = data.get("doi")
        if isinstance(doi, list):
            if len(doi) != 1:
                logger.warning(
                    f"Discarding list of DOIs {doi} due to it not having one value,"
                    f" full data was {data}."
                )
                doi = None
            else:
                doi = doi[0]
        if doi:
            for url_prefix_to_remove in cls.DOI_URL_FORMATS:
                if doi.startswith(url_prefix_to_remove):
                    doi = doi.replace(url_prefix_to_remove, "")
            data["doi"] = doi.lower()
            data["doc_id"] = encode_id(doi.lower())
        else:
            data["doc_id"] = encode_id(uuid4())

        if "dockey" in data.get(
            "fields_to_overwrite_from_metadata",
            DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA,
        ):
            data["dockey"] = data["doc_id"]

        return data

    @staticmethod
    def is_bibtex_complete(bibtex: str, fields: list[str] | None = None) -> bool:
        """Validate bibtex entries have certain fields."""
        if fields is None:
            fields = ["doi", "title"]
        return all(field + "=" in bibtex for field in fields)

    @staticmethod
    def merge_bibtex_entries(entry1: Entry, entry2: Entry) -> Entry:
        """Merge two bibtex entries into one, preferring entry2 fields."""
        merged_entry = Entry(entry1.type)

        for field, value in entry1.fields.items():
            merged_entry.fields[field] = value
        for field, value in entry2.fields.items():
            merged_entry.fields[field] = value

        return merged_entry

    @staticmethod
    def misc_string_cleaning(data: dict[str, Any]) -> dict[str, Any]:
        """Clean strings before the enter the validation process."""
        if pages := data.get("pages"):
            data["pages"] = pages.replace("--", "-").replace(" ", "")
        return data

    @staticmethod
    def inject_clean_doi_url_into_data(data: dict[str, Any]) -> dict[str, Any]:
        """Ensure doi_url is present in data (since non-default arguments are not included)."""
        doi_url, doi = data.get("doi_url"), data.get("doi")

        if doi and not doi_url:
            doi_url = "https://doi.org/" + doi

        # ensure the modern doi url is used
        if doi_url:
            data["doi_url"] = doi_url.replace(
                "http://dx.doi.org/", "https://doi.org/"
            ).lower()

        return data

    @staticmethod
    def add_preprint_journal_from_doi_if_missing(
        data: dict[str, Any],
    ) -> dict[str, Any]:
        if not data.get("journal"):
            doi = data.get("doi", "") or ""
            if "10.48550/" in doi or "ArXiv" in (
                (data.get("other", {}) or {}).get("externalIds", {}) or {}
            ):
                data["journal"] = "ArXiv"
            elif "10.26434/" in doi:
                data["journal"] = "ChemRxiv"
            elif (
                "10.1101/" in doi
                and len(data.get("doi", "")) == JOURNAL_EXPECTED_DOI_LENGTHS["BioRxiv"]
            ):
                data["journal"] = "BioRxiv"
            elif (
                "10.1101/" in doi
                and len(data.get("doi", "")) == JOURNAL_EXPECTED_DOI_LENGTHS["MedRxiv"]
            ):
                data["journal"] = "MedRxiv"
            elif "10.31224/" in doi:
                data["journal"] = "EngRxiv"
        return data

    @classmethod
    def remove_invalid_authors(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Capture and cull strange author names."""
        if authors := data.get("authors"):
            # On 10/29/2024 while indexing 19k PDFs, a provider (unclear which one)
            # returned an author of None. The vast majority of the time authors are str
            authors = cast("list[str | None]", authors)
            data["authors"] = [
                a for a in authors if a and a.lower() not in cls.AUTHOR_NAMES_TO_REMOVE
            ]

        return data

    @staticmethod
    def overwrite_docname_dockey_for_compatibility_w_doc(
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Overwrite fields from metadata if specified."""
        overwrite_fields = {"key": "docname", "doc_id": "dockey"}
        fields_to_overwrite = data.get(
            "fields_to_overwrite_from_metadata",
            DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA,
        )
        for field in overwrite_fields.keys() & fields_to_overwrite:
            if data.get(field):
                data[overwrite_fields[field]] = data[field]
        return data

    @classmethod
    def populate_bibtex_key_citation(  # noqa: PLR0912
        cls, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Add or modify bibtex, key, and citation fields.

        Missing values, 'unknown' keys, and incomplete bibtex entries are regenerated.

        When fields_to_overwrite_from_metadata:
            If bibtex is regenerated, the citation field is also regenerated.

            Otherwise we keep the citation field as is.

        """
        # we try to regenerate the key if unknowns are present, maybe they have been found
        if not data.get("key") or "unknown" in data["key"].lower():
            data["key"] = create_bibtex_key(
                data.get("authors") or CITATION_FALLBACK_DATA["authors"],  # type: ignore[arg-type]
                data.get("year") or CITATION_FALLBACK_DATA["year"],  # type: ignore[arg-type]
                data.get("title") or CITATION_FALLBACK_DATA["title"],  # type: ignore[arg-type]
            )
            if "docname" in data.get(
                "fields_to_overwrite_from_metadata",
                DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA,
            ):
                data["docname"] = data["key"]

        # even if we have a bibtex, it may not be complete, thus we need to add to it
        if not data.get("bibtex") or not cls.is_bibtex_complete(data["bibtex"]):
            existing_entry = None
            # if our bibtex already exists, but is incomplete, we add self_generated to metadata
            if data.get("bibtex"):
                if data.get("other"):
                    if (
                        "bibtex_source" in data["other"]
                        and "self_generated" not in data["other"]["bibtex_source"]
                    ):
                        data["other"]["bibtex_source"].append("self_generated")
                    else:
                        data["other"]["bibtex_source"] = ["self_generated"]
                else:
                    data["other"] = {"bibtex_source": ["self_generated"]}
                try:
                    existing_entry = next(
                        iter(Parser().parse_string(data["bibtex"]).entries.values())
                    )
                except PybtexSyntaxError:
                    logger.warning(f"Failed to parse bibtex for {data['bibtex']}.")
                    existing_entry = None

            entry_data = {
                "title": data.get("title") or CITATION_FALLBACK_DATA["title"],
                "year": (
                    CITATION_FALLBACK_DATA["year"]
                    if not data.get("year")
                    else str(data["year"])
                ),
                "journal": data.get("journal") or CITATION_FALLBACK_DATA["journal"],
                "volume": data.get("volume"),
                "pages": data.get("pages"),
                "month": (
                    None
                    if not (maybe_date := maybe_get_date(data.get("publication_date")))
                    else maybe_date.strftime("%b")
                ),
                "doi": data.get("doi"),
                "url": data.get("doi_url"),
                "publisher": data.get("publisher"),
                "issue": data.get("issue"),
                "issn": data.get("issn"),
            }
            entry_data = {k: v for k, v in entry_data.items() if v}
            try:
                new_entry = Entry(
                    data.get("bibtex_type", "article") or "article", fields=entry_data
                )
                if existing_entry:
                    new_entry = cls.merge_bibtex_entries(existing_entry, new_entry)
                # add in authors manually into the entry
                authors = [Person(a) for a in data.get("authors", ["Unknown authors"])]
                for a in authors:
                    new_entry.add_person(a, "author")
                data["bibtex"] = BibliographyData(
                    entries={data["key"]: new_entry}
                ).to_string("bibtex")
                # clear out the citation, since it will be regenerated
                if "citation" in data.get(
                    "fields_to_overwrite_from_metadata",
                    DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA,
                ):
                    data["citation"] = None
            except Exception:
                logger.warning(
                    "Failed to generate bibtex for"
                    f" {data.get('docname') or data.get('citation')}"
                )
        if data.get("citation") is None and data.get("bibtex") is not None:
            data["citation"] = format_bibtex(
                data["bibtex"], missing_replacements=CITATION_FALLBACK_DATA  # type: ignore[arg-type]
            )
        elif data.get("citation") is None:
            data["citation"] = data.get("title") or CITATION_FALLBACK_DATA["title"]
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_all_fields(cls, data: Mapping[str, Any]) -> dict[str, Any]:

        data = deepcopy(data)  # Avoid mutating input
        data = dict(data)
        if isinstance(data.get("fields_to_overwrite_from_metadata"), str):
            data["fields_to_overwrite_from_metadata"] = {
                s.strip()
                for s in data.get("fields_to_overwrite_from_metadata", "").split(",")
            }
        data = cls.lowercase_doi_and_populate_doc_id(data)
        data = cls.remove_invalid_authors(data)
        data = cls.misc_string_cleaning(data)
        data = cls.inject_clean_doi_url_into_data(data)
        data = cls.add_preprint_journal_from_doi_if_missing(data)
        data = cls.populate_bibtex_key_citation(data)
        return cls.overwrite_docname_dockey_for_compatibility_w_doc(data)

    def __getitem__(self, item: str):
        """Allow for dictionary-like access, falling back on other."""
        try:
            return getattr(self, item)
        except AttributeError:
            return self.other[item]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def formatted_citation(self) -> str:

        if self.is_retracted:
            base_message = "**RETRACTED ARTICLE**"
            retract_info = "Retrieved from http://retractiondatabase.org/."
            citation_message = (
                f"Citation: {self.citation}"
                if self.citation
                else f"Original DOI: {self.doi}"
            )
            return f"{base_message} {citation_message} {retract_info}"

        if self.citation_count is None or self.source_quality is None:
            logger.debug("citation_count and source_quality are not set.")
            return self.citation

        if self.source_quality_message:
            return (
                f"{self.citation} This article has {self.citation_count} citations and"
                f" is from a {self.source_quality_message}."
            )
        return f"{self.citation} This article has {self.citation_count} citations."

    @property
    def source_quality_message(self) -> str:
        return (
            SOURCE_QUALITY_MESSAGES[self.source_quality]
            if self.source_quality is not None
            and self.source_quality
            != DocDetails.UNDEFINED_JOURNAL_QUALITY  # note - zero is a valid value
            else ""
        )

    OPTIONAL_HYDRATION_FIELDS: ClassVar[Collection[str]] = {"url"}

    def is_hydration_needed(
        self,
        exclusion: Collection[str] = OPTIONAL_HYDRATION_FIELDS,
        inclusion: Collection[str] = (),
    ) -> bool:
        """Determine if we have unfilled attributes."""
        if inclusion:
            return any(
                v is None for k, v in self.model_dump().items() if k in inclusion
            )
        return any(
            v is None for k, v in self.model_dump().items() if k not in exclusion
        )

    def repopulate_doc_id_from_doi(self) -> None:
        # TODO: should this be a hash of the doi?
        if self.doi:
            self.doc_id = encode_id(self.doi)

    def __add__(self, other: DocDetails | int) -> DocDetails:  # noqa: PLR0912
        """Merge two DocDetails objects together."""
        # control for usage w. Python's sum() function
        if isinstance(other, int):
            return self

        # first see if one of the entries is newer, which we will prefer
        PREFER_OTHER = True
        if self.publication_date and other.publication_date:
            PREFER_OTHER = self.publication_date <= other.publication_date

        merged_data = {}
        # pylint: disable-next=not-an-iterable  # pylint bug: https://github.com/pylint-dev/pylint/issues/10144
        for field in type(self).model_fields:
            self_value = getattr(self, field)
            other_value = getattr(other, field)

            if field == "other":
                # Merge 'other' dictionaries
                merged_data[field] = {**self.other, **other.other}
                # handle the bibtex / sources as special fields
                for field_to_combine in ("bibtex_source", "client_source"):
                    # Ensure the fields are lists before combining
                    if self.other.get(field_to_combine) and not isinstance(
                        self.other[field_to_combine], list
                    ):
                        self.other[field_to_combine] = [self.other[field_to_combine]]
                    if other.other.get(field_to_combine) and not isinstance(
                        other.other[field_to_combine], list
                    ):
                        other.other[field_to_combine] = [other.other[field_to_combine]]

                    if self.other.get(field_to_combine) and other.other.get(
                        field_to_combine
                    ):
                        # Note: these should always be lists
                        merged_data[field][field_to_combine] = (
                            self.other[field_to_combine] + other.other[field_to_combine]
                        )

            elif field == "authors" and self_value and other_value:
                # Combine authors lists, removing duplicates
                # Choose whichever author list is longer
                best_authors = (
                    self.authors
                    if (
                        sum(len(a) for a in (self.authors or []))
                        >= sum(len(a) for a in (other.authors or []))
                    )
                    else other.authors
                )
                merged_data[field] = best_authors or None  # type: ignore[assignment]

            elif field == "key" and self_value is not None and other_value is not None:
                # if we have multiple keys, we wipe them and allow regeneration
                merged_data[field] = None  # type: ignore[assignment]

            elif field in {"citation_count", "year", "publication_date"}:
                # get the latest data
                # this conditional is written in a way to handle if multiple doc objects
                # are provided, we'll use the highest value
                # if there's only one valid value, we'll use that regardless even if
                # that value is 0
                if self_value is None or other_value is None:
                    merged_data[field] = (
                        self_value
                        if self_value is not None  # Dance around 0
                        else other_value
                    )
                else:
                    merged_data[field] = max(self_value, other_value)

            else:
                # Prefer non-null values, default preference for 'other' object.
                # Note: if PREFER_OTHER = False then even if 'other' data exists
                # we will use 'self' data. This is to control for when we have
                # pre-prints / arXiv versions of papers that are not as up-to-date
                merged_data[field] = (
                    other_value
                    if (
                        (other_value is not None and other_value != []) and PREFER_OTHER
                    )
                    else self_value
                )

        # Recalculate doc_id if doi has changed
        if merged_data["doi"] != self.doi:
            merged_data["doc_id"] = (
                encode_id(merged_data["doi"].lower()) if merged_data["doi"] else None  # type: ignore[attr-defined,assignment]
            )

        # Create and return new DocDetails instance
        return DocDetails(**merged_data)

    def __radd__(self, other: DocDetails | int) -> DocDetails:
        # other == 0 captures the first call of sum()
        if isinstance(other, int) and other == 0:
            return self
        return self.__add__(other)

    def __iadd__(self, other: DocDetails | int) -> DocDetails:  # noqa: PYI034
        # only includes int to align with __radd__ and __add__
        if isinstance(other, int):
            return self
        return self.__add__(other)



================================================
FILE: paperqa/utils.py
================================================
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import logging.config
import math
import os
import re
import string
import unicodedata
from collections.abc import Awaitable, Collection, Iterable, Iterator
from datetime import datetime
from functools import reduce
from http import HTTPStatus
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, TypeVar
from uuid import UUID

import aiohttp
import httpx
import pymupdf
from lmi import configure_llm_logs
from pybtex.database import Person, parse_string
from pybtex.database.input.bibtex import Parser
from pybtex.style.formatting import unsrtalpha
from pybtex.style.template import FieldIsMissing
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_incrementing,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ImpossibleParsingError(Exception):
    """Error to throw when a parsing is impossible."""

    LOG_METHOD_NAME: ClassVar[str] = "warning"


def name_in_text(name: str, text: str) -> bool:
    sname = name.strip()
    pattern = rf"\b({re.escape(sname)})\b(?!\w)"
    return bool(re.search(pattern, text))


def maybe_is_text(s: str, thresh: float = 2.5) -> bool:
    """
    Calculate the entropy of the string to discard files with excessively repeated symbols.

    PDF parsing sometimes represents horizontal distances between words on title pages
    and in tables with spaces, which should therefore not be included in this calculation.
    """
    if not s:
        return False

    entropy = 0.0
    s_wo_spaces = s.replace(" ", "")
    for c in string.printable:
        p = s_wo_spaces.count(c) / len(s_wo_spaces)
        if p > 0:
            entropy += -p * math.log2(p)

    # Check if the entropy is within a reasonable range for text
    return entropy > thresh


def maybe_is_pdf(file: BinaryIO) -> bool:
    magic_number = file.read(4)
    file.seek(0)
    return magic_number == b"%PDF"


def maybe_is_html(file: BinaryIO) -> bool:
    magic_number = file.read(4)
    file.seek(0)
    return magic_number in {b"<htm", b"<!DO", b"<xsl", b"<!X"}


def strings_similarity(s1: str, s2: str, case_insensitive: bool = True) -> float:
    if not s1 or not s2:
        return 0

    # break the strings into words
    ss1 = set(s1.lower().split()) if case_insensitive else set(s1.split())
    ss2 = set(s2.lower().split()) if case_insensitive else set(s2.split())

    # return the similarity ratio
    return len(ss1.intersection(ss2)) / len(ss1.union(ss2))


def count_pdf_pages(file_path: str | os.PathLike) -> int:
    with pymupdf.open(file_path) as doc:
        return len(doc)


def hexdigest(data: str | bytes) -> str:
    if isinstance(data, str):
        return hashlib.md5(data.encode("utf-8")).hexdigest()  # noqa: S324
    return hashlib.md5(data).hexdigest()  # noqa: S324


def md5sum(file_path: str | os.PathLike) -> str:
    return hexdigest(Path(file_path).read_bytes())


def strip_citations(text: str) -> str:
    # Combined regex for identifying citations (see unit tests for examples)
    citation_regex = r"\b[\w\-]+\set\sal\.\s\([0-9]{4}\)|\((?:[^\)]*?[a-zA-Z][^\)]*?[0-9]{4}[^\)]*?)\)"
    # Remove the citations from the text
    return re.sub(citation_regex, "", text, flags=re.MULTILINE)


def extract_score(text: str) -> int:
    """
    Extract an integer score from the text in 0 to 10.

    Note: score is 1-10, and we use 0 as a sentinel for not applicable.
    """
    # Check for N/A, not applicable, not relevant.
    # Don't check for NA, as there can be genes containing "NA"
    last_line = text.split("\n")[-1]
    if (
        "n/a" in last_line.lower()
        or "not applicable" in text.lower()
        or "not relevant" in text.lower()
    ):
        return 0

    score = re.search(r"[sS]core[:is\s]+([0-9]+)", text)
    if not score:
        score = re.search(r"\(([0-9])\w*\/", text)
    if not score:
        score = re.search(r"([0-9]+)\w*\/", text)
    if score:
        s = int(score.group(1))
        if s > 10:  # noqa: PLR2004
            s = int(s / 10)  # sometimes becomes out of 100
        return s
    last_few = text[-15:]
    scores = re.findall(r"([0-9]+)", last_few)
    if scores:
        s = int(scores[-1])
        if s > 10:  # noqa: PLR2004
            s = int(s / 10)  # sometimes becomes out of 100
        return s
    if len(text) < 100:  # noqa: PLR2004
        return 1
    return 5


def get_citenames(text: str) -> set[str]:
    # Combined regex for identifying citations (see unit tests for examples)
    citation_regex = r"\b[\w\-]+\set\sal\.\s\([0-9]{4}\)|\((?:[^\)]*?[a-zA-Z][^\)]*?[0-9]{4}[^\)]*?)\)"
    results = re.findall(citation_regex, text, flags=re.MULTILINE)
    # now find None patterns
    none_citation_regex = r"(\(None[a-f]{0,1} pages [0-9]{1,10}-[0-9]{1,10}\))"
    none_results = re.findall(none_citation_regex, text, flags=re.MULTILINE)
    results.extend(none_results)
    values = []
    for citation in results:
        citation = citation.strip("() ")
        for c in re.split(r",|;", citation):
            if c == "Extra background information":
                continue
            # remove leading/trailing spaces
            c = c.strip()
            values.append(c)
    return set(values)


def extract_doi(reference: str) -> str:
    """
    Extracts DOI from the reference string using regex.

    :param reference: A string containing the reference.
    :return: A string containing the DOI link or a message if DOI is not found.
    """
    # DOI regex pattern
    doi_pattern = r"10.\d{4,9}/[-._;()/:A-Z0-9]+"
    doi_match = re.search(doi_pattern, reference, re.IGNORECASE)

    # If DOI is found in the reference, return the DOI link
    if doi_match:
        return "https://doi.org/" + doi_match.group()
    return ""


def batch_iter(iterable: list, n: int = 1) -> Iterator[list]:
    """
    Batch an iterable into chunks of size n.

    :param iterable: The iterable to batch
    :param n: The size of the batches
    :return: A list of batches
    """
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


def get_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def run_or_ensure(coro: Awaitable[T]) -> T | asyncio.Task[T]:
    """Run a coroutine or convert to a future if an event loop is running."""
    loop = get_loop()
    if loop.is_running():  # In async contexts (e.g., Jupyter notebook), return a Task
        return asyncio.ensure_future(coro)
    return loop.run_until_complete(coro)


def encode_id(value: str | bytes | UUID, maxsize: int | None = 16) -> str:
    """Encode a value (e.g. a DOI) optionally with a max length."""
    if isinstance(value, UUID):
        value = str(value)
    if isinstance(value, str):
        value = value.lower().encode()
    return hashlib.md5(value).hexdigest()[:maxsize]  # noqa: S324


def get_year(ts: datetime | None = None) -> str:
    """Get the year from the input datetime, otherwise using the current datetime."""
    if ts is None:
        ts = datetime.now()
    return ts.strftime("%Y")


class CitationConversionError(Exception):
    """Exception to throw when we can't process a citation from a BibTeX."""


def clean_upbibtex(bibtex: str) -> str:

    if not bibtex:
        return bibtex

    mapping = {
        "None": "article",
        "Article": "article",
        "JournalArticle": "article",
        "Review": "article",
        "Book": "book",
        "BookSection": "inbook",
        "ConferencePaper": "inproceedings",
        "Conference": "inproceedings",
        "Dataset": "misc",
        "Dissertation": "phdthesis",
        "Journal": "article",
        "Patent": "patent",
        "Preprint": "article",
        "Report": "techreport",
        "Thesis": "phdthesis",
        "WebPage": "misc",
        "Plain": "article",
    }
    if "@None" in bibtex:
        return bibtex.replace("@None", "@article")
    match = re.findall(r"@\['(.*)'\]", bibtex)
    if not match:
        match = re.findall(r"@(\w+)\{", bibtex)
        bib_type = match[0]
        current = f"@{match[0]}"
    else:
        bib_type = match[0]
        current = f"@['{bib_type}']"
    for k, v in mapping.items():
        # can have multiple
        if k in bib_type:
            bibtex = bibtex.replace(current, f"@{v}")
            break
    return bibtex


def format_bibtex(
    bibtex: str,
    key: str | None = None,
    clean: bool = True,
    missing_replacements: dict[str, str] | None = None,
) -> str:
    """Transform bibtex entry into a citation, potentially adding missing fields."""
    if missing_replacements is None:
        missing_replacements = {}
    if key is None:
        key = bibtex.split("{")[1].split(",")[0]
    style = unsrtalpha.Style()
    try:
        bd = parse_string(clean_upbibtex(bibtex) if clean else bibtex, "bibtex")
    except Exception:
        return "Ref " + key
    try:
        entry = bd.entries[key]
    except KeyError as exc:  # Let's check if key is a non-empty prefix
        try:
            entry = next(
                iter(v for k, v in bd.entries.items() if k.startswith(key) and key)
            )
        except StopIteration:
            raise CitationConversionError(
                f"Failed to process{' and clean up' if clean else ''} bibtex {bibtex}"
                f" due to failed lookup of key {key}."
            ) from exc
    try:
        # see if we can insert missing fields
        for field, replacement_value in missing_replacements.items():
            # Deal with special case for author, since it needs to be parsed
            # into Person objects. This reorganizes the names automatically.
            if field == "author" and "author" not in entry.persons:
                tmp_author_bibtex = f"@misc{{tmpkey, author={{{replacement_value}}}}}"
                authors: list[Person] = (
                    Parser()
                    .parse_string(tmp_author_bibtex)
                    .entries["tmpkey"]
                    .persons["author"]
                )
                for a in authors:
                    entry.add_person(a, "author")
            elif field not in entry.fields:
                entry.fields.update({field: replacement_value})
        entry = style.format_entry(label="1", entry=entry)
        return entry.text.render_as("text")
    except (FieldIsMissing, UnicodeDecodeError):
        try:
            return entry.fields["title"]
        except KeyError as exc:
            raise CitationConversionError(
                f"Failed to process{' and clean up' if clean else ''} bibtex {bibtex}"
                " due to missing a 'title' field."
            ) from exc


def remove_substrings(target: str, substr_removal_list: Collection[str]) -> str:
    """Remove substrings from a target string."""
    if all(len(w) == 1 for w in substr_removal_list):
        return target.translate(str.maketrans("", "", "".join(substr_removal_list)))

    for substr in substr_removal_list:
        target = target.replace(substr, "")
    return target


def mutate_acute_accents(text: str, replace: bool = False) -> str:
    """
    Replaces or removes acute accents in a string based on the boolean flag.

    Args:
        text: The input string.
        replace: A flag to determine whether to replace (True) or remove (False) acute accents.

            If 'replace' is True, acute accents on vowels are replaced with an apostrophe (e.g., "á" becomes "'a").

            If 'replace' is False, all acute accents are removed from the string.

    Returns:
        The modified string with acute accents either replaced or removed.
    """
    if replace:

        def replace_acute(match):
            return f"'{match.group(1)}"

        nfd = unicodedata.normalize("NFD", text)
        converted = re.sub(r"([aeiouAEIOU])\u0301", replace_acute, nfd)
        return unicodedata.normalize("NFC", converted)
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def bibtex_field_extract(
    bibtex: str, field: str, missing_replacements: dict[str, str] | None = None
) -> str:
    """Get a field from a bibtex entry.

    Args:
        bibtex: bibtex entry
        field: field to extract
        missing_replacements: replacement extract for field if not present in the bibtex string
    """
    if missing_replacements is None:
        missing_replacements = {}
    try:
        pattern = rf"{field}\s*=\s*{{(.*?)}},"
        # note: we intentionally have an attribute error if no match
        return re.search(pattern, bibtex, re.IGNORECASE).group(1).strip()  # type: ignore[union-attr]
    except AttributeError:
        return missing_replacements.get(field, "")


UNKNOWN_AUTHOR_KEY: str = "unknownauthors"


def create_bibtex_key(author: list[str], year: str, title: str) -> str:
    FORBIDDEN_KEY_CHARACTERS = {"_", " ", "-", "/", "'", "`", ":", ",", "\n"}
    try:
        author_rep = (
            # casefold will not remove accutes
            mutate_acute_accents(text=author[0].split()[-1].casefold())
            if "Unknown" not in author[0]
            else UNKNOWN_AUTHOR_KEY
        )
    except IndexError:
        author_rep = UNKNOWN_AUTHOR_KEY
    # we don't want a bibtex-parsing induced line break in the key
    # so we cap it to 100+50+4 = 154 characters max
    # 50 for the author, 100 for the first three title words, 4 for the year
    # the first three title words are just emulating the s2 convention
    key = f"{author_rep[:50]}{year}{''.join([t.casefold() for t in title.split()[:3]])[:100]}"
    return remove_substrings(key, FORBIDDEN_KEY_CHARACTERS)


def is_retryable(exc: BaseException) -> bool:
    """Check if an exception is known to be a retryable HTTP issue."""
    if isinstance(
        exc, aiohttp.ServerDisconnectedError | aiohttp.ClientConnectionResetError
    ):
        # Seen with Semantic Scholar:
        # > aiohttp.client_exceptions.ClientConnectionResetError:
        # > Cannot write to closing transport
        return True
    return isinstance(exc, aiohttp.ClientResponseError) and exc.status in {
        httpx.codes.INTERNAL_SERVER_ERROR.value,
        httpx.codes.GATEWAY_TIMEOUT.value,
    }


@retry(
    retry=retry_if_exception(is_retryable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    stop=stop_after_attempt(5),
    wait=wait_incrementing(0.1, 0.1),
)
async def _get_with_retrying(
    url: str,
    session: aiohttp.ClientSession,
    http_exception_mappings: dict[HTTPStatus | int, Exception] | None = None,
    **get_kwargs,
) -> dict[str, Any]:
    """Get from a URL with retrying protection."""
    try:
        async with session.get(url, **get_kwargs) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientResponseError as e:
        if http_exception_mappings and e.status in http_exception_mappings:
            raise http_exception_mappings[e.status] from e
        raise


def union_collections_to_ordered_list(collections: Iterable) -> list:
    return sorted(reduce(lambda x, y: set(x) | set(y), collections))


def pqa_directory(name: str) -> Path:
    if pqa_home := os.environ.get("PQA_HOME"):
        directory = Path(pqa_home) / ".pqa" / name
    else:
        directory = Path.home() / ".pqa" / name

    directory.mkdir(parents=True, exist_ok=True)
    return directory


def setup_default_logs() -> None:
    """Configure logs to reasonable defaults."""
    # Trigger PyMuPDF to use Python logging
    # SEE: https://pymupdf.readthedocs.io/en/latest/app3.html#diagnostics
    pymupdf.set_messages(pylogging=True)
    configure_llm_logs()


def extract_thought(content: str | None) -> str:
    """Extract an Anthropic thought from a message's content."""
    # SEE: https://regex101.com/r/bpJt05/1
    return re.sub(r"<\/?thinking>", "", content or "")


BIBTEX_MAPPING: dict[str, str] = {
    """Maps client bibtex types to pybtex types""" "journal-article": "article",
    "journal-issue": "misc",  # No direct equivalent, so 'misc' is used
    "journal-volume": "misc",  # No direct equivalent, so 'misc' is used
    "journal": "misc",  # No direct equivalent, so 'misc' is used
    "proceedings-article": "inproceedings",
    "proceedings": "proceedings",
    "dataset": "misc",  # No direct equivalent, so 'misc' is used
    "component": "misc",  # No direct equivalent, so 'misc' is used
    "report": "techreport",
    "report-series": (  # 'series' implies multiple tech reports, but each is still a 'techreport'
        "techreport"
    ),
    "standard": "misc",  # No direct equivalent, so 'misc' is used
    "standard-series": "misc",  # No direct equivalent, so 'misc' is used
    "edited-book": "book",  # Edited books are considered books in BibTeX
    "monograph": "book",  # Monographs are considered books in BibTeX
    "reference-book": "book",  # Reference books are considered books in BibTeX
    "book": "book",
    "book-series": "book",  # Series of books can be considered as 'book' in BibTeX
    "book-set": "book",  # Set of books can be considered as 'book' in BibTeX
    "book-chapter": "inbook",
    "book-section": "inbook",  # Sections in books can be considered as 'inbook'
    "book-part": "inbook",  # Parts of books can be considered as 'inbook'
    "book-track": "inbook",  # Tracks in books can be considered as 'inbook'
    "reference-entry": (  # Entries in reference books can be considered as 'inbook'
        "inbook"
    ),
    "dissertation": "phdthesis",  # Dissertations are usually PhD thesis
    "posted-content": "misc",  # No direct equivalent, so 'misc' is used
    "peer-review": "misc",  # No direct equivalent, so 'misc' is used
    "other": "article",  # Assume an article if we don't know the type
}


@contextlib.contextmanager
def logging_filters(
    loggers: Collection[str], filters: Collection[type[logging.Filter]]
):
    """Temporarily add a filter to each specified logger."""
    filters_added: dict[str, list[logging.Filter]] = {}
    try:
        for logger_name in loggers:
            log_to_filter = logging.getLogger(logger_name)
            for log_filter in filters:
                _filter = log_filter()
                log_to_filter.addFilter(_filter)
                if logger_name not in filters_added:
                    filters_added[logger_name] = [_filter]
                else:
                    filters_added[logger_name] += [_filter]
        yield
    finally:
        for logger_name, log_filters_to_remove in filters_added.items():
            log_with_filter = logging.getLogger(logger_name)
            for log_filter_to_remove in log_filters_to_remove:
                log_with_filter.removeFilter(log_filter_to_remove)


def citation_to_docname(citation: str) -> str:
    """Create a docname that follows MLA parenthetical in-text citation."""
    # get first name and year from citation
    match = re.search(r"([A-Z][a-z]+)", citation)
    if match is not None:
        author = match.group(1)
    else:
        # panicking - no word??
        raise ValueError(
            f"Could not parse docname from citation {citation}. "
            "Consider just passing key explicitly - e.g. docs.py "
            "(path, citation, key='mykey')"
        )
    year = ""
    match = re.search(r"(\d{4})", citation)
    if match is not None:
        year = match.group(1)
    return f"{author}{year}"


def maybe_get_date(date: str | datetime | None) -> datetime | None:
    if not date:
        return None
    if isinstance(date, str):
        # Try common date formats in sequence
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",  # ISO with timezone: 2023-01-31T14:30:00+0000
            "%Y-%m-%d %H:%M:%S",  # ISO with time: 2023-01-31 14:30:00
            "%B %d, %Y",  # Full month day, year: January 31, 2023
            "%b %d, %Y",  # Month day, year: Jan 31, 2023
            "%Y-%m-%d",  # ISO format: 2023-01-31
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date, fmt)
            except ValueError:
                continue
        return None
    return date



================================================
FILE: paperqa/agents/__init__.py
================================================
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from aviary.utils import MultipleChoiceQuestion
from pydantic_settings import CliSettingsSource
from rich.logging import RichHandler

from paperqa.settings import Settings, get_settings
from paperqa.utils import pqa_directory, run_or_ensure, setup_default_logs
from paperqa.version import __version__

from .main import agent_query, index_search
from .models import AnswerResponse
from .search import SearchIndex, get_directory_index

logger = logging.getLogger(__name__)

LOG_VERBOSITY_MAP: dict[int, dict[str, int]] = {
    0: {
        "paperqa.agents": logging.INFO,
        "paperqa.agents.helpers": logging.WARNING,
        "paperqa.agents.main": logging.WARNING,
        "paperqa.agents.main.agent_callers": logging.INFO,
        "paperqa.agents.models": logging.WARNING,
        "paperqa.agents.search": logging.INFO,
        "anthropic": logging.WARNING,
        "openai": logging.WARNING,
        "httpcore": logging.WARNING,
        "httpx": logging.WARNING,
        "LiteLLM": logging.WARNING,
        "LiteLLM Router": logging.WARNING,
        "LiteLLM Proxy": logging.WARNING,
    }
}
LOG_VERBOSITY_MAP[1] = LOG_VERBOSITY_MAP[0] | {
    "paperqa.models": logging.INFO,
    "paperqa.agents.main": logging.INFO,
}
LOG_VERBOSITY_MAP[2] = LOG_VERBOSITY_MAP[1] | {
    "paperqa.models": logging.DEBUG,
    "paperqa.agents.helpers": logging.DEBUG,
    "paperqa.agents.main": logging.DEBUG,
    "paperqa.agents.main.agent_callers": logging.DEBUG,
    "paperqa.agents.search": logging.DEBUG,
    "LiteLLM": logging.INFO,
    "LiteLLM Router": logging.INFO,
    "LiteLLM Proxy": logging.INFO,
}
LOG_VERBOSITY_MAP[3] = LOG_VERBOSITY_MAP[2] | {
    "LiteLLM": logging.DEBUG,  # <-- every single LLM call
}
_MAX_PRESET_VERBOSITY: int = max(k for k in LOG_VERBOSITY_MAP)

_PAPERQA_PKG_ROOT_LOGGER = logging.getLogger(__name__.split(".", maxsplit=1)[0])
_INITIATED_FROM_CLI = False


def is_running_under_cli() -> bool:
    """Check if the current Python process comes from the CLI."""
    return _INITIATED_FROM_CLI


def set_up_rich_handler(install: bool = True) -> RichHandler:
    """Add a RichHandler to the paper-qa "root" logger, and return it."""
    rich_handler = RichHandler(
        rich_tracebacks=True, markup=True, show_path=False, show_level=False
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    if install and not any(
        isinstance(h, RichHandler) for h in _PAPERQA_PKG_ROOT_LOGGER.handlers
    ):
        _PAPERQA_PKG_ROOT_LOGGER.addHandler(rich_handler)
    return rich_handler


def configure_log_verbosity(verbosity: int = 0) -> None:
    key = min(verbosity, _MAX_PRESET_VERBOSITY)
    for logger_name, logger_ in logging.Logger.manager.loggerDict.items():
        if isinstance(logger_, logging.Logger) and (
            log_level := LOG_VERBOSITY_MAP.get(key, {}).get(logger_name)
        ):
            logger_.setLevel(log_level)


def configure_cli_logging(verbosity: int | Settings = 0) -> None:
    """Suppress loquacious loggers according to the settings' verbosity level."""
    setup_default_logs()
    set_up_rich_handler()
    if isinstance(verbosity, Settings):
        verbosity = verbosity.verbosity
    configure_log_verbosity(verbosity)
    if verbosity > 0:
        print(f"PaperQA version: {__version__}")


def ask(
    query: str | MultipleChoiceQuestion, settings: Settings
) -> AnswerResponse | asyncio.Task[AnswerResponse]:
    """Query PaperQA via an agent."""
    configure_cli_logging(settings)
    return run_or_ensure(
        coro=agent_query(query, settings, agent_type=settings.agent.agent_type)
    )


def search_query(
    query: str | MultipleChoiceQuestion,
    index_name: str,
    settings: Settings,
) -> (
    list[tuple[AnswerResponse, str] | tuple[Any, str]]
    | asyncio.Task[list[tuple[AnswerResponse, str] | tuple[Any, str]]]
):
    """Search using a pre-built PaperQA index."""
    configure_cli_logging(settings)
    if index_name == "default":
        index_name = settings.get_index_name()
    return run_or_ensure(
        coro=index_search(
            query if isinstance(query, str) else query.question_prompt,
            index_name=index_name,
            index_directory=settings.agent.index.index_directory,
        )
    )


def build_index(
    index_name: str | None = None,
    directory: str | os.PathLike | None = None,
    settings: Settings | None = None,
) -> SearchIndex | asyncio.Task[SearchIndex]:
    """Build a PaperQA search index, this will also happen automatically upon using `ask`."""
    settings = get_settings(settings)
    if index_name == "default":
        settings.agent.index.name = None
    elif isinstance(index_name, str):
        settings.agent.index.name = index_name
    configure_cli_logging(settings)
    if directory:
        settings.agent.index.paper_directory = directory
    return run_or_ensure(coro=get_directory_index(settings=settings))


def save_settings(settings: Settings, settings_path: str | os.PathLike) -> None:
    """Save the settings to a file."""
    configure_cli_logging(settings)
    # check if this could be interpreted at an absolute path
    if os.path.isabs(settings_path):
        full_settings_path = os.path.expanduser(settings_path)
    else:
        full_settings_path = os.path.join(pqa_directory("settings"), settings_path)
        if not full_settings_path.endswith(".json"):
            full_settings_path += ".json"

    is_overwrite = os.path.exists(full_settings_path)

    Path(full_settings_path).write_text(settings.model_dump_json(indent=2))

    if is_overwrite:
        logger.info(f"Settings overwritten to: {full_settings_path}")
    else:
        logger.info(f"Settings saved to: {full_settings_path}")


def main() -> None:

    parser = argparse.ArgumentParser(description="PaperQA CLI")
    parser.add_argument(
        "--settings",
        "-s",
        default="high_quality",
        help=(
            "Named settings to use. Will search in local, pqa directory, and package"
            " last"
        ),
    )
    parser.add_argument(
        "--index", "-i", default="default", help="Index name to search or create"
    )

    subparsers = parser.add_subparsers(
        title="commands", dest="command", description="Available commands"
    )

    subparsers.add_parser("view", help="View the chosen settings")

    save_parser = subparsers.add_parser("save", help="View the chosen settings")
    save_parser.add_argument(
        "location", help="Location for new settings (name or an absolute path)"
    )

    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question of current index (based on settings)"
    )
    ask_parser.add_argument("query", help="Question to ask")

    search_parser = subparsers.add_parser(
        "search",
        help=(
            "Search the index specified by --index."
            " Pass `--index answers` to search previous answers."
        ),
    )
    search_parser.add_argument("query", help="Keyword search")

    build_parser = subparsers.add_parser(
        "index", help="Build a search index from given directory"
    )
    build_parser.add_argument("directory", help="Directory to build index from")

    # Create CliSettingsSource instance
    cli_settings = CliSettingsSource[argparse.ArgumentParser](
        Settings, root_parser=parser
    )

    # Now use argparse to parse the remaining arguments
    args, remaining_args = parser.parse_known_args()
    # Parse arguments using CliSettingsSource
    settings = Settings.from_name(
        args.settings, cli_source=cli_settings(args=remaining_args)
    )

    match args.command:
        case "ask":
            ask(args.query, settings)
        case "view":
            configure_cli_logging(settings)
            logger.info(f"Viewing: {args.settings}")
            logger.info(settings.model_dump_json(indent=2))
        case "save":
            save_settings(settings, args.location)
        case "search":
            search_query(args.query, args.index, settings)
        case "index":
            build_index(args.index, args.directory, settings)
        case _:
            commands = ", ".join({"view", "ask", "search", "index"})
            brief_help = f"\nRun with commands: {{{commands}}}\n\n"
            brief_help += "For more information, run with --help"
            print(brief_help)


if __name__ == "__main__":
    _INITIATED_FROM_CLI = True
    main()



================================================
FILE: paperqa/agents/env.py
================================================
import logging
from copy import deepcopy
from typing import Any, ClassVar, Self, cast
from uuid import UUID

from aviary.core import (
    Environment,
    Frame,
    Message,
    Messages,
    Tool,
    ToolRequestMessage,
    ToolResponseMessage,
)
from aviary.env import ENV_REGISTRY
from aviary.utils import MultipleChoiceQuestion
from lmi import EmbeddingModel, LiteLLMModel

from paperqa.docs import Docs
from paperqa.settings import Settings
from paperqa.sources.clinical_trials import (
    CLINICAL_TRIALS_BASE,
    partition_clinical_trials_by_source,
)
from paperqa.types import PQASession
from paperqa.utils import get_year

from .tools import (
    AVAILABLE_TOOL_NAME_TO_CLASS,
    DEFAULT_TOOL_NAMES,
    ClinicalTrialsSearch,
    Complete,
    EnvironmentState,
    GatherEvidence,
    GenerateAnswer,
    PaperSearch,
    Reset,
)

logger = logging.getLogger(__name__)

POPULATE_FROM_SETTINGS = None


def settings_to_tools(  # noqa: PLR0912
    settings: Settings,
    llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
    summary_llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
    embedding_model: EmbeddingModel | None = POPULATE_FROM_SETTINGS,
) -> list[Tool]:
    """
    Convert a Settings into tools, confirming the complete tool is present.

    NOTE: the last element of the return will always be Complete.
    """
    llm_model = llm_model or settings.get_llm()
    summary_llm_model = summary_llm_model or settings.get_summary_llm()
    embedding_model = embedding_model or settings.get_embedding_model()
    tools: list[Tool] = []
    for tool_type in (
        [AVAILABLE_TOOL_NAME_TO_CLASS[name] for name in DEFAULT_TOOL_NAMES]
        if settings.agent.tool_names is None
        else [
            AVAILABLE_TOOL_NAME_TO_CLASS[name]
            for name in set(settings.agent.tool_names)
        ]
    ):
        if issubclass(tool_type, PaperSearch):
            tool = Tool.from_function(
                PaperSearch(
                    settings=settings, embedding_model=embedding_model
                ).paper_search
            )
            for pname in ("min_year", "max_year"):
                tool.info.get_properties()[pname]["description"] = cast(
                    "str", tool.info.get_properties()[pname]["description"]
                ).format(current_year=get_year())
        elif issubclass(tool_type, GatherEvidence):
            gather_evidence_tool = GatherEvidence(
                settings=settings,
                summary_llm_model=summary_llm_model,
                embedding_model=embedding_model,
            )

            # if we're using the SearchClinicalTrialsTool,
            # we override this tool's docstring/prompt
            # because the default prompt is unaware of the clinical trials tool

            if ClinicalTrialsSearch.TOOL_FN_NAME in (
                settings.agent.tool_names or DEFAULT_TOOL_NAMES
            ):
                gather_evidence_tool.gather_evidence.__func__.__doc__ = (  # type: ignore[attr-defined]
                    ClinicalTrialsSearch.GATHER_EVIDENCE_TOOL_PROMPT_OVERRIDE
                )
                gather_evidence_tool.partitioning_fn = (
                    partition_clinical_trials_by_source
                )

            tool = Tool.from_function(gather_evidence_tool.gather_evidence)

        elif issubclass(tool_type, GenerateAnswer):
            generate_answer_tool = GenerateAnswer(
                settings=settings,
                llm_model=llm_model,
                summary_llm_model=summary_llm_model,
                embedding_model=embedding_model,
            )

            if ClinicalTrialsSearch.TOOL_FN_NAME in (
                settings.agent.tool_names or DEFAULT_TOOL_NAMES
            ):
                generate_answer_tool.partitioning_fn = (
                    partition_clinical_trials_by_source
                )

            tool = Tool.from_function(generate_answer_tool.gen_answer)

        elif issubclass(tool_type, Reset):
            tool = Tool.from_function(Reset().reset)
        elif issubclass(tool_type, Complete):
            tool = Tool.from_function(Complete().complete)
        elif issubclass(tool_type, ClinicalTrialsSearch):
            tool = Tool.from_function(
                ClinicalTrialsSearch(
                    search_count=settings.agent.search_count,
                    settings=settings,
                ).clinical_trials_search
            )
        else:
            raise NotImplementedError(f"Didn't handle tool type {tool_type}.")
        if tool.info.name == Complete.complete.__name__:
            tools.append(tool)  # Place at the end
        else:
            tools.insert(0, tool)
    return tools


def make_clinical_trial_status(
    total_paper_count: int,
    relevant_paper_count: int,
    total_clinical_trials: int,
    relevant_clinical_trials: int,
    evidence_count: int,
    cost: float,
) -> str:
    return (
        f"Status: Paper Count={total_paper_count}"
        f" | Relevant Papers={relevant_paper_count}"
        f" | Clinical Trial Count={total_clinical_trials}"
        f" | Relevant Clinical Trials={relevant_clinical_trials}"
        f" | Current Evidence={evidence_count}"
        f" | Current Cost=${cost:.4f}"
    )


# SEE: https://regex101.com/r/L0L5MH/1
CLINICAL_STATUS_SEARCH_REGEX_PATTERN: str = (
    r"Status: Paper Count=(\d+) \| Relevant Papers=(\d+)(?:\s\|\sClinical Trial"
    r" Count=(\d+)\s\|\sRelevant Clinical Trials=(\d+))?\s\|\sCurrent Evidence=(\d+)"
)


def clinical_trial_status(state: "EnvironmentState") -> str:
    relevant_contexts = state.get_relevant_contexts()
    return make_clinical_trial_status(
        total_paper_count=len(
            {
                d.dockey
                for d in state.docs.docs.values()
                if CLINICAL_TRIALS_BASE
                not in getattr(d, "other", {}).get("client_source", [])
            }
        ),
        relevant_paper_count=len(
            {
                c.text.doc.dockey
                for c in relevant_contexts
                if CLINICAL_TRIALS_BASE
                not in getattr(c.text.doc, "other", {}).get("client_source", [])
            }
        ),
        total_clinical_trials=len(
            {
                d.dockey
                for d in state.docs.docs.values()
                if CLINICAL_TRIALS_BASE
                in getattr(d, "other", {}).get("client_source", [])
            }
        ),
        relevant_clinical_trials=len(
            {
                c.text.doc.dockey
                for c in relevant_contexts
                if CLINICAL_TRIALS_BASE
                in getattr(c.text.doc, "other", {}).get("client_source", [])
            }
        ),
        evidence_count=len(relevant_contexts),
        cost=state.session.cost,
    )


class PaperQAEnvironment(Environment[EnvironmentState]):
    """Environment connecting paper-qa's tools with state."""

    def __init__(
        self,
        query: str | MultipleChoiceQuestion,
        settings: Settings,
        docs: Docs,
        llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
        summary_llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
        embedding_model: EmbeddingModel | None = POPULATE_FROM_SETTINGS,
        session_id: UUID | None = None,
        **env_kwargs,
    ):
        super().__init__(**env_kwargs)
        self._query = query
        self._settings = settings
        self._docs = docs
        self._llm_model = llm_model
        self._summary_llm_model = summary_llm_model
        self._embedding_model = embedding_model
        self._session_id = session_id

    @classmethod
    def from_task(cls, task: str) -> Self:
        return cls(query=task, settings=Settings(), docs=Docs())

    def make_tools(self) -> list[Tool]:
        return settings_to_tools(
            settings=self._settings,
            llm_model=self._llm_model,
            summary_llm_model=self._summary_llm_model,
            embedding_model=self._embedding_model,
        )

    def make_initial_state(self) -> EnvironmentState:
        status_fn = None

        if ClinicalTrialsSearch.TOOL_FN_NAME in (
            self._settings.agent.tool_names or DEFAULT_TOOL_NAMES
        ):
            status_fn = clinical_trial_status

        session_kwargs: dict[str, Any] = {}
        if self._session_id:
            session_kwargs["id"] = self._session_id
        return EnvironmentState(
            docs=self._docs,
            session=PQASession(
                question=(
                    self._query
                    if isinstance(self._query, str)
                    else self._query.question_prompt
                ),
                config_md5=self._settings.md5,
                **session_kwargs,
            ),
            status_fn=status_fn,
        )

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        # NOTE: don't build the index here, as sometimes we asyncio.gather over this
        # method, and our current design (as of v5.0.10) could hit race conditions
        # because index building does not use file locks
        self._docs.clear_docs()
        self.state, self.tools = self.make_initial_state(), self.make_tools()
        return (
            [
                Message(
                    content=self._settings.agent.agent_prompt.format(
                        question=self.state.session.question,
                        status=self.state.status,
                        complete_tool_name=Complete.TOOL_FN_NAME,
                    ),
                )
            ],
            self.tools,
        )

    def export_frame(self) -> Frame:
        return Frame(state=self.state, info={"query": self._query})

    def _has_excess_answer_failures(self) -> bool:
        if self._settings.answer.max_answer_attempts is None:
            return False
        return (
            sum(
                tn == GenerateAnswer.gen_answer.__name__
                for s in self.state.session.tool_history
                for tn in s
            )
            > self._settings.answer.max_answer_attempts
        )

    USE_POST_PROCESSED_REWARD: ClassVar[float] = 0.0

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[Messages, float, bool, bool]:
        self.state.record_action(action)

        response_messages = cast(
            "list[Message]",
            await self.exec_tool_calls(
                action,
                concurrency=False,  # PQA tools aren't yet concurrency safe
                state=self.state,
                handle_tool_exc=True,
            ),
        ) or [Message(content=f"No tool calls input in tool request {action}.")]
        done = any(
            isinstance(msg, ToolResponseMessage)
            and msg.name == Complete.complete.__name__
            for msg in response_messages
        )
        if not done and self._has_excess_answer_failures():
            # If the caller set max_answer_attempts, and the agent has tried to answer
            # too many times, we consider this done, but we cannot determine success
            # because we're not calling the complete tool
            self.state.session.has_successful_answer = None
            done = True
        return (
            response_messages,
            self.USE_POST_PROCESSED_REWARD,
            done,
            False,  # Let caller determine truncations
        )

    def __deepcopy__(self, memo) -> Self:
        copy_state = deepcopy(self.state, memo)
        # We don't know the side effects of deep copying a litellm.Router,
        # so we force a shallow copy of these LiteLLMModels
        env_model_kwargs: dict[str, Any] = {
            name: model if model is None else type(model)(**model.model_dump())
            for name, model in (
                ("llm_model", self._llm_model),
                ("summary_llm_model", self._summary_llm_model),
                ("embedding_model", self._embedding_model),
            )
        }
        copy_self = type(self)(
            query=self._query,  # No need to copy since we read only
            settings=deepcopy(self._settings, memo),  # Deepcopy just to be safe
            docs=copy_state.docs,
            **env_model_kwargs,
        )
        copy_self.state = copy_state
        # Because we shallow copied the LiteLLMModels, we need to re-make the
        # tool functions within the tools
        copy_self.tools = copy_self.make_tools()
        return copy_self


ENV_REGISTRY["paperqa"] = "paperqa.agents.env", PaperQAEnvironment.__name__



================================================
FILE: paperqa/agents/helpers.py
================================================
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import cast

from aviary.core import Message
from lmi import LiteLLMModel, LLMModel
from rich.table import Table

from paperqa.docs import Docs
from paperqa.types import DocDetails

from .models import AnswerResponse

logger = logging.getLogger(__name__)


def get_year(ts: datetime | None = None) -> str:
    """Get the year from the input datetime, otherwise using the current datetime."""
    if ts is None:
        ts = datetime.now()
    return ts.strftime("%Y")


async def litellm_get_search_query(
    question: str,
    count: int,
    template: str | None = None,
    llm: LLMModel | str = "gpt-4o-mini",
    temperature: float = 1.0,
) -> list[str]:
    search_prompt = ""
    if isinstance(template, str) and all(
        x in template for x in ("{count}", "{question}", "{date}")
    ):
        # partial formatting
        search_prompt = template.replace("{date}", get_year())
    elif isinstance(template, str):
        logger.warning(
            "Template does not contain {count}, {question} and {date} variables."
            " Ignoring template and using default search prompt."
        )
    if not search_prompt:
        # TODO: move to use tools instead of DIY schema in prompt
        search_prompt = (
            "We want to answer the following question: {question}\nProvide"
            " {count} unique keyword searches (one search per line) and year ranges"
            " that will find papers to help answer the question. Do not use boolean"
            " operators. Make sure not to repeat searches without changing the"
            " keywords or year ranges. Make some searches broad and some narrow. Use"
            " this format: [keyword search], [start year]-[end year]. where end year"
            f" is optional. The current year is {get_year()}."
        )

    if isinstance(llm, str):
        model: LLMModel = LiteLLMModel(name=llm)
        model.config["model_list"][0]["litellm_params"].update(
            {"temperature": temperature}
        )
    else:
        model = llm
    messages = [
        Message(content=search_prompt.format(question=question, count=count)),
    ]
    result = await model.call_single(
        messages=messages,
    )
    search_query = cast("str", result.text)
    queries = [s for s in search_query.split("\n") if len(s) > 3]  # noqa: PLR2004
    # remove "2.", "3.", etc. -- https://regex101.com/r/W2f7F1/1
    queries = [re.sub(r"^\d+\.\s*", "", q) for q in queries]
    # remove quotes
    return [re.sub(r'["\[\]]', "", q) for q in queries]


def table_formatter(
    objects: list[tuple[AnswerResponse | Docs, str]], max_chars_per_column: int = 2000
) -> Table:
    example_object, _ = objects[0]
    if isinstance(example_object, AnswerResponse):
        table = Table(title="Prior Answers")
        table.add_column("Question", style="cyan")
        table.add_column("Answer", style="magenta")
        for obj, _ in objects:
            table.add_row(
                cast("AnswerResponse", obj).session.question[:max_chars_per_column],
                cast("AnswerResponse", obj).session.answer[:max_chars_per_column],
            )
        return table
    if isinstance(example_object, Docs):
        table = Table(title="PDF Search")
        table.add_column("Title", style="cyan")
        table.add_column("File", style="magenta")
        for obj, filename in objects:
            docs = cast("Docs", obj)  # Assume homogeneous objects
            doc = docs.texts[0].doc
            if isinstance(doc, DocDetails) and doc.title:
                display_name: str = doc.title  # Prefer title if available
            else:
                display_name = doc.formatted_citation
            table.add_row(display_name[:max_chars_per_column], filename)
        return table
    raise NotImplementedError(
        f"Object type {type(example_object)} can not be converted to table."
    )



================================================
FILE: paperqa/agents/main.py
================================================
import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from aviary.core import (
    MalformedMessageError,
    Message,
    Tool,
    ToolCall,
    ToolRequestMessage,
    ToolSelector,
    ToolSelectorLedger,
)
from aviary.utils import MultipleChoiceQuestion
from pydantic import BaseModel
from rich.console import Console
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
)

from paperqa._ldp_shims import Callback, RolloutManager
from paperqa.docs import Docs
from paperqa.settings import AgentSettings, Settings
from paperqa.types import PQASession

from .env import PaperQAEnvironment
from .helpers import litellm_get_search_query, table_formatter
from .models import AgentStatus, AnswerResponse, SimpleProfiler
from .search import SearchDocumentStorage, SearchIndex, get_directory_index
from .tools import (
    DEFAULT_TOOL_NAMES,
    Complete,
    EnvironmentState,
    GatherEvidence,
    GenerateAnswer,
    PaperSearch,
)

if TYPE_CHECKING:
    from aviary.core import Environment
    from ldp.agent import Agent, SimpleAgentState
    from ldp.graph.ops import OpResult

logger = logging.getLogger(__name__)
agent_logger = logging.getLogger(__name__ + ".agent_callers")

DEFAULT_AGENT_TYPE = AgentSettings.model_fields["agent_type"].default


async def agent_query(
    query: str | MultipleChoiceQuestion,
    settings: Settings,
    docs: Docs | None = None,
    agent_type: str | type = DEFAULT_AGENT_TYPE,
    **runner_kwargs,
) -> AnswerResponse:
    if docs is None:
        docs = Docs()

    answers_index = SearchIndex(
        fields=[*SearchIndex.REQUIRED_FIELDS, "question"],
        index_name="answers",
        index_directory=settings.agent.index.index_directory,
        storage=SearchDocumentStorage.JSON_MODEL_DUMP,
    )

    response = await run_agent(docs, query, settings, agent_type, **runner_kwargs)
    agent_logger.debug(f"agent_response: {response}")

    agent_logger.info(f"[bold blue]Answer: {response.session.answer}[/bold blue]")

    await answers_index.add_document(
        {
            "file_location": str(response.session.id),
            "body": response.session.answer,
            "question": response.session.question,
        },
        document=response,
    )
    await answers_index.save_index()
    return response


FAKE_AGENT_TYPE = "fake"  # No agent, just invoke tools in deterministic order


async def run_agent(
    docs: Docs,
    query: str | MultipleChoiceQuestion,
    settings: Settings,
    agent_type: str | type = DEFAULT_AGENT_TYPE,
    **runner_kwargs,
) -> AnswerResponse:
    """
    Run an agent.

    Args:
        docs: Docs to run upon.
        query: Query to answer.
        settings: Settings to use.
        agent_type: Agent type (or fully qualified name to the type) to pass to
            AgentType.get_agent, or "fake" to TODOC.
        runner_kwargs: Keyword arguments to pass to the runner.

    Returns:
        Tuple of resultant answer, token counts, and agent status.
    """
    profiler = SimpleProfiler()
    outer_profile_name = f"agent-{agent_type}-{settings.agent.agent_llm}"
    profiler.start(outer_profile_name)
    question = query if isinstance(query, str) else query.question_prompt
    logger.info(
        f"Beginning agent {agent_type!r} run with question {question!r} and full"
        f" settings {settings.model_dump()}."
    )

    # Build the index once here, and then all tools won't need to rebuild it
    # only build if the a search tool is requested
    if PaperSearch.TOOL_FN_NAME in (settings.agent.tool_names or DEFAULT_TOOL_NAMES):
        await get_directory_index(settings=settings, build=settings.agent.rebuild_index)

    if isinstance(agent_type, str) and agent_type.lower() == FAKE_AGENT_TYPE:
        session, agent_status = await run_fake_agent(
            query, settings, docs, **runner_kwargs
        )
    elif tool_selector_or_none := settings.make_aviary_tool_selector(agent_type):
        session, agent_status = await run_aviary_agent(
            query, settings, docs, tool_selector_or_none, **runner_kwargs
        )
    elif ldp_agent_or_none := await settings.make_ldp_agent(agent_type):
        session, agent_status = await run_ldp_agent(
            query, settings, docs, ldp_agent_or_none, **runner_kwargs
        )
    else:
        raise NotImplementedError(f"Didn't yet handle agent type {agent_type}.")

    if agent_status != AgentStatus.TRUNCATED and session.has_successful_answer is False:
        agent_status = AgentStatus.UNSURE
    # stop after, so overall isn't reported as long-running step.
    logger.info(
        f"Finished agent {agent_type!r} run with question {question!r} and status"
        f" {agent_status}."
    )
    return AnswerResponse(session=session, status=agent_status)


async def _run_with_timeout_failure(
    rollout: Callable[[], Awaitable[AgentStatus]],
    settings: Settings,
    env: PaperQAEnvironment,
) -> tuple[PQASession, AgentStatus]:
    try:
        async with asyncio.timeout(settings.agent.timeout):
            status = await rollout()
    except TimeoutError:
        logger.warning(
            f"Agent timeout after {settings.agent.timeout}-sec, just answering."
        )
        status = AgentStatus.TRUNCATED
    except Exception:
        logger.exception("Trajectory failed.")
        status = AgentStatus.FAIL
    if status == AgentStatus.TRUNCATED or not env.state.query_tool_history(
        GenerateAnswer.TOOL_FN_NAME
    ):
        # Fail over after truncation (too many steps, timeout): just answer
        generate_answer_tool = next(
            filter(lambda x: x.info.name == GenerateAnswer.TOOL_FN_NAME, env.tools)
        )
        action = ToolRequestMessage(
            tool_calls=[ToolCall.from_tool(generate_answer_tool)]
        )
        await env.exec_tool_calls(message=action, state=env.state, handle_tool_exc=True)
        env.state.record_action(action)
    return env.state.session, status


async def run_fake_agent(
    query: str | MultipleChoiceQuestion,
    settings: Settings,
    docs: Docs,
    env_class: type[PaperQAEnvironment] = PaperQAEnvironment,
    on_env_reset_callback: Callable[[EnvironmentState], Awaitable] | None = None,
    on_agent_action_callback: (
        Callable[[ToolRequestMessage, BaseModel], Awaitable] | None
    ) = None,
    on_env_step_callback: (
        Callable[[list[Message], float, bool, bool], Awaitable] | None
    ) = None,
    **env_kwargs,
) -> tuple[PQASession, AgentStatus]:
    if settings.agent.max_timesteps is not None:
        logger.warning(
            f"Max timesteps (configured {settings.agent.max_timesteps}) is not"
            " applicable with the fake agent, ignoring it."
        )
    env = env_class(query, settings, docs, **env_kwargs)
    obs, tools = await env.reset()
    settings.adjust_tools_for_agent_llm(tools)

    if on_env_reset_callback:
        await on_env_reset_callback(env.state)

    question = env.state.session.question
    search_tool = next(filter(lambda x: x.info.name == PaperSearch.TOOL_FN_NAME, tools))
    gather_evidence_tool = next(
        filter(lambda x: x.info.name == GatherEvidence.TOOL_FN_NAME, tools)
    )
    generate_answer_tool = next(
        filter(lambda x: x.info.name == GenerateAnswer.TOOL_FN_NAME, tools)
    )
    complete_tool = next(filter(lambda x: x.info.name == Complete.TOOL_FN_NAME, tools))
    agent_messages = obs.copy()  # Copy just to be safe

    async def step(action: list[ToolCall] | ToolRequestMessage) -> None:
        action = (
            action
            if isinstance(action, ToolRequestMessage)
            else ToolRequestMessage(tool_calls=action)
        )
        agent_messages.append(action)
        if on_agent_action_callback:
            await on_agent_action_callback(action, env.state)
        obs, reward, done, truncated = await env.step(action)
        agent_messages.extend(obs)
        if on_env_step_callback:
            await on_env_step_callback(obs, reward, done, truncated)

    async def rollout() -> AgentStatus:
        llm_model = settings.get_llm()

        # Seed docs with a few LLM-proposed search calls
        # TODO: make properly support year ranges
        for search in await litellm_get_search_query(question, llm=llm_model, count=3):
            search_tcs = [
                ToolCall.from_tool(
                    search_tool, query=search, min_year=None, max_year=None
                )
            ]
            await step(search_tcs)
        await step([ToolCall.from_tool(gather_evidence_tool, question=question)])
        await step([ToolCall.from_tool(generate_answer_tool)])
        # Complete with an LLM-proposed complete call
        complete_action = await llm_model.select_tool(
            messages=agent_messages, tools=tools, tool_choice=complete_tool
        )
        await step(complete_action)
        return (
            AgentStatus.SUCCESS
            if env.state.session.has_successful_answer is not False
            else AgentStatus.UNSURE
        )

    return await _run_with_timeout_failure(rollout, settings, env)


async def run_aviary_agent(
    query: str | MultipleChoiceQuestion,
    settings: Settings,
    docs: Docs,
    agent: ToolSelector,
    env_class: type[PaperQAEnvironment] = PaperQAEnvironment,
    on_env_reset_callback: Callable[[EnvironmentState], Awaitable] | None = None,
    on_agent_action_callback: (
        Callable[[ToolRequestMessage, BaseModel], Awaitable] | None
    ) = None,
    on_env_step_callback: (
        Callable[[list[Message], float, bool, bool], Awaitable] | None
    ) = None,
    **env_kwargs,
) -> tuple[PQASession, AgentStatus]:
    env = env_class(query, settings, docs, **env_kwargs)

    async def rollout() -> AgentStatus:
        obs, tools = await env.reset()
        settings.adjust_tools_for_agent_llm(tools)

        if on_env_reset_callback:
            await on_env_reset_callback(env.state)

        agent_state = ToolSelectorLedger(
            messages=(
                [
                    Message(
                        role="system",
                        content=settings.agent.agent_system_prompt,
                    )
                ]
                if settings.agent.agent_system_prompt
                else []
            ),
            tools=tools,
        )

        timestep, max_timesteps = 0, settings.agent.max_timesteps
        done = False
        while not done:
            if max_timesteps is not None and timestep >= max_timesteps:
                logger.warning(
                    f"Agent didn't finish within {max_timesteps} timesteps, just"
                    " answering."
                )
                return AgentStatus.TRUNCATED
            agent_state.messages += obs
            for attempt in Retrying(
                stop=stop_after_attempt(5),
                retry=retry_if_exception_type(MalformedMessageError),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:  # Retrying if ToolSelector fails to select a tool
                    action = await agent(agent_state.messages, tools)
            agent_state.messages = [*agent_state.messages, action]
            if on_agent_action_callback:
                await on_agent_action_callback(action, agent_state)

            obs, reward, done, truncated = await env.step(action)
            if on_env_step_callback:
                await on_env_step_callback(obs, reward, done, truncated)
            timestep += 1
        return AgentStatus.SUCCESS

    return await _run_with_timeout_failure(rollout, settings, env)


class LDPRolloutCallback(Callback):
    """Shim connecting ldp RolloutManager Callbacks with paperqa runner callbacks."""

    def __init__(
        self,
        env: "Environment",
        on_env_reset_callback: Callable[[EnvironmentState], Awaitable] | None = None,
        on_agent_action_callback: "Callable[[OpResult[ToolRequestMessage], SimpleAgentState, float], Awaitable] | None" = None,  # noqa: E501
        on_env_step_callback: (
            Callable[[list[Message], float, bool, bool], Awaitable] | None
        ) = None,
    ):
        self.env = env
        self.on_env_reset_callback = on_env_reset_callback
        self.on_agent_action_callback = on_agent_action_callback
        self.on_env_step_callback = on_env_step_callback

    async def after_agent_get_asv(self, traj_id: str, *args) -> None:  # noqa: ARG002
        if self.on_agent_action_callback is not None:
            await self.on_agent_action_callback(*args)

    async def after_env_reset(self, traj_id: str, *_) -> None:  # noqa: ARG002
        if self.on_env_reset_callback is not None:
            await self.on_env_reset_callback(self.env.state)

    async def after_env_step(self, traj_id: str, *args) -> None:  # noqa: ARG002
        if self.on_env_step_callback is not None:
            await self.on_env_step_callback(*args)


class LDPAdjustToolsForAgentCallback(Callback):
    def __init__(self, settings: Settings):
        self._settings = settings

    async def after_env_reset(
        self, traj_id: str, obs: list[Message], tools: list[Tool]  # noqa: ARG002
    ) -> None:
        self._settings.adjust_tools_for_agent_llm(tools)


async def run_ldp_agent(
    query: str | MultipleChoiceQuestion,
    settings: Settings,
    docs: Docs,
    agent: "Agent[SimpleAgentState]",
    env_class: type[PaperQAEnvironment] = PaperQAEnvironment,
    on_env_reset_callback: Callable[[EnvironmentState], Awaitable] | None = None,
    on_agent_action_callback: "Callable[[OpResult[ToolRequestMessage], SimpleAgentState, float], Awaitable] | None" = None,  # noqa: E501
    on_env_step_callback: (
        Callable[[list[Message], float, bool, bool], Awaitable] | None
    ) = None,
    ldp_callback_type: type[LDPRolloutCallback] = LDPRolloutCallback,
    **env_kwargs,
) -> tuple[PQASession, AgentStatus]:
    env = env_class(query, settings, docs, **env_kwargs)
    # NOTE: don't worry about ldp import checks, because we know Settings.make_ldp_agent
    # has already taken place, which checks that ldp is installed

    async def rollout() -> AgentStatus:
        rollout_manager = RolloutManager(
            agent,
            callbacks=[
                ldp_callback_type(
                    env,
                    on_env_reset_callback,
                    on_agent_action_callback,
                    on_env_step_callback,
                ),
                LDPAdjustToolsForAgentCallback(settings),
            ],
        )
        trajs = await rollout_manager.sample_trajectories(
            environments=[env], max_steps=settings.agent.max_timesteps
        )
        traj = trajs[0]
        if traj.steps[-1].truncated:
            return AgentStatus.TRUNCATED
        return AgentStatus.SUCCESS

    return await _run_with_timeout_failure(rollout, settings, env)


async def index_search(
    query: str, index_name: str = "answers", **index_kwargs
) -> list[tuple[AnswerResponse, str] | tuple[Any, str]]:
    fields = [*SearchIndex.REQUIRED_FIELDS]
    if index_name == "answers":
        fields.append("question")
    index_to_query = SearchIndex(
        fields=fields,
        index_name=index_name,
        storage=(
            SearchDocumentStorage.JSON_MODEL_DUMP
            if index_name == "answers"
            else SearchDocumentStorage.PICKLE_COMPRESSED
        ),
        **index_kwargs,
    )

    results = [
        (AnswerResponse(**a[0]) if index_name == "answers" else a[0], a[1])
        for a in await index_to_query.query(query=query, keep_filenames=True)
    ]
    if results:
        console = Console(record=True)
        # Render the table to a string
        console.print(table_formatter(results))
    else:
        count = await index_to_query.count
        agent_logger.info(f"No results found. Searched {count} docs.")

    return results



================================================
FILE: paperqa/agents/models.py
================================================
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Any, ClassVar, Protocol, cast
from uuid import UUID, uuid4

from aviary.core import Message
from lmi import LiteLLMModel, LLMModel
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

from paperqa.types import PQASession
from paperqa.version import __version__

logger = logging.getLogger(__name__)


class SupportsPickle(Protocol):
    """Type protocol for typing any object that supports pickling."""

    def __reduce__(self) -> str | tuple[Any, ...]: ...
    def __getstate__(self) -> object: ...
    def __setstate__(self, state: object) -> None: ...


class AgentStatus(StrEnum):  # TODO: rename to AnswerStatus or RolloutStatus
    # FAIL - during the trajectory encountered an unhandled exception
    FAIL = "fail"
    # SUCCESS - answer was generated
    SUCCESS = "success"
    # TRUNCATED - agent didn't finish naturally (e.g. timeout, too many actions),
    # so we just generated an answer after the unnatural finish
    TRUNCATED = "truncated"
    # UNSURE - the gen_answer did not succeed, but an answer is present
    UNSURE = "unsure"


class MismatchedModelsError(Exception):
    """Error to throw when model clients clash ."""

    LOG_METHOD_NAME: ClassVar[str] = "warning"


class AnswerResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session: PQASession = Field(alias="answer")
    bibtex: dict[str, str] | None = None
    status: AgentStatus
    timing_info: dict[str, dict[str, float]] | None = None
    duration: float = 0.0
    # A placeholder for interesting statistics we can show users
    # about the answer, such as the number of sources used, etc.
    stats: dict[str, str] | None = None

    @field_validator("session")
    def strip_answer(
        cls, v: PQASession, info: ValidationInfo  # noqa: ARG002, N805
    ) -> PQASession:
        # This modifies in place, this is fine
        # because when a response is being constructed,
        # we should be done with the Answer object
        v.filter_content_for_user()
        return v

    async def get_summary(self, llm_model: LLMModel | str = "gpt-4o") -> str:
        sys_prompt = (
            "Revise the answer to a question to be a concise SMS message. "
            "Use abbreviations or emojis if necessary."
        )
        model = (
            LiteLLMModel(name=llm_model) if isinstance(llm_model, str) else llm_model
        )
        prompt_template = "{question}\n\n{answer}"
        messages = [
            Message(role="system", content=sys_prompt),
            Message(
                role="user",
                content=prompt_template.format(
                    question=self.session.question, answer=self.session.answer
                ),
            ),
        ]
        result = await model.call_single(
            messages=messages,
        )
        return cast("str", result.text).strip()


class TimerData(BaseModel):
    start_time: float = Field(default_factory=time.time)  # noqa: FURB111
    durations: list[float] = Field(default_factory=list)


class SimpleProfiler(BaseModel):
    """Basic profiler with start/stop and named timers.

    The format for this logger needs to be strictly followed, as downstream google
    cloud monitoring is based on the following
    # [Profiling] {**name** of timer} | {**elapsed** time of function} | {**__version__** of PaperQA}
    """

    timers: dict[str, list[float]] = Field(default_factory=dict)
    running_timers: dict[str, TimerData] = Field(default_factory=dict)
    uid: UUID = Field(default_factory=uuid4)

    @asynccontextmanager
    async def timer(self, name: str):
        start_time = asyncio.get_running_loop().time()
        try:
            yield
        finally:
            end_time = asyncio.get_running_loop().time()
            elapsed = end_time - start_time
            self.timers.setdefault(name, []).append(elapsed)
            logger.info(
                f"[Profiling] | UUID: {self.uid} | NAME: {name} | TIME: {elapsed:.3f}s"
                f" | VERSION: {__version__}"
            )

    def start(self, name: str) -> None:
        try:
            self.running_timers[name] = TimerData()
        except RuntimeError:  # No running event loop (not in async)
            self.running_timers[name] = TimerData(start_time=time.time())

    def stop(self, name: str) -> None:
        timer_data = self.running_timers.pop(name, None)
        if timer_data:
            try:
                t_stop: float = asyncio.get_running_loop().time()
            except RuntimeError:  # No running event loop (not in async)
                t_stop = time.time()
            elapsed = t_stop - timer_data.start_time
            self.timers.setdefault(name, []).append(elapsed)
            logger.info(
                f"[Profiling] | UUID: {self.uid} | NAME: {name} | TIME: {elapsed:.3f}s"
                f" | VERSION: {__version__}"
            )
        else:
            logger.warning(f"Timer {name} not running")

    def results(self) -> dict[str, dict[str, float]]:
        result = {}
        for name, durations in self.timers.items():
            mean = sum(durations) / len(durations)
            result[name] = {
                "low": min(durations),
                "mean": mean,
                "max": max(durations),
                "total": sum(durations),
            }
        return result



================================================
FILE: paperqa/agents/search.py
================================================
from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import pathlib
import pickle
import re
import warnings
import zlib
from collections import Counter
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import datetime
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID

import anyio
from pydantic import BaseModel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tantivy import (  # pylint: disable=no-name-in-module
    Document,
    Index,
    Schema,
    SchemaBuilder,
    Searcher,
)
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from paperqa.docs import Docs
from paperqa.settings import IndexSettings, get_settings
from paperqa.types import VAR_MATCH_LOOKUP, DocDetails
from paperqa.utils import ImpossibleParsingError, hexdigest

from .models import SupportsPickle

if TYPE_CHECKING:
    from tantivy import IndexWriter

    from paperqa.settings import MaybeSettings, Settings

logger = logging.getLogger(__name__)


class AsyncRetryError(Exception):
    """Flags a retry for another tenacity attempt."""


class RobustEncoder(json.JSONEncoder):
    """JSON encoder that can handle UUID and set objects."""

    def default(self, o):
        if isinstance(o, UUID):
            # if the obj is uuid, we simply return the value of uuid
            return str(o)
        if isinstance(o, set):
            return list(o)
        if isinstance(o, os.PathLike):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


class SearchDocumentStorage(StrEnum):
    """Method to serialize a document."""

    JSON_MODEL_DUMP = auto()  # utf-8 JSON dump
    PICKLE_COMPRESSED = auto()  # pickle + zlib compression
    PICKLE_UNCOMPRESSED = auto()  # pickle

    def extension(self) -> str:
        if self == SearchDocumentStorage.JSON_MODEL_DUMP:
            return "json"
        if self == SearchDocumentStorage.PICKLE_COMPRESSED:
            return "zip"
        return "pkl"

    def write_to_string(self, data: BaseModel | SupportsPickle) -> bytes:
        if self == SearchDocumentStorage.JSON_MODEL_DUMP:
            if isinstance(data, BaseModel):
                return json.dumps(data.model_dump(), cls=RobustEncoder).encode("utf-8")
            raise ValueError("JSON_MODEL_DUMP requires a BaseModel object.")
        if self == SearchDocumentStorage.PICKLE_COMPRESSED:
            return zlib.compress(pickle.dumps(data))
        return pickle.dumps(data)

    def read_from_string(self, data: str | bytes) -> BaseModel | SupportsPickle:
        if self == SearchDocumentStorage.JSON_MODEL_DUMP:
            return json.loads(data)
        if self == SearchDocumentStorage.PICKLE_COMPRESSED:
            return pickle.loads(zlib.decompress(data))  # type: ignore[arg-type] # noqa: S301
        return pickle.loads(data)  # type: ignore[arg-type] # noqa: S301


# Cache keys are a two-tuple of index name and absolute index directory
# Cache values are a two-tuple of an opened Index instance and the count
# of SearchIndex instances currently referencing that Index
_OPENED_INDEX_CACHE: dict[tuple[str, str], tuple[Index, int]] = {}
DONT_USE_OPENED_INDEX_CACHE = (
    os.environ.get("PQA_INDEX_DONT_CACHE_INDEXES", "").lower() in VAR_MATCH_LOOKUP
)


def reap_opened_index_cache() -> None:
    """Delete any unreferenced Index instances from the Index cache."""
    for index_name, (index, count) in _OPENED_INDEX_CACHE.items():
        if count == 0:
            _OPENED_INDEX_CACHE.pop(index_name)
            del index


class SearchIndex:
    """Wrapper around a tantivy.Index exposing higher-level behaviors for documents."""

    REQUIRED_FIELDS: ClassVar[list[str]] = ["file_location", "body"]

    def __init__(
        self,
        fields: Sequence[str] | None = None,
        index_name: str = "pqa_index",
        index_directory: str | os.PathLike = IndexSettings.model_fields[
            "index_directory"
        ].default,
        storage: SearchDocumentStorage = SearchDocumentStorage.PICKLE_COMPRESSED,
    ):
        if fields is None:
            fields = self.REQUIRED_FIELDS
        self.fields = fields
        if not all(f in self.fields for f in self.REQUIRED_FIELDS):
            raise ValueError(
                f"{self.REQUIRED_FIELDS} must be included in search index fields."
            )
        self.index_name = index_name
        self._index_directory = index_directory
        self._schema: Schema | None = None
        self._index: Index | None = None
        self._searcher: Searcher | None = None
        self._writer: IndexWriter | None = None
        self._index_files: dict[str, str] = {}
        self.changed = False
        self.storage = storage

    @property
    async def index_directory(  # TODO: rename to index_root_directory
        self,
    ) -> anyio.Path:
        directory = anyio.Path(self._index_directory).joinpath(self.index_name)
        await directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    async def index_filename(  # TODO: rename to index_meta_directory
        self,
    ) -> anyio.Path:
        """Directory to store files used to house index internals."""
        index_dir = (await self.index_directory) / "index"
        await index_dir.mkdir(exist_ok=True)
        return index_dir

    @property
    async def docs_index_directory(self) -> anyio.Path:
        """Directory to store documents (e.g. chunked PDFs) given the storage type."""
        docs_dir = (await self.index_directory) / "docs"
        await docs_dir.mkdir(exist_ok=True)
        return docs_dir

    @property
    async def file_index_filename(self) -> anyio.Path:
        """File containing a zlib-compressed pickle of the index_files."""
        return (await self.index_directory) / "files.zip"

    @property
    def schema(self) -> Schema:
        if not self._schema:
            schema_builder = SchemaBuilder()
            for field in self.fields:
                schema_builder.add_text_field(field, stored=True)
            self._schema = schema_builder.build()
        return self._schema

    @property
    async def index(self) -> Index:
        if not self._index:
            index_meta_directory = await self.index_filename
            if await (index_meta_directory / "meta.json").exists():
                if DONT_USE_OPENED_INDEX_CACHE:
                    self._index = Index.open(path=str(index_meta_directory))
                else:
                    key = self.index_name, str(await index_meta_directory.absolute())
                    # NOTE: now we know we're using the cache and have created the cache
                    # key. And we know we're in asyncio.gather race condition risk land.
                    # All of the following operations are *synchronous* so we are not
                    # giving the opportunity for an await to switch to another parallel
                    # version of this code. Otherwise, we risk counts being incorrect
                    # due to race conditions
                    if key not in _OPENED_INDEX_CACHE:  # open a new Index
                        self._index = Index.open(path=str(index_meta_directory))
                        prev_count: int = 0
                    else:  # reuse Index
                        self._index, prev_count = _OPENED_INDEX_CACHE[key]
                    _OPENED_INDEX_CACHE[key] = self._index, prev_count + 1
            else:
                # NOTE: this creates the above meta.json file
                self._index = Index(self.schema, path=str(index_meta_directory))
        return self._index

    def __del__(self) -> None:
        index_meta_directory = (
            pathlib.Path(self._index_directory) / self.index_name / "index"
        )
        key = self.index_name, str(index_meta_directory.absolute())
        if key in _OPENED_INDEX_CACHE:
            index, count = _OPENED_INDEX_CACHE[key]
            _OPENED_INDEX_CACHE[key] = index, count - 1

    @property
    async def searcher(self) -> Searcher:
        if not self._searcher:
            index = await self.index
            index.reload()
            self._searcher = index.searcher()
        return self._searcher

    @contextlib.asynccontextmanager
    async def writer(self, reset: bool = False) -> AsyncIterator[IndexWriter]:
        if not self._writer:
            index = await self.index
            self._writer = index.writer()
        yield self._writer
        if reset:
            self._writer = None

    @property
    async def count(self) -> int:
        return (await self.searcher).num_docs

    @property
    async def index_files(self) -> dict[str, str]:
        if not self._index_files:
            file_index_path = await self.file_index_filename
            if await file_index_path.exists():
                async with await anyio.open_file(file_index_path, "rb") as f:
                    content = await f.read()
                    try:
                        self._index_files = pickle.loads(  # noqa: S301
                            zlib.decompress(content)
                        )
                    except Exception:
                        logger.exception(
                            f"Failed to load index file {file_index_path}."
                        )
                        raise
        return self._index_files

    @staticmethod
    def filehash(body: str) -> str:
        return hexdigest(body)

    async def filecheck(self, filename: str, body: str | None = None) -> bool:
        """Check if this index contains the filename and if the body's filehash matches."""
        filehash: str | None = self.filehash(body) if body else None
        index_files = await self.index_files
        return bool(
            index_files.get(filename)
            and (filehash is None or index_files[filename] == filehash)
        )

    async def mark_failed_document(self, path: str | os.PathLike) -> None:
        (await self.index_files)[str(path)] = FAILED_DOCUMENT_ADD_ID
        self.changed = True

    async def add_document(
        self,
        index_doc: dict[str, Any],  # TODO: rename to something more intuitive
        document: Any | None = None,
        lock_acquisition_max_retries: int = 1000,
    ) -> None:
        """
        Add the input document to this index.

        Args:
            index_doc: "Document" (thinking types.Doc) of metadata such as 'title' to
                use in the index.
            document: Document to store according to the specified storage method.
            lock_acquisition_max_retries: Amount of retries to acquire a file lock. A
                large default of 1000 is used because lock acquisition can take a while.
        """

        @retry(
            stop=stop_after_attempt(lock_acquisition_max_retries),
            wait=wait_random_exponential(multiplier=0.25, max=60),
            retry=retry_if_exception_type(AsyncRetryError),
        )
        async def _add_document() -> None:
            if not await self.filecheck(index_doc["file_location"], index_doc["body"]):
                try:
                    async with self.writer() as writer:
                        # Let caller handle commit to allow for batching
                        writer.add_document(Document.from_dict(index_doc))

                    filehash = self.filehash(index_doc["body"])
                    (await self.index_files)[index_doc["file_location"]] = filehash

                    if document:
                        docs_index_dir = await self.docs_index_directory
                        async with await anyio.open_file(
                            docs_index_dir / f"{filehash}.{self.storage.extension()}",
                            "wb",
                        ) as f:
                            await f.write(self.storage.write_to_string(document))

                    self.changed = True
                except ValueError as e:
                    if "Failed to acquire Lockfile: LockBusy." in str(e):
                        raise AsyncRetryError("Failed to acquire lock.") from e
                    raise

        try:
            await _add_document()  # If this runs, we succeeded
        except RetryError:
            logger.exception(
                f"Failed to add document to {index_doc['file_location']}"
                f" within {lock_acquisition_max_retries} attempts."
            )
            raise

    @retry(
        stop=stop_after_attempt(1000),
        wait=wait_random_exponential(multiplier=0.25, max=60),
        retry=retry_if_exception_type(AsyncRetryError),
        reraise=True,
    )
    async def delete_document(self, file_location: str) -> None:
        try:
            async with self.writer() as writer:
                writer.delete_documents("file_location", file_location)
            await self.save_index()
        except ValueError as e:
            if "Failed to acquire Lockfile: LockBusy." in str(e):
                raise AsyncRetryError("Failed to acquire lock") from e
            raise

    async def remove_from_index(self, file_location: str) -> None:
        index_files = await self.index_files
        if index_files.get(file_location):
            await self.delete_document(file_location)
            filehash = index_files.pop(file_location)
            docs_index_dir = await self.docs_index_directory
            # TODO: since the directory is part of the filehash these
            # are always missing. Unsure of how to get around this.
            await (docs_index_dir / f"{filehash}.{self.storage.extension()}").unlink(
                missing_ok=True
            )

            self.changed = True

    @retry(
        stop=stop_after_attempt(1000),
        wait=wait_random_exponential(multiplier=0.25, max=60),
        retry=retry_if_exception_type(AsyncRetryError),
        reraise=True,
    )
    async def save_index(self) -> None:
        try:
            async with self.writer(reset=True) as writer:
                writer.commit()
                writer.wait_merging_threads()
        except ValueError as e:
            if "Failed to acquire Lockfile: LockBusy." in str(e):
                raise AsyncRetryError("Failed to acquire lock") from e
            raise
        file_index_path = await self.file_index_filename
        async with await anyio.open_file(file_index_path, "wb") as f:
            await f.write(zlib.compress(pickle.dumps(await self.index_files)))
        self.changed = False

    async def get_saved_object(
        self, file_location: str, keep_filenames: bool = False
    ) -> Any | tuple[Any, str] | None:
        filehash = (await self.index_files).get(file_location)
        if filehash:
            docs_index_dir = await self.docs_index_directory
            async with await anyio.open_file(
                docs_index_dir / f"{filehash}.{self.storage.extension()}", "rb"
            ) as f:
                content = await f.read()
                if keep_filenames:
                    return self.storage.read_from_string(content), file_location
                return self.storage.read_from_string(content)
        return None

    def clean_query(self, query: str) -> str:
        # SEE: https://regex101.com/r/DoLMoa/3
        return re.sub(r'[*\[\]:(){}~^><+"\\]', "", query)

    async def query(
        self,
        query: str,
        top_n: int = 10,
        offset: int = 0,
        min_score: float = 0.0,
        keep_filenames: bool = False,
        field_subset: list[str] | None = None,
    ) -> list[Any]:
        query_fields = list(field_subset or self.fields)
        searcher = await self.searcher
        index = await self.index
        addresses = [
            s[1]
            for s in searcher.search(
                index.parse_query(self.clean_query(query), query_fields),
                top_n,
                offset=offset,
            ).hits
            if s[0] > min_score
        ]
        search_index_docs = [searcher.doc(address) for address in addresses]
        return [
            result
            for result in [
                await self.get_saved_object(
                    doc["file_location"][0], keep_filenames=keep_filenames  # type: ignore[index]
                )
                for doc in search_index_docs
            ]
            if result is not None
        ]


def fetch_kwargs_from_manifest(
    file_location: str, manifest: dict[str, Any], manifest_fallback_location: str
) -> dict[str, Any]:
    manifest_entry: dict[str, Any] | None = manifest.get(file_location) or manifest.get(
        manifest_fallback_location
    )
    if manifest_entry:
        return DocDetails(**manifest_entry).model_dump()
    return {}


async def maybe_get_manifest(
    filename: anyio.Path | None = None,
) -> dict[str, dict[str, Any]]:
    if not filename:
        return {}
    if filename.suffix == ".csv":
        try:
            async with await anyio.open_file(filename, mode="r") as file:
                content = await file.read()
            file_loc_to_records = {
                str(r.get("file_location")): r
                for r in csv.DictReader(content.splitlines())
                if r.get("file_location")
            }
            if not file_loc_to_records:
                raise ValueError(  # noqa: TRY301
                    "No mapping of file location to details extracted from manifest"
                    f" file {filename}."
                )
            logger.debug(
                f"Found manifest file at {filename}, read"
                f" {len(file_loc_to_records)} records from it, which maps to"
                f" {len(file_loc_to_records)} locations."
            )
        except FileNotFoundError:
            logger.warning(f"Manifest file at {filename} could not be found.")
        except Exception:
            logger.exception(f"Error reading manifest file {filename}.")
        else:
            return file_loc_to_records
    else:
        logger.error(f"Invalid manifest file type: {filename.suffix}")

    return {}


FAILED_DOCUMENT_ADD_ID = "ERROR"


async def process_file(
    rel_file_path: anyio.Path,
    search_index: SearchIndex,
    manifest: dict[str, Any],
    semaphore: anyio.Semaphore,
    settings: Settings,
    processed_counter: Counter[str],
    progress_bar_update: Callable[[], Any] | None = None,
) -> None:

    abs_file_path = (
        pathlib.Path(settings.agent.index.paper_directory).absolute() / rel_file_path
    )
    fallback_title = rel_file_path.name
    if settings.agent.index.use_absolute_paper_directory:
        file_location = str(abs_file_path)
        manifest_fallback_location = str(rel_file_path)
    else:
        file_location = str(rel_file_path)
        manifest_fallback_location = str(abs_file_path)

    async with semaphore:
        if not await search_index.filecheck(filename=file_location):
            logger.info(f"New file to index: {file_location}...")

            kwargs = fetch_kwargs_from_manifest(
                file_location, manifest, manifest_fallback_location
            )

            tmp_docs = Docs()
            try:
                await tmp_docs.aadd(
                    path=abs_file_path,
                    fields=["title", "author", "journal", "year"],
                    settings=settings,
                    **kwargs,
                )
            except Exception as e:
                # We handle any exception here because we want to save_index so we
                # 1. can resume the build without rebuilding this file if a separate
                # process_file invocation leads to a segfault or crash.
                # 2. don't have deadlock issues after.
                logger.exception(
                    f"Error parsing {file_location}, skipping index for this file."
                )
                await search_index.mark_failed_document(file_location)
                await search_index.save_index()
                if progress_bar_update:
                    progress_bar_update()

                if not isinstance(e, ValueError | ImpossibleParsingError):
                    # ImpossibleParsingError: parsing failure, don't retry
                    # ValueError: TODOC
                    raise
                return

            this_doc = next(iter(tmp_docs.docs.values()))

            if isinstance(this_doc, DocDetails):
                title = this_doc.title or fallback_title
                year = this_doc.year or "Unknown year"
            else:
                title, year = fallback_title, "Unknown year"

            await search_index.add_document(
                {
                    "title": title,
                    "year": year,
                    "file_location": file_location,
                    "body": "".join(t.text for t in tmp_docs.texts),
                },
                document=tmp_docs,
            )

            processed_counter["batched_save_counter"] += 1
            if (
                processed_counter["batched_save_counter"]
                == settings.agent.index.batch_size
            ):
                await search_index.save_index()
                processed_counter["batched_save_counter"] = 0

            logger.info(f"Complete ({title}).")

        # Update progress bar for either a new or previously indexed file
        if progress_bar_update:
            progress_bar_update()


WARN_IF_INDEXING_MORE_THAN = 999


def _make_progress_bar_update(
    sync_index_w_directory: bool, total: int
) -> tuple[contextlib.AbstractContextManager, Callable[[], Any] | None]:
    # Disable should override enable
    env_var_disable = (
        os.environ.get("PQA_INDEX_DISABLE_PROGRESS_BAR", "").lower() in VAR_MATCH_LOOKUP
    )
    env_var_enable = (
        os.environ.get("PQA_INDEX_ENABLE_PROGRESS_BAR", "").lower() in VAR_MATCH_LOOKUP
    )
    try:
        is_cli = is_running_under_cli()  # pylint: disable=used-before-assignment
    except NameError:  # Work around circular import
        from . import is_running_under_cli

        is_cli = is_running_under_cli()

    if sync_index_w_directory and not env_var_disable and (is_cli or env_var_enable):
        # Progress.get_default_columns with a few more
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        task_id = progress.add_task("Indexing...", total=total)

        def progress_bar_update() -> None:
            progress.update(task_id, advance=1)

        return progress, progress_bar_update
    return contextlib.nullcontext(), None


async def get_directory_index(  # noqa: PLR0912
    index_name: str | None = None,
    sync_index_w_directory: bool = True,
    settings: MaybeSettings = None,
    build: bool = True,
) -> SearchIndex:
    """
    Create a Tantivy index by reading from a directory of text files.

    This function only reads from the source directory, not edits or writes to it.

    Args:
        index_name: Deprecated override on the name of the index. If unspecified,
            the default behavior is to generate the name from the input settings.
        sync_index_w_directory: Opt-out flag to sync the index (add or delete index
            files) with the source paper directory.
        settings: Application settings.
        build: Opt-out flag (default is True) to read the contents of the source paper
            directory and if sync_index_w_directory is enabled also update the index.
    """
    _settings = get_settings(settings)
    index_settings = _settings.agent.index
    if index_name:
        warnings.warn(
            "The index_name argument has been moved to"
            f" {type(_settings.agent.index).__name__},"
            " this deprecation will conclude in version 6.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        index_settings.name = index_name
    del index_name

    search_index = SearchIndex(
        fields=[*SearchIndex.REQUIRED_FIELDS, "title", "year"],
        index_name=index_settings.name or _settings.get_index_name(),
        index_directory=index_settings.index_directory,
    )
    # NOTE: if the index was not previously built, its index_files will be empty.
    # Otherwise, the index_files will not be empty
    if not build:
        if not await search_index.index_files:
            raise RuntimeError(
                f"Index {search_index.index_name} was empty, please rebuild it."
            )
        return search_index

    if not sync_index_w_directory:
        warnings.warn(
            "The sync_index_w_directory argument has been moved to"
            f" {type(_settings.agent.index).__name__},"
            " this deprecation will conclude in version 6.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        index_settings.sync_with_paper_directory = sync_index_w_directory
    del sync_index_w_directory

    paper_directory = anyio.Path(index_settings.paper_directory)
    manifest = await maybe_get_manifest(
        filename=await index_settings.finalize_manifest_file()
    )
    valid_papers_rel_file_paths = [
        file.relative_to(paper_directory)
        async for file in (
            paper_directory.rglob("*")
            if index_settings.recurse_subdirectories
            else paper_directory.iterdir()
        )
        if file.suffix in {".txt", ".pdf", ".html", ".md"}
    ]
    if len(valid_papers_rel_file_paths) > WARN_IF_INDEXING_MORE_THAN:
        logger.warning(
            f"Indexing {len(valid_papers_rel_file_paths)} files into the index"
            f" {search_index.index_name}, may take a few minutes."
        )

    index_unique_file_paths: set[str] = set((await search_index.index_files).keys())
    if extra_index_files := (
        index_unique_file_paths - {str(f) for f in valid_papers_rel_file_paths}
    ):
        if index_settings.sync_with_paper_directory:
            for extra_file in extra_index_files:
                logger.warning(
                    f"[bold red]Removing {extra_file} from index.[/bold red]"
                )
                await search_index.remove_from_index(extra_file)
            logger.warning("[bold red]Files removed![/bold red]")
        else:
            logger.warning(
                f"[bold red]Indexed files {extra_index_files} are missing from paper"
                f" folder ({paper_directory}).[/bold red]"
            )

    semaphore = anyio.Semaphore(index_settings.concurrency)
    progress_bar, progress_bar_update_fn = _make_progress_bar_update(
        index_settings.sync_with_paper_directory, total=len(valid_papers_rel_file_paths)
    )
    with progress_bar:
        async with anyio.create_task_group() as tg:
            processed_counter: Counter[str] = Counter()
            for rel_file_path in valid_papers_rel_file_paths:
                if index_settings.sync_with_paper_directory:
                    tg.start_soon(
                        process_file,
                        rel_file_path,
                        search_index,
                        manifest,
                        semaphore,
                        _settings,
                        processed_counter,
                        progress_bar_update_fn,
                    )
                else:
                    logger.debug(
                        f"File {rel_file_path} found in paper directory"
                        f" {paper_directory}."
                    )

    if search_index.changed:
        await search_index.save_index()
    else:
        logger.debug("No changes to index.")

    return search_index



================================================
FILE: paperqa/agents/tools.py
================================================
"""Base classes for tools, implemented in a functional manner."""

import asyncio
import inspect
import logging
import os
import re
import sys
from collections.abc import Callable
from itertools import chain
from typing import ClassVar, Self, cast

from aviary.core import ToolRequestMessage
from lmi import Embeddable, EmbeddingModel, LiteLLMModel
from pydantic import BaseModel, ConfigDict, Field, computed_field

from paperqa.docs import Docs
from paperqa.settings import Settings
from paperqa.sources.clinical_trials import add_clinical_trials_to_docs
from paperqa.types import Context, DocDetails, PQASession

from .search import get_directory_index

logger = logging.getLogger(__name__)


def make_status(
    total_paper_count: int, relevant_paper_count: int, evidence_count: int, cost: float
) -> str:
    return (
        f"Status: Paper Count={total_paper_count}"
        f" | Relevant Papers={relevant_paper_count} | Current Evidence={evidence_count}"
        f" | Current Cost=${cost:.4f}"
    )


def default_status(state: "EnvironmentState") -> str:
    relevant_contexts = state.get_relevant_contexts()
    return make_status(
        total_paper_count=len(state.docs.docs),
        relevant_paper_count=len({c.text.doc.dockey for c in relevant_contexts}),
        evidence_count=len(relevant_contexts),
        cost=state.session.cost,
    )


class EnvironmentState(BaseModel):
    """State here contains documents and answer being populated."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    docs: Docs
    session: PQASession = Field(..., alias="answer")
    status_fn: Callable[[Self], str] | None = Field(
        default=None,
        description=(
            "Function used to generate status,"
            " uses `paperqa.agents.tools.default_status` "
            "if not provided."
        ),
    )

    # SEE: https://regex101.com/r/RmuVdC/1
    STATUS_SEARCH_REGEX_PATTERN: ClassVar[str] = (
        r"Status: Paper Count=(\d+) \| Relevant Papers=(\d+) \| Current Evidence=(\d+)"
    )
    RELEVANT_SCORE_CUTOFF: ClassVar[int] = 5

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> str:
        if self.status_fn is not None:
            return self.status_fn(cast("Self", self))
        return default_status(self)

    def get_relevant_contexts(self) -> list[Context]:
        return [
            c for c in self.session.contexts if c.score > self.RELEVANT_SCORE_CUTOFF
        ]

    def record_action(self, action: ToolRequestMessage) -> None:
        self.session.add_tokens(action)
        self.session.tool_history.append([tc.function.name for tc in action.tool_calls])

    def query_tool_history(self, tool_name: str) -> bool:
        """Return true if the tool is has been called in history."""
        return tool_name in set(chain.from_iterable(self.session.tool_history))


class NamedTool(BaseModel):
    """Base class to make looking up tools easier."""

    TOOL_FN_NAME: ClassVar[str] = (
        "# unpopulated"  # Comment symbol ensures no collisions
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class PaperSearch(NamedTool):
    TOOL_FN_NAME = "paper_search"

    settings: Settings
    embedding_model: EmbeddingModel
    previous_searches: dict[tuple[str, str | None], int] = Field(default_factory=dict)

    async def paper_search(
        self,
        query: str,
        min_year: int | None,
        max_year: int | None,
        state: EnvironmentState,
    ) -> str:
        """
        Search for papers to increase the paper count.

        Repeat previous calls with the same query and years to continue a search. Only repeat a maximum of twice.
        This tool can be called concurrently.
        This tool introduces novel papers, so invoke this tool when just beginning or when unsatisfied with the current evidence.

        Args:
            query: A search query, which can be a specific phrase, complete sentence,
                or general keywords, e.g. 'machine learning for immunology'. Also can be
                given search operators.
            min_year: Filter for minimum publication year, or None for no minimum year.
                The current year is {current_year}.
            max_year: Filter for maximum publication year, or None for no maximum year.
                The current year is {current_year}.
            state: Current state.

        Returns:
            String describing searched papers and the current status.
        """  # noqa: E501,W505
        # Convert to date range (e.g. 2022-2022) if date is present
        year = (
            f"{min_year if min_year else ''}-{max_year if max_year else ''}"  # noqa: FURB110
            if (min_year or max_year)
            else None
        )
        # get offset if we've done this search before (continuation of search)
        # or mark this search as new (so offset 0)
        search_key = query, year
        try:
            offset = self.previous_searches[search_key]
        except KeyError:
            offset = self.previous_searches[search_key] = 0

        logger.info(f"Starting paper search for {query!r}.")
        index = await get_directory_index(settings=self.settings, build=False)
        results: list[Docs] = await index.query(
            query,
            top_n=self.settings.agent.search_count,
            offset=offset,
            field_subset=[f for f in index.fields if f != "year"],
        )
        logger.info(
            f"{self.TOOL_FN_NAME} for query {query!r} and offset {offset} returned"
            f" {len(results)} papers."
        )

        # combine all the resulting doc objects into one and update the state
        all_doc_details: list[DocDetails] = []
        for r in results:
            # there's only one doc per result, so just take the first one
            this_doc_details = cast("DocDetails", next(iter(r.docs.values())))
            all_doc_details.append(this_doc_details)
            await state.docs.aadd_texts(
                texts=r.texts,
                doc=this_doc_details,
                settings=self.settings,
                embedding_model=self.embedding_model,
            )

        status = state.status
        logger.info(status)
        # mark how far we've searched so that continuation will start at the right place
        self.previous_searches[search_key] += self.settings.agent.search_count
        if self.settings.agent.return_paper_metadata:
            retrieved_papers = "\n".join(
                [f"{x.title} ({x.year})" for x in all_doc_details]
            )
            return f"Retrieved Papers:\n{retrieved_papers}\n\n{status}"
        return status


class EmptyDocsError(RuntimeError):
    """Error to throw when we needed docs to be present."""


class GatherEvidence(NamedTool):
    TOOL_FN_NAME = "gather_evidence"

    settings: Settings
    summary_llm_model: LiteLLMModel
    embedding_model: EmbeddingModel
    partitioning_fn: Callable[[Embeddable], int] | None = None

    async def gather_evidence(self, question: str, state: EnvironmentState) -> str:
        """
        Gather evidence from previous papers given a specific question to increase evidence and relevant paper counts.

        A valuable time to invoke this tool is right after another tool increases paper count.
        Feel free to invoke this tool in parallel with other tools, but do not call this tool in parallel with itself.
        Only invoke this tool when the paper count is above zero, or this tool will be useless.

        Args:
            question: Specific question to gather evidence for.
            state: Current state.

        Returns:
            String describing gathered evidence and the current status.
        """
        if not state.docs.docs:
            raise EmptyDocsError("Not gathering evidence due to having no papers.")

        if f"{self.TOOL_FN_NAME}_initialized" in self.settings.agent.callbacks:
            await asyncio.gather(
                *(
                    c(state)
                    for c in self.settings.agent.callbacks[
                        f"{self.TOOL_FN_NAME}_initialized"
                    ]
                )
            )

        logger.info(f"{self.TOOL_FN_NAME} starting for question {question!r}.")
        original_question = state.session.question
        l1 = l0 = len(state.session.contexts)
        l1_relevant = l0_relevant = len(state.get_relevant_contexts())

        try:
            # Swap out the question with the more specific question
            # TODO: remove this swap, as it prevents us from supporting parallel calls
            state.session.question = question

            # TODO: refactor answer out of this...
            state.session = await state.docs.aget_evidence(
                query=state.session,
                settings=self.settings,
                embedding_model=self.embedding_model,
                summary_llm_model=self.summary_llm_model,
                partitioning_fn=self.partitioning_fn,
                callbacks=self.settings.agent.callbacks.get(
                    f"{self.TOOL_FN_NAME}_aget_evidence"
                ),
            )
            l1 = len(state.session.contexts)
            l1_relevant = len(state.get_relevant_contexts())
        finally:
            state.session.question = original_question

        status = state.status
        logger.info(status)
        # only show top n contexts for this particular question to the agent
        sorted_contexts = sorted(
            [
                c
                for c in state.session.contexts
                if (c.question is None or c.question == question)
            ],
            key=lambda x: x.score,
            reverse=True,
        )

        top_contexts = "\n".join(
            [
                f"{n + 1}. {sc.context}\n"
                for n, sc in enumerate(
                    sorted_contexts[: self.settings.agent.agent_evidence_n]
                )
            ]
        )

        best_evidence = f" Best evidence(s):\n\n{top_contexts}" if top_contexts else ""

        if f"{self.TOOL_FN_NAME}_completed" in self.settings.agent.callbacks:
            await asyncio.gather(
                *(
                    callback(state)
                    for callback in self.settings.agent.callbacks[
                        f"{self.TOOL_FN_NAME}_completed"
                    ]
                )
            )

        return (
            f"Added {l1 - l0} pieces of evidence, {l1_relevant - l0_relevant} of which"
            f" were relevant.{best_evidence}\n\n" + status
        )


class GenerateAnswer(NamedTool):
    TOOL_FN_NAME = "gen_answer"

    settings: Settings
    llm_model: LiteLLMModel
    summary_llm_model: LiteLLMModel
    embedding_model: EmbeddingModel
    partitioning_fn: Callable[[Embeddable], int] | None = None

    async def gen_answer(self, state: EnvironmentState) -> str:
        """
        Generate an answer using current evidence.

        The tool may fail, indicating that better or different evidence should be found.
        Aim for at least five pieces of evidence from multiple sources before invoking this tool.
        Feel free to invoke this tool in parallel with other tools, but do not call this tool in parallel with itself.

        Args:
            state: Current state.
        """
        logger.info(f"Generating answer for '{state.session.question}'.")

        if f"{self.TOOL_FN_NAME}_initialized" in self.settings.agent.callbacks:
            await asyncio.gather(
                *(
                    callback(state)
                    for callback in self.settings.agent.callbacks[
                        f"{self.TOOL_FN_NAME}_initialized"
                    ]
                )
            )

        state.session = await state.docs.aquery(
            query=state.session,
            settings=self.settings,
            llm_model=self.llm_model,
            summary_llm_model=self.summary_llm_model,
            embedding_model=self.embedding_model,
            partitioning_fn=self.partitioning_fn,
            callbacks=self.settings.agent.callbacks.get(
                f"{self.TOOL_FN_NAME}_aget_query"
            ),
        )

        answer = state.session.answer
        status = state.status
        logger.info(status)

        if f"{self.TOOL_FN_NAME}_completed" in self.settings.agent.callbacks:
            await asyncio.gather(
                *(
                    callback(state)
                    for callback in self.settings.agent.callbacks[
                        f"{self.TOOL_FN_NAME}_completed"
                    ]
                )
            )

        return f"{answer} | {status}"

    # Use to separate answer from status
    # NOTE: can match failure to answer or an actual answer
    ANSWER_SPLIT_REGEX_PATTERN: ClassVar[str] = (
        r" \| " + EnvironmentState.STATUS_SEARCH_REGEX_PATTERN
    )

    @classmethod
    def extract_answer_from_message(cls, content: str) -> str:
        """Extract the answer from a message content."""
        answer, *rest = re.split(
            pattern=cls.ANSWER_SPLIT_REGEX_PATTERN, string=content, maxsplit=1
        )
        return answer if len(rest) == 4 else ""  # noqa: PLR2004


class Reset(NamedTool):
    TOOL_FN_NAME = "reset"

    async def reset(self, state: EnvironmentState) -> None:
        """
        Reset by clearing all current evidence from the system.

        This tool is useful when repeatedly failing to answer because the existing evidence may unsuitable for the question.
        It does not make sense to call this tool in parallel with other tools, as its resetting all state.
        Only invoke this tool when the current evidence is above zero, or this tool will be useless.
        """  # noqa: E501,W505
        logger.info(f"Resetting '{state.session.question}'.")
        state.session.contexts = []
        state.session.context = ""


class Complete(NamedTool):
    TOOL_FN_NAME = "complete"

    # Use to separate certainty from status
    CERTAINTY_SPLIT_REGEX_PATTERN: ClassVar[str] = (
        r" \| " + EnvironmentState.STATUS_SEARCH_REGEX_PATTERN
    )

    NO_ANSWER_PHRASE: ClassVar[str] = "No answer generated."

    async def complete(
        self, has_successful_answer: bool, state: EnvironmentState
    ) -> str:
        """
        Terminate using the last proposed answer.

        Do not invoke this tool in parallel with other tools or itself.

        Args:
            has_successful_answer: Set True if an answer that addresses all parts of the
                task has been generated, otherwise set False to indicate unsureness.
            state: Current state.
        """
        # TODO: eliminate race condition here if agent calls 2+ times in parallel
        # with opposite has_successful_answer values
        state.session.has_successful_answer = has_successful_answer

        if not state.session.answer:
            state.session.answer = self.NO_ANSWER_PHRASE

        logger.info(
            f"Completing '{state.session.question}' as"
            f" '{'certain' if has_successful_answer else 'unsure'}'."
        )
        # Return answer and status to simplify postprocessing of tool response
        return f"{'Certain' if has_successful_answer else 'Unsure'} | {state.status}"


class ClinicalTrialsSearch(NamedTool):
    TOOL_FN_NAME = "clinical_trials_search"

    model_config = ConfigDict(extra="forbid")

    search_count: int = 8
    previous_searches: dict[str, int] = Field(default_factory=dict)
    settings: Settings = Field(default_factory=Settings)

    # Gather evidence tool must be modified to understand the new evidence
    GATHER_EVIDENCE_TOOL_PROMPT_OVERRIDE: ClassVar[str] = (
        """Gather evidence from previous papers and clinical trials given a specific question.

        Will increase evidence, relevant paper counts, and relevant clinical trial counts.
        A valuable time to invoke this tool is right after another tool increases paper or clinical trials count.
        Feel free to invoke this tool in parallel with other tools, but do not call this tool in parallel with itself.
        Only invoke this tool when the paper count or clinical trial count is above zero, or this tool will be useless.

        Args:
            question: Specific question to gather evidence for.
            state: Current state.

        Returns:
            String describing gathered evidence and the current status.
        """
    )

    async def clinical_trials_search(self, query: str, state: EnvironmentState) -> str:
        r"""Search for clinical trials, with support for repeated calls and concurrent execution.

        Will add new clinical trials to the state, and return metadata about the number of trials found.

        Args:
            query: The search query string. Supports complex boolean expressions, field-specific
                searches, and query modifiers through operators. All configuration is done through
                operators in the query string.
                Query Syntax:
                    Basic Search:
                        Simple text automatically uses default EXPANSION[Relaxation] and COVERAGE[Contains]
                        >>> "heart attack"

                    Modified Search:
                        Use operators to modify search behavior:
                        >>> 'EXPANSION[None]COVERAGE[FullMatch]"exact phrase"'
                        >>> 'EXPANSION[Concept]heart attack'

                    Field Search:
                        Specify fields using AREA operator:
                        >>> 'AREA[InterventionName]aspirin'
                        >>> 'AREA[Phase]PHASE3'

                    Location Search:
                        Use SEARCH operator for compound location queries:
                        >>> 'cancer AND SEARCH[Location](AREA[LocationCity]Boston AND AREA[LocationState]Massachusetts)'

                    Complex Boolean:
                        Combine terms with AND, OR, NOT and parentheses:
                        >>> '(cancer OR tumor) AND NOT (EXPANSION[None]pediatric OR AREA[StdAge]CHILD)'

                    Date Ranges:
                        Use RANGE to specify date ranges with formats like "yyyy-MM" or "yyyy-MM-dd".
                        Note that MIN and MAX can be used for open-ended ranges:
                        >>> AREA[ResultsFirstPostDate]RANGE[2015-01-01, MAX]

                Operators:
                    EXPANSION[type]: Controls term expansion
                        - None: Exact match only, case and accent sensitive
                        - Term: Includes lexical variants (plurals, spellings)
                        - Concept: Includes UMLS synonyms
                        - Relaxation: Relaxes adjacency requirements (default)
                        - Lossy: Allows missing partial terms

                    COVERAGE[type]: Controls text matching
                        - FullMatch: Must match entire field
                        - StartsWith: Must match beginning of field
                        - EndsWith: Must match end of field
                        - Contains: Must match part of field (default)

                    AREA[field]: Specifies field to search
                        - See Field Reference for available fields

                    SEARCH[type]: Groups field searches
                        - Location: Groups location-related fields
                        - Study: Groups study-related fields

                Usage Notes:
                    - All search expressions are implicitly OR expressions
                    - Operator precedence (highest to lowest): terms/source operators, NOT/context operators, AND, OR
                    - Use quotes for exact phrase matching: "heart attack"
                    - Use parentheses for grouping: (heart OR cardiac) AND attack
                    - Use backslash to escape operators: \AND
                    - Default expansion is EXPANSION[Relaxation]
                    - Default coverage is COVERAGE[Contains]

                Field Reference:
                    High Priority Fields (weight >= 0.8):
                        - NCTId (1.0): Trial identifier
                        - Acronym (1.0): Study acronym
                        - BriefTitle (0.89): Short title
                        - OfficialTitle (0.85): Full official title
                        - Condition (0.81): Medical condition
                        - InterventionName (0.8): Primary intervention name
                        - OverallStatus: Trial status

                    Medium Priority Fields (0.5-0.79):
                        - InterventionOtherName (0.75): Alternative intervention names
                        - Phase (0.65): Trial phase
                        - StdAge (0.65): Standard age groups
                        - Keyword (0.6): Study keywords
                        - BriefSummary (0.6): Short description
                        - SecondaryOutcomeMeasure (0.5): Secondary outcomes

                    Low Priority Fields (< 0.5):
                        - DesignPrimaryPurpose (0.3): Primary purpose of study
                        - StudyType (0.3)
                        - Various descriptive, location, and administrative fields

                Supported Enums:
                    Phase:
                        - EARLY_PHASE1: Early Phase 1
                        - PHASE1: Phase 1
                        - PHASE2: Phase 2
                        - PHASE3: Phase 3
                        - PHASE4: Phase 4
                        - NA: Not Applicable

                    StandardAge:
                        - CHILD: Child
                        - ADULT: Adult
                        - OLDER_ADULT: Older Adult

                    Status:
                        - RECRUITING: Currently recruiting participants
                        - ACTIVE_NOT_RECRUITING: Active but not recruiting
                        - COMPLETED: Study completed
                        - ENROLLING_BY_INVITATION: Enrolling by invitation only
                        - NOT_YET_RECRUITING: Not yet recruiting
                        - SUSPENDED: Study suspended
                        - TERMINATED: Study terminated
                        - WITHDRAWN: Study withdrawn
                        - AVAILABLE: Available
                        - NO_LONGER_AVAILABLE: No longer available
                        - TEMPORARILY_NOT_AVAILABLE: Temporarily not available
                        - APPROVED_FOR_MARKETING: Approved for marketing
                        - WITHHELD: Withheld
                        - UNKNOWN: Unknown status

                    StudyType:
                        - INTERVENTIONAL: Interventional studies
                        - OBSERVATIONAL: Observational studies
                        - EXPANDED_ACCESS: Expanded access studies

                    PrimaryPurpose:
                        - TREATMENT: Treatment
                        - PREVENTION: Prevention
                        - DIAGNOSTIC: Diagnostic
                        - ECT: Educational/Counseling/Training
                        - SUPPORTIVE_CARE: Supportive Care
                        - SCREENING: Screening
                        - HEALTH_SERVICES_RESEARCH: Health Services Research
                        - BASIC_SCIENCE: Basic Science
                        - DEVICE_FEASIBILITY: Device Feasibility
                        - OTHER: Other

                    InterventionType:
                        - BEHAVIORAL: Behavioral interventions
                        - BIOLOGICAL: Biological interventions
                        - COMBINATION_PRODUCT: Combination product interventions
                        - DEVICE: Device interventions
                        - DIAGNOSTIC_TEST: Diagnostic test interventions
                        - DIETARY_SUPPLEMENT: Dietary supplement interventions
                        - DRUG: Drug interventions
                        - GENETIC: Genetic interventions
                        - PROCEDURE: Procedure interventions
                        - RADIATION: Radiation interventions
                        - OTHER: Other interventions

                    DesignAllocation:
                        - RANDOMIZED: Randomized allocation
                        - NON_RANDOMIZED: Non-randomized allocation
                        - NA: Not applicable

                    InterventionalAssignment:
                        - SINGLE_GROUP: Single group assignment
                        - PARALLEL: Parallel assignment
                        - CROSSOVER: Crossover assignment
                        - FACTORIAL: Factorial assignment
                        - SEQUENTIAL: Sequential assignment

                    ObservationalModel:
                        - COHORT: Cohort
                        - CASE_CONTROL: Case-Control
                        - CASE_ONLY: Case-Only
                        - CASE_CROSSOVER: Case-Crossover
                        - ECOLOGIC_OR_COMMUNITY: Ecologic or Community
                        - FAMILY_BASED: Family-Based
                        - DEFINED_POPULATION: Defined Population
                        - NATURAL_HISTORY: Natural History
                        - OTHER: Other

                    DesignMasking:
                        - NONE: None (Open Label)
                        - SINGLE: Single
                        - DOUBLE: Double
                        - TRIPLE: Triple
                        - QUADRUPLE: Quadruple

                    WhoMasked:
                        - PARTICIPANT: Participant
                        - CARE_PROVIDER: Care Provider
                        - INVESTIGATOR: Investigator
                        - OUTCOMES_ASSESSOR: Outcomes Assessor

            state: Current state

        Returns:
            String describing current status
        """
        # get offset if we've done this search before (continuation of search)
        # or mark this search as new (so offset 0)
        try:
            offset = self.previous_searches[query]
        except KeyError:
            offset = self.previous_searches[query] = 0

        total_result_count, new_result_count, error_message = (
            await add_clinical_trials_to_docs(
                query,
                state.docs,
                self.settings,
                limit=self.search_count,
                offset=offset,
            )
        )
        # mark how far we've searched so that continuation will start at the right place
        self.previous_searches[query] += self.search_count
        if error_message is None:
            return (
                f"Found clinical trial search results from search {offset} to"
                f" {offset + new_result_count} among {total_result_count} total"
                f" results. {state.status}"
            )
        return f"Error in clinical trial query syntax: {error_message}"


AVAILABLE_TOOL_NAME_TO_CLASS: dict[str, type[NamedTool]] = {
    cls.TOOL_FN_NAME: cls
    for _, cls in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda v: inspect.isclass(v)
        and issubclass(v, NamedTool)
        and v is not NamedTool,
    )
}


DEFAULT_TOOL_NAMES: list[str] = [
    name.strip()
    for name in os.environ.get("PAPERQA_DEFAULT_TOOL_NAMES", "").split(",")
    if name.strip()
] or [
    PaperSearch.TOOL_FN_NAME,
    GatherEvidence.TOOL_FN_NAME,
    GenerateAnswer.TOOL_FN_NAME,
    Reset.TOOL_FN_NAME,
    Complete.TOOL_FN_NAME,
]



================================================
FILE: paperqa/clients/__init__.py
================================================
from __future__ import annotations

import copy
import logging
from collections.abc import Awaitable, Collection, Coroutine, Sequence
from typing import Any, cast

import aiohttp
from lmi.utils import gather_with_concurrency
from pydantic import BaseModel, ConfigDict

from paperqa.types import Doc, DocDetails

from .client_models import MetadataPostProcessor, MetadataProvider
from .crossref import CrossrefProvider
from .journal_quality import JournalQualityPostProcessor
from .openalex import OpenAlexProvider
from .retractions import RetractionDataPostProcessor
from .semantic_scholar import SemanticScholarProvider
from .unpaywall import UnpaywallProvider

logger = logging.getLogger(__name__)

DEFAULT_CLIENTS: Collection[type[MetadataPostProcessor | MetadataProvider]] = {
    CrossrefProvider,
    SemanticScholarProvider,
    JournalQualityPostProcessor,
}

ALL_CLIENTS: Collection[type[MetadataPostProcessor | MetadataProvider]] = {
    *DEFAULT_CLIENTS,
    OpenAlexProvider,
    UnpaywallProvider,
    RetractionDataPostProcessor,
}


class DocMetadataTask(BaseModel):
    """Holder for provider and processor tasks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    providers: Collection[MetadataProvider]
    processors: Collection[MetadataPostProcessor]

    def provider_queries(
        self, query: dict
    ) -> list[Coroutine[Any, Any, DocDetails | None]]:
        return [p.query(query) for p in self.providers]

    def processor_queries(
        self, doc_details: DocDetails, session: aiohttp.ClientSession
    ) -> list[Coroutine[Any, Any, DocDetails]]:
        return [
            p.process(copy.copy(doc_details), session=session) for p in self.processors
        ]

    def __repr__(self) -> str:
        return (
            f"DocMetadataTask(providers={self.providers}, processors={self.processors})"
        )


class DocMetadataClient:
    def __init__(
        self,
        session: aiohttp.ClientSession | None = None,
        clients: (
            Collection[type[MetadataPostProcessor | MetadataProvider]]
            | Sequence[Collection[type[MetadataPostProcessor | MetadataProvider]]]
        ) = DEFAULT_CLIENTS,
    ) -> None:
        """Metadata client for querying multiple metadata providers and processors.

        Args:
            session: outer scope aiohttp session to allow for connection pooling
            clients: list of MetadataProvider and MetadataPostProcessor classes to query;
                if nested, will query in order looking for termination criteria after each.
                Will terminate early if either DocDetails.is_hydration_needed is False OR if
                all requested fields are present in the DocDetails object.

        """
        self._session = session
        self.tasks: list[DocMetadataTask] = []

        # first see if we are nested; i.e. we want order
        if isinstance(clients, Sequence) and all(
            isinstance(sub_clients, Collection) for sub_clients in clients
        ):
            for sub_clients in clients:
                self.tasks.append(
                    DocMetadataTask(
                        providers=[
                            c if isinstance(c, MetadataProvider) else c()
                            for c in sub_clients
                            if (isinstance(c, type) and issubclass(c, MetadataProvider))
                            or isinstance(c, MetadataProvider)
                        ],
                        processors=[
                            c if isinstance(c, MetadataPostProcessor) else c()
                            for c in sub_clients
                            if (
                                isinstance(c, type)
                                and issubclass(c, MetadataPostProcessor)
                            )
                            or isinstance(c, MetadataPostProcessor)
                        ],
                    )
                )
        # otherwise, we are a flat collection
        if not self.tasks and all(not isinstance(c, Collection) for c in clients):
            self.tasks.append(
                DocMetadataTask(
                    providers=[
                        c if isinstance(c, MetadataProvider) else c()  # type: ignore[redundant-expr]
                        for c in clients
                        if (isinstance(c, type) and issubclass(c, MetadataProvider))
                        or isinstance(c, MetadataProvider)
                    ],
                    processors=[
                        c if isinstance(c, MetadataPostProcessor) else c()  # type: ignore[redundant-expr]
                        for c in clients
                        if (
                            isinstance(c, type) and issubclass(c, MetadataPostProcessor)
                        )
                        or isinstance(c, MetadataPostProcessor)
                    ],
                )
            )

        if not self.tasks or (self.tasks and not self.tasks[0].providers):
            raise ValueError("At least one MetadataProvider must be provided.")

    async def query(self, **kwargs) -> DocDetails | None:

        session = aiohttp.ClientSession() if self._session is None else self._session

        query_args = kwargs if "session" in kwargs else kwargs | {"session": session}

        all_doc_details: DocDetails | None = None

        for ti, task in enumerate(self.tasks):

            logger.debug(
                f"Attempting to populate metadata query: {query_args} via {task}"
            )

            # first query all client_models and aggregate the results
            doc_details = (
                sum(
                    p
                    for p in (
                        await gather_with_concurrency(
                            len(task.providers), task.provider_queries(query_args)
                        )
                    )
                    if p
                )
                or None
            )
            # then process and re-aggregate the results
            if doc_details and task.processors:
                doc_details = (
                    sum(
                        await gather_with_concurrency(
                            len(task.processors),
                            task.processor_queries(doc_details, session),
                        )
                    )
                    or None
                )

            if doc_details:
                # abuse int handling in __add__ for empty all_doc_details, None types won't work
                all_doc_details = doc_details + (all_doc_details or 0)

                if not all_doc_details.is_hydration_needed(
                    inclusion=kwargs.get("fields", [])
                ):
                    logger.debug(
                        "All requested fields are present in the DocDetails "
                        f"object{', stopping early.' if ti != len(self.tasks) - 1 else '.'}"
                    )
                    break

        if self._session is None:
            await session.close()

        return all_doc_details

    async def bulk_query(
        self, queries: Collection[dict[str, Any]], concurrency: int = 10
    ) -> list[DocDetails]:
        return await gather_with_concurrency(
            concurrency,
            [cast("Awaitable[DocDetails]", self.query(**kwargs)) for kwargs in queries],
        )

    async def upgrade_doc_to_doc_details(self, doc: Doc, **kwargs) -> DocDetails:

        # note we have some extra fields which may have come from reading the doc text,
        # but aren't in the doc object, we add them here too.
        extra_fields = {
            k: v for k, v in kwargs.items() if k in set(DocDetails.model_fields)
        }
        # abuse our doc_details object to be an int if it's empty
        # our __add__ operation supports int by doing nothing
        extra_doc: int | DocDetails = (
            0 if not extra_fields else DocDetails(**extra_fields)
        )

        if doc_details := await self.query(**kwargs):

            # hard overwrite the details from the prior object
            if "dockey" in doc.fields_to_overwrite_from_metadata:
                doc_details.dockey = doc.dockey
            if "doc_id" in doc.fields_to_overwrite_from_metadata:
                doc_details.doc_id = doc.dockey
            if "docname" in doc.fields_to_overwrite_from_metadata:
                doc_details.docname = doc.docname
            if "key" in doc.fields_to_overwrite_from_metadata:
                doc_details.key = doc.docname
            if "citation" in doc.fields_to_overwrite_from_metadata:
                doc_details.citation = doc.citation
            return extra_doc + doc_details

        # if we can't get metadata, just return the doc, but don't overwrite any fields
        prior_doc = doc.model_dump()
        prior_doc["fields_to_overwrite_from_metadata"] = set()
        return DocDetails(**(prior_doc | extra_fields))



================================================
FILE: paperqa/clients/client_models.py
================================================
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Any, Generic, TypeVar

import aiohttp
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from tenacity import RetryError

from paperqa.types import DocDetails

from .exceptions import DOINotFoundError

logger = logging.getLogger(__name__)


# ClientQuery is a base class for all queries to the client_models
class ClientQuery(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: aiohttp.ClientSession


class TitleAuthorQuery(ClientQuery):
    title: str
    authors: list[str] = []
    title_similarity_threshold: float = 0.75
    fields: Collection[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def ensure_fields_are_present(cls, data: dict[str, Any]) -> dict[str, Any]:
        if fields := data.get("fields"):
            if "doi" not in fields:
                fields.append("doi")
            if "title" not in fields:
                fields.append("title")
            if data.get("authors") is not None and "authors" not in fields:
                fields.append("authors")
            # ensure these are ranked the same for caching purposes
            data["fields"] = sorted(fields)
        return data

    @field_validator("title_similarity_threshold")
    @classmethod
    def zero_and_one(cls, v: float, info: ValidationInfo) -> float:  # noqa: ARG003
        if v < 0.0 or v > 1.0:
            raise ValueError(
                "title_similarity_threshold must be between 0 and 1. (inclusive)"
            )
        return v


class DOIQuery(ClientQuery):
    doi: str
    fields: Collection[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def add_doi_to_fields_and_validate(cls, data: dict[str, Any]) -> dict[str, Any]:

        if (fields := data.get("fields")) and "doi" not in fields:
            fields.append("doi")

        # sometimes the DOI has a URL prefix, remove it
        remove_urls = ["https://doi.org/", "http://dx.doi.org/"]
        for url in remove_urls:
            if data["doi"].startswith(url):
                data["doi"] = data["doi"].replace(url, "")

        return data


class JournalQuery(ClientQuery):
    journal: str


ClientQueryType = TypeVar("ClientQueryType", bound=ClientQuery)


class MetadataProvider(ABC, Generic[ClientQueryType]):
    """Provide metadata from a query by any means necessary."""

    async def query(self, query: dict) -> DocDetails | None:
        return await self._query(self.query_transformer(query))

    @abstractmethod
    async def _query(self, query: ClientQueryType) -> DocDetails | None:
        pass

    @abstractmethod
    def query_transformer(self, query: dict) -> ClientQueryType:
        pass


class DOIOrTitleBasedProvider(MetadataProvider[DOIQuery | TitleAuthorQuery]):

    async def query(self, query: dict) -> DocDetails | None:
        try:
            client_query = self.query_transformer(query)
            return await self._query(client_query)
        # We allow graceful failures, i.e. return "None" for both DOI errors and timeout errors
        # DOINotFoundError means the paper doesn't exist in the source, the timeout is to prevent
        # this service from failing us when it's down or slow.
        except DOINotFoundError:
            logger.warning(
                "Metadata not found for"
                f" {client_query.doi if isinstance(client_query, DOIQuery) else client_query.title} in"
                f" {self.__class__.__name__}."
            )
        except RetryError:
            logger.warning(
                "Metadata service is down for"
                f" {client_query.doi if isinstance(client_query, DOIQuery) else client_query.title} in"
                f" {self.__class__.__name__}."
            )
        except TimeoutError:
            logger.warning(
                f"Request to {self.__class__.__name__} for"
                f" {client_query.doi if isinstance(client_query, DOIQuery) else client_query.title} timed"
                " out."
            )
        return None

    @abstractmethod
    async def _query(self, query: DOIQuery | TitleAuthorQuery) -> DocDetails | None:
        """
        Query the source using either a DOI or title/author search.

        None should be returned if the DOI or title is not a good match.

        Raises:
            DOINotFoundError: This is when the DOI or title is not found in the sources
            TimeoutError: When the request takes too long on the client side
        """

    def query_transformer(self, query: dict) -> DOIQuery | TitleAuthorQuery:
        try:
            if "doi" in query:
                return DOIQuery(**query)
            if "title" in query:
                return TitleAuthorQuery(**query)
        except ValidationError as e:
            raise ValueError(
                f"Query {query} format not supported by {self.__class__.__name__}."
            ) from e

        raise ValueError("Provider query missing 'doi' or 'title' field.")


class MetadataPostProcessor(ABC, Generic[ClientQueryType]):
    """Post-process metadata from a query.

    MetadataPostProcessor should be idempotent and not order-dependent, i.e.
    all MetadataPostProcessor instances should be able to run in parallel.

    """

    async def process(self, doc_details: DocDetails, **kwargs) -> DocDetails:
        if query := self.query_creator(doc_details, **kwargs):
            return await self._process(query, doc_details)
        return doc_details

    @abstractmethod
    async def _process(
        self, query: ClientQueryType, doc_details: DocDetails
    ) -> DocDetails:
        pass

    @abstractmethod
    def query_creator(
        self, doc_details: DocDetails, **kwargs
    ) -> ClientQueryType | None:
        pass



================================================
FILE: paperqa/clients/crossref.py
================================================
from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
from collections.abc import Collection
from datetime import datetime
from typing import Any
from urllib.parse import quote

import aiohttp
from anyio import open_file
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from paperqa.types import CITATION_FALLBACK_DATA, DocDetails
from paperqa.utils import BIBTEX_MAPPING as CROSSREF_CONTENT_TYPE_TO_BIBTEX_MAPPING
from paperqa.utils import (
    bibtex_field_extract,
    create_bibtex_key,
    remove_substrings,
    strings_similarity,
    union_collections_to_ordered_list,
)

from .client_models import DOIOrTitleBasedProvider, DOIQuery, TitleAuthorQuery
from .exceptions import DOINotFoundError, make_flaky_ssl_error_predicate

logger = logging.getLogger(__name__)

CROSSREF_HOST = "api.crossref.org"
CROSSREF_BASE_URL = f"https://{CROSSREF_HOST}"
CROSSREF_HEADER_KEY = "Crossref-Plus-API-Token"
CROSSREF_API_REQUEST_TIMEOUT = 5.0
CROSSREF_API_MAPPING: dict[str, Collection[str]] = {
    "title": {"title"},
    "doi": {"DOI"},
    "authors": {"author"},
    "publication_date": {"published"},
    "year": {"published"},
    "volume": {"volume"},
    "issue": {"issue"},
    "publisher": {"publisher"},
    "issn": {"ISSN"},
    "pages": {"page"},
    "journal": {"container-title"},
    "doi_url": {"URL"},
    "url": {"URL"},
    "bibtex": {"bibtex", "type"},
    "citation_count": {"is-referenced-by-count"},
    "bibtex_type": {"type"},
    "citation": {
        "title",
        "DOI",
        "published",
        "volume",
        "issue",
        "publisher",
        "ISSN",
        "page",
        "container-title",
        "is-referenced-by-count",
        "type",
    },
    "source_quality": {"container-title"},
    "doc_id": {"DOI"},
}

_ISSUED_WARNINGS = [False, False]  # 0 is API key, 1 is email


def crossref_headers() -> dict[str, str]:
    """Crossref API key if available, otherwise nothing."""
    try:
        return {CROSSREF_HEADER_KEY: f"Bearer {os.environ['CROSSREF_API_KEY']}"}
    except KeyError:
        if not _ISSUED_WARNINGS[0]:
            _ISSUED_WARNINGS[0] = True
            logger.warning(
                "CROSSREF_API_KEY environment variable not set."
                " Crossref API rate limits may apply."
            )
        return {}


def get_crossref_mailto() -> str:
    """Crossref mailto if available, otherwise a default."""
    try:
        return os.environ["CROSSREF_MAILTO"]
    except KeyError:
        if not _ISSUED_WARNINGS[1]:
            logger.warning(
                "CROSSREF_MAILTO environment variable not set."
                " Crossref API rate limits may apply."
            )
            _ISSUED_WARNINGS[1] = True
        return "example@papercrow.ai"


@retry(
    retry=retry_if_exception(make_flaky_ssl_error_predicate(CROSSREF_HOST)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    stop=stop_after_attempt(5),
)
async def doi_to_bibtex(
    doi: str,
    session: aiohttp.ClientSession,
    missing_replacements: dict[str, str] | None = None,
) -> str:
    """Get a bibtex entry from a DOI via Crossref, replacing the key if possible.

    `missing_replacements` can optionally be used to fill missing fields in the bibtex key.
        these fields are NOT replaced or inserted into the bibtex string otherwise.

    """
    if missing_replacements is None:
        missing_replacements = {}
    FORBIDDEN_KEY_CHARACTERS = {"_", " ", "-", "/"}
    # get DOI via crossref
    async with session.get(
        f"https://api.crossref.org/works/{quote(doi, safe='')}/transform/application/x-bibtex",
        headers=crossref_headers(),
    ) as r:
        if not r.ok:
            raise DOINotFoundError(
                f"Per HTTP status code {r.status}, could not resolve DOI {doi}."
            )
        data = await r.text()
    # must make new key
    key = data.split("{")[1].split(",")[0]
    new_key = remove_substrings(key, FORBIDDEN_KEY_CHARACTERS)
    substrings_to_remove_per_field = {"author": [" and ", ","]}
    fragments = [
        remove_substrings(
            bibtex_field_extract(
                data, field, missing_replacements=missing_replacements
            ),
            substrings_to_remove_per_field.get(field, []),
        )
        for field in ("author", "year", "title")
    ]
    # replace the key if all the fragments are present
    if all(fragments):
        new_key = create_bibtex_key(
            author=fragments[0].split(), year=fragments[1], title=fragments[2]
        )
    # we use the count parameter below to ensure only the 1st entry is replaced
    return data.replace(key, new_key, 1)


async def parse_crossref_to_doc_details(
    message: dict[str, Any],
    session: aiohttp.ClientSession,
    query_bibtex: bool = True,
) -> DocDetails:

    bibtex_source = "self_generated"
    bibtex = None

    with contextlib.suppress(DOINotFoundError):
        # get the title from the message, if it exists
        # rare circumstance, but bibtex may not have a title
        fallback_data = copy.copy(CITATION_FALLBACK_DATA)
        if title := (
            None if not message.get("title") else message.get("title", [None])[0]
        ):
            fallback_data["title"] = title

        # TODO: we keep this for robustness, but likely not needed anymore,
        # since we now create the bibtex from scratch
        if query_bibtex:
            bibtex = await doi_to_bibtex(
                message["DOI"], session, missing_replacements=fallback_data  # type: ignore[arg-type]
            )
            # track the origin of the bibtex entry for debugging
            bibtex_source = "crossref"

    authors = [
        f"{author.get('given', '')} {author.get('family', '')}".strip()
        for author in message.get("author", [])
    ]

    publication_date = None
    if "published" in message and "date-parts" in message["published"]:
        date_parts = message["published"]["date-parts"][0]
        if len(date_parts) >= 3:  # noqa: PLR2004
            publication_date = datetime(date_parts[0], date_parts[1], date_parts[2])
        elif len(date_parts) == 2:  # noqa: PLR2004
            publication_date = datetime(date_parts[0], date_parts[1], 1)
        elif len(date_parts) == 1:
            publication_date = datetime(date_parts[0], 1, 1)

    doc_details = DocDetails(
        key=None if not bibtex else bibtex.split("{")[1].split(",")[0],
        bibtex_type=CROSSREF_CONTENT_TYPE_TO_BIBTEX_MAPPING.get(
            message.get("type", "other"), "misc"
        ),
        bibtex=bibtex,
        authors=authors,
        publication_date=publication_date,
        year=message.get("published", {}).get("date-parts", [[None]])[0][0],
        volume=message.get("volume"),
        issue=message.get("issue"),
        publisher=message.get("publisher"),
        issn=message.get("ISSN", [None])[0],
        pages=message.get("page"),
        journal=(
            None
            if not message.get("container-title")
            else message["container-title"][0]
        ),
        url=message.get("URL"),
        title=None if not message.get("title") else message.get("title", [None])[0],
        citation_count=message.get("is-referenced-by-count"),
        doi=message.get("DOI"),
        other={},  # Initialize empty dict for other fields
    )

    # Add any additional fields to the 'other' dict
    for key, value in (
        message | {"client_source": ["crossref"], "bibtex_source": [bibtex_source]}
    ).items():
        if key not in type(doc_details).model_fields:
            if key in doc_details.other:
                doc_details.other[key] = [doc_details.other[key], value]
            else:
                doc_details.other[key] = value

    return doc_details


@retry(
    retry=retry_if_exception(make_flaky_ssl_error_predicate(CROSSREF_HOST)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    stop=stop_after_attempt(5),
)
async def get_doc_details_from_crossref(  # noqa: PLR0912
 