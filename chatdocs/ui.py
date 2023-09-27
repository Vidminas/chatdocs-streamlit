import argparse
from typing import Any
from uuid import UUID

import langchain  # unused but needed to avoid circular import errors
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import LLMResult

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit import runtime

# allow relative imports when running with streamlit
if runtime.exists() and not __package__:
    from pathlib import Path

    __package__ = Path(__file__).parent.name

from .chains import get_retrieval_qa
from .st_utils import load_config


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = run_id
            return
        self.status = self.container.status(
            label="Thinking...", state="running", expanded=True
        )
        with self.status.chat_message("assistant"):
            self.placeholder = st.empty()

    def on_llm_new_token(self, token: str, *, run_id: UUID, **kwargs: Any) -> None:
        if self.run_id_ignore_token == run_id:
            return
        self.text += token
        self.placeholder.markdown(self.text)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> Any:
        if self.run_id_ignore_token == run_id:
            return
        self.status.update(state="complete")


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container: DeltaGenerator):
        self.container = container

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status = self.container.status(
            label=f"**Context Retrieval:** {query}", state="running"
        )
        self.status.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source, content = doc.metadata["source"], doc.page_content
            self.status.divider()
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(content)
        self.status.update(state="complete")


def init_messages(msgs: StreamlitChatMessageHistory) -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or len(msgs.messages) == 0:
        msgs.clear()


def print_state_messages(msgs: StreamlitChatMessageHistory):
    roles = {
        "human": "user",
        "ai": "assistant",
    }

    for message in msgs.messages:
        with st.chat_message(roles[message.type]):
            st.markdown(message.content)


@st.cache_resource
def load_qa_chain(config):
    return get_retrieval_qa(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        type=str,
        nargs="?",
        help="Custom path to a chatdocs.yml configuration file.",
    )
    args = parser.parse_args()

    st.session_state["config_path"] = args.config_path

    st.title("ChatDocs")
    st.sidebar.title("Options")

    msgs = StreamlitChatMessageHistory(key="messages")
    init_messages(msgs)
    print_state_messages(msgs)

    config = load_config()
    qa = load_qa_chain(config)

    if prompt := st.chat_input("Enter a query"):
        with st.chat_message("user"):
            st.markdown(prompt)
        msgs.add_user_message(prompt)

        retrieve_callback = PrintRetrievalHandler(st.container())
        print_callback = StreamHandler(st.empty())
        stdout_callback = StreamingStdOutCallbackHandler()
        response = qa(
            prompt, callbacks=[retrieve_callback, print_callback, stdout_callback]
        )
        msgs.add_ai_message(response["result"])


if __name__ == "__main__":
    st.set_page_config(page_title="ChatDocs", page_icon="📚")
    main()
