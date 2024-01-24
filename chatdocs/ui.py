import argparse
from typing import Any
from uuid import UUID, uuid4

import langchain  # unused but needed to avoid circular import errors
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.schema.output import LLMResult

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit import runtime

# allow relative imports when running with streamlit
if runtime.exists() and not __package__:
    from pathlib import Path

    __package__ = Path(__file__).parent.name

from .chains import make_conversation_chain
from .st_utils import load_config
from memory.solid_message_history import SolidChatMessageHistory, CssAccount, create_css_account


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
        if prompts[0][:20] == CONDENSE_QUESTION_PROMPT.template[:20]:
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
        if not self.text: # for models that produce all output at once at the end
            self.text = response.flatten()[0].generations[0][0].text
            self.placeholder.markdown(self.text)
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


def init_messages(history: BaseChatMessageHistory) -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or len(history.messages) == 0:
        history.clear()


def print_state_messages(history: BaseChatMessageHistory):
    roles = {
        "human": "user",
        "ai": "assistant",
    }

    for message in history.messages:
        with st.chat_message(roles[message.type]):
            st.markdown(message.content)


@st.cache_resource
def load_llm(config, selected_llm):
    return make_conversation_chain(config, selected_llm_index=selected_llm)


@st.cache_data
def create_random_solid_account(css_base_url: str) -> CssAccount:
    name = f"test-{uuid4()}"
    email = f"{name}@example.org"
    password = "12345"

    return create_css_account(
        css_base_url=css_base_url, name=name, email=email, password=password
    )


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

    st.set_page_config(page_title="ChatDocs", page_icon="📚")
    st.title("ChatDocs")
    st.sidebar.title("Options")

    solid_server_url = st.sidebar.text_input("Solid Server URL", "https://localhost:1234/")
    css_account = create_random_solid_account(solid_server_url)

    history_type = st.sidebar.radio("Message history", ("Local", "Solid"))
    if history_type == "Local":
        history = StreamlitChatMessageHistory(key="messages")
    else:
        history = SolidChatMessageHistory(solid_server_url, css_account)
    init_messages(history)
    print_state_messages(history)

    config = load_config()
    selected_llm = st.sidebar.radio("LLM", range(len(config["llms"])), format_func=lambda idx: config["llms"][idx]["model"])
    llm = load_llm(config, selected_llm)

    if prompt := st.chat_input("Enter a query"):
        with st.chat_message("user"):
            st.markdown(prompt)
        history.add_user_message(prompt)

        retrieve_callback = PrintRetrievalHandler(st.container())
        print_callback = StreamHandler(st.empty())
        stdout_callback = StreamingStdOutCallbackHandler()
        response = llm(
            { "question": prompt, "chat_history": history.messages },
            callbacks=[retrieve_callback, print_callback, stdout_callback],
        )
        history.add_ai_message(response["answer"])


if __name__ == "__main__":
    main()
