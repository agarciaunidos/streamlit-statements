# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from datetime import datetime
from llm_bedrock import retrieval_answer

# Constants for date range and document options

YEAR = ['ALL', '2024','2023', '2022']

def run():
    # Display the application title and caption
    st.title("Policy Statements Document Assistant")
    st.caption("A Digital Services Project")

    # Initialize session state for chat messages if not already present
    st.session_state["messages"] = [{"role": "user", "content": "UUS Assistant"}]

    # Display chat messages from session state
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Sidebar for filtering documents by time period and type
    with st.sidebar:
        st.image('https://unidosus.org/wp-content/themes/unidos/images/unidosus-logo-color-2x.png', use_column_width=True)
        st.title("Filter Data")
        selected_years = st.multiselect('Select Type:', YEAR)

    # Input field for user queries
    prompt = st.chat_input()
    if prompt and len(prompt) > 0:
        st.info("Your Input: " + prompt)
        # Retrieve answer and metadata based on the user's query, selected years, and document types
        answer, metadata = retrieval_answer(prompt,selected_years)
        st.subheader('Answer:')
        st.write(answer)
        st.subheader('Sources:')
        st.data_editor(
            metadata,
            column_config={
                "Source": st.column_config.LinkColumn("Source")
            },
            hide_index=True,
            )
    else:
        st.error("Please enter a query.")

if __name__ == "__main__":
    run()
