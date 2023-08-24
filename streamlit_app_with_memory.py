import streamlit as st
import time
from query_data import get_basic_qa_chain

agent = get_basic_qa_chain()

# App title
st.set_page_config(page_title="🔗💬 Ask the FCO App")

st.title('🦜🔗 Ask the FCO App')
st.info("This is an attempt to create an application using [LangChain](https://github.com/langchain-ai/langchain) to let you ask questions of the UK Foreign Travel Advice. For this demo application, we will use the country specific data. Please explore an example [here](https://www.gov.uk/foreign-travel-advice/italy) to get a sense for what questions you can ask. This is just an example country and you are able to ask questions for any country. Please leave feedback on how well the question is answered, and we will use that improve the application!")


# Store generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
def generate_response(prompt_input):
    time.sleep(2)
    return f"{prompt_input}: botresponse"

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant" and st.session_state.messages[-1]["content"] != None:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent({"query": prompt}, include_run_info=True)
            result = response["result"]
            st.write(result)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
