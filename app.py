import streamlit as st
import pandas as pd

import numpy as np
from openai.embeddings_utils import distances_from_embeddings

import openai

# Load CSV data
@st.cache_data  # Decorate the function with st.cache to cache its result
def load_data():
    df=pd.read_csv('processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)  # Replace with your CSV file path
    return df

def create_context(
    question, df, max_len=1800, size="ada"
    ):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )

    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


# import streamlit as st

# def clear_text():
#     st.session_state["text"] = ""

# def main():
#     st.title("Chat Interface")

#     # data = load_data()

#     # Initialize session history to store messages
#     if "session_history" not in st.session_state:
#         st.session_state.session_history = []

#     # Create a placeholder for displaying session history
#     chat_history_placeholder = st.empty()

#     user_input = st.text_input("Ask a question", key="text")

#     # Create a button to simulate user response
#     if st.button("Send"):
#         # Append user input to session history
#         st.session_state.session_history.append(("Question", user_input))


#         # Process user input and generate bot response (replace this with your logic)
#         bot_response = f"Bot: You said: '{user_input}'"
#         # bot_response = answer_question(data, question=user_input)

#         # Append bot response to session history
#         st.session_state.session_history.append(("Answer", bot_response))

#         # Update the chat history placeholder with the updated session history
#         chat_history_placeholder.text("\n".join([f"{sender}: {message}" for sender, message in st.session_state.session_history]))

#         # Clear the user input after processing
#         st.button("clear text input", on_click=clear_text)

#     # Display session history by default
#     chat_history_placeholder.text("\n\n".join([f"{sender}: {message}" for sender, message in st.session_state.session_history]))

# if __name__ == "__main__":
#     main()


import streamlit as st

def clear_text():
    st.session_state["text"] = ""

def format_session_history(session_history, max_characters=70):
    formatted_messages = []

    for i, (sender, message) in enumerate(session_history):
        # Insert line breaks while preserving words
        message_with_line_breaks = insert_line_breaks(message, max_characters)

        formatted_messages.append(f"{sender}: {message_with_line_breaks}")

        # Add a newline every other message to alternate between gap and no gap
        if i % 2 != 0:
            formatted_messages.append("")

    return "\n".join(formatted_messages)

def insert_line_breaks(message, max_characters):
    words = message.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) <= max_characters:
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "

    lines.append(current_line)
    return "\n".join(lines)


def main():
    st.title("Travel support chatbot")
    st.write("#### I can help with any info you would get from the FCDO travel advice country pages e.g. Do i need a visa to travel to india?")
    # st.write("#### e.g. Do i need a visa to travel to india?")

    data = load_data()

    # Initialize session history to store messages
    if "session_history" not in st.session_state:
        st.session_state.session_history = []

    # Create a placeholder for displaying session history
    chat_history_placeholder = st.empty()

    user_input = st.text_input("Ask me a question", key="text")

    col1, col2, col3, col4, col5  = st.columns(5)

    with col1:
    # Create a button to simulate user response
        if st.button("Submit"):
            # Append user input to session history
            st.session_state.session_history.append(("Question", user_input))

            # Process user input and generate bot response (replace this with your logic)
            if user_input == "":
                bot_response = "Please ask a question."
            else:
                bot_response = user_input                                                    # used for UI testing
                # bot_response = answer_question(data, question=user_input)

            # Append bot response to session history
            st.session_state.session_history.append(("Answer", bot_response))

            # Update the chat history placeholder with the updated session history
            formatted_history = format_session_history(st.session_state.session_history, max_characters=70)
            chat_history_placeholder.text(formatted_history)



            if user_input != "":
                with col2:
                    # Clear the user input after processing
                    st.button("Clear Question", on_click=clear_text)

        # Display session history by default
        formatted_history = format_session_history(st.session_state.session_history)
        chat_history_placeholder.text(formatted_history)
        st.write(formatted_history)


    # a button to clear and dislpay empty chat history
    if st.button("Clear chat history"):
        st.session_state.session_history = []
        formatted_history = format_session_history(st.session_state.session_history)
        chat_history_placeholder.text(formatted_history)


if __name__ == "__main__":
    main()
