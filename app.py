import streamlit as st
import pandas as pd

import numpy as np
from openai.embeddings_utils import distances_from_embeddings

import openai

# df=pd.read_csv('processed/embeddings.csv', index_col=0)
# df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

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


# # Create a text input widget
# user_input = st.text_input("Ask a question", "")

# if user_input:
#     answer = answer_question(df, question=user_input)

#     # Display the input
#     st.write("Answer:", answer)


import streamlit as st

def main():
    st.title("Chat Interface")

    # Initialize session history to store messages
    if "session_history" not in st.session_state:
        st.session_state.session_history = []

    # Create a placeholder for displaying session history
    chat_history_placeholder = st.empty()

    # Create a text area for user input
    user_input = st.text_area("You:", value="", height=100)

    # Create a button to simulate user response
    if st.button("Send"):
        # Append user input to session history
        st.session_state.session_history.append(("User", user_input))

        # Process user input and generate bot response (replace this with your logic)
        bot_response = f"Bot: You said: '{user_input}'"

        # Append bot response to session history
        st.session_state.session_history.append(("Bot", bot_response))

        # Update the chat history placeholder with the updated session history
        chat_history_placeholder.text("\n".join([f"{sender}: {message}" for sender, message in st.session_state.session_history]))

    # Display session history by default
    chat_history_placeholder.text("\n".join([f"{sender}: {message}" for sender, message in st.session_state.session_history]))

if __name__ == "__main__":
    main()
