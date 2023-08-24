import streamlit as st

from query_data import get_basic_qa_chain

agent = get_basic_qa_chain()


st.set_page_config(page_title='🦜🔗 Ask the FCO App')
st.title('🦜🔗 Ask the FCO App')
st.info("This is an attempt to create an application using [LangChain](https://github.com/langchain-ai/langchain) to let you ask questions of the UK Foreign Travel Advice. For this demo application, we will use the country specific data. Please explore it [here](https://www.gov.uk/foreign-travel-advice/italy) to get a sense for what questions you can ask. This is just an example country and you are able to ask questions for any country. Please leave feedback on how well the question is answered, and we will use that improve the application!")

query_text = st.text_input('Enter your question:', placeholder = 'Do i need a vist to visit Argentina?')
# Form input and query
result = None
with st.form('myform', clear_on_submit=True):
	submitted = st.form_submit_button('Submit')
	if submitted:
		with st.spinner('Calculating...'):
			response = agent({"query": query_text}, include_run_info=True)
			result = response["result"]
			run_id = response["__run"].run_id

if result is not None:
	st.info(result)
