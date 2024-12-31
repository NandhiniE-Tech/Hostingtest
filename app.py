import streamlit as st
from RetrievalPrompt import get_answer  # Import the updated retrieval code

st.title('Your Travel Assistant Chatbot!')

# Function for generating LLM response
def generate_response(input_text):
    response = get_answer(input_text)
    return response

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your travel assistant. How can I help you today with best trips ?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if input := st.chat_input():
    # Add user input to session state
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

    # Generate and display assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Fetching the best travel recommendations..."):
            response = generate_response(input)
            st.write(response)
            # Add assistant's response to session state
            st.session_state.messages.append({"role": "assistant", "content": response})
