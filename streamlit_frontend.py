import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage
from backend import chatbot

if 'msg_state' not in st.session_state:
    st.session_state['msg_state'] = []
else:
    for msg in st.session_state['msg_state']:
        with st.chat_message(msg['role']):
            st.text(msg['msg'])


uinput = st.chat_input('Type Here')
if uinput:
    st.session_state['msg_state'].append({'role':'user','msg':uinput})
    with st.chat_message('user'):
        st.text(uinput)
    res = chatbot.invoke({'messages':HumanMessage(content=uinput)},config={'configurable':{'thread_id':'1'}})['messages'][-1].content
    st.session_state['msg_state'].append({'role':'ai','msg':res}) 
    with st.chat_message('ai'):
        st.text(res)