import streamlit as st
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from backend import chatbot,model
import uuid
from database import (
    init_db,
    create_thread,
    update_title,
    get_all_threads,
    get_thread_title
)


def generate_title(user_message):

    prompt = [
        SystemMessage(content="""
        You are a conversation title generator.

        Generate a short title for the conversation.

        Rules:
        - Maximum 6 words.
        - Capture the main topic.
        - No explanations.
        - No quotation marks.
        - Return only the title.
        """),

        HumanMessage(content=user_message)
    ]

    response = model.invoke(prompt)

    return response.content.strip()

# gen uuid
def gen_uuid():
    return str(uuid.uuid4())

# reset button
def reset_chat():
    uid = gen_uuid()
    create_thread(uid)
    st.session_state['thread_id'] = uid
    st.session_state['msg_state'] = []
    st.session_state['thread_chat'] = get_all_threads()


        
def load_conver(thread_id):
    state = chatbot.get_state({"configurable":{"thread_id":thread_id}})
    if not state or not state.values:
        return []
    return state.values.get('messages',[])


# session Setup

init_db()

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = None

if 'thread_chat' not in st.session_state:
    st.session_state['thread_chat'] = get_all_threads()

if 'msg_state' not in st.session_state:
    st.session_state['msg_state'] = []

# Sidebar Setup

st.sidebar.title("Langraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversation")

for thread_id,title in st.session_state['thread_chat']:
    if st.sidebar.button(title,key=f"thread_{thread_id}"):
        tmp_msg = []
        
        messages = load_conver(thread_id)
        
        for msg in messages:
            if isinstance(msg,HumanMessage):
                role = 'user'
            elif isinstance(msg,AIMessage):
                role = 'assistant'
            else:
                continue
            
            tmp_msg.append({"role":role,"msg":msg.content})
        st.session_state['thread_id'] = thread_id
        st.session_state['msg_state'] = tmp_msg

# Main UI
for msg in st.session_state['msg_state']:
    with st.chat_message(msg['role']):
        st.markdown(msg['msg'])

uinput = st.chat_input('Type Here')

if uinput:
    if st.session_state['thread_id'] is None:
        tid = gen_uuid()
        create_thread(tid)
        st.session_state['thread_id'] = tid
        
    tid = st.session_state['thread_id']
    if get_thread_title(tid) == 'NewChat':
        try:
            title = generate_title(uinput)
        except Exception as e:
            print(e)
            title =  uinput[:30]
        update_title(tid,title)
        st.session_state['thread_chat'] = get_all_threads()
    
    st.session_state['msg_state'].append({'role':'user','msg':uinput})
    with st.chat_message('user'):
        st.markdown(uinput)
    
    with st.chat_message('assistant'):
        CONFIG={'configurable':{'thread_id':st.session_state['thread_id']}}
        ai_msg = st.write_stream(
            msg_chunk.content for msg_chunk,metadata in  chatbot.stream(
                {'messages':[HumanMessage(content=uinput)]},
                config=CONFIG,
                stream_mode='messages'
                )
        )
    
    st.session_state['msg_state'].append({'role':'assistant','msg':ai_msg})