import streamlit as st
import pypdf
from rag_pipeline import answer_query, retrieve_docs, llm


#Step 1 : Set up Upload PDF functionality
#Step 2 : Chatbot Skeleton

st.set_page_config(
    page_title= "LexIQ 🧑‍⚖️",
    page_icon= "⚖️" ,
    layout= "centered" ,
    initial_sidebar_state= "collapsed"
)
st.markdown("""
    <h1 style='text-align: center;'>LexIQ 🧑‍⚖️</h1>
    <p style='text-align: center; color: gray;'>Upload a legal document and ask any question related to it</p>
    <hr style="margin-top: 0;"/>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload one or more PDF files : ",
                                  type="pdf",
                                  accept_multiple_files= True
                                  )

# If you want to display PDF content 
# if uploaded_file is not None :
#     pdf_reader = pypdf.PdfReader(uploaded_file)
#     content = " "

#     for page in range(pdf_reader.get_num_pages()) :
#         content += pdf_reader.get_page(page).extract_text()

#     lines = content.strip().split("\n")
#     first_3_lines = "\n".join(lines[:10])

#     st.write("### First 3 lines from the PDF:")
#     st.write(first_3_lines)

user_query = st.text_area("💬 Enter your legal question : ", height=90, placeholder="Ask Anything !")

ask_question = st.button("🔍 Ask AI Lawyer")

if ask_question :
    if not uploaded_file and not user_query.strip() :
        st.error("⚠️ Kindly upload a valid PDF file and enter a query!")
    elif not uploaded_file :
        st.error("⚠️ Kindly upload a valid PDF file first !")
    elif not user_query.strip() :
        st.error("⚠️ Please enter a query first !")
    else :
        st.chat_message(name="User", avatar="👤").write(user_query)

        #RAG Pipeline
        retrieved = retrieve_docs(user_query)  
        response = answer_query(documents=retrieved, model=llm, query= user_query)
        # fixed_response = "Hi, this is a fixed response !"
        st.chat_message("AI Lawyer", avatar="⚖️").write(response)


