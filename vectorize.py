import os
import pdfplumber
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.qa_with_sources.vector_db import VectorDBQAWithSourcesChain
from langchain_openai import OpenAI
import gradio as gr
from gtts import gTTS
from langdetect import detect


def get_pdf_content(pdf_files):
    """Get contents from a PDF file."""
    pdf_text = []
    pdf_sources = []
    filtered_lines = ""
    filtered_lines_tables = ""
    # Delete sentences with those words in contents
    forbidden = ["http", "https", "html", ".fr", ".pdf", ".php", ".gov"]

    for pdf in pdf_files:
        with pdfplumber.open(pdf) as pdf_file:
            for page in pdf_file.pages:
                # Get texts from the page
                content = page.extract_text()
                # Keep only sentences without forbidden words
                filtered_lines += " ".join([line for line in content.splitlines() if not any(word.lower() in line.lower() for word in forbidden)])
                # Get the texts from tables
                content_tables = get_tables_content(page)
                filtered_lines_tables += " ".join([line for line in content_tables.splitlines() if not any(word.lower() in line.lower() for word in forbidden)])
 
            concat = filtered_lines + filtered_lines_tables
            
            chunk_size = 4096 # Maximum context length of the model
            chunks = [concat[i:i+chunk_size] for i in range(0, len(concat), chunk_size)]
            sources = [pdf for _ in range(len(chunks))]
            pdf_text.extend(chunks)
            pdf_sources.extend(sources)

            filtered_lines = ""
            filtered_lines_tables = ""

    return pdf_text, pdf_sources


def get_tables_content(page):
    """Get contents from cells in tables in a PDF page."""
    result = ""
    tables = page.extract_tables()
    if tables:
        # Iterate over each row in the table
        for table in tables:
            for row in table:
                combined_row = []
                # Iterate over each cell in the row
                for i in range(len(row)):
                    # Check if the current cell is not None and if the next cell is also not None
                    if row[i] is not None and (i + 1 < len(row) and row[i + 1] is not None):
                        combined_row.append(str(row[i]) + " ")
                    else:
                        combined_row.append(str(row[i]) if row[i] is not None else " ")

                result += " ".join(combined_row) + "\n"
    return result


def dectect_lang(text):
    """Detect the language of a text."""
    try:
        return detect(text)
    except:
        return 'en'
    

def get_response(query):
    """Get the response of the model for a query."""
    pdf_directory = "PDF database" 
    pdf_files = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith('.pdf')]
    contents, sources = get_pdf_content(pdf_files)

    # Set the OpenAI API key as an environment variable
    os.environ['API_KEY'] = 'YOUR_OPENAI_API_KEY'

    # Set the embeddings model
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['API_KEY'])

    # Create a Chroma vectorstore.
    dir = "settings"
    vectordb = Chroma.from_texts(contents, embeddings, metadatas=[{'source': s} for s in sources], persist_directory=dir)

    # Initialize Chroma vector store
    vectordb = Chroma(persist_directory=dir, embedding_function=embeddings)

    # Initialize QA model
    qa = VectorDBQAWithSourcesChain.from_chain_type(llm=OpenAI(openai_api_key=os.environ['API_KEY']), k=1, chain_type="stuff", vectorstore=vectordb)

    # Get the answer
    res = qa(query)

    answer = res['answer']
    source = res['sources']

    # Generation the audio description of the answer
    if os.path.exists("response_audio.mp3"):
        os.remove("response_audio.mp3")

    audio_response = gTTS(text=answer, lang=dectect_lang(answer))
    audio_response.save("response_audio.mp3")

    return answer, source, "response_audio.mp3"

css = ".gradio-container {background: url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQEAtkYmbQu9OMP1j7bjv0G0rgO12JhLXFQ1uDMSYlUyQ&s)}"

# Gradio interface
interface = gr.Interface(
    inputs=[
        gr.Textbox(label="Question"),
    ],
    fn=get_response,
    outputs=[gr.Text(label="Answer"), gr.Text(label="Source"),  gr.Audio(label="Audio transcription")],
    title="Q&A",
    theme=gr.themes.Monochrome(),
    css=css
)

if os.path.exists("response_audio.mp3"):
    os.remove("response_audio.mp3")

os.system("start \"\" http://127.0.0.1:7860")
interface.launch()
