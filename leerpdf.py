
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


class PDFReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_pdf(self):
        loader = PyPDFLoader(self.file_path)
        pages = loader.load_and_split()
        return pages

    def text(self, tam):
        loader = PyPDFLoader(self.file_path)
        pages = loader.load_and_split()
        # Objeto que va a hacer los cortes en el texto
        split = CharacterTextSplitter(chunk_size=tam, separator='.\n')
        text = split.split_documents(pages)  # Lista de textos
        return text





