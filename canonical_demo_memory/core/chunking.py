import nltk
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

nltk.download('punkt')
import nltk.data
from core.parsing import File


def chunk_sentences(sentences, chunk_size=512):
  sents = []
  current_sent = ""

  for sentence in sentences:
    # If adding the next sentence doesn't exceed the chunk_size,
    # we add the sentence to the current chunk.
    if len(current_sent) + len(sentence) <= chunk_size:
      current_sent += " " + sentence
    else:
      # If adding the sentence would make the chunk too long,
      # we add the current_sent chunk to the list of chunks and start a new chunk.
      sents.append(current_sent)
      current_sent = sentence

  # After going through all the sentences, there may be a chunk that hasn't yet been added to the list.
  # We add it now:
  if current_sent:
    sents.append(current_sent)

  return sents

def chunk_file(
    file: File, chunk_size: int, chunk_overlap: int = 0, model_name="gpt-3.5-turbo"
) -> File:
    """Chunks each document in a file into smaller documents
    according to the specified chunk size and overlap
    where the size is determined by the number of token for the specified model.
    """

    # sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # split each document into chunks
    chunked_docs = []
    for doc in file.docs:
        # sentences = sent_detector.tokenize(doc.page_content)
        # chunks = chunk_sentences(sentences, chunk_size=chunk_size)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = text_splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page": doc.metadata.get("page", 1),
                    "chunk": i + 1,
                    "source": f"{doc.metadata.get('page', 1)}-{i + 1}",
                },
            )
            chunked_docs.append(doc)

    chunked_file = file.copy()
    chunked_file.docs = chunked_docs
    return chunked_file
