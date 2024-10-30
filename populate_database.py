import argparse
import os
import shutil
from pathlib import Path
import pandas as pd
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function


# CHROMA_PATH = Path('chroma')              # 800 - 80
# CHROMA_PATH = Path('chroma_1000_150')     # 1000 - 150
# CHROMA_PATH = Path('chroma_csv')          # 1000 - 150
CHROMA_PATH = Path('chroma_csv_pdf')

PDF_PATH = Path('data/pdf')
CSV_PATH = Path('data/csv')


def main():
    # Check if the database should be cleared (using the --reset flag)
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    # Create (or update) the data store.
    chunks = load_documents()
    add_to_chroma(chunks)


def load_documents():
    """
    Load PDF and CSV documents from the predefined directories.

    Returns:
        list: A list containing the chunks extracted from the files.
    """
    pdf_documents = pdf_loader()
    pdf_chunks = split_documents(pdf_documents)
    csv_chuncks = csv_loader()
    return pdf_chunks + csv_chuncks


def pdf_loader():
    """
    Processes a list of PDF files, loads data, and returns the aggregated data.

    Returns:
        list: A list containing the content of the PDF files.
    """
    document_loader = PyPDFDirectoryLoader(str(PDF_PATH))
    return document_loader.load()


def parse_csv_to_list(file_path):
    """
    Processes the content of a CSV file, loads data, and create a list containing one element per row. 
    Each element of the list corresponds to a string. The string content is structured as follows:

    "
    col_1: df[x][1]
    ...
    col_n: df[x][n]
    "

    If the content of a cell is missing, then the corresponding row in the string is removed.

    Returns:
        list: A list containing the content of the CSV file. Every entry in the list correspond to a row of the CSV
    """
    df = pd.read_csv(file_path)
    formatted_rows = []
    
    for index, row in df.iterrows():
        
        row_content = []
        for col in df.columns:
            cell_content = row[col]
            # Only include cells that are not empty
            if pd.notna(cell_content):
                row_content.append(f"{col}: {cell_content}")
        
        # Join each column's content with a newline and add to the list
        document = Document(
            page_content="\n".join(row_content),
            metadata={'source': file_path, 'row': index}
        )
        formatted_rows.append(document)
    
    return formatted_rows


def csv_loader():
    """
    Processes a list of CSV files, loads data, and returns the aggregated data.

    Returns:
        list: A list containing the aggregated data from all CSV files. Every entry in the list correspond to a row of the CSV
    """
    file_paths = [f for f in CSV_PATH.iterdir() if f.is_file() and f.suffix == '.csv']
    data = []
    print(f"Loading {len(file_paths)} files.\nFile names: {file_paths}")

    for file_path in file_paths:
        file_data = parse_csv_to_list(file_path=str(file_path))  # Ensure loader receives a string path
        data.extend(file_data)

        print(file_data[0], "\n\n", file_data[-1], "\n\n")

    return data


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=str(CHROMA_PATH), embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks):
    """
    Create chunks IDs like "data/example.pdf:6:2"
    [Page Source : Page Number : Chunk Index]

    Returns:
        list: A list of chunks with their IDs.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(str(CHROMA_PATH)):
        shutil.rmtree(str(CHROMA_PATH))


if __name__ == "__main__":
    main()

