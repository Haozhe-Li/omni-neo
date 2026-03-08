import os
import boto3
import tempfile
import logging
from upstash_search import Search
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from core.database.db_user_files import (
    update_file_ready,
    get_file_record,
    update_file_failed,
)

logger = logging.getLogger(__name__)

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("S3_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
    region_name="auto",  # Often needed for R2/Custom endpoints
)

# Initialize Upstash Search Client
try:
    upstash_client = Search.from_env()
    upstash_index = upstash_client.index("omni-documents")
except Exception as e:
    logger.error(f"Failed to initialize Upstash Search: {e}")
    upstash_index = None


def _download_from_s3(bucket: str, key: str, local_path: str):
    s3_client.download_file(bucket, key, local_path)


def process_uploaded_file(file_id: str):
    record = get_file_record(file_id)
    if not record:
        logger.error(f"Record {file_id} not found.")
        return

    if record["category"] == "image":
        update_file_ready(file_id)
        return

    # Download file for parsing
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(temp_dir, record["original_filename"])
        try:
            _download_from_s3(record["s3_bucket"], file_id, local_path)
        except Exception as e:
            logger.error(f"S3 download failed for {file_id}: {e}")
            update_file_failed(file_id)
            return

        try:
            full_text = ""
            if record["file_type"] == "application/pdf":
                loader = PyMuPDF4LLMLoader(local_path)
                docs = loader.load()
                full_text = "\n".join([doc.page_content for doc in docs])
            else:
                with open(local_path, "r", encoding="utf-8") as f:
                    full_text = f.read()

            print(f"Full text length: {len(full_text)}")
            if not upstash_index:
                logger.error("Upstash Index not initialized, marking file as failed.")
                update_file_failed(file_id)
                return

            # Setup splitting
            file_ext = record["original_filename"].lower()
            if record["file_type"] == "text/x-python" or file_ext.endswith(".py"):
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
                )
            elif record["file_type"] in [
                "text/markdown",
                "text/md",
            ] or file_ext.endswith((".md", ".markdown")):
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.MARKDOWN, chunk_size=1000, chunk_overlap=200
                )
            elif file_ext.endswith((".js", ".jsx", ".ts", ".tsx")):
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.JS, chunk_size=1000, chunk_overlap=200
                )
            elif file_ext.endswith(".html"):
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.HTML, chunk_size=1000, chunk_overlap=200
                )
            elif file_ext.endswith((".java")):
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.JAVA, chunk_size=1000, chunk_overlap=200
                )
            elif file_ext.endswith((".cpp", ".c", ".h", ".hpp")):
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.CPP, chunk_size=1000, chunk_overlap=200
                )
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
            chunks = splitter.create_documents([full_text])

            upstash_documents = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_id}-{i}"
                upstash_documents.append(
                    {
                        "id": chunk_id,
                        "content": {"text": chunk.page_content},
                        "metadata": {
                            "thread_id": record["thread_id"],
                            "user_id": record["user_id"],
                            "filename": record["original_filename"],
                        },
                    }
                )

            # Upsert into Upstash Search in batches of 100
            batch_size = 100
            for i in range(0, len(upstash_documents), batch_size):
                batch = upstash_documents[i : i + batch_size]
                upstash_index.upsert(documents=batch)

            update_file_ready(file_id, extracted_text=None, is_rag_indexed=True)
            logger.info(f"File {file_id} processed and indexed into Upstash Search.")

        except Exception as e:
            logger.error(f"Parsing/Indexing failed for {file_id}: {e}")
            update_file_failed(file_id)
            import traceback

            traceback.print_exc()


def get_read_presigned_url(file_id: str) -> str | None:
    """Generate a short-lived presigned URL for LLM Vision models to read images"""
    record = get_file_record(file_id)
    if not record or not record.get("s3_bucket"):
        return None
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": record["s3_bucket"], "Key": file_id},
            ExpiresIn=3600,
        )
        return url
    except Exception as e:
        logger.error(f"Generate presigned URL failed: {e}")
        return None


def get_put_presigned_url(s3_bucket: str, file_id: str, file_type: str) -> str | None:
    """Generate a short-lived presigned URL for frontend direct upload"""
    try:
        url = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": s3_bucket, "Key": file_id, "ContentType": file_type},
            ExpiresIn=600,
        )
        return url
    except Exception as e:
        logger.error(f"Generate put presigned URL failed: {e}")
        return None
