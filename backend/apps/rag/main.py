# FastAPI를 사용하여 API 서버를 구축하는데 필요한 모듈을 임포트합니다.
from fastapi import (
    FastAPI,  # FastAPI 애플리케이션 생성에 사용
    Depends,  # 의존성 주입에 사용
    HTTPException,  # HTTP 예외 처리에 사용
    status,  # HTTP 상태 코드에 사용
    UploadFile, File,  # 파일 업로드를 처리하기 위해 사용
    Form,  # 폼 데이터 처리에 사용
)
from fastapi.middleware.cors import CORSMiddleware  # CORS 정책 설정에 사용

import os, shutil, logging, re  # 파일 시스템 조작, 로깅 및 정규 표현식에 사용

from pathlib import Path  # 파일 시스템 경로를 객체 지향적으로 취급
from typing import List  # 타입 힌트에 리스트 사용

# batch 처리 및 다양한 문서 로더를 위한 사용자 정의 모듈 임포트
from chromadb.utils.batch_utils import create_batches

# 다양한 형식의 문서를 로드하는데 사용되는 클래스들
from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    BSHTMLLoader,
    Docx2txtLoader,
    UnstructuredEPubLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredRSTLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    YoutubeLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 텍스트 분할에 사용

import validators  # URL 유효성 검사에 사용
import urllib.parse  # URL 파싱에 사용
import socket  # 네트워크 소켓 작업에 사용


from pydantic import BaseModel  # 데이터 검증 및 설정 관리에 사용
from typing import Optional  # 타입 힌트에 선택적(Optional) 타입 사용
import mimetypes  # MIME 타입 처리에 사용
import uuid  # UUID 생성에 사용
import json  # JSON 데이터 처리에 사용

import sentence_transformers  # 문장 레벨의 임베딩을 생성하기 위해 사용

# 문서 처리와 관련된 모델과 응답 형태를 정의한 사용자 정의 모듈
from apps.web.models.documents import (
    Documents,
    DocumentForm,
    DocumentResponse,
)

# RAG(Retrieval Augmented Generation)와 관련된 유틸리티 함수들
from apps.rag.utils import (
    get_model_path,
    get_embedding_function,
    query_doc,
    query_doc_with_hybrid_search,
    query_collection,
    query_collection_with_hybrid_search,
)

# 다양한 유틸리티 함수들을 포함한 사용자 정의 모듈
from utils.misc import (
    calculate_sha256,
    calculate_sha256_string,
    sanitize_filename,
    extract_folders_after_data_docs,
)
from utils.utils import get_current_user, get_admin_user  # 사용자 인증과 관련된 유틸리티 함수

# 애플리케이션 설정과 관련된 변수들을 포함한 모듈
from config import (
    ENV,
    SRC_LOG_LEVELS,
    UPLOAD_DIR,
    DOCS_DIR,
    RAG_TOP_K,
    RAG_RELEVANCE_THRESHOLD,
    RAG_EMBEDDING_ENGINE,
    RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
    ENABLE_RAG_HYBRID_SEARCH,
    ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
    RAG_RERANKING_MODEL,
    PDF_EXTRACT_IMAGES,
    RAG_RERANKING_MODEL_AUTO_UPDATE,
    RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
    RAG_OPENAI_API_BASE_URL,
    RAG_OPENAI_API_KEY,
    DEVICE_TYPE,
    CHROMA_CLIENT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAG_TEMPLATE,
    ENABLE_RAG_LOCAL_WEB_FETCH,
    YOUTUBE_LOADER_LANGUAGE,
    AppConfig,
)

# 필요한 라이브러리와 모듈을 임포트합니다.
from constants import ERROR_MESSAGES
import logging
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import sentence_transformers

# 로거 설정을 초기화합니다.
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])  # 로깅 레벨을 설정합니다.

app = FastAPI()  # FastAPI 앱 인스턴스를 생성합니다.

# 애플리케이션의 상태 객체에 AppConfig 인스턴스를 할당하여 설정을 관리합니다.
app.state.config = AppConfig()

# 검색 관련 설정 값을 AppConfig 인스턴스에 할당합니다.
app.state.config.TOP_K = RAG_TOP_K
app.state.config.RELEVANCE_THRESHOLD = RAG_RELEVANCE_THRESHOLD

app.state.config.ENABLE_RAG_HYBRID_SEARCH = ENABLE_RAG_HYBRID_SEARCH
app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION = ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION

app.state.config.CHUNK_SIZE = CHUNK_SIZE
app.state.config.CHUNK_OVERLAP = CHUNK_OVERLAP

app.state.config.RAG_EMBEDDING_ENGINE = RAG_EMBEDDING_ENGINE
app.state.config.RAG_EMBEDDING_MODEL = RAG_EMBEDDING_MODEL
app.state.config.RAG_RERANKING_MODEL = RAG_RERANKING_MODEL
app.state.config.RAG_TEMPLATE = RAG_TEMPLATE

app.state.config.OPENAI_API_BASE_URL = RAG_OPENAI_API_BASE_URL
app.state.config.OPENAI_API_KEY = RAG_OPENAI_API_KEY

app.state.config.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

app.state.config.YOUTUBE_LOADER_LANGUAGE = YOUTUBE_LOADER_LANGUAGE
app.state.YOUTUBE_LOADER_TRANSLATION = None  # 유튜브 로더 번역 관련 설정을 None으로 초기화합니다.

# 임베딩 모델을 업데이트하는 함수입니다.
def update_embedding_model(embedding_model: str, update_model: bool = False):
    if embedding_model and app.state.config.RAG_EMBEDDING_ENGINE == "":
        app.state.sentence_transformer_ef = sentence_transformers.SentenceTransformer(
            get_model_path(embedding_model, update_model),
            device=DEVICE_TYPE,
            trust_remote_code=RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
        )
    else:
        app.state.sentence_transformer_ef = None

# 리랭킹 모델을 업데이트하는 함수입니다.
def update_reranking_model(reranking_model: str, update_model: bool = False):
    if reranking_model:
        app.state.sentence_transformer_rf = sentence_transformers.CrossEncoder(
            get_model_path(reranking_model, update_model),
            device=DEVICE_TYPE,
            trust_remote_code=RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
        )
    else:
        app.state.sentence_transformer_rf = None

# 애플리케이션 시작 시 임베딩 모델과 리랭킹 모델을 초기화합니다.
update_embedding_model(app.state.config.RAG_EMBEDDING_MODEL, RAG_EMBEDDING_MODEL_AUTO_UPDATE)
update_reranking_model(app.state.config.RAG_RERANKING_MODEL, RAG_RERANKING_MODEL_AUTO_UPDATE)

# 임베딩 기능을 설정합니다.
app.state.EMBEDDING_FUNCTION = get_embedding_function(
    app.state.config.RAG_EMBEDDING_ENGINE,
    app.state.config.RAG_EMBEDDING_MODEL,
    app.state.sentence_transformer_ef,
    app.state.config.OPENAI_API_KEY,
    app.state.config.OPENAI_API_BASE_URL,
)

origins = ["*"]  # CORS 정책에서 허용할 출처를 설정합니다.

# CORS 미들웨어를 애플리케이션에 추가합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# FastAPI와 관련된 기본 모듈을 가져옵니다.
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

# FastAPI 인스턴스를 생성합니다.
app = FastAPI()

# 컬렉션 이름을 나타내는 폼 모델을 정의합니다.
class CollectionNameForm(BaseModel):
    collection_name: Optional[str] = "test"

# URL을 포함한 폼 모델을 정의합니다. CollectionNameForm을 상속받습니다.
class UrlForm(CollectionNameForm):
    url: str

# 루트 엔드포인트를 정의합니다. 상태 정보를 반환합니다.
@app.get("/")
async def get_status():
    return {
        "status": True,
        "chunk_size": app.state.config.CHUNK_SIZE,
        "chunk_overlap": app.state.config.CHUNK_OVERLAP,
        "template": app.state.config.RAG_TEMPLATE,
        "embedding_engine": app.state.config.RAG_EMBEDDING_ENGINE,
        "embedding_model": app.state.config.RAG_EMBEDDING_MODEL,
        "reranking_model": app.state.config.RAG_RERANKING_MODEL,
    }

# 임베딩 설정 정보를 반환하는 엔드포인트를 정의합니다. 관리자 사용자만 접근 가능합니다.
@app.get("/embedding")
async def get_embedding_config(user=Depends(get_admin_user)):
    return {
        "status": True,
        "embedding_engine": app.state.config.RAG_EMBEDDING_ENGINE,
        "embedding_model": app.state.config.RAG_EMBEDDING_MODEL,
        "openai_config": {
            "url": app.state.config.OPENAI_API_BASE_URL,
            "key": app.state.config.OPENAI_API_KEY,
        },
    }

# 리랭킹 설정 정보를 반환하는 엔드포인트를 정의합니다. 관리자 사용자만 접근 가능합니다.
@app.get("/reranking")
async def get_reraanking_config(user=Depends(get_admin_user)):
    return {
        "status": True,
        "reranking_model": app.state.config.RAG_RERANKING_MODEL,
    }

# OpenAI 설정 정보를 포함하는 폼 모델을 정의합니다.
class OpenAIConfigForm(BaseModel):
    url: str
    key: str

# 임베딩 모델 업데이트 폼 모델을 정의합니다.
class EmbeddingModelUpdateForm(BaseModel):
    openai_config: Optional[OpenAIConfigForm] = None
    embedding_engine: str
    embedding_model: str

# 임베딩 설정을 업데이트하는 엔드포인트를 정의합니다. 관리자 사용자만 접근 가능합니다.
@app.post("/embedding/update")
async def update_embedding_config(
    form_data: EmbeddingModelUpdateForm, user=Depends(get_admin_user)
):
    log.info(
        f"Updating embedding model: {app.state.config.RAG_EMBEDDING_MODEL} to {form_data.embedding_model}"
    )
    try:
        # 새로운 임베딩 엔진과 모델을 설정합니다.
        app.state.config.RAG_EMBEDDING_ENGINE = form_data.embedding_engine
        app.state.config.RAG_EMBEDDING_MODEL = form_data.embedding_model

        # 만약 임베딩 엔진이 "ollama" 또는 "openai"라면, OpenAI 설정을 업데이트합니다.
        if app.state.config.RAG_EMBEDDING_ENGINE in ["ollama", "openai"]:
            if form_data.openai_config != None:
                app.state.config.OPENAI_API_BASE_URL = form_data.openai_config.url
                app.state.config.OPENAI_API_KEY = form_data.openai_config.key

        # 임베딩 모델을 업데이트하는 비즈니스 로직을 호출합니다.
        update_embedding_model(app.state.config.RAG_EMBEDDING_MODEL)

        # 새로운 임베딩 함수 설정을 업데이트합니다.
        app.state.EMBEDDING_FUNCTION = get_embedding_function(
            app.state.config.RAG_EMBEDDING_ENGINE,
            app.state.config.RAG_EMBEDDING_MODEL,
            app.state.sentence_transformer_ef,
            app.state.config.OPENAI_API_KEY,
            app.state.config.OPENAI_API_BASE_URL,
        )

        return {
            "status": True,
            "embedding_engine": app.state.config.RAG_EMBEDDING_ENGINE,
            "embedding_model": app.state.config.RAG_EMBEDDING_MODEL,
            "openai_config": {
                "url": app.state.config.OPENAI_API_BASE_URL,
                "key": app.state.config.OPENAI_API_KEY,
            },
        }
    except Exception as e:
        log.exception(f"Problem updating embedding model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )

# 리랭킹 모델 업데이트 폼 모델을 정의합니다.
class RerankingModelUpdateForm(BaseModel):
    reranking_model: str

# 리랭킹 설정을 업데이트하는 엔드포인트를 정의합니다. 관리자 사용자만 접근 가능합니다.
@app.post("/reranking/update")
async def update_reranking_config(
    form_data: RerankingModelUpdateForm, user=Depends(get_admin_user)
):
    log.info(
        f"Updating reranking model: {app.state.config.RAG_RERANKING_MODEL} to {form_data.reranking_model}"
    )
    try:
        # 새로운 리랭킹 모델을 설정합니다.
        app.state.config.RAG_RERANKING_MODEL = form_data.reranking_model

        # 리랭킹 모델을 업데이트하는 비즈니스 로직을 호출합니다.
        update_reranking_model(app.state.config.RAG_RERANKING_MODEL), True

        return {
            "status": True,
            "reranking_model": app.state.config.RAG_RERANKING_MODEL,
        }
    except Exception as e:
        log.exception(f"Problem updating reranking model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )
# FastAPI 애플리케이션의 인스턴스를 가져옵니다.
@app.get("/config")
# "/config" 경로에 GET 요청이 오면 실행되는 비동기 함수입니다.
async def get_rag_config(user=Depends(get_admin_user)):
    # 사용자 검증을 위해 get_admin_user 종속성을 사용합니다.
    return {
        # 구성 상태를 반환합니다.
        "status": True,
        # PDF 이미지 추출 설정을 반환합니다.
        "pdf_extract_images": app.state.config.PDF_EXTRACT_IMAGES,
        # 청크 설정을 반환합니다.
        "chunk": {
            "chunk_size": app.state.config.CHUNK_SIZE,
            "chunk_overlap": app.state.config.CHUNK_OVERLAP,
        },
        # 웹 로더 SSL 검증 설정을 반환합니다.
        "web_loader_ssl_verification": app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
        # 유튜브 로더 설정을 반환합니다.
        "youtube": {
            "language": app.state.config.YOUTUBE_LOADER_LANGUAGE,
            "translation": app.state.YOUTUBE_LOADER_TRANSLATION,
        },
    }

# 청크 매개변수 업데이트 폼을 정의하는 Pydantic 모델입니다.
class ChunkParamUpdateForm(BaseModel):
    chunk_size: int
    chunk_overlap: int

# 유튜브 로더 설정을 정의하는 Pydantic 모델입니다.
class YoutubeLoaderConfig(BaseModel):
    language: List[str]
    translation: Optional[str] = None

# 구성 업데이트 폼을 정의하는 Pydantic 모델입니다.
class ConfigUpdateForm(BaseModel):
    pdf_extract_images: Optional[bool] = None
    chunk: Optional[ChunkParamUpdateForm] = None
    web_loader_ssl_verification: Optional[bool] = None
    youtube: Optional[YoutubeLoaderConfig] = None

@app.post("/config/update")
# "/config/update" 경로에 POST 요청이 오면 실행되는 비동기 함수입니다.
async def update_rag_config(form_data: ConfigUpdateForm, user=Depends(get_admin_user)):
    # 사용자 검증을 위해 get_admin_user 종속성을 사용합니다.
    app.state.config.PDF_EXTRACT_IMAGES = (
        form_data.pdf_extract_images
        if form_data.pdf_extract_images is not None
        else app.state.config.PDF_EXTRACT_IMAGES
    )

    app.state.config.CHUNK_SIZE = (
        form_data.chunk.chunk_size
        if form_data.chunk is not None
        else app.state.config.CHUNK_SIZE
    )

    app.state.config.CHUNK_OVERLAP = (
        form_data.chunk.chunk_overlap
        if form_data.chunk is not None
        else app.state.config.CHUNK_OVERLAP
    )

    app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION = (
        form_data.web_loader_ssl_verification
        if form_data.web_loader_ssl_verification != None
        else app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION
    )

    app.state.config.YOUTUBE_LOADER_LANGUAGE = (
        form_data.youtube.language
        if form_data.youtube is not None
        else app.state.config.YOUTUBE_LOADER_LANGUAGE
    )

    app.state.YOUTUBE_LOADER_TRANSLATION = (
        form_data.youtube.translation
        if form_data.youtube is not None
        else app.state.YOUTUBE_LOADER_TRANSLATION
    )

    # 업데이트된 설정을 반환합니다.
    return {
        "status": True,
        "pdf_extract_images": app.state.config.PDF_EXTRACT_IMAGES,
        "chunk": {
            "chunk_size": app.state.config.CHUNK_SIZE,
            "chunk_overlap": app.state.config.CHUNK_OVERLAP,
        },
        "web_loader_ssl_verification": app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
        "youtube": {
            "language": app.state.config.YOUTUBE_LOADER_LANGUAGE,
            "translation": app.state.YOUTUBE_LOADER_TRANSLATION,
        },
    }

@app.get("/template")
# "/template" 경로에 GET 요청이 오면 실행되는 비동기 함수입니다.
async def get_rag_template(user=Depends(get_current_user)):
    # 사용자 검증을 위해 get_current_user 종속성을 사용합니다.
    return {
        # 템플릿 상태를 반환합니다.
        "status": True,
        "template": app.state.config.RAG_TEMPLATE,
    }

@app.get("/query/settings")
# "/query/settings" 경로에 GET 요청이 오면 실행되는 비동기 함수입니다.
async def get_query_settings(user=Depends(get_admin_user)):
    # 사용자 검증을 위해 get_admin_user 종속성을 사용합니다.
    return {
        # 쿼리 설정 상태를 반환합니다.
        "status": True,
        "template": app.state.config.RAG_TEMPLATE,
        "k": app.state.config.TOP_K,
        "r": app.state.config.RELEVANCE_THRESHOLD,
        "hybrid": app.state.config.ENABLE_RAG_HYBRID_SEARCH,
    }

# 쿼리 설정 업데이트 폼을 정의하는 Pydantic 모델입니다.
class QuerySettingsForm(BaseModel):
    k: Optional[int] = None
    r: Optional[float] = None
    template: Optional[str] = None
    hybrid: Optional[bool] = None

@app.post("/query/settings/update")
# "/query/settings/update" 경로에 POST 요청이 오면 실행되는 비동기 함수입니다.
async def update_query_settings(
    form_data: QuerySettingsForm, user=Depends(get_admin_user)
):
    # 사용자 검증을 위해 get_admin_user 종속성을 사용합니다.
    app.state.config.RAG_TEMPLATE = (
        form_data.template if form_data.template else RAG_TEMPLATE
    )
    app.state.config.TOP_K = form_data.k if form_data.k else 4
    app.state.config.RELEVANCE_THRESHOLD = form_data.r if form_data.r else 0.0
    app.state.config.ENABLE_RAG_HYBRID_SEARCH = (
        form_data.hybrid if form_data.hybrid else False
    )
    # 업데이트된 쿼리 설정을 반환합니다.
    return {
        "status": True,
        "template": app.state.config.RAG_TEMPLATE,
        "k": app.state.config.TOP_K,
        "r": app.state.config.RELEVANCE_THRESHOLD,
        "hybrid": app.state.config.ENABLE_RAG_HYBRID_SEARCH,
    }

class QuerySettingsForm(BaseModel):
    k: Optional[int] = None
    r: Optional[float] = None
    template: Optional[str] = None
    hybrid: Optional[bool] = None


@app.post("/query/settings/update")
async def update_query_settings(
    form_data: QuerySettingsForm, user=Depends(get_admin_user)
):
    app.state.config.RAG_TEMPLATE = (
        form_data.template if form_data.template else RAG_TEMPLATE
    )
    app.state.config.TOP_K = form_data.k if form_data.k else 4
    app.state.config.RELEVANCE_THRESHOLD = form_data.r if form_data.r else 0.0
    app.state.config.ENABLE_RAG_HYBRID_SEARCH = (
        form_data.hybrid if form_data.hybrid else False
    )
    return {
        "status": True,
        "template": app.state.config.RAG_TEMPLATE,
        "k": app.state.config.TOP_K,
        "r": app.state.config.RELEVANCE_THRESHOLD,
        "hybrid": app.state.config.ENABLE_RAG_HYBRID_SEARCH,
    }


class QueryDocForm(BaseModel):
    collection_name: str
    query: str
    k: Optional[int] = None
    r: Optional[float] = None
    hybrid: Optional[bool] = None


@app.post("/query/doc")
def query_doc_handler(
    form_data: QueryDocForm,
    user=Depends(get_current_user),
):
    try:
        if app.state.config.ENABLE_RAG_HYBRID_SEARCH:
            return query_doc_with_hybrid_search(
                collection_name=form_data.collection_name,
                query=form_data.query,
                embedding_function=app.state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app.state.config.TOP_K,
                reranking_function=app.state.sentence_transformer_rf,
                r=(
                    form_data.r if form_data.r else app.state.config.RELEVANCE_THRESHOLD
                ),
            )
        else:
            return query_doc(
                collection_name=form_data.collection_name,
                query=form_data.query,
                embedding_function=app.state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app.state.config.TOP_K,
            )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


class QueryCollectionsForm(BaseModel):
    collection_names: List[str]
    query: str
    k: Optional[int] = None
    r: Optional[float] = None
    hybrid: Optional[bool] = None


@app.post("/query/collection")
def query_collection_handler(
    form_data: QueryCollectionsForm,
    user=Depends(get_current_user),
):
    try:
        if app.state.config.ENABLE_RAG_HYBRID_SEARCH:
            return query_collection_with_hybrid_search(
                collection_names=form_data.collection_names,
                query=form_data.query,
                embedding_function=app.state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app.state.config.TOP_K,
                reranking_function=app.state.sentence_transformer_rf,
                r=(
                    form_data.r if form_data.r else app.state.config.RELEVANCE_THRESHOLD
                ),
            )
        else:
            return query_collection(
                collection_names=form_data.collection_names,
                query=form_data.query,
                embedding_function=app.state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app.state.config.TOP_K,
            )

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@app.post("/youtube")
def store_youtube_video(form_data: UrlForm, user=Depends(get_current_user)):
    try:
        loader = YoutubeLoader.from_youtube_url(
            form_data.url,
            add_video_info=True,
            language=app.state.config.YOUTUBE_LOADER_LANGUAGE,
            translation=app.state.YOUTUBE_LOADER_TRANSLATION,
        )
        data = loader.load()

        collection_name = form_data.collection_name
        if collection_name == "":
            collection_name = calculate_sha256_string(form_data.url)[:63]

        store_data_in_vector_db(data, collection_name, overwrite=True)
        return {
            "status": True,
            "collection_name": collection_name,
            "filename": form_data.url,
        }
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@app.post("/web")
def store_web(form_data: UrlForm, user=Depends(get_current_user)):
    # "https://www.gutenberg.org/files/1727/1727-h/1727-h.htm"
    try:
        loader = get_web_loader(
            form_data.url,
            verify_ssl=app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
        )
        data = loader.load()

        collection_name = form_data.collection_name
        if collection_name == "":
            collection_name = calculate_sha256_string(form_data.url)[:63]

        store_data_in_vector_db(data, collection_name, overwrite=True)
        return {
            "status": True,
            "collection_name": collection_name,
            "filename": form_data.url,
        }
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


def get_web_loader(url: str, verify_ssl: bool = True):
    # Check if the URL is valid
    if isinstance(validators.url(url), validators.ValidationError):
        raise ValueError(ERROR_MESSAGES.INVALID_URL)
    if not ENABLE_RAG_LOCAL_WEB_FETCH:
        # Local web fetch is disabled, filter out any URLs that resolve to private IP addresses
        parsed_url = urllib.parse.urlparse(url)
        # Get IPv4 and IPv6 addresses
        ipv4_addresses, ipv6_addresses = resolve_hostname(parsed_url.hostname)
        # Check if any of the resolved addresses are private
        # This is technically still vulnerable to DNS rebinding attacks, as we don't control WebBaseLoader
        for ip in ipv4_addresses:
            if validators.ipv4(ip, private=True):
                raise ValueError(ERROR_MESSAGES.INVALID_URL)
        for ip in ipv6_addresses:
            if validators.ipv6(ip, private=True):
                raise ValueError(ERROR_MESSAGES.INVALID_URL)
    return WebBaseLoader(url, verify_ssl=verify_ssl)


def resolve_hostname(hostname):
    # Get address information
    addr_info = socket.getaddrinfo(hostname, None)

    # Extract IP addresses from address information
    ipv4_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET]
    ipv6_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET6]

    return ipv4_addresses, ipv6_addresses


def store_data_in_vector_db(data, collection_name, overwrite: bool = False) -> bool:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=app.state.config.CHUNK_SIZE,
        chunk_overlap=app.state.config.CHUNK_OVERLAP,
        add_start_index=True,
    )

    docs = text_splitter.split_documents(data)

    if len(docs) > 0:
        log.info(f"store_data_in_vector_db {docs}")
        return store_docs_in_vector_db(docs, collection_name, overwrite), None
    else:
        raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)


def store_text_in_vector_db(
    text, metadata, collection_name, overwrite: bool = False
) -> bool:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=app.state.config.CHUNK_SIZE,
        chunk_overlap=app.state.config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    docs = text_splitter.create_documents([text], metadatas=[metadata])
    return store_docs_in_vector_db(docs, collection_name, overwrite)


def store_docs_in_vector_db(docs, collection_name, overwrite: bool = False) -> bool:
    log.info(f"store_docs_in_vector_db {docs} {collection_name}")

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    try:
        if overwrite:
            for collection in CHROMA_CLIENT.list_collections():
                if collection_name == collection.name:
                    log.info(f"deleting existing collection {collection_name}")
                    CHROMA_CLIENT.delete_collection(name=collection_name)

        collection = CHROMA_CLIENT.create_collection(name=collection_name)

        embedding_func = get_embedding_function(
            app.state.config.RAG_EMBEDDING_ENGINE,
            app.state.config.RAG_EMBEDDING_MODEL,
            app.state.sentence_transformer_ef,
            app.state.config.OPENAI_API_KEY,
            app.state.config.OPENAI_API_BASE_URL,
        )

        embedding_texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = embedding_func(embedding_texts)

        for batch in create_batches(
            api=CHROMA_CLIENT,
            ids=[str(uuid.uuid4()) for _ in texts],
            metadatas=metadatas,
            embeddings=embeddings,
            documents=texts,
        ):
            collection.add(*batch)

        return True
    except Exception as e:
        log.exception(e)
        if e.__class__.__name__ == "UniqueConstraintError":
            return True

        return False


def get_loader(filename: str, file_content_type: str, file_path: str):
    file_ext = filename.split(".")[-1].lower()
    known_type = True

    known_source_ext = [
        "go",
        "py",
        "java",
        "sh",
        "bat",
        "ps1",
        "cmd",
        "js",
        "ts",
        "css",
        "cpp",
        "hpp",
        "h",
        "c",
        "cs",
        "sql",
        "log",
        "ini",
        "pl",
        "pm",
        "r",
        "dart",
        "dockerfile",
        "env",
        "php",
        "hs",
        "hsc",
        "lua",
        "nginxconf",
        "conf",
        "m",
        "mm",
        "plsql",
        "perl",
        "rb",
        "rs",
        "db2",
        "scala",
        "bash",
        "swift",
        "vue",
        "svelte",
    ]

    if file_ext == "pdf":
        loader = PyPDFLoader(
            file_path, extract_images=app.state.config.PDF_EXTRACT_IMAGES
        )
    elif file_ext == "csv":
        loader = CSVLoader(file_path)
    elif file_ext == "rst":
        loader = UnstructuredRSTLoader(file_path, mode="elements")
    elif file_ext == "xml":
        loader = UnstructuredXMLLoader(file_path)
    elif file_ext in ["htm", "html"]:
        loader = BSHTMLLoader(file_path, open_encoding="unicode_escape")
    elif file_ext == "md":
        loader = UnstructuredMarkdownLoader(file_path)
    elif file_content_type == "application/epub+zip":
        loader = UnstructuredEPubLoader(file_path)
    elif (
        file_content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or file_ext in ["doc", "docx"]
    ):
        loader = Docx2txtLoader(file_path)
    elif file_content_type in [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ] or file_ext in ["xls", "xlsx"]:
        loader = UnstructuredExcelLoader(file_path)
    elif file_content_type in [
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ] or file_ext in ["ppt", "pptx"]:
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_ext in known_source_ext or (
        file_content_type and file_content_type.find("text/") >= 0
    ):
        loader = TextLoader(file_path, autodetect_encoding=True)
    else:
        loader = TextLoader(file_path, autodetect_encoding=True)
        known_type = False

    return loader, known_type


@app.post("/doc")
def store_doc(
    collection_name: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    # "https://www.gutenberg.org/files/1727/1727-h/1727-h.htm"

    log.info(f"file.content_type: {file.content_type}")
    try:
        unsanitized_filename = file.filename
        filename = os.path.basename(unsanitized_filename)

        file_path = f"{UPLOAD_DIR}/{filename}"

        contents = file.file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
            f.close()

        f = open(file_path, "rb")
        if collection_name == None:
            collection_name = calculate_sha256(f)[:63]
        f.close()

        loader, known_type = get_loader(filename, file.content_type, file_path)
        data = loader.load()

        try:
            result = store_data_in_vector_db(data, collection_name)

            if result:
                return {
                    "status": True,
                    "collection_name": collection_name,
                    "filename": filename,
                    "known_type": known_type,
                }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=e,
            )
    except Exception as e:
        log.exception(e)
        if "No pandoc was found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e),
            )


class TextRAGForm(BaseModel):
    name: str
    content: str
    collection_name: Optional[str] = None


@app.post("/text")
def store_text(
    form_data: TextRAGForm,
    user=Depends(get_current_user),
):

    collection_name = form_data.collection_name
    if collection_name == None:
        collection_name = calculate_sha256_string(form_data.content)

    result = store_text_in_vector_db(
        form_data.content,
        metadata={"name": form_data.name, "created_by": user.id},
        collection_name=collection_name,
    )

    if result:
        return {"status": True, "collection_name": collection_name}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(),
        )


@app.get("/scan")
def scan_docs_dir(user=Depends(get_admin_user)):
    for path in Path(DOCS_DIR).rglob("./**/*"):
        try:
            if path.is_file() and not path.name.startswith("."):
                tags = extract_folders_after_data_docs(path)
                filename = path.name
                file_content_type = mimetypes.guess_type(path)

                f = open(path, "rb")
                collection_name = calculate_sha256(f)[:63]
                f.close()

                loader, known_type = get_loader(
                    filename, file_content_type[0], str(path)
                )
                data = loader.load()

                try:
                    result = store_data_in_vector_db(data, collection_name)

                    if result:
                        sanitized_filename = sanitize_filename(filename)
                        doc = Documents.get_doc_by_name(sanitized_filename)

                        if doc == None:
                            doc = Documents.insert_new_doc(
                                user.id,
                                DocumentForm(
                                    **{
                                        "name": sanitized_filename,
                                        "title": filename,
                                        "collection_name": collection_name,
                                        "filename": filename,
                                        "content": (
                                            json.dumps(
                                                {
                                                    "tags": list(
                                                        map(
                                                            lambda name: {"name": name},
                                                            tags,
                                                        )
                                                    )
                                                }
                                            )
                                            if len(tags)
                                            else "{}"
                                        ),
                                    }
                                ),
                            )
                except Exception as e:
                    log.exception(e)
                    pass

        except Exception as e:
            log.exception(e)

    return True


@app.get("/reset/db")
def reset_vector_db(user=Depends(get_admin_user)):
    CHROMA_CLIENT.reset()


@app.get("/reset")
def reset(user=Depends(get_admin_user)) -> bool:
    folder = f"{UPLOAD_DIR}"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            log.error("Failed to delete %s. Reason: %s" % (file_path, e))

    try:
        CHROMA_CLIENT.reset()
    except Exception as e:
        log.exception(e)

    return True


if ENV == "dev":

    @app.get("/ef")
    async def get_embeddings():
        return {"result": app.state.EMBEDDING_FUNCTION("hello world")}

    @app.get("/ef/{text}")
    async def get_embeddings_text(text: str):
        return {"result": app.state.EMBEDDING_FUNCTION(text)}
