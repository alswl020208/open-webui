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
# QuerySettingsForm 클래스는 쿼리 설정 폼 데이터를 정의합니다.
class QuerySettingsForm(BaseModel):
    k: Optional[int] = None  # 검색 결과 상위 k개의 문서를 반환하는 변수
    r: Optional[float] = None  # 검색 결과의 관련성 임계값
    template: Optional[str] = None  # 검색 템플릿
    hybrid: Optional[bool] = None  # 하이브리드 검색 사용 여부

# /query/settings/update 엔드포인트에 대한 POST 요청을 처리하는 비동기 함수입니다.
@app.post("/query/settings/update")
async def update_query_settings(
    form_data: QuerySettingsForm, user=Depends(get_admin_user)
):
    # 설정 값을 업데이트합니다. 폼 데이터가 제공되지 않으면 기본값을 사용합니다.
    app.state.config.RAG_TEMPLATE = (
        form_data.template if form_data.template else RAG_TEMPLATE
    )
    app.state.config.TOP_K = form_data.k if form_data.k else 4
    app.state.config.RELEVANCE_THRESHOLD = form_data.r if form_data.r else 0.0
    app.state.config.ENABLE_RAG_HYBRID_SEARCH = (
        form_data.hybrid if form_data.hybrid else False
    )
    # 업데이트된 설정 값을 반환합니다.
    return {
        "status": True,
        "template": app.state.config.RAG_TEMPLATE,
        "k": app.state.config.TOP_K,
        "r": app.state.config.RELEVANCE_THRESHOLD,
        "hybrid": app.state.config.ENABLE_RAG_HYBRID_SEARCH,
    }

# QueryDocForm 클래스는 문서 쿼리 폼 데이터를 정의합니다.
class QueryDocForm(BaseModel):
    collection_name: str  # 쿼리할 컬렉션 이름
    query: str  # 쿼리 문자열
    k: Optional[int] = None  # 검색 결과 상위 k개의 문서를 반환하는 변수
    r: Optional[float] = None  # 검색 결과의 관련성 임계값
    hybrid: Optional[bool] = None  # 하이브리드 검색 사용 여부

# /query/doc 엔드포인트에 대한 POST 요청을 처리하는 함수입니다.
@app.post("/query/doc")
def query_doc_handler(
    form_data: QueryDocForm,
    user=Depends(get_current_user),
):
    try:
        # 하이브리드 검색이 활성화된 경우
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
        else:  # 일반 검색
            return query_doc(
                collection_name=form_data.collection_name,
                query=form_data.query,
                embedding_function=app.state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app.state.config.TOP_K,
            )
    except Exception as e:  # 예외 처리
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )

# QueryCollectionsForm 클래스는 컬렉션 쿼리 폼 데이터를 정의합니다.
class QueryCollectionsForm(BaseModel):
    collection_names: List[str]  # 쿼리할 컬렉션 이름들의 리스트
    query: str  # 쿼리 문자열
    k: Optional[int] = None  # 검색 결과 상위 k개의 문서를 반환하는 변수
    r: Optional[float] = None  # 검색 결과의 관련성 임계값
    hybrid: Optional[bool] = None  # 하이브리드 검색 사용 여부

# /query/collection 엔드포인트에 대한 POST 요청을 처리하는 함수입니다.
@app.post("/query/collection")
def query_collection_handler(
    form_data: QueryCollectionsForm,
    user=Depends(get_current_user),
):
    try:
        # 하이브리드 검색이 활성화된 경우
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
        else:  # 일반 검색
            return query_collection(
                collection_names=form_data.collection_names,
                query=form_data.query,
                embedding_function=app.state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app.state.config.TOP_K,
            )
    except Exception as e:  # 예외 처리
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@app.post("/youtube")
def store_youtube_video(form_data: UrlForm, user=Depends(get_current_user)):
    try:
        # YoutubeLoader를 사용하여 YouTube URL에서 비디오 정보와 데이터를 가져옴
        loader = YoutubeLoader.from_youtube_url(
            form_data.url,
            add_video_info=True,
            language=app.state.config.YOUTUBE_LOADER_LANGUAGE,
            translation=app.state.YOUTUBE_LOADER_TRANSLATION,
        )
        data = loader.load()

        # collection_name이 제공되지 않으면 URL을 해시하여 컬렉션 이름을 생성
        collection_name = form_data.collection_name
        if collection_name == "":
            collection_name = calculate_sha256_string(form_data.url)[:63]

        # 데이터를 벡터 데이터베이스에 저장
        store_data_in_vector_db(data, collection_name, overwrite=True)
        return {
            "status": True,
            "collection_name": collection_name,
            "filename": form_data.url,
        }
    except Exception as e:
        # 예외 발생 시 로그를 남기고 HTTP 400 에러를 반환
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@app.post("/web")
def store_web(form_data: UrlForm, user=Depends(get_current_user)):
    try:
        # get_web_loader 함수를 사용하여 웹 페이지 로더를 가져옴
        loader = get_web_loader(
            form_data.url,
            verify_ssl=app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
        )
        data = loader.load()

        # collection_name이 제공되지 않으면 URL을 해시하여 컬렉션 이름을 생성
        collection_name = form_data.collection_name
        if collection_name == "":
            collection_name = calculate_sha256_string(form_data.url)[:63]

        # 데이터를 벡터 데이터베이스에 저장
        store_data_in_vector_db(data, collection_name, overwrite=True)
        return {
            "status": True,
            "collection_name": collection_name,
            "filename": form_data.url,
        }
    except Exception as e:
        # 예외 발생 시 로그를 남기고 HTTP 400 에러를 반환
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


def get_web_loader(url: str, verify_ssl: bool = True):
    # URL이 유효한지 검사
    if isinstance(validators.url(url), validators.ValidationError):
        raise ValueError(ERROR_MESSAGES.INVALID_URL)

    # 로컬 웹 페치가 비활성화된 경우, 개인 IP 주소로 해석되는 URL을 필터링
    if not ENABLE_RAG_LOCAL_WEB_FETCH:
        # URL을 파싱하여 호스트 이름을 가져옴
        parsed_url = urllib.parse.urlparse(url)
        # 호스트 이름을 통해 IPv4 및 IPv6 주소를 해석
        ipv4_addresses, ipv6_addresses = resolve_hostname(parsed_url.hostname)
        # 해석된 주소 중 개인 IP 주소가 있는지 확인
        for ip in ipv4_addresses:
            if validators.ipv4(ip, private=True):
                raise ValueError(ERROR_MESSAGES.INVALID_URL)
        for ip in ipv6_addresses:
            if validators.ipv6(ip, private=True):
                raise ValueError(ERROR_MESSAGES.INVALID_URL)

    # WebBaseLoader 인스턴스를 반환
    return WebBaseLoader(url, verify_ssl=verify_ssl)


def resolve_hostname(hostname):
    # 호스트 이름에 대한 주소 정보를 가져옴
    addr_info = socket.getaddrinfo(hostname, None)

    # 주소 정보에서 IPv4 및 IPv6 주소를 추출
    ipv4_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET]
    ipv6_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET6]

    return ipv4_addresses, ipv6_addresses


def store_data_in_vector_db(data, collection_name, overwrite: bool = False) -> bool:
    # 텍스트 분할기를 설정 (chunk_size와 chunk_overlap을 사용)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=app.state.config.CHUNK_SIZE,
        chunk_overlap=app.state.config.CHUNK_OVERLAP,
        add_start_index=True,
    )

    # 데이터를 분할하여 문서 리스트를 생성
    docs = text_splitter.split_documents(data)

    # 분할된 문서가 있을 경우 벡터 데이터베이스에 저장
    if len(docs) > 0:
        log.info(f"store_data_in_vector_db {docs}")
        return store_docs_in_vector_db(docs, collection_name, overwrite), None
    else:
        # 분할된 문서가 없을 경우 예외 발생
        raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)


def store_text_in_vector_db(
        text, metadata, collection_name, overwrite: bool = False
) -> bool:
    # 텍스트 분할기를 설정 (chunk_size와 chunk_overlap을 사용)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=app.state.config.CHUNK_SIZE,
        chunk_overlap=app.state.config.CHUNK_OVERLAP,
        add_start_index=True,
    )

    # 텍스트와 메타데이터를 분할하여 문서 리스트를 생성
    docs = text_splitter.create_documents([text], metadatas=[metadata])

    # 분할된 문서를 벡터 데이터베이스에 저장
    return store_docs_in_vector_db(docs, collection_name, overwrite)

# 벡터 DB에 문서 저장 기능을 구현한 함수입니다.
def store_docs_in_vector_db(docs, collection_name, overwrite: bool = False) -> bool:
    # 문서 저장 작업 시작을 로그에 기록합니다.
    log.info(f"store_docs_in_vector_db {docs} {collection_name}")

    # 문서들에서 페이지 내용을 추출하여 리스트에 저장합니다.
    texts = [doc.page_content for doc in docs]
    # 문서들에서 메타데이터를 추출하여 리스트에 저장합니다.
    metadatas = [doc.metadata for doc in docs]

    try:
        # overwrite 옵션이 True이면 기존 컬렉션을 삭제합니다.
        if overwrite:
            for collection in CHROMA_CLIENT.list_collections():
                if collection_name == collection.name:
                    log.info(f"deleting existing collection {collection_name}")
                    CHROMA_CLIENT.delete_collection(name=collection_name)

        # 새로운 컬렉션을 생성합니다.
        collection = CHROMA_CLIENT.create_collection(name=collection_name)

        # 임베딩 함수를 가져옵니다. 이 함수는 문서의 텍스트를 벡터로 변환하는 데 사용됩니다.
        embedding_func = get_embedding_function(
            app.state.config.RAG_EMBEDDING_ENGINE,
            app.state.config.RAG_EMBEDDING_MODEL,
            app.state.sentence_transformer_ef,
            app.state.config.OPENAI_API_KEY,
            app.state.config.OPENAI_API_BASE_URL,
        )

        # 텍스트에서 줄바꿈 문자를 공백으로 대체합니다.
        embedding_texts = list(map(lambda x: x.replace("\n", " "), texts))
        # 임베딩 함수를 이용하여 텍스트를 벡터로 변환합니다.
        embeddings = embedding_func(embedding_texts)

        # 생성된 벡터, 메타데이터, 문서 ID를 배치로 생성하여 컬렉션에 추가합니다.
        for batch in create_batches(
            api=CHROMA_CLIENT,
            ids=[str(uuid.uuid4()) for _ in texts],
            metadatas=metadatas,
            embeddings=embeddings,
            documents=texts,
        ):
            collection.add(*batch)

        # 모든 작업이 성공적으로 완료되면 True를 반환합니다.
        return True
    except Exception as e:
        # 예외 발생 시 로그에 기록합니다.
        log.exception(e)
        # 고유 제약 조건 오류가 발생한 경우 True를 반환합니다.
        if e.__class__.__name__ == "UniqueConstraintError":
            return True

        # 그 외의 경우 False를 반환합니다.
        return False


# 파일 로더를 선택하는 함수입니다.
def get_loader(filename: str, file_content_type: str, file_path: str):
    # 파일 확장자를 소문자로 변환하여 추출합니다.
    file_ext = filename.split(".")[-1].lower()
    known_type = True

    # 지원하는 소스 코드 파일 확장자 목록입니다.
    known_source_ext = [
        "go", "py", "java", "sh", "bat", "ps1", "cmd", "js", "ts", "css", "cpp", "hpp",
        "h", "c", "cs", "sql", "log", "ini", "pl", "pm", "r", "dart", "dockerfile", "env",
        "php", "hs", "hsc", "lua", "nginxconf", "conf", "m", "mm", "plsql", "perl", "rb",
        "rs", "db2", "scala", "bash", "swift", "vue", "svelte",
    ]

    # 파일 확장자나 컨텐츠 타입에 따라 적절한 로더를 선택합니다.
    if file_ext == "pdf":
        loader = PyPDFLoader(file_path, extract_images=app.state.config.PDF_EXTRACT_IMAGES)
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
    elif (file_content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file_ext in ["doc", "docx"]):
        loader = Docx2txtLoader(file_path)
    elif file_content_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"] or file_ext in ["xls", "xlsx"]:
        loader = UnstructuredExcelLoader(file_path)
    elif file_content_type in ["application/vnd.ms-powerpoint", "application/vnd.openxmlformats-officedocument.presentationml.presentation"] or file_ext in ["ppt", "pptx"]:
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_ext in known_source_ext or (file_content_type and file_content_type.find("text/") >= 0):
        loader = TextLoader(file_path, autodetect_encoding=True)
    else:
        loader = TextLoader(file_path, autodetect_encoding=True)
        known_type = False

    # 선택된 로더와 파일 타입이 알려져 있는지 여부를 반환합니다.
    return loader, known_type

@app.post("/doc")
def store_doc(
    collection_name: Optional[str] = Form(None),  # 선택적으로 컬렉션 이름을 Form 데이터로 받습니다.
    file: UploadFile = File(...),  # 업로드된 파일을 받습니다.
    user=Depends(get_current_user),  # 현재 사용자를 확인하는 의존성을 주입합니다.
):
    # 파일의 내용 유형을 로그로 기록합니다.
    log.info(f"file.content_type: {file.content_type}")
    try:
        unsanitized_filename = file.filename  # 파일 이름을 가져옵니다.
        filename = os.path.basename(unsanitized_filename)  # 경로나 디렉토리를 제외한 파일 이름만 추출합니다.

        file_path = f"{UPLOAD_DIR}/{filename}"  # 파일을 저장할 경로를 설정합니다.

        contents = file.file.read()  # 파일의 내용을 읽습니다.
        with open(file_path, "wb") as f:  # 파일을 바이너리 쓰기 모드로 엽니다.
            f.write(contents)  # 파일 내용을 씁니다.
            f.close()  # 파일을 닫습니다.

        f = open(file_path, "rb")  # 파일을 다시 바이너리 읽기 모드로 엽니다.
        if collection_name == None:  # 컬렉션 이름이 제공되지 않았다면,
            collection_name = calculate_sha256(f)[:63]  # 파일의 내용으로부터 SHA-256 해시를 계산하여 컬렉션 이름으로 사용합니다.
        f.close()  # 파일을 닫습니다.

        loader, known_type = get_loader(filename, file.content_type, file_path)  # 파일을 읽기 위한 로더를 얻습니다.
        data = loader.load()  # 파일의 내용을 로드합니다.

        try:
            result = store_data_in_vector_db(data, collection_name)  # 데이터를 벡터 데이터베이스에 저장합니다.

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
        if "No pandoc was found" in str(e):  # Pandoc이 설치되지 않은 경우에 대한 예외 처리입니다.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
            )
        else:  # 기타 예외에 대한 처리입니다.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e),
            )


class TextRAGForm(BaseModel):
    name: str  # 텍스트 데이터의 이름입니다.
    content: str  # 실제 텍스트 데이터입니다.
    collection_name: Optional[str] = None  # 선택적으로 컬렉션 이름을 지정할 수 있습니다.


@app.post("/text")
def store_text(
    form_data: TextRAGForm,  # 텍스트 데이터와 관련 정보를 담고 있는 폼 데이터입니다.
    user=Depends(get_current_user),  # 현재 사용자를 확인하는 의존성을 주입합니다.
):

    collection_name = form_data.collection_name
    if collection_name == None:  # 컬렉션 이름이 제공되지 않았다면,
        collection_name = calculate_sha256_string(form_data.content)  # 텍스트 내용으로부터 SHA-256 해시를 계산하여 컬렉션 이름으로 사용합니다.

    result = store_text_in_vector_db(
        form_data.content,
        metadata={"name": form_data.name, "created_by": user.id},  # 메타데이터로 이름과 생성자 ID를 저장합니다.
        collection_name=collection_name,
    )

    if result:
        return {"status": True, "collection_name": collection_name}
    else:  # 데이터 저장에 실패한 경우,
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(),
        )

# 문서 디렉토리를 스캔하는 엔드포인트
@app.get("/scan")
def scan_docs_dir(user=Depends(get_admin_user)):  # 관리자 사용자만 이용할 수 있도록 의존성 주입을 사용합니다.
    for path in Path(DOCS_DIR).rglob("./**/*"):  # DOCS_DIR 경로 내 모든 파일을 재귀적으로 탐색합니다.
        try:
            if path.is_file() and not path.name.startswith("."):  # 숨김 파일이 아닌 일반 파일인 경우
                tags = extract_folders_after_data_docs(path)  # 파일 경로로부터 태그 추출
                filename = path.name
                file_content_type = mimetypes.guess_type(path)  # 파일의 MIME 타입 추정

                f = open(path, "rb")
                collection_name = calculate_sha256(f)[:63]  # 파일 내용을 기반으로한 SHA256 해시값 생성
                f.close()

                loader, known_type = get_loader(filename, file_content_type[0], str(path))
                data = loader.load()  # 파일 로더를 이용하여 파일 내용 로드

                try:
                    result = store_data_in_vector_db(data, collection_name)  # 벡터 데이터베이스에 데이터 저장 시도

                    if result:
                        sanitized_filename = sanitize_filename(filename)  # 안전한 파일 이름 생성
                        doc = Documents.get_doc_by_name(sanitized_filename)

                        if doc == None:  # 문서가 데이터베이스에 존재하지 않는 경우 새로운 문서로 추가
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
                    log.exception(e)  # 데이터 저장 중 발생한 예외 로깅
                    pass

        except Exception as e:
            log.exception(e)  # 파일 처리 중 발생한 예외 로깅

    return True


# 벡터 데이터베이스를 초기화하는 엔드포인트
@app.get("/reset/db")
def reset_vector_db(user=Depends(get_admin_user)):  # 관리자 사용자만 이용할 수 있도록 의존성 주입
    CHROMA_CLIENT.reset()  # 벡터 데이터베이스 클라이언트를 이용하여 데이터베이스 초기화


# 업로드 디렉토리를 초기화하는 엔드포인트
@app.get("/reset")
def reset(user=Depends(get_admin_user)) -> bool:  # 관리자 사용자만 이용할 수 있도록 의존성 주입
    folder = f"{UPLOAD_DIR}"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 파일 또는 링크 삭제
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 디렉토리 삭제
        except Exception as e:
            log.error("Failed to delete %s. Reason: %s" % (file_path, e))  # 삭제 실패 로깅

    try:
        CHROMA_CLIENT.reset()  # 추가적으로 벡터 데이터베이스 초기화 시도
    except Exception as e:
        log.exception(e)

    return True


# 개발 환경에서만 사용되는 엔드포인트 예시
if ENV == "dev":

    # 'hello world' 텍스트의 임베딩 결과 반환
    @app.get("/ef")
    async def get_embeddings():
        return {"result": app.state.EMBEDDING_FUNCTION("hello world")}

    # 주어진 텍스트의 임베딩 결과 반환
    @app.get("/ef/{text}")
    async def get_embeddings_text(text: str):
        return {"result": app.state.EMBEDDING_FUNCTION(text)}