import os
import logging
import requests

from typing import List

from apps.ollama.main import (
    generate_ollama_embeddings,
    GenerateEmbeddingsForm,
)

from huggingface_hub import snapshot_download

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)

from typing import Optional
from config import SRC_LOG_LEVELS, CHROMA_CLIENT


# 로깅 설정
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


# 문서 쿼리 함수
def query_doc(
    collection_name: str,
    query: str,
    embedding_function,
    k: int,
):
    try:
        # 컬렉션 가져오기
        collection = CHROMA_CLIENT.get_collection(name=collection_name)
        # 쿼리 임베딩 생성
        query_embeddings = embedding_function(query)

        # 쿼리 실행
        result = collection.query(
            query_embeddings=[query_embeddings],
            n_results=k,
        )

        log.info(f"query_doc:result {result}")
        return result
    except Exception as e:
        raise e


# 하이브리드 검색을 통한 문서 쿼리 함수
def query_doc_with_hybrid_search(
    collection_name: str,
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    r: float,
):
    try:
        # 컬렉션 가져오기
        collection = CHROMA_CLIENT.get_collection(name=collection_name)
        # 모든 문서 가져오기
        documents = collection.get()  # get all documents

        # BM25 검색기 설정
        bm25_retriever = BM25Retriever.from_texts(
            texts=documents.get("documents"),
            metadatas=documents.get("metadatas"),
        )
        bm25_retriever.k = k

        # Chroma 검색기 설정
        chroma_retriever = ChromaRetriever(
            collection=collection,
            embedding_function=embedding_function,
            top_n=k,
        )

        # 앙상블 검색기 설정
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
        )

        # 재랭킹 압축기 설정
        compressor = RerankCompressor(
            embedding_function=embedding_function,
            top_n=k,
            reranking_function=reranking_function,
            r_score=r,
        )

        # 문맥 압축 검색기 설정
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )

        # 쿼리 실행
        result = compression_retriever.invoke(query)
        result = {
            "distances": [[d.metadata.get("score") for d in result]],
            "documents": [[d.page_content for d in result]],
            "metadatas": [[d.metadata for d in result]],
        }

        log.info(f"query_doc_with_hybrid_search:result {result}")
        return result
    except Exception as e:
        raise e


# 쿼리 결과 병합 및 정렬 함수
def merge_and_sort_query_results(query_results, k, reverse=False):
    # 결합된 데이터를 저장할 리스트 초기화
    combined_distances = []
    combined_documents = []
    combined_metadatas = []

    for data in query_results:
        combined_distances.extend(data["distances"][0])
        combined_documents.extend(data["documents"][0])
        combined_metadatas.extend(data["metadatas"][0])

    # (거리, 문서, 메타데이터) 튜플 리스트 생성
    combined = list(zip(combined_distances, combined_documents, combined_metadatas))

    # 거리 기준으로 리스트 정렬
    combined.sort(key=lambda x: x[0], reverse=reverse)

    # 결합된 데이터가 없는 경우
    if not combined:
        sorted_distances = []
        sorted_documents = []
        sorted_metadatas = []
    else:
        # 정렬된 리스트 풀기
        sorted_distances, sorted_documents, sorted_metadatas = zip(*combined)

        # k개 요소만 포함하도록 슬라이스
        sorted_distances = list(sorted_distances)[:k]
        sorted_documents = list(sorted_documents)[:k]
        sorted_metadatas = list(sorted_metadatas)[:k]

    # 출력 딕셔너리 생성
    result = {
        "distances": [sorted_distances],
        "documents": [sorted_documents],
        "metadatas": [sorted_metadatas],
    }

    return result


# 컬렉션에 쿼리 함수
def query_collection(
    collection_names: List[str],
    query: str,
    embedding_function,
    k: int,
):
    results = []
    for collection_name in collection_names:
        try:
            result = query_doc(
                collection_name=collection_name,
                query=query,
                k=k,
                embedding_function=embedding_function,
            )
            results.append(result)
        except:
            pass
    return merge_and_sort_query_results(results, k=k)


# 하이브리드 검색을 통한 컬렉션 쿼리 함수
def query_collection_with_hybrid_search(
    collection_names: List[str],
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    r: float,
):
    results = []
    for collection_name in collection_names:
        try:
            result = query_doc_with_hybrid_search(
                collection_name=collection_name,
                query=query,
                embedding_function=embedding_function,
                k=k,
                reranking_function=reranking_function,
                r=r,
            )
            results.append(result)
        except:
            pass
    return merge_and_sort_query_results(results, k=k, reverse=True)

#------------------------------------------------------------------------#
# 템플릿 문자열에서 [context]와 [query]를 실제 값으로 대체하는 함수
def rag_template(template: str, context: str, query: str):
    template = template.replace("[context]", context)
    template = template.replace("[query]", query)
    return template

# 임베딩 함수를 반환하는 함수. 임베딩 엔진과 모델에 따라 다른 함수를 반환함.
def get_embedding_function(
    embedding_engine,
    embedding_model,
    embedding_function,
    openai_key,
    openai_url,
):
    if embedding_engine == "":
        # 기본 임베딩 함수 반환
        return lambda query: embedding_function.encode(query).tolist()
    elif embedding_engine in ["ollama", "openai"]:
        # ollama 또는 openai 임베딩 함수 반환
        if embedding_engine == "ollama":
            func = lambda query: generate_ollama_embeddings(
                GenerateEmbeddingsForm(
                    **{
                        "model": embedding_model,
                        "prompt": query,
                    }
                )
            )
        elif embedding_engine == "openai":
            func = lambda query: generate_openai_embeddings(
                model=embedding_model,
                text=query,
                key=openai_key,
                url=openai_url,
            )

        # 다수의 쿼리에 대해 임베딩을 생성하는 함수
        def generate_multiple(query, f):
            if isinstance(query, list):
                return [f(q) for q in query]
            else:
                return f(query)

        return lambda query: generate_multiple(query, func)

# RAG 메시지를 생성하는 함수
def rag_messages(
    docs,
    messages,
    template,
    embedding_function,
    k,
    reranking_function,
    r,
    hybrid_search,
):
    log.debug(f"docs: {docs} {messages} {embedding_function} {reranking_function}")

    # 마지막 사용자 메시지의 인덱스를 찾음
    last_user_message_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            last_user_message_idx = i
            break

    user_message = messages[last_user_message_idx]

    # 사용자 메시지의 내용을 분석하여 쿼리를 추출함
    if isinstance(user_message["content"], list):
        # 리스트 형태의 콘텐츠 처리
        content_type = "list"
        query = ""
        for content_item in user_message["content"]:
            if content_item["type"] == "text":
                query = content_item["text"]
                break
    elif isinstance(user_message["content"], str):
        # 텍스트 형태의 콘텐츠 처리
        content_type = "text"
        query = user_message["content"]
    else:
        # 예상되지 않은 형태의 콘텐츠 처리
        content_type = None
        query = ""

    extracted_collections = []
    relevant_contexts = []

    # 문서들에서 관련된 컨텍스트를 추출함
    for doc in docs:
        context = None

        # 컬렉션 이름들을 추출함
        collection_names = (
            doc["collection_names"]
            if doc["type"] == "collection"
            else [doc["collection_name"]]
        )

        # 이미 추출된 컬렉션은 제외함
        collection_names = set(collection_names).difference(extracted_collections)
        if not collection_names:
            log.debug(f"skipping {doc} as it has already been extracted")
            continue

        try:
            if doc["type"] == "text":
                context = doc["content"]
            else:
                if hybrid_search:
                    context = query_collection_with_hybrid_search(
                        collection_names=collection_names,
                        query=query,
                        embedding_function=embedding_function,
                        k=k,
                        reranking_function=reranking_function,
                        r=r,
                    )
                else:
                    context = query_collection(
                        collection_names=collection_names,
                        query=query,
                        embedding_function=embedding_function,
                        k=k,
                    )
        except Exception as e:
            log.exception(e)
            context = None

        if context:
            relevant_contexts.append({**context, "source": doc})

        extracted_collections.extend(collection_names)

    context_string = ""

    citations = []
    for context in relevant_contexts:
        try:
            if "documents" in context:
                context_string += "\n\n".join(
                    [text for text in context["documents"][0] if text is not None]
                )

                if "metadatas" in context:
                    citations.append(
                        {
                            "source": context["source"],
                            "document": context["documents"][0],
                            "metadata": context["metadatas"][0],
                        }
                    )
        except Exception as e:
            log.exception(e)

    context_string = context_string.strip()

    # 템플릿을 사용하여 최종 콘텐츠를 생성함
    ra_content = rag_template(
        template=template,
        context=context_string,
        query=query,
    )

    log.debug(f"ra_content: {ra_content}")

    # 새로운 사용자 메시지를 생성함
    if content_type == "list":
        new_content = []
        for content_item in user_message["content"]:
            if content_item["type"] == "text":
                # 텍스트 항목의 콘텐츠를 ra_content로 업데이트
                new_content.append({"type": "text", "text": ra_content})
            else:
                # 다른 유형의 콘텐츠는 그대로 유지함
                new_content.append(content_item)
        new_user_message = {**user_message, "content": new_content}
    else:
        new_user_message = {
            **user_message,
            "content": ra_content,
        }

    messages[last_user_message_idx] = new_user_message

    return messages, citations

#------------------------------------------------------------------------#
import os
import requests
from huggingface_hub import snapshot_download
from typing import List, Any
from langchain_core.documents import Document

# 모델 경로를 가져오는 함수
def get_model_path(model: str, update_model: bool = False):
    # 환경 변수에서 캐시 디렉토리를 가져옴
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")

    # 모델을 업데이트할지 여부에 따라 로컬 파일만 사용할지 결정
    local_files_only = not update_model

    # snapshot_download 함수에 전달할 인수 설정
    snapshot_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }

    log.debug(f"model: {model}")
    log.debug(f"snapshot_kwargs: {snapshot_kwargs}")

    # 모델 경로가 존재하거나 로컬 파일만 사용해야 하고 모델 경로에 슬래시가 포함된 경우
    if (
        os.path.exists(model)
        or ("\\" in model or model.count("/") > 1)
        and local_files_only
    ):
        # 경로가 존재하면 해당 경로 반환, 그렇지 않으면 repo_id로 설정
        return model
    elif "/" not in model:
        # 모델 이름이 짧은 경우 기본 레포지토리 경로로 설정
        model = "sentence-transformers" + "/" + model

    snapshot_kwargs["repo_id"] = model

    # huggingface_hub 라이브러리를 사용하여 로컬 경로를 가져오거나 업데이트 시도
    try:
        model_repo_path = snapshot_download(**snapshot_kwargs)
        log.debug(f"model_repo_path: {model_repo_path}")
        return model_repo_path
    except Exception as e:
        log.exception(f"Cannot determine model snapshot path: {e}")
        return model

# OpenAI 임베딩을 생성하는 함수
def generate_openai_embeddings(
    model: str, text: str, key: str, url: str = "https://api.openai.com/v1"
):
    try:
        # OpenAI API에 POST 요청을 보내 임베딩 생성
        r = requests.post(
            f"{url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            json={"input": text, "model": model},
        )
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            return data["data"][0]["embedding"]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        print(e)
        return None

from typing import Any

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# 문서 검색을 위한 ChromaRetriever 클래스
class ChromaRetriever(BaseRetriever):
    collection: Any
    embedding_function: Any
    top_n: int

    # 관련 문서를 검색하는 내부 함수
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # 쿼리 임베딩 생성
        query_embeddings = self.embedding_function(query)

        # 컬렉션에서 쿼리 임베딩을 사용하여 검색
        results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=self.top_n,
        )

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        results = []
        # 검색 결과를 Document 객체로 변환하여 리스트에 추가
        for idx in range(len(ids)):
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=documents[idx],
                )
            )
        return results

import operator
from typing import Optional, Sequence
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.callbacks import Callbacks
from langchain_core.pydantic_v1 import Extra
from sentence_transformers import util

# 문서 압축을 위한 RerankCompressor 클래스
class RerankCompressor(BaseDocumentCompressor):
    embedding_function: Any
    top_n: int
    reranking_function: Any
    r_score: float

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    # 문서를 압축하는 함수
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        reranking = self.reranking_function is not None

        if reranking:
            # 재랭킹 기능이 있는 경우 재랭킹 함수로 점수 계산
            scores = self.reranking_function.predict(
                [(query, doc.page_content) for doc in documents]
            )
        else:
            # 재랭킹 기능이 없는 경우 임베딩을 사용하여 점수 계산
            query_embedding = self.embedding_function(query)
            document_embedding = self.embedding_function(
                [doc.page_content for doc in documents]
            )
            scores = util.cos_sim(query_embedding, document_embedding)[0]

        # 문서와 점수를 쌍으로 묶음
        docs_with_scores = list(zip(documents, scores.tolist()))
        if self.r_score:
            # r_score 이상의 점수만 필터링
            docs_with_scores = [
                (d, s) for d, s in docs_with_scores if s >= self.r_score
            ]

        # 점수를 기준으로 문서 정렬
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        final_results = []
        for doc, doc_score in result[: self.top_n]:
            metadata = doc.metadata
            metadata["score"] = doc_score
            doc = Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
            final_results.append(doc)
        return final_results

