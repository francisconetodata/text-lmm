import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Processamento de documentos
import PyPDF2
import docx
import mammoth
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import unicodedata

# NLP e pré-processamento
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import RSLPStemmer
import spacy

# Download de recursos necessários do NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AdvancedDocumentAI:
    def __init__(self, config_path='config.json'):
        """Sistema avançado de IA para documentos"""
        self.load_config(config_path)
        self.setup_logging()
        self.initialize_models()
        self.documents = []
        self.embeddings = None
        self.metadata = []
        self.index = None
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'languages_detected': set(),
            'file_types': {},
            'processing_time': 0
        }
        
    def load_config(self, config_path):
        """Carrega configurações do sistema"""
        default_config = {
            "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "min_chunk_size": 100,
            "max_chunks_per_doc": 100,
            "similarity_threshold": 0.3,
            "top_k_results": 5,
            "languages": ["portuguese", "english", "spanish"],
            "file_types": [".pdf", ".txt", ".docx", ".doc", ".rtf"],
            "index_type": "FAISS",
            "embedding_dim": 384
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = default_config
            self.save_config(config_path)
    
    def save_config(self, config_path):
        """Salva configurações"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def setup_logging(self):
        """Configura sistema de logs"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('document_ai.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_models(self):
        """Inicializa modelos de IA"""
        try:
            self.logger.info("Carregando modelo de embeddings...")
            self.model = SentenceTransformer(self.config['model_name'])
            
            # Carrega processador de linguagem para português
            try:
                self.nlp = spacy.load('pt_core_news_sm')
            except OSError:
                self.logger.warning("Modelo spaCy português não encontrado. Usando processamento básico.")
                self.nlp = None
            
            # Stemmer para português
            self.stemmer = RSLPStemmer()
            
            # Stopwords
            self.stop_words = set()
            for lang in self.config['languages']:
                try:
                    self.stop_words.update(stopwords.words(lang))
                except:
                    pass
                    
            self.logger.info("Modelos carregados com sucesso!")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelos: {e}")
            raise
    
    def clean_text(self, text):
        """Limpa e normaliza texto"""
        # Remove caracteres especiais e normaliza unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove caracteres de controle e espaços extras
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.,!?;:()\-]', '', text)
        
        # Remove URLs e emails
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def extract_text_from_pdf(self, file_path):
        """Extrai texto de PDF"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Página {page_num + 1} ---\n{page_text}"
        except Exception as e:
            self.logger.error(f"Erro ao processar PDF {file_path}: {e}")
        return text
    
    def extract_text_from_docx(self, file_path):
        """Extrai texto de DOCX"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Erro ao processar DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_doc(self, file_path):
        """Extrai texto de DOC usando mammoth"""
        try:
            with open(file_path, "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                return result.value
        except Exception as e:
            self.logger.error(f"Erro ao processar DOC {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path):
        """Extrai texto de TXT"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        self.logger.error(f"Não foi possível decodificar {file_path}")
        return ""
    
    def extract_text_from_file(self, file_path):
        """Extrai texto baseado na extensão do arquivo"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif extension in ['.doc', '.rtf']:
            return self.extract_text_from_doc(file_path)
        elif extension == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            self.logger.warning(f"Tipo de arquivo não suportado: {extension}")
            return ""
    
    def intelligent_chunking(self, text, file_name):
        """Divisão inteligente de texto em chunks"""
        # Limpa o texto
        text = self.clean_text(text)
        
        chunks = []
        
        # Primeiro tenta dividir por parágrafos
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Se adicionar este parágrafo não exceder o limite
            if len(current_chunk) + len(paragraph) <= self.config['chunk_size']:
                current_chunk += paragraph + "\n\n"
            else:
                # Salva o chunk atual se for grande o suficiente
                if len(current_chunk.strip()) >= self.config['min_chunk_size']:
                    chunks.append(current_chunk.strip())
                
                # Se o parágrafo é muito grande, divide por sentenças
                if len(paragraph) > self.config['chunk_size']:
                    sentences = sent_tokenize(paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) <= self.config['chunk_size']:
                            temp_chunk += sentence + " "
                        else:
                            if len(temp_chunk.strip()) >= self.config['min_chunk_size']:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + " "
                    
                    if len(temp_chunk.strip()) >= self.config['min_chunk_size']:
                        chunks.append(temp_chunk.strip())
                    current_chunk = ""
                else:
                    current_chunk = paragraph + "\n\n"
        
        # Adiciona o último chunk
        if len(current_chunk.strip()) >= self.config['min_chunk_size']:
            chunks.append(current_chunk.strip())
        
        # Limita o número de chunks por documento
        if len(chunks) > self.config['max_chunks_per_doc']:
            chunks = chunks[:self.config['max_chunks_per_doc']]
            self.logger.warning(f"Arquivo {file_name} teve chunks limitados a {self.config['max_chunks_per_doc']}")
        
        return chunks
    
    def detect_language(self, text):
        """Detecta idioma do texto (básico)"""
        # Método simples baseado em stopwords
        words = word_tokenize(text.lower())
        
        lang_scores = {}
        for lang in self.config['languages']:
            try:
                lang_stopwords = set(stopwords.words(lang))
                score = len([w for w in words if w in lang_stopwords])
                lang_scores[lang] = score
            except:
                lang_scores[lang] = 0
        
        detected_lang = max(lang_scores.keys(), key=lambda k: lang_scores[k])
        return detected_lang if lang_scores[detected_lang] > 0 else 'unknown'
    
    def process_documents(self, folder_path, progress_callback=None):
        """Processa todos os documentos de uma pasta"""
        start_time = time.time()
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Pasta não encontrada: {folder_path}")
        
        # Encontra todos os arquivos suportados
        all_files = []
        for ext in self.config['file_types']:
            all_files.extend(folder.glob(f"*{ext}"))
        
        self.logger.info(f"Encontrados {len(all_files)} arquivos para processar")
        
        processed_files = 0
        total_chunks = 0
        
        for i, file_path in enumerate(all_files):
            try:
                if progress_callback:
                    progress_callback(i / len(all_files), f"Processando: {file_path.name}")
                
                self.logger.info(f"Processando: {file_path.name}")
                
                # Extrai texto
                text = self.extract_text_from_file(file_path)
                
                if not text or len(text.strip()) < self.config['min_chunk_size']:
                    self.logger.warning(f"Arquivo {file_path.name} muito pequeno ou vazio")
                    continue
                
                # Detecta idioma
                language = self.detect_language(text)
                self.stats['languages_detected'].add(language)
                
                # Cria chunks
                chunks = self.intelligent_chunking(text, file_path.name)
                
                # Adiciona metadados
                for chunk_idx, chunk in enumerate(chunks):
                    self.documents.append(chunk)
                    self.metadata.append({
                        'file_name': file_path.name,
                        'file_path': str(file_path),
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'language': language,
                        'file_size': file_path.stat().st_size,
                        'processed_at': datetime.now().isoformat(),
                        'chunk_length': len(chunk)
                    })
                
                # Atualiza estatísticas
                file_ext = file_path.suffix.lower()
                self.stats['file_types'][file_ext] = self.stats['file_types'].get(file_ext, 0) + 1
                total_chunks += len(chunks)
                processed_files += 1
                
            except Exception as e:
                self.logger.error(f"Erro ao processar {file_path.name}: {e}")
                continue
        
        # Atualiza estatísticas finais
        self.stats['total_documents'] = processed_files
        self.stats['total_chunks'] = total_chunks
        self.stats['processing_time'] = time.time() - start_time
        
        self.logger.info(f"Processamento concluído: {processed_files} arquivos, {total_chunks} chunks")
        
        if progress_callback:
            progress_callback(1.0, "Processamento concluído!")
    
    def create_embeddings(self, progress_callback=None):
        """Cria embeddings dos documentos"""
        if not self.documents:
            raise ValueError("Nenhum documento carregado")
        
        self.logger.info("Criando embeddings...")
        
        # Cria embeddings em lotes para eficiência
        batch_size = 32
        embeddings_list = []
        
        for i in range(0, len(self.documents), batch_size):
            if progress_callback:
                progress = i / len(self.documents)
                progress_callback(progress, f"Criando embeddings: {i}/{len(self.documents)}")
            
            batch = self.documents[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings_list.append(batch_embeddings)
        
        self.embeddings = np.vstack(embeddings_list)
        
        # Cria índice FAISS para busca rápida
        self.create_faiss_index()
        
        if progress_callback:
            progress_callback(1.0, "Embeddings criados!")
        
        self.logger.info(f"Embeddings criados para {len(self.documents)} chunks")
    
    def create_faiss_index(self):
        """Cria índice FAISS para busca eficiente"""
        if self.embeddings is None:
            raise ValueError("Embeddings não criados")
        
        # Normaliza embeddings para busca por similaridade de cosseno
        faiss.normalize_L2(self.embeddings)
        
        # Cria índice FAISS
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        self.index.add(self.embeddings)
        
        self.logger.info(f"Índice FAISS criado com {self.index.ntotal} vetores")
    
    def search(self, query, top_k=None, filter_metadata=None):
        """Busca documentos relevantes"""
        if self.index is None:
            raise ValueError("Índice não criado. Execute create_embeddings() primeiro.")
        
        if top_k is None:
            top_k = self.config['top_k_results']
        
        # Cria embedding da query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Busca no índice FAISS
        similarities, indices = self.index.search(query_embedding, top_k * 2)  # Busca mais para filtrar
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim < self.config['similarity_threshold']:
                continue
            
            metadata = self.metadata[idx].copy()
            
            # Aplica filtros se fornecidos
            if filter_metadata:
                skip = False
                for key, value in filter_metadata.items():
                    if key in metadata and metadata[key] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            metadata['similarity'] = float(sim)
            metadata['content'] = self.documents[idx]
            metadata['index'] = int(idx)
            
            results.append(metadata)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_statistics(self):
        """Retorna estatísticas do sistema"""
        stats = self.stats.copy()
        stats['languages_detected'] = list(stats['languages_detected'])
        
        if self.embeddings is not None:
            stats['embedding_dimension'] = self.embeddings.shape[1]
            stats['index_size'] = self.index.ntotal if self.index else 0
        
        return stats
    
    def save_knowledge_base(self, save_path='knowledge_base'):
        """Salva base de conhecimento"""
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        # Salva documentos e metadados
        with open(save_dir / 'documents.json', 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'stats': self.get_statistics(),
                'config': self.config
            }, f, ensure_ascii=False, indent=2)
        
        # Salva embeddings
        if self.embeddings is not None:
            np.save(save_dir / 'embeddings.npy', self.embeddings)
        
        # Salva índice FAISS
        if self.index is not None:
            faiss.write_index(self.index, str(save_dir / 'faiss_index.idx'))
        
        self.logger.info(f"Base de conhecimento salva em: {save_path}")
    
    def load_knowledge_base(self, load_path='knowledge_base'):
        """Carrega base de conhecimento"""
        load_dir = Path(load_path)
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Base de conhecimento não encontrada: {load_path}")
        
        # Carrega documentos e metadados
        with open(load_dir / 'documents.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = data['documents']
        self.metadata = data['metadata']
        self.stats = data['stats']
        
        # Carrega embeddings
        embeddings_path = load_dir / 'embeddings.npy'
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        
        # Carrega índice FAISS
        index_path = load_dir / 'faiss_index.idx'
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        
        self.logger.info(f"Base de conhecimento carregada de: {load_path}")

def create_streamlit_app():
    """Cria interface web com Streamlit"""
    
    st.set_page_config(
        page_title="🧠 IA Avançada para Documentos",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Sistema Avançado de IA para Documentos")
    st.markdown("### Transforme seus documentos em uma base de conhecimento inteligente")
    
    # Inicializa estado da sessão
    if 'ai_system' not in st.session_state:
        st.session_state.ai_system = AdvancedDocumentAI()
    
    if 'knowledge_base_loaded' not in st.session_state:
        st.session_state.knowledge_base_loaded = False
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Configurações do modelo
        with st.expander("🤖 Configurações do Modelo"):
            chunk_size = st.slider("Tamanho do Chunk", 200, 1000, 512)
            top_k = st.slider("Número de Resultados", 1, 20, 5)
            similarity_threshold = st.slider("Limite de Similaridade", 0.0, 1.0, 0.3)
            
            st.session_state.ai_system.config['chunk_size'] = chunk_size
            st.session_state.ai_system.config['top_k_results'] = top_k
            st.session_state.ai_system.config['similarity_threshold'] = similarity_threshold
        
        # Informações do sistema
        st.header("📊 Status do Sistema")
        if st.session_state.knowledge_base_loaded:
            stats = st.session_state.ai_system.get_statistics()
            st.success("✅ Base carregada")
            st.write(f"📄 {stats['total_documents']} documentos")
            st.write(f"🧩 {stats['total_chunks']} chunks")
            st.write(f"🌐 {len(stats['languages_detected'])} idiomas")
        else:
            st.warning("⚠️ Base não carregada")
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Busca", "📁 Processar Documentos", "📊 Estatísticas", "⚙️ Gerenciar Base"])
    
    with tab1:
        st.header("🔍 Buscar na Base de Conhecimento")
        
        if not st.session_state.knowledge_base_loaded:
            st.warning("⚠️ Carregue ou crie uma base de conhecimento primeiro!")
            return
        
        # Interface de busca
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "💬 Faça sua pergunta:",
                placeholder="Digite sua pergunta aqui...",
                key="search_query"
            )
        
        with col2:
            search_button = st.button("🔍 Buscar", type="primary", use_container_width=True)
        
        # Filtros avançados
        with st.expander("🎯 Filtros Avançados"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Filtro por arquivo
                file_names = list(set([meta['file_name'] for meta in st.session_state.ai_system.metadata]))
                selected_file = st.selectbox("📄 Filtrar por arquivo:", ["Todos"] + file_names)
            
            with col2:
                # Filtro por idioma
                languages = list(st.session_state.ai_system.stats['languages_detected'])
                selected_lang = st.selectbox("🌐 Filtrar por idioma:", ["Todos"] + languages)
        
        # Executa busca
        if search_button and query:
            with st.spinner("🔍 Buscando..."):
                # Prepara filtros
                filters = {}
                if selected_file != "Todos":
                    filters['file_name'] = selected_file
                if selected_lang != "Todos":
                    filters['language'] = selected_lang
                
                # Busca
                results = st.session_state.ai_system.search(
                    query, 
                    filter_metadata=filters if filters else None
                )
                
                if results:
                    st.success(f"✅ Encontrados {len(results)} resultados relevantes")
                    
                    for i, result in enumerate(results, 1):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"### 📋 Resultado {i}")
                                st.markdown(f"**📄 Arquivo:** {result['file_name']}")
                                st.markdown(f"**🧩 Chunk:** {result['chunk_index'] + 1}/{result['total_chunks']}")
                            
                            with col2:
                                # Medidor de relevância
                                relevance = result['similarity']
                                color = "green" if relevance > 0.7 else "orange" if relevance > 0.5 else "red"
                                st.markdown(f"**🎯 Relevância:** <span style='color:{color}'>{relevance:.2%}</span>", unsafe_allow_html=True)
                                st.markdown(f"**🌐 Idioma:** {result['language']}")
                            
                            # Conteúdo
                            with st.expander("📖 Ver conteúdo completo"):
                                st.write(result['content'])
                            
                            st.divider()
                else:
                    st.warning("⚠️ Nenhum resultado encontrado para sua busca.")
    
    with tab2:
        st.header("📁 Processar Documentos")
        
        folder_path = st.text_input(
            "📂 Caminho da pasta com documentos:",
            value="meus_documentos",
            help="Digite o caminho para a pasta contendo seus PDFs, DOCX, TXT, etc."
        )
        
        if st.button("🚀 Processar Documentos", type="primary"):
            if not os.path.exists(folder_path):
                st.error(f"❌ Pasta não encontrada: {folder_path}")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                try:
                    # Processa documentos
                    st.session_state.ai_system.process_documents(folder_path, update_progress)
                    
                    # Cria embeddings
                    status_text.text("🧠 Criando embeddings...")
                    st.session_state.ai_system.create_embeddings(update_progress)
                    
                    # Salva base
                    st.session_state.ai_system.save_knowledge_base()
                    st.session_state.knowledge_base_loaded = True
                    
                    st.success("✅ Processamento concluído! Base de conhecimento criada.")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Erro durante o processamento: {e}")
                finally:
                    progress_bar.empty()
                    status_text.empty()
    
    with tab3:
        st.header("📊 Estatísticas da Base de Conhecimento")
        
        if not st.session_state.knowledge_base_loaded:
            st.warning("⚠️ Nenhuma base de conhecimento carregada!")
            return
        
        stats = st.session_state.ai_system.get_statistics()
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Documentos", stats['total_documents'])
        
        with col2:
            st.metric("🧩 Chunks", stats['total_chunks'])
        
        with col3:
            st.metric("🌐 Idiomas", len(stats['languages_detected']))
        
        with col4:
            st.metric("⏱️ Tempo de Processamento", f"{stats['processing_time']:.1f}s")
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição por tipo de arquivo
            if stats['file_types']:
                df_types = pd.DataFrame(list(stats['file_types'].items()), columns=['Tipo', 'Quantidade'])
                fig = px.pie(df_types, values='Quantidade', names='Tipo', title='📊 Distribuição por Tipo de Arquivo')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribuição por idioma
            if stats['languages_detected']:
                lang_counts = {}
                for meta in st.session_state.ai_system.metadata:
                    lang = meta['language']
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                
                df_langs = pd.DataFrame(list(lang_counts.items()), columns=['Idioma', 'Chunks'])
                fig = px.bar(df_langs, x='Idioma', y='Chunks', title='🌐 Distribuição por Idioma')
                st.plotly_chart(fig, use_container_width=True)
        
        # Detalhes dos arquivos
        st.subheader("📋 Detalhes dos Arquivos")
        
        file_details = {}
        for meta in st.session_state.ai_system.metadata:
            file_name = meta['file_name']
            if file_name not in file_details:
                file_details[file_name] = {
                    'chunks': 0,
                    'size': meta['file_size'],
                    'language': meta['language']
                }
            file_details[file_name]['chunks'] += 1
        
        df_files = pd.DataFrame([
            {
                'Arquivo': name,
                'Chunks': details['chunks'],
                'Tamanho (KB)': details['size'] / 1024,
                'Idioma': details['language']
            }
            for name, details in file_details.items()
        ])
        
        st.dataframe(df_files, use_container_width=True)
    
    with tab4:
        st.header("⚙️ Gerenciar Base de Conhecimento")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💾 Salvar Base")
            save_path = st.text_input("📂 Nome da base:", "knowledge_base")
            
            if st.button("💾 Salvar Base Atual"):
                try:
                    st.session_state.ai_system.save_knowledge_base(save_path)
                    st.success(f"✅ Base salva em: {save_path}")
                except Exception as e:
                    st.error(f"❌ Erro ao salvar: {e}")
        
        with col2:
            st.subheader("📁 Carregar Base")
            load_path = st.text_input("📂 Caminho da base:", "knowledge_base")
            
            if st.button("📁 Carregar Base"):
                try:
                    st.session_state.ai_system.load_knowledge_base(load_path)
                    st.session_state.knowledge_base_loaded = True
                    st.success(f"✅ Base carregada de: {load_path}")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Erro ao carregar: {e}")
        
        st.divider()
        
        # Limpar sistema
        if st.button("🗑️ Limpar Sistema", type="secondary"):
            st.session_state.ai_system = AdvancedDocumentAI()
            st.session_state.knowledge_base_loaded = False
            st.success("✅ Sistema limpo!")
            st.experimental_rerun()

if __name__ == "__main__":
    # Executa a aplicação Streamlit
    create_streamlit_app()
