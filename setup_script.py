#!/usr/bin/env python3
"""
Script de Setup Automático - Sistema Avançado de IA para Documentos
Instala automaticamente todas as dependências e configura o sistema
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Exibe cabeçalho do setup"""
    print("=" * 60)
    print("🧠 SISTEMA AVANÇADO DE IA PARA DOCUMENTOS")
    print("=" * 60)
    print("🚀 Setup Automático")
    print("📧 Suporte: Gratuito e Open Source")
    print("=" * 60)

def check_python_version():
    """Verifica versão do Python"""
    print("\n🐍 Verificando versão do Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ é necessário!")
        print(f"   Versão atual: {version.major}.{version.minor}.{version.micro}")
        print("   Baixe em: https://python.org")
        sys.exit(1)
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK!")

def install_package(package):
    """Instala um pacote via pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def setup_virtual_environment():
    """Configura ambiente virtual (opcional)"""
    print("\n🏗️  Configuração do Ambiente")
    
    use_venv = input("Criar ambiente virtual? (s/n) [recomendado]: ").lower().strip()
    
    if use_venv in ['s', 'sim', 'y', 'yes', '']:
        venv_name = "venv_document_ai"
        
        if Path(venv_name).exists():
            print(f"⚠️  Ambiente '{venv_name}' já existe!")
            overwrite = input("Sobrescrever? (s/n): ").lower().strip()
            if overwrite not in ['s', 'sim', 'y', 'yes']:
                return False
        
        print(f"📦 Criando ambiente virtual: {venv_name}")
        
        try:
            subprocess.check_call([sys.executable, "-m", "venv", venv_name])
            
            # Instruções para ativação
            system = platform.system().lower()
            if system == "windows":
                activate_cmd = f"{venv_name}\\Scripts\\activate"
            else:
                activate_cmd = f"source {venv_name}/bin/activate"
            
            print(f"✅ Ambiente criado com sucesso!")
            print(f"📌 Para ativar: {activate_cmd}")
            print("⚠️  IMPORTANTE: Ative o ambiente e execute este script novamente!")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao criar ambiente virtual: {e}")
            return False
    
    return False

def install_dependencies():
    """Instala todas as dependências"""
    print("\n📦 Instalando dependências...")
    
    # Lista de pacotes essenciais
    packages = [
        "streamlit>=1.28.0",
        "plotly>=5.15.0", 
        "pandas>=2.0.0",
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11",
        "mammoth>=1.6.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "nltk>=3.8.1",
        "spacy>=3.6.0"
    ]
    
    # Atualiza pip primeiro
    print("🔧 Atualizando pip...")
    if not install_package("--upgrade pip"):
        print("⚠️  Falha ao atualizar pip, continuando...")
    
    # Instala pacotes
    failed_packages = []
    for package in packages:
        print(f"📥 Instalando {package.split('>=')[0]}...")
        if not install_package(package):
            failed_packages.append(package)
            print(f"❌ Falha: {package}")
        else:
            print(f"✅ OK: {package.split('>=')[0]}")
    
    if failed_packages:
        print(f"\n⚠️  {len(failed_packages)} pacote(s) falharam:")
        for package in failed_packages:
            print(f"   - {package}")
        return False
    
    print("\n✅ Todas as dependências instaladas com sucesso!")
    return True

def download_nltk_data():
    """Baixa dados necessários do NLTK"""
    print("\n📚 Baixando recursos do NLTK...")
    
    try:
        import nltk
        print("📥 Baixando punkt...")
        nltk.download('punkt', quiet=True)
        print("📥 Baixando stopwords...")
        nltk.download('stopwords', quiet=True)
        print("✅ Recursos NLTK baixados!")
        return True
    except Exception as e:
        print(f"❌ Erro ao baixar recursos NLTK: {e}")
        return False

def install_spacy_model():
    """Instala modelo spaCy para português"""
    print("\n🌐 Instalando modelo spaCy para português...")
    
    install_model = input("Instalar modelo pt_core_news_sm? (s/n) [recomendado]: ").lower().strip()
    
    if install_model in ['s', 'sim', 'y', 'yes', '']:
        try:
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "pt_core_news_sm"
            ])
            print("✅ Modelo spaCy instalado!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao instalar modelo spaCy: {e}")
            print("ℹ️  Você pode instalar depois com:")
            print("   python -m spacy download pt_core_news_sm")
            return False
    else:
        print("⏭️  Pulando instalação do modelo spaCy")
        return True

def create_directories():
    """Cria diretórios necessários"""
    print("\n📁 Criando estrutura de diretórios...")
    
    directories = [
        "meus_documentos",
        "knowledge_base",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📂 {directory}/")
    
    print("✅ Diretórios criados!")

def create_example_files():
    """Cria arquivos de exemplo"""
    print("\n📝 Criando arquivos de exemplo...")
    
    # Arquivo de configuração exemplo
    config_example = {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "min_chunk_size": 100,
        "max_chunks_per_doc": 100,
        "similarity_threshold": 0.3,
        "top_k_results": 5,
        "languages": ["portuguese", "english", "spanish"],
        "file_types": [".pdf", ".txt", ".docx", ".doc", ".rtf"]
    }
    
    with open("config_example.json", "w", encoding="utf-8") as f:
        import json
        json.dump(config_example, f, indent=2, ensure_ascii=False)
    
    # Arquivo README com instruções
    readme_content = """# 🧠 Sistema Avançado de IA para Documentos

## 🚀 Como usar:

1. **Adicione seus documentos** na pasta `meus_documentos/`
2. **Execute a interface web**: `streamlit run advanced_document_ai.py`
3. **Processe os documentos** na aba "Processar Documentos"
4. **Faça suas buscas** na aba "Busca"

## 📁 Estrutura de pastas:
- `meus_documentos/` - Coloque seus PDFs, DOCX, TXT aqui
- `knowledge_base/` - Bases de conhecimento salvas
- `logs/` - Logs do sistema

## ⚙️ Configurações:
- Edite `config.json` para personalizar o comportamento
- Use `config_example.json` como referência

## 🆘 Problemas?
- Verifique os logs em `logs/`
- Reinicie o sistema se necessário
- Consulte a documentação completa

Boa sorte! 🎉
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Arquivo de exemplo de documento
    example_doc = """Exemplo de Documento

Este é um arquivo de exemplo para testar o sistema de IA.

Você pode fazer perguntas como:
- "O que é este documento?"
- "Qual o propósito deste arquivo?"
- "Como usar o sistema?"

O sistema irá encontrar este conteúdo e responder baseado nele.

Adicione seus próprios documentos na pasta meus_documentos/ para começar!
"""
    
    with open("meus_documentos/exemplo.txt", "w", encoding="utf-8") as f:
        f.write(example_doc)
    
    print("✅ Arquivos de exemplo criados!")

def run_tests():
    """Executa testes básicos"""
    print("\n🧪 Executando testes básicos...")
    
    try:
        # Testa imports principais
        print("📦 Testando imports...")
        import streamlit
        import sentence_transformers
        import faiss
        import nltk
        import pandas
        import plotly
        print("✅ Todos os imports OK!")
        
        # Testa modelo
        print("🧠 Testando modelo de IA...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        test_embedding = model.encode(["Teste"])
        print(f"✅ Modelo funcionando! Dimensão: {test_embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nos testes: {e}")
        return False

def main():
    """Função principal do setup"""
    print_header()
    
    # Verifica Python
    check_python_version()
    
    # Pergunta sobre ambiente virtual
    if setup_virtual_environment():
        print("\n🔄 Reinicie este script após ativar o ambiente virtual!")
        return
    
    # Instala dependências
    if not install_dependencies():
        print("\n❌ Setup falhou na instalação de dependências!")
        return
    
    # Configura NLTK
    if not download_nltk_data():
        print("⚠️  Recursos NLTK podem não funcionar corretamente")
    
    # Instala modelo spaCy
    install_spacy_model()
    
    # Cria estrutura
    create_directories()
    create_example_files()
    
    # Executa testes
    if not run_tests():
        print("\n⚠️  Alguns testes falharam, mas o sistema pode ainda funcionar")
    
    # Finalização
    print("\n" + "=" * 60)
    print("🎉 SETUP CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print("\n📋 Próximos passos:")
    print("1. 📁 Adicione seus documentos em 'meus_documentos/'")
    print("2. 🚀 Execute: streamlit run advanced_document_ai.py")
    print("3. 🌐 Abra o navegador no endereço mostrado")
    print("4. 📊 Processe seus documentos e faça buscas!")
    print("\n💡 Dica: Comece processando o arquivo exemplo.txt!")
    print("\n🆘 Problemas? Verifique os logs/ ou consulte a documentação")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup cancelado pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        print("🆘 Execute novamente ou verifique as dependências manualmente")
