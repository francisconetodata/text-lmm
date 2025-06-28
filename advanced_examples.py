"""
Exemplos Avançados de Uso - Sistema de IA para Documentos
Este arquivo mostra como usar recursos avançados do sistema
"""

from advanced_document_ai import AdvancedDocumentAI
import json
import time
from pathlib import Path

# ============================================================================
# EXEMPLO 1: USO BÁSICO PROGRAMÁTICO
# ============================================================================

def exemplo_basico():
    """Exemplo básico de uso programático"""
    print("🚀 EXEMPLO 1: Uso Básico")
    print("-" * 40)
    
    # Inicializa sistema
    ai = AdvancedDocumentAI()
    
    # Processa documentos
    print("📁 Processando documentos...")
    ai.process_documents('meus_documentos')
    
    # Cria embeddings
    print("🧠 Criando embeddings...")
    ai.create_embeddings()
    
    # Busca
    query = "Como usar este sistema?"
    print(f"🔍 Buscando: '{query}'")
    results = ai.search(query)
    
    # Mostra resultados
    for i, result in enumerate(results, 1):
        print(f"\n📋 Resultado {i}:")
        print(f"   📄 Arquivo: {result['file_name']}")
        print(f"   🎯 Relevância: {result['similarity']:.2%}")
        print(f"   📝 Trecho: {result['content'][:100]}...")

# ============================================================================
# EXEMPLO 2: CONFIGURAÇÃO PERSONALIZADA
# ============================================================================

def exemplo_configuracao_personalizada():
    """Exemplo com configurações personalizadas"""
    print("\n🚀 EXEMPLO 2: Configuração Personalizada")
    print("-" * 40)
    
    # Configuração personalizada
    config_personalizada = {
        "model_name": "all-MiniLM-L6-v2",  # Modelo mais rápido
        "chunk_size": 300,                  # Chunks menores
        "chunk_overlap": 30,
        "similarity_threshold": 0.4,        # Mais restritivo
        "top_k_results": 3,                 # Menos resultados
        "languages": ["portuguese"]         # Apenas português
    }
    
    # Salva configuração personalizada
    with open('config_personalizada.json', 'w') as f:
        json.dump(config_personalizada, f, indent=2)
    
    # Inicializa com configuração personalizada
    ai = AdvancedDocumentAI('config_personalizada.json')
    
    print(f"⚙️ Usando modelo: {ai.config['model_name']}")
    print(f"📏 Tamanho do chunk: {ai.config['chunk_size']}")
    print(f"🎯 Threshold: {ai.config['similarity_threshold']}")

# ============================================================================
# EXEMPLO 3: PROCESSAMENTO EM LOTE COM MÚLTIPLAS BASES
# ============================================================================

def exemplo_multiplas_bases():
    """Exemplo com múltiplas bases de conhecimento"""
    print("\n🚀 EXEMPLO 3: Múltiplas Bases")
    print("-" * 40)
    
    bases = {
        'documentos_tecnicos': 'documentos_tech/',
        'manuais_usuario': 'manuais/',
        'artigos_academicos': 'papers/'
    }
    
    resultados_combinados = []
    
    for nome_base, pasta in bases.items():
        if not Path(pasta).exists():
            print(f"⚠️ Pasta {pasta} não existe, criando exemplo...")
            Path(pasta).mkdir(exist_ok=True)
            continue
            
        print(f"📚 Processando base: {nome_base}")
        
        # Cria sistema específico para esta base
        ai = AdvancedDocumentAI()
        ai.process_documents(pasta)
        ai.create_embeddings()
        
        # Salva base específica
        ai.save_knowledge_base(f'knowledge_base_{nome_base}')
        
        # Busca em cada base
        query = "instalação e configuração"
        results = ai.search(query, top_k=2)
        
        # Adiciona identificação da base aos resultados
        for result in results:
            result['base_origem'] = nome_base
            resultados_combinados.append(result)
    
    # Ordena resultados combinados por relevância
    resultados_combinados.sort(key=lambda x: x['similarity'], reverse=True)
    
    print(f"\n🔍 Resultados combinados para: 'instalação e configuração'")
    for i, result in enumerate(resultados_combinados[:5], 1):
        print(f"{i}. [{result['base_origem']}] {result['file_name']} - {result['similarity']:.2%}")

# ============================================================================
# EXEMPLO 4: ANÁLISE DE PERFORMANCE E ESTATÍSTICAS
# ============================================================================

def exemplo_analise_performance():
    """Exemplo de análise de performance e estatísticas"""
    print("\n🚀 EXEMPLO 4: Análise de Performance")
    print("-" * 40)
    
    ai = AdvancedDocumentAI()
    
    # Mede tempo de processamento
    start_time = time.time()
    ai.process_documents('meus_documentos')
    processing_time = time.time() - start_time
    
    # Mede tempo de criação de embeddings
    start_time = time.time()
    ai.create_embeddings()
    embedding_time = time.time() - start_time
    
    # Obtém estatísticas
    stats = ai.get_statistics()
    
    print("📊 ESTATÍSTICAS DE PERFORMANCE:")
    print(f"   ⏱️ Tempo de processamento: {processing_time:.2f}s")
    print(f"   🧠 Tempo de embeddings: {embedding_time:.2f}s")
    print(f"   📄 Total de documentos: {stats['total_documents']}")
    print(f"   🧩 Total de chunks: {stats['total_chunks']}")
    print(f"   📏 Chunks por documento: {stats['total_chunks']/max(stats['total_documents'],1):.1f}")
    print(f"   🌐 Idiomas detectados: {stats['languages_detected']}")
    print(f"   💾 Tipos de arquivo: {stats['file_types']}")
    
    # Teste de velocidade de busca
    queries = [
        "configuração do sistema",
        "como instalar",
        "solução de problemas",
        "documentação técnica",
        "manual do usuário"
    ]
    
    print("\n🏃 TESTE DE VELOCIDADE DE BUSCA:")
    total_search_time = 0
    
    for query in queries:
        start_time = time.time()
        results = ai.search(query)
        search_time = time.time() - start_time
        total_search_time += search_time
        
        print(f"   🔍 '{query}': {search_time*1000:.1f}ms ({len(results)} resultados)")
    
    print(f"   ⚡ Tempo médio de busca: {(total_search_time/len(queries))*1000:.1f}ms")

# ============================================================================
# EXEMPLO 5: BUSCA AVANÇADA COM FILTROS
# ============================================================================

def exemplo_busca_avancada():
    """Exemplo de busca avançada com filtros"""
    print("\n🚀 EXEMPLO 5: Busca Avançada")
    print("-" * 40)
    
    # Carrega base existente (se houver)
    ai = AdvancedDocumentAI()
    
    try:
        ai.load_knowledge_base()
        print("✅ Base de conhecimento carregada")
    except:
        print("⚠️ Criando nova base...")
        ai.process_documents('meus_documentos')
        ai.create_embeddings()
        ai.save_knowledge_base()
    
    # Busca simples
    print("\n1️⃣ BUSCA SIMPLES:")
    results = ai.search("configuração", top_k=3)
    for r in results:
        print(f"   📄 {r['file_name']}: {r['similarity']:.2%}")
    
    # Busca com filtro por arquivo específico
    print("\n2️⃣ BUSCA FILTRADA POR ARQUIVO:")
    if ai.metadata:
        primeiro_arquivo = ai.metadata[0]['file_name']
        results = ai.search(
            "configuração", 
            filter_metadata={'file_name': primeiro_arquivo}
        )
        print(f"   🎯 Buscando apenas em: {primeiro_arquivo}")
        for r in results:
            print(f"   📝 Chunk {r['chunk_index']}: {r['similarity']:.2%}")
    
    # Busca com filtro por idioma
    print("\n3️⃣ BUSCA FILTRADA POR IDIOMA:")
    results = ai.search(
        "installation", 
        filter_metadata={'language': 'english'}
    )
    if results:
        print("   🇺🇸 Resultados em inglês:")
        for r in results:
            print(f"   📄 {r['file_name']}: {r['similarity']:.2%}")
    else:
        print("   ℹ️ Nenhum documento em inglês encontrado")
    
    # Busca com threshold personalizado
    print("\n4️⃣ BUSCA COM THRESHOLD ALTO:")
    ai.config['similarity_threshold'] = 0.7  # Muito restritivo
    results = ai.search("sistema")
    print(f"   🎯 Threshold: {ai.config['similarity_threshold']}")
    print(f"   📊 Resultados: {len(results)}")

# ============================================================================
# EXEMPLO 6: COMPARAÇÃO COM VERSÃO ANTERIOR
# ============================================================================

def exemplo_comparacao_versoes():
    """Compara performance com versão anterior (simulado)"""
    print("\n🚀 EXEMPLO 6: Comparação de Versões")
    print("-" * 40)
    
    print("📈 MELHORIAS DA VERSÃO AVANÇADA:")
    print()
    
    melhorias = [
        {
            "feature": "Interface Web",
            "anterior": "❌ Apenas linha de comando",
            "atual": "✅ Interface web completa com Streamlit"
        },
        {
            "feature": "Tipos de Arquivo",
            "anterior": "📄 PDF e TXT apenas", 
            "atual": "📚 PDF, DOCX, DOC, RTF, TXT"
        },
        {
            "feature": "Busca",
            "anterior": "🔍 Similaridade simples (O(n))",
            "atual": "⚡ FAISS indexing (O(log n))"
        },
        {
            "feature": "Processamento",
            "anterior": "📝 Chunks fixos",
            "atual": "🧠 Chunking inteligente por parágrafos/sentenças"
        },
        {
            "feature": "Configuração",
            "anterior": "⚙️ Parâmetros fixos no código",
            "atual": "🎛️ Configuração JSON dinâmica"
        },
        {
            "feature": "Metadados",
            "anterior": "📊 Informações básicas",
            "atual": "📈 Metadados ricos + estatísticas"
        },
        {
            "feature": "Idiomas",
            "anterior": "🌐 Suporte básico",
            "atual": "🗣️ Detecção automática + multilíngue"
        },
        {
            "feature": "Visualização",
            "anterior": "📋 Texto simples",
            "atual": "📊 Gráficos interativos + métricas"
        },
        {
            "feature": "Persistência",
            "anterior": "💾 Pickle simples",
            "atual": "🗄️ Base estruturada + FAISS index"
        },
        {
            "feature": "Logs",
            "anterior": "🔇 Sem logging",
            "atual": "📝 Sistema completo de logs"
        }
    ]
    
    for melhoria in melhorias:
        print(f"🎯 {melhoria['feature']}:")
        print(f"   Antes: {melhoria['anterior']}")
        print(f"   Agora: {melhoria['atual']}")
        print()

# ============================================================================
# EXEMPLO 7: CASO DE USO REAL - BIBLIOTECA JURÍDICA
# ============================================================================

def exemplo_biblioteca_juridica():
    """Exemplo de caso de uso real: biblioteca jurídica"""
    print("\n🚀 EXEMPLO 7: Caso de Uso - Biblioteca Jurídica")
    print("-" * 40)
    
    # Simula estrutura de uma biblioteca jurídica
    estrutura_juridica = {
        'leis/': ['lei_8112.pdf', 'constituicao.pdf', 'codigo_civil.pdf'],
        'jurisprudencia/': ['stf_2023.docx', 'stj_precedentes.pdf'],
        'doutrina/': ['direito_admin.pdf', 'processo_civil.docx'],
        'peticoes/': ['modelo_mandado.docx', 'contestacao.pdf']
    }
    
    print("📚 ESTRUTURA DA BIBLIOTECA JURÍDICA:")
    for pasta, arquivos in estrutura_juridica.items():
        print(f"   📁 {pasta}")
        for arquivo in arquivos:
            print(f"      📄 {arquivo}")
    
    print("\n🔍 CONSULTAS TÍPICAS:")
    consultas_juridicas = [
        "prazo para contestação",
        "requisitos do mandado de segurança", 
        "estabilidade do servidor público",
        "precedentes sobre direito adquirido",
        "competência do STF",
        "prescrição em direito administrativo"
    ]
    
    for i, consulta in enumerate(consultas_juridicas, 1):
        print(f"   {i}. '{consulta}'")
    
    print("\n💡 VANTAGENS PARA ÁREA JURÍDICA:")
    vantagens = [
        "⚡ Busca instantânea em milhares de documentos",
        "🎯 Encontra jurisprudência relevante automaticamente", 
        "📋 Localiza precedentes por similaridade semântica",
        "🔍 Busca cruzada em leis, doutrinas e decisões",
        "📊 Estatísticas de uso dos documentos",
        "💾 Diferentes bases por área do direito",
        "🌐 Funciona offline (dados sensíveis seguros)"
    ]
    
    for vantagem in vantagens:
        print(f"   {vantagem}")

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Executa todos os exemplos"""
    print("🧠 SISTEMA AVANÇADO DE IA PARA DOCUMENTOS")
    print("📚 Exemplos Avançados de Uso")
    print("=" * 60)
    
    exemplos = [
        ("Uso Básico", exemplo_basico),
        ("Configuração Personalizada", exemplo_configuracao_personalizada),
        ("Múltiplas Bases", exemplo_multiplas_bases),
        ("Análise de Performance", exemplo_analise_performance),
        ("Busca Avançada", exemplo_busca_avancada),
        ("Comparação de Versões", exemplo_comparacao_versoes),
        ("Caso de Uso Jurídico", exemplo_biblioteca_juridica)
    ]
    
    print("\n📋 EXEMPLOS DISPONÍVEIS:")
    for i, (nome, _) in enumerate(exemplos, 1):
        print(f"   {i}. {nome}")
    
    print("\n" + "="*60)
    
    # Executa exemplo de comparação por padrão
    exemplo_comparacao_versoes()
    
    print("\n💡 DICAS PARA USAR OS EXEMPLOS:")
    print("   1. Certifique-se de ter documentos em 'meus_documentos/'")
    print("   2. Execute 'python exemplos_avancados.py' para ver comparações")
    print("   3. Modifique os exemplos conforme sua necessidade")
    print("   4. Use 'streamlit run advanced_document_ai.py' para interface web")
    
    print("\n🎯 PRÓXIMOS PASSOS:")
    print("   • Adicione seus documentos")
    print("   • Configure parâmetros específicos") 
    print("   • Teste diferentes queries")
    print("   • Explore a interface web")
    print("   • Crie múltiplas bases especializadas")

if __name__ == "__main__":
    main()
