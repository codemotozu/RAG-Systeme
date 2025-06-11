# RAG From Scratch / RAG von Grund auf

LLMs are trained on a large but fixed corpus of data, limiting their ability to reason about private or recent information. Fine-tuning is one way to mitigate this, but is often [not well-suited for facutal recall](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts) and [can be costly](https://www.glean.com/blog/how-to-build-an-ai-assistant-for-the-enterprise).
Retrieval augmented generation (RAG) has emerged as a popular and powerful mechanism to expand an LLM's knowledge base, using documents retrieved from an external data source to ground the LLM generation via in-context learning. 
These notebooks accompany a [video playlist](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&feature=shared) that builds up an understanding of RAG from scratch, starting with the basics of indexing, retrieval, and generation. 
![rag_detail_v2](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)
 
---

## 🎥 Tutorial Source / Tutorial-Quelle
This project follows the comprehensive tutorial: [**RAG From Scratch**](https://www.youtube.com/watch?v=sVcwVQRHIc8&t=3195s)

Dieses Projekt folgt dem umfassenden Tutorial: [**RAG From Scratch**](https://www.youtube.com/watch?v=sVcwVQRHIc8&t=3195s)

---

## 🇺🇸 English

### Overview
This project implements a comprehensive Retrieval-Augmented Generation (RAG) system built from scratch following LangChain's educational tutorial series. The implementation covers all major RAG components including indexing, retrieval, query transformations, routing, and generation techniques.

### 🎯 Project Purpose
This project was developed as a learning exercise to gain deep understanding of RAG systems by following [this YouTube tutorial series](https://www.youtube.com/watch?v=sVcwVQRHIc8&t=3195s). The original tutorial content is from [LangChain's RAG from scratch repository](https://github.com/langchain-ai/rag-from-scratch/tree/main).

**Learning Objectives:**
- Master fundamental RAG concepts and implementation
- Understand advanced query transformation techniques
- Explore different indexing and retrieval strategies
- Practice with real-world RAG applications
- Document learning in both German and English for language skill development

### 🚀 Features Implemented

#### **Core RAG Pipeline**
- **Document Loading**: Web-based content ingestion with BeautifulSoup parsing
- **Text Splitting**: Intelligent chunking with RecursiveCharacterTextSplitter
- **Vector Storage**: Chroma database with OpenAI embeddings
- **Retrieval**: Similarity-based document retrieval
- **Generation**: LLM-powered answer generation

#### **Query Transformations**
- **Multi-Query**: Generate multiple query variations for parallel retrieval
- **RAG-Fusion**: Combine multiple queries with reciprocal rank fusion
- **Decomposition**: Break complex questions into sub-questions
- **Step-Back**: Create abstract questions for broader context retrieval
- **HyDE**: Generate hypothetical documents for improved similarity matching

#### **Advanced Techniques**
- **Routing**: Logical and semantic query routing
- **Query Construction**: Structured search with metadata filters
- **Multi-Representation Indexing**: Summary-based retrieval with full document storage
- **ColBERT**: Token-level embeddings for precise matching
- **Re-ranking**: Cohere-powered result refinement

### 📁 Project Structure

#### **Notebooks 1-4: Fundamentals** (`rag_from_scratch_1_to_4.ipynb`)
- **Overview**: Complete RAG system architecture understanding
- **Indexing**: Document loading, splitting, embedding, and vector storage
- **Retrieval**: Similarity search and document ranking
- **Generation**: Prompt engineering and LLM integration
- **Implementation**: Basic RAG pipeline with LangChain

#### **Notebooks 5-9: Query Transformations** (`rag_from_scratch_5_to_9.ipynb`)
- **Multi-Query**: Parallel query generation and document deduplication
- **RAG-Fusion**: Query fusion with reciprocal rank scoring
- **Decomposition**: Recursive and individual sub-question answering
- **Step-Back**: Abstract question generation for conceptual retrieval
- **HyDE**: Hypothetical document embedding for improved matching

#### **Notebooks 10-11: Routing** (`rag_from_scratch_10_and_11.ipynb`)
- **Logical Routing**: Function calling for datasource classification
- **Semantic Routing**: Embedding-based prompt selection
- **Query Structuring**: Metadata filter extraction for targeted search

#### **Notebooks 12-14: Advanced Indexing** (`rag_from_scratch_12_to_14.ipynb`)
- **Multi-Representation**: Summary creation with GPT-4V for enhanced retrieval
- **RAPTOR**: Hierarchical document clustering and summarization
- **ColBERT**: Token-level contextual embeddings with RAGatouille

#### **Notebooks 15-18: Retrieval Optimization** (`rag_from_scratch_15_to_18.ipynb`)
- **Re-ranking**: Cohere and RankGPT result optimization
- **CRAG**: Corrective retrieval-augmented generation
- **Self-RAG**: Self-reflective generation with quality control
- **Long Context**: Impact analysis and optimization strategies

### 🛠️ Installation

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd rag-from-scratch
```

2. **Install required packages**
```bash
pip install langchain langchain_community tiktoken langchain-openai
pip install langchainhub chromadb langchain cohere
pip install youtube-transcript-api pytube
pip install ragatouille  # For ColBERT implementation
```

3. **Set up environment variables**
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

4. **Run the notebooks**
Execute notebooks in sequence:
- Start with `rag_from_scratch_1_to_4.ipynb` for fundamentals
- Progress through query transformations in `rag_from_scratch_5_to_9.ipynb`
- Explore routing in `rag_from_scratch_10_and_11.ipynb`
- Study advanced indexing in `rag_from_scratch_12_to_14.ipynb`
- Master retrieval optimization in `rag_from_scratch_15_to_18.ipynb`

### 🤖 Technologies Used
- **LangChain**: Framework for LLM application development
- **OpenAI**: GPT models and embeddings (GPT-3.5-turbo, GPT-4)
- **Chroma**: Vector database for similarity search
- **Cohere**: Advanced re-ranking and embeddings
- **RAGatouille**: ColBERT implementation for token-level matching
- **LangSmith**: Monitoring and debugging platform
- **BeautifulSoup**: HTML parsing for web content
- **Tiktoken**: Token counting and management

### 🎯 Key Learning Outcomes
1. **RAG Fundamentals**: Understanding of indexing, retrieval, and generation pipeline
2. **Query Engineering**: Multiple techniques for query optimization and transformation
3. **Advanced Retrieval**: Multi-representation indexing and specialized embeddings
4. **Result Optimization**: Re-ranking and fusion techniques for improved accuracy
5. **System Architecture**: Routing and query construction for complex applications

---

## 🇩🇪 Deutsch

### Überblick
Dieses Projekt implementiert ein umfassendes Retrieval-Augmented Generation (RAG) System, das von Grund auf nach LangChains Bildungs-Tutorial-Serie erstellt wurde. Die Implementierung deckt alle wichtigen RAG-Komponenten ab, einschließlich Indexierung, Retrieval, Query-Transformationen, Routing und Generierungstechniken.

### 🎯 Projektzweck
Dieses Projekt wurde als Lernübung entwickelt, um ein tiefes Verständnis von RAG-Systemen zu erlangen, indem [dieser YouTube-Tutorial-Serie](https://www.youtube.com/watch?v=sVcwVQRHIc8&t=3195s) gefolgt wurde. Der ursprüngliche Tutorial-Inhalt stammt aus [LangChains RAG from scratch Repository](https://github.com/langchain-ai/rag-from-scratch/tree/main).

**Lernziele:**
- Grundlegende RAG-Konzepte und -Implementierung meistern
- Erweiterte Query-Transformationstechniken verstehen
- Verschiedene Indexierungs- und Retrieval-Strategien erkunden
- Praxis mit realen RAG-Anwendungen
- Lernen in Deutsch und Englisch dokumentieren für Sprachkompetenzentwicklung

### 🚀 Implementierte Funktionen

#### **Kern-RAG-Pipeline**
- **Dokumentenladung**: Web-basierte Inhaltsaufnahme mit BeautifulSoup-Parsing
- **Textaufteilung**: Intelligente Chunking mit RecursiveCharacterTextSplitter
- **Vektorspeicherung**: Chroma-Datenbank mit OpenAI-Embeddings
- **Retrieval**: Ähnlichkeitsbasierte Dokumentwiederauffindung
- **Generierung**: LLM-gestützte Antwortgenerierung

#### **Query-Transformationen**
- **Multi-Query**: Mehrere Query-Variationen für paralleles Retrieval generieren
- **RAG-Fusion**: Mehrere Queries mit reciprocal rank fusion kombinieren
- **Decomposition**: Komplexe Fragen in Teilfragen aufteilen
- **Step-Back**: Abstrakte Fragen für breiteren Kontext-Retrieval erstellen
- **HyDE**: Hypothetische Dokumente für verbesserte Ähnlichkeitsabgleiche generieren

#### **Erweiterte Techniken**
- **Routing**: Logisches und semantisches Query-Routing
- **Query-Konstruktion**: Strukturierte Suche mit Metadaten-Filtern
- **Multi-Representation-Indexierung**: Zusammenfassungsbasiertes Retrieval mit vollständiger Dokumentspeicherung
- **ColBERT**: Token-Level-Embeddings für präzise Übereinstimmung
- **Re-ranking**: Cohere-gestützte Ergebnisverfeinerung

### 📁 Projektstruktur

#### **Notebooks 1-4: Grundlagen** (`rag_from_scratch_1_to_4.ipynb`)
- **Überblick**: Vollständiges Verständnis der RAG-Systemarchitektur
- **Indexierung**: Dokumentenladen, -aufteilung, -embedding und Vektorspeicherung
- **Retrieval**: Ähnlichkeitssuche und Dokumentenranking
- **Generierung**: Prompt-Engineering und LLM-Integration
- **Implementierung**: Grundlegende RAG-Pipeline mit LangChain

#### **Notebooks 5-9: Query-Transformationen** (`rag_from_scratch_5_to_9.ipynb`)
- **Multi-Query**: Parallele Query-Generierung und Dokumentendeduplizierung
- **RAG-Fusion**: Query-Fusion mit reciprocal rank scoring
- **Decomposition**: Rekursive und individuelle Teilfragen-Beantwortung
- **Step-Back**: Abstrakte Fragengenerierung für konzeptionelles Retrieval
- **HyDE**: Hypothetisches Dokument-Embedding für verbesserte Übereinstimmung

#### **Notebooks 10-11: Routing** (`rag_from_scratch_10_and_11.ipynb`)
- **Logical Routing**: Function Calling für Datenquellen-Klassifikation
- **Semantic Routing**: Embedding-basierte Prompt-Auswahl
- **Query Structuring**: Metadaten-Filter-Extraktion für gezielte Suche

#### **Notebooks 12-14: Erweiterte Indexierung** (`rag_from_scratch_12_to_14.ipynb`)
- **Multi-Representation**: Zusammenfassungserstellung mit GPT-4V für verbessertes Retrieval
- **RAPTOR**: Hierarchische Dokumentenclustering und -zusammenfassung
- **ColBERT**: Token-Level kontextuelle Embeddings mit RAGatouille

#### **Notebooks 15-18: Retrieval-Optimierung** (`rag_from_scratch_15_to_18.ipynb`)
- **Re-ranking**: Cohere und RankGPT Ergebnisoptimierung
- **CRAG**: Corrective retrieval-augmented generation
- **Self-RAG**: Selbstreflektive Generierung mit Qualitätskontrolle
- **Long Context**: Auswirkungsanalyse und Optimierungsstrategien

### 🛠️ Installation

1. **Repository klonen**
```bash
git clone <ihre-repository-url>
cd rag-from-scratch
```

2. **Erforderliche Pakete installieren**
```bash
pip install langchain langchain_community tiktoken langchain-openai
pip install langchainhub chromadb langchain cohere
pip install youtube-transcript-api pytube
pip install ragatouille  # Für ColBERT-Implementierung
```

3. **Umgebungsvariablen einrichten**
Erstellen Sie eine `.env` Datei im Projektroot:
```
OPENAI_API_KEY=ihr_openai_api_schluessel_hier
COHERE_API_KEY=ihr_cohere_api_schluessel_hier
LANGCHAIN_API_KEY=ihr_langsmith_api_schluessel_hier
```

4. **Notebooks ausführen**
Notebooks in Reihenfolge ausführen:
- Beginnen Sie mit `rag_from_scratch_1_to_4.ipynb` für Grundlagen
- Fortschritt durch Query-Transformationen in `rag_from_scratch_5_to_9.ipynb`
- Routing erkunden in `rag_from_scratch_10_and_11.ipynb`
- Erweiterte Indexierung studieren in `rag_from_scratch_12_to_14.ipynb`
- Retrieval-Optimierung meistern in `rag_from_scratch_15_to_18.ipynb`

### 🤖 Verwendete Technologien
- **LangChain**: Framework für LLM-Anwendungsentwicklung
- **OpenAI**: GPT-Modelle und Embeddings (GPT-3.5-turbo, GPT-4)
- **Chroma**: Vektordatenbank für Ähnlichkeitssuche
- **Cohere**: Erweiterte Re-ranking und Embeddings
- **RAGatouille**: ColBERT-Implementierung für Token-Level-Matching
- **LangSmith**: Monitoring- und Debugging-Plattform
- **BeautifulSoup**: HTML-Parsing für Web-Inhalte
- **Tiktoken**: Token-Zählung und -Verwaltung

### 🎯 Wichtige Lernergebnisse
1. **RAG-Grundlagen**: Verständnis der Indexierungs-, Retrieval- und Generierungs-Pipeline
2. **Query-Engineering**: Mehrere Techniken für Query-Optimierung und -Transformation
3. **Erweiterte Wiederauffindung**: Multi-Representation-Indexierung und spezialisierte Embeddings
4. **Ergebnisoptimierung**: Re-ranking und Fusion-Techniken für verbesserte Genauigkeit
5. **Systemarchitektur**: Routing und Query-Konstruktion für komplexe Anwendungen

---

## 🙏 Acknowledgments / Danksagungen

**English:**
- Original tutorial content from [LangChain's RAG from scratch tutorial series](https://www.youtube.com/watch?v=sVcwVQRHIc8&t=3195s)
- Source repository: [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch/tree/main)
- Learning approach guided by LangChain's comprehensive educational materials
- Documentation enhanced with bilingual German-English explanations for language learning

**Deutsch:**
- Ursprünglicher Tutorial-Inhalt von [LangChains RAG from scratch Tutorial-Serie](https://www.youtube.com/watch?v=sVcwVQRHIc8&t=3195s)
- Quell-Repository: [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch/tree/main)
- Lernansatz geleitet von LangChains umfassenden Bildungsmaterialien
- Dokumentation erweitert mit zweisprachigen Deutsch-Englisch-Erklärungen für Sprachenlernen

## 📄 License / Lizenz
This project is for educational purposes. Please respect the original licenses of the referenced repositories and datasets.

Dieses Projekt dient Bildungszwecken. Bitte respektieren Sie die ursprünglichen Lizenzen der referenzierten Repositories und Datensätze.
