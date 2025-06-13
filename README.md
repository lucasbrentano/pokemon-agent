# 🤖 Pokémon AI Agent (RAG-Enabled)

This project implements an advanced AI agent capable of answering a wide range of questions about the Pokémon universe. It uses a modern **RAG-First, Tools-as-Fallback** architecture, combining a local knowledge base with live API calls to provide accurate and context-aware responses.

The agent is built with Python, leveraging the **LangChain** and **Ollama** ecosystems to run local language models. The user interface is a web application created with **Streamlit**.

## ✨ Core Features

* **Hybrid RAG & Tools Architecture**: The agent first searches a local vector database (RAG) for answers. If the information isn't found, it falls back to a suite of specialized tools that call the live PokéAPI.
* **Local LLM Powered**: Runs entirely on your local machine using models from Ollama (e.g., Mistral, Gemma), ensuring privacy and no API costs.
* **Comprehensive Knowledge**: Can answer questions about:
    * Basic Pokémon stats, types, and abilities.
    * Evolutionary lines.
    * Type effectiveness and battle matchups.
    * Detailed move analysis.
    * Complex team-building and battle strategies.
* **Interactive Web Interface**: A user-friendly chat interface built with Streamlit allows for easy interaction with the agent.
* **Publicly Sharable**: Includes a `launch.py` script using `pyngrok` to create a temporary public URL, making it easy to share your running agent with others.

## 🏛️ Project Architecture

The agent's intelligence is structured around a "chain of thought" process that prioritizes efficiency and accuracy:

1.  **RAG First**: When a query is received, the agent first searches its `ChromaDB` vectorstore for relevant information. This local knowledge base is ideal for answering complex or conceptual questions.
2.  **Query Classification**: If the RAG system cannot find a confident answer, the query is passed to a classifier LLM. This classifier determines the user's intent (e.g., `BATTLE_ANALYSIS`, `EVOLUTION_ANALYSIS`).
3.  **Tools as Fallback**: Based on the classification, a specialized agent is chosen to handle the query. This agent uses a set of tools that make live calls to the PokéAPI to fetch structured data. This ensures that information like stats and move details is always up-to-date.

This hybrid approach allows the agent to answer "what is" questions with API data and "how-to" or "why" questions using its broader, embedded knowledge.

## 🛠️ Setup and Installation

Follow these steps to get the Pokémon AI Agent running on your local machine.

### Prerequisites

1.  **Python 3.8+**: Ensure you have a modern version of Python installed.
2.  **Ollama**: You must have [Ollama](https://ollama.com/) installed and running on your machine.
3.  **Required LLM Models**: Before launching the app, pull the necessary models by running these commands in your terminal:
    ```bash
    ollama pull mistral:7b-instruct
    ollama pull gemma2:12b
    ollama pull mxbai-embed-large
    ```

### Hardware Requirements

Running multiple language models locally is resource-intensive. Performance will vary greatly depending on your hardware. The key is having enough available RAM or VRAM to load the largest model (`gemma2:9b`), which requires ~9 GB of memory on its own.

* **✅ Recommended (GPU)**: A dedicated NVIDIA GPU with at least **12 GB of VRAM** (e.g., RTX 3060 12GB, RTX 4070) is highly recommended for a fast and smooth experience.
* **👍 Minimum (GPU)**: A dedicated GPU with at least **8 GB of VRAM** (e.g., RTX 3050, RTX 2060 Super) can run the models, but may be slower, especially with the 9B parameter model.
* **🍏 Apple Silicon**: Macs with Apple Silicon (M1/M2/M3) and at least **16 GB of unified memory** are excellent for running these models. 24 GB is recommended for best performance.
* **🐌 CPU Fallback**: If you do not have a dedicated GPU, you can run the models on your CPU, but it will be **significantly slower**. You should have at least **32 GB of system RAM** for a reasonable experience.

### Installation Steps

1.  **Clone the Repository**:
    Clone this repository to your local machine. Make sure you download the entire `pokemon_knowledge` folder along with the Python files.
    ```bash
    git clone [https://github.com/YOUR_USERNAME/pokemon-ai-agent.git](https://github.com/YOUR_USERNAME/pokemon-ai-agent.git)
    cd pokemon-ai-agent
    ```

2.  **Verify the Knowledge Base**:
    Ensure the `pokemon_knowledge` folder is present in the root directory of the project (the same folder as `app.py`). This folder contains the pre-built vector database that the agent uses for its RAG capabilities.

3.  **Create a Virtual Environment** (Recommended):
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    Install all required Python libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Running the Application

There are two ways to run the agent:

### 1. Local Development

This runs the Streamlit app on your local machine, accessible only to you.

```bash
streamlit run app.py
```

### 2. Public Sharing with Ngrok

To share a live, temporary link to your running application, use the `launch.py` script. This is perfect for demos.

```bash
python launch.py
```

This will start the Streamlit app and print a public `ngrok.io` URL in your terminal that you can share with others.

## ©️ Licensing and Attributions

This project is made possible by the incredible work of the Pokémon community and open-source developers. Please note that different components of this project are covered by different licenses.

**The project as a whole is intended for non-commercial, educational, and fan-use only.**

* **Source Code**: The original source code of this project is licensed under the **MIT License**. You can find the full license text in the `LICENSE` file.

* **Data Sources**:
    * **PokéAPI**: This project uses the free and open [PokéAPI](https://pokeapi.co/) for live Pokémon data. We thank the PokéAPI community for their amazing work.
    * **Bulbapedia**: The local knowledge base (RAG) was built using information sourced from [Bulbapedia](https://bulbapedia.bulbagarden.net/), which is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 2.5 Generic (CC BY-NC-SA 2.5)](https://creativecommons.org/licenses/by-nc-sa/2.5/) license.

* **Core Libraries**:
    * **PokéBase**: API interactions are simplified by the `pokebase` library, a Python wrapper for the PokéAPI.
    * **LangChain, Streamlit, & Ollama**: This project stands on the shoulders of giants in the open-source AI community. Their powerful tools made this agent possible.

