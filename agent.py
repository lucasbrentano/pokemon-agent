# agent.py

import json
import re
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# --- LangChain Imports (Updated) ---
# Note the new 'langchain_ollama' imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from pydantic import BaseModel

# (Your other standard imports like pokebase, etc., remain the same)
import pokebase

# --- Global Configuration ---

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pokemon_agent_academic.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- Helper Functions ---

def normalize_name(name: Any) -> str:
    """
    Ensures the passed name is a simple string, handles regional forms,
    and formats it for the PokéAPI standard (lowercase, with a hyphen).
    """
    if isinstance(name, dict):
        if "title" in name:
            name = name["title"]
        elif "name" in name:
            name = name["name"]
        else:
            # Fallback for unexpected dictionary structures
            for v in name.values():
                if isinstance(v, str):
                    name = v
                    break

    name_str = str(name).lower()

    regional_forms = {
        "alolan": "alola",
        "galarian": "galar",
        "hisuian": "hisui",
        "paldean": "paldea"
    }

    parts = name_str.split()
    for regional_long, regional_short in regional_forms.items():
        if regional_long in parts:
            parts.remove(regional_long)
            pokemon_name = "-".join(parts)
            return f"{pokemon_name}-{regional_short}"

    return name_str.replace(" ", "-")


# --- Data Models ---

@dataclass
class APIMetrics:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_response_time: float = 0.0
    endpoints_used: Dict[str, int] = None

    def __post_init__(self):
        if self.endpoints_used is None:
            self.endpoints_used = {}


class QueryType(Enum):
    BASIC_INFO = "basic_info"
    BATTLE_ANALYSIS = "battle_analysis"
    TEAM_COMPOSITION = "team_composition"
    EVOLUTION_ANALYSIS = "evolution_analysis"
    TYPE_EFFECTIVENESS = "type_effectiveness"
    MOVE_ANALYSIS = "move_analysis"


class PokemonBasicData(BaseModel):
    id: int
    name: str
    types: List[str]
    abilities: List[str]
    base_stats: Dict[str, int]


class PokemonSpeciesData(BaseModel):
    id: int
    name: str
    evolution_chain_url: Optional[str] = None
    habitat: Optional[str] = None
    generation: Optional[str] = None
    legendary: Optional[bool] = None
    mythical: Optional[bool] = None
    base_happiness: Optional[int] = None
    capture_rate: Optional[int] = None


class TypeEffectivenessData(BaseModel):
    name: str
    double_damage_to: List[str]
    double_damage_from: List[str]
    half_damage_to: List[str]
    half_damage_from: List[str]
    no_damage_to: List[str]
    no_damage_from: List[str]


# --- API Client ---

class PokeAPIClient:
    """Optimized and robust client for the PokéAPI."""

    def __init__(self):
        """Initializes the PokeAPIClient and its metrics tracker."""
        self.metrics = APIMetrics()

    def _track_api_call(self, endpoint: str, start_time: datetime):
        """Helper function to track metrics for successful API calls."""
        response_time = (datetime.now() - start_time).total_seconds()
        self.metrics.total_calls += 1
        self.metrics.successful_calls += 1
        self.metrics.total_response_time += response_time
        self.metrics.endpoints_used[endpoint] = self.metrics.endpoints_used.get(endpoint, 0) + 1

    def get_pokemon(self, name_or_id: str) -> Optional[PokemonBasicData]:
        normalized_name = normalize_name(name_or_id)
        start_time = datetime.now()
        try:
            p = pokebase.pokemon(normalized_name)
            if not hasattr(p, 'id'): return None
            self._track_api_call('pokemon', start_time)
            return PokemonBasicData(
                id=p.id, name=p.name, types=[t.type.name for t in p.types],
                abilities=[a.ability.name for a in p.abilities],
                base_stats={s.stat.name: s.base_stat for s in p.stats}
            )
        except Exception as e:
            self.metrics.failed_calls += 1
            logger.error(f"Error fetching pokemon/{normalized_name}: {e}")
            return None

    def get_pokemon_species(self, name_or_id: str) -> Optional[PokemonSpeciesData]:
        normalized_name = normalize_name(name_or_id)
        start_time = datetime.now()
        try:
            s = pokebase.pokemon_species(normalized_name)
            if not hasattr(s, 'id'):
                logger.warning(f"pokebase returned an invalid object for pokemon-species/{normalized_name}")
                return None
            self._track_api_call('pokemon-species', start_time)
            return PokemonSpeciesData(
                id=s.id, name=s.name,
                habitat=s.habitat.name if hasattr(s, 'habitat') and s.habitat else "Unknown",
                generation=s.generation.name if hasattr(s, 'generation') and s.generation else "Unknown",
                legendary=getattr(s, 'is_legendary', False),
                mythical=getattr(s, 'is_mythical', False),
                base_happiness=getattr(s, 'base_happiness', None),
                capture_rate=getattr(s, 'capture_rate', None)
            )
        except Exception as e:
            self.metrics.failed_calls += 1
            logger.error(f"Error fetching pokemon-species/{normalized_name}: {e}")
            return None

    def get_evolution_chain(self, pokemon_name: str) -> dict:
        normalized_name = normalize_name(pokemon_name)
        start_time = datetime.now()
        try:
            species_obj = pokebase.pokemon_species(normalized_name)
            if not hasattr(species_obj, 'evolution_chain') or not species_obj.evolution_chain:
                return {"pokemon": pokemon_name, "evolution_chain": [pokemon_name],
                        "notes": "This Pokémon does not evolve."}
            self._track_api_call('pokemon-species', start_time)

            evo_chain_id = species_obj.evolution_chain.id
            if not evo_chain_id:
                evo_chain_id = int(species_obj.evolution_chain.resource_uri.strip('/').split('/')[-1])

            chain_obj = pokebase.evolution_chain(evo_chain_id)
            self._track_api_call('evolution-chain', datetime.now())

            def parse_chain(node):
                result = [node.species.name]
                for evolution in node.evolves_to:
                    result.extend(parse_chain(evolution))
                return result

            return {"pokemon": pokemon_name, "evolution_chain": parse_chain(chain_obj.chain)}
        except AttributeError:
            self.metrics.failed_calls += 1
            logger.warning(f"Invalid object returned for species '{normalized_name}'. Pokémon may not exist.")
            return {"error": f"Species '{normalized_name}' not found."}
        except Exception as e:
            self.metrics.failed_calls += 1
            logger.error(f"Internal error processing evolution chain for {normalized_name}: {str(e)}")
            return {"error": f"Internal error processing evolution chain: {str(e)}"}

    def get_type_effectiveness(self, type_name: str) -> Optional[TypeEffectivenessData]:
        normalized_name = normalize_name(type_name)
        start_time = datetime.now()
        try:
            t = pokebase.type_(normalized_name)
            if not hasattr(t, 'id'): return None
            self._track_api_call('type', start_time)
            dr = t.damage_relations
            return TypeEffectivenessData(
                name=t.name, double_damage_to=[x.name for x in dr.double_damage_to],
                double_damage_from=[x.name for x in dr.double_damage_from],
                half_damage_to=[x.name for x in dr.half_damage_to],
                half_damage_from=[x.name for x in dr.half_damage_from],
                no_damage_to=[x.name for x in dr.no_damage_to],
                no_damage_from=[x.name for x in dr.no_damage_from]
            )
        except Exception as e:
            self.metrics.failed_calls += 1
            logger.error(f"Error fetching type/{normalized_name}: {e}")
            return None

    def get_move_details(self, move_name: str) -> Optional[Dict]:
        normalized_name = normalize_name(move_name)
        start_time = datetime.now()
        try:
            m = pokebase.move(normalized_name)
            if not hasattr(m, 'id'): return None
            self._track_api_call('move', start_time)
            return {
                "name": m.name, "type": m.type.name, "power": m.power,
                "accuracy": m.accuracy, "pp": m.pp, "damage_class": m.damage_class.name,
                "effect": next((e.effect for e in m.effect_entries if e.language.name == 'en'),
                               "No effect description available.")
            }
        except Exception as e:
            self.metrics.failed_calls += 1
            logger.error(f"Error fetching move/{normalized_name}: {e}")
            return None


# --- Agent Tools ---

# Shared client instance for all tools
poke_client = PokeAPIClient()


@tool
def pokemon_info(pokemon_name: str) -> str:
    """
    Gets basic information about a Pokémon including ID, name, types, abilities, and base stats.
    Input should be the name or ID of the Pokémon.
    """
    try:
        data = poke_client.get_pokemon(pokemon_name)
        if not data:
            return json.dumps({"error": f"Pokémon '{pokemon_name}' not found"})
        return json.dumps(data.model_dump(), indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error in pokemon_info tool for {pokemon_name}: {e}")
        return json.dumps({"error": f"Internal tool error: {str(e)}"})


@tool
def pokemon_species(pokemon_name: str) -> str:
    """
    Gets species information about a Pokémon including legendary/mythical status, generation, and habitat.
    Input should be the name or ID of the Pokémon.
    """
    try:
        data = poke_client.get_pokemon_species(pokemon_name)
        if not data:
            return json.dumps({"error": f"Species '{pokemon_name}' not found"})
        return json.dumps(data.model_dump(), indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error in pokemon_species tool for {pokemon_name}: {e}")
        return json.dumps({"error": f"Internal tool error: {str(e)}"})


@tool
def type_effectiveness(type_name: str) -> str:
    """
    Analyzes the effectiveness of a Pokémon type, including weaknesses, resistances, and immunities.
    Input should be the name of the type (e.g., fire, water, grass).
    """
    try:
        data = poke_client.get_type_effectiveness(type_name)
        if not data:
            return json.dumps({"error": f"Type '{type_name}' not found"})
        return json.dumps(data.model_dump(), indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error in type_effectiveness tool for {type_name}: {e}")
        return json.dumps({"error": f"Internal tool error: {str(e)}"})


@tool
def move_analysis(move_name: str) -> str:
    """
    Analyzes details of a Pokémon move including its type, power, accuracy, PP, damage class, and effect.
    Input should be the name of the move.
    """
    try:
        data = poke_client.get_move_details(move_name)
        if not data:
            return json.dumps({"error": f"Move '{move_name}' not found"})
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error in move_analysis tool for {move_name}: {e}")
        return json.dumps({"error": f"Internal tool error: {str(e)}"})


@tool
def evolution_chain(pokemon_name: str) -> str:
    """

    Gets the complete evolution chain for a Pokémon in order.
    Returns a list of Pokémon in the evolutionary order.
    Example: Charmander → Charmeleon → Charizard.
    Input should be the name or ID of the Pokémon.
    """
    try:
        data = poke_client.get_evolution_chain(pokemon_name)
        if not data:
            return json.dumps({"error": f"Evolution chain for '{pokemon_name}' not found"})
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error in evolution_chain tool for {pokemon_name}: {e}")
        return json.dumps({"error": f"Internal tool error: {str(e)}"})


class PokemonReasoningAgent:
    """
    Main Agent: Implements a RAG-First, Tools-as-Fallback architecture.
    """

    # In agent.py, inside the PokemonReasoningAgent class

    def __init__(self, model_name: str = "mistral:7b-instruct"):
        self.model_name = model_name
        self.llm = ChatOllama(model=self.model_name, temperature=0.1)
        self.api_client = poke_client
        self.tools = [pokemon_info, pokemon_species, type_effectiveness, move_analysis, evolution_chain]

        # These lines are correct
        self.rag_answerer = self._setup_rag_answerer()
        self.query_classifier = self._setup_classifier()
        self.tool_agents = self._setup_tool_agents()

        # --- THIS IS THE MISSING PIECE ---
        # Initialize the session_metrics dictionary here, before the try block.
        self.session_metrics = {
            "queries_processed": 0, "successful_queries": 0,
            "reasoning_chains": [], "model_decisions": []
        }
        # --- END OF FIX ---

        try:
            print("🧠 Loading Pokémon knowledge base...")
            embeddings = OllamaEmbeddings(model="mxbai-embed-large")
            self.vectorstore = Chroma(
                persist_directory="./pokemon_knowledge",
                embedding_function=embeddings
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            print("✅ Knowledge base loaded successfully.")
        except Exception as e:
            print(f"⚠️ WARNING: Could not load vector database. RAG features will be disabled. Error: {e}")
            self.retriever = None

        logger.info(f"RAG-First agent initialized with base model: {model_name}")

    def _setup_rag_answerer(self) -> LLMChain:
        """Sets up a simple chain to answer ONLY from context."""
        rag_prompt_template = """You are a helpful Pokémon expert. Based ONLY on the CONTEXT provided below, answer the user's QUESTION.
        If the context does not contain the answer, you MUST respond with the exact phrase "I DON'T KNOW". Do not use any other information.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        prompt = PromptTemplate(template=rag_prompt_template, input_variables=["context", "question"])
        return LLMChain(llm=self.llm, prompt=prompt)

    def _setup_classifier(self) -> LLMChain:
        """Sets up a classifier using a fast LLM."""
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""TASK: Classify the Pokémon query into ONE category. AVAILABLE CATEGORIES: BASIC_INFO, BATTLE_ANALYSIS, TEAM_COMPOSITION, EVOLUTION_ANALYSIS, TYPE_EFFECTIVENESS, MOVE_ANALYSIS. QUERY: "{query}". CLASSIFICATION (respond only with the category):"""
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def _setup_tool_agents(self) -> Dict[str, Any]:
        """Sets up specialized agents with appropriate models for each task."""
        llm_simple = ChatOllama(model="mistral:7b-instruct", temperature=0.1)
        llm_complex = ChatOllama(model="gemma3:12b", temperature=0.2)

        # --- UPDATED, RAG-AWARE PROMPTS ---
        # Note the new first line and the {context} variable.

        battle_analysis_prompt = """You are a meticulous Pokémon Battle Analyst. Your task is to compare two Pokémon. You MUST follow this sequence of steps STRICTLY:
        STEP 1: Identify the two Pokémon from the user's query.
        STEP 2: Use the `pokemon_info` tool for the FIRST Pokémon.
        STEP 3: Use the `pokemon_info` tool for the SECOND Pokémon.
        STEP 4: ONLY AFTER you have the data for both, write a final analysis comparing their stats and types, then conclude which has an advantage.
        CRITICAL RULE: Do not provide a final answer until you have successfully executed tool calls for BOTH Pokémon."""

        team_composition_prompt = """You are a meticulous Team Builder. Your task is to build a team of up to 3 Pokémon. You MUST follow this RIGID AND SEQUENTIAL process:
        STEP 1: Identify the core Pokémon mentioned in the user's query.
        STEP 2: Suggest the FIRST Pokémon for the team and IMMEDIATELY use the `pokemon_info` tool to get its data.
        STEP 3: Based on the data from the previous steps, suggest the NEXT Pokémon and IMMEDIATELY use the `pokemon_info` tool for it.
        STEP 4: ONLY when you have the data for all suggested members, provide a final summary of the team.
        CRITICAL RULE: It is FORBIDDEN to suggest a Pokémon without immediately using the `pokemon_info` tool to get its data."""

        type_effectiveness_prompt = """You are a precise Pokémon Type Analyst. Your only task is to use the `type_effectiveness` tool to answer the user's question.
        CRITICAL INSTRUCTIONS:
        - If the user asks 'What is effective against type X?' or 'What are X's weaknesses?', you MUST look at the `double_damage_from` field in the tool's output.
        - If they ask 'What is type X effective against?', you MUST look at the `double_damage_to` field.
        Do not confuse the two."""

        agent_configs = {
            QueryType.BASIC_INFO.value: {"llm": llm_simple, "system_message": "Use the `pokemon_info` tool to get data."},
            QueryType.EVOLUTION_ANALYSIS.value: {"llm": llm_complex, "system_message": "Use the `evolution_chain` tool to determine a Pokémon's evolution line."},
            QueryType.TYPE_EFFECTIVENESS.value: {"llm": llm_complex, "system_message": type_effectiveness_prompt},
            QueryType.MOVE_ANALYSIS.value: {"llm": llm_simple, "system_message": "Use the `move_analysis` tool to get data."},
            QueryType.BATTLE_ANALYSIS.value: {"llm": llm_complex, "system_message": battle_analysis_prompt},
            QueryType.TEAM_COMPOSITION.value: {"llm": llm_complex, "system_message": team_composition_prompt},
        }

        agents = {}
        for query_type, config in agent_configs.items():
            agents[query_type] = {
                "agent": initialize_agent(
                    tools=self.tools,
                    llm=config["llm"],
                    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    max_iterations=8,
                    handle_parsing_errors=True,
                    agent_kwargs={"system_message": config["system_message"]}
                ),
                "llm_model": config["llm"].model
            }
        return agents

    def classify_query(self, user_query: str) -> QueryType:
        """Classifies the query using the base LLM."""
        try:
            result = self.query_classifier.invoke({"query": user_query})
            classification = result['text'].strip().upper().split('.')[-1].strip()
            return QueryType[classification]
        except (KeyError, IndexError):
            logger.warning(f"Could not classify query '{user_query}'. Defaulting to BASIC_INFO.")
            return QueryType.BASIC_INFO

    # In agent.py -> class PokemonReasoningAgent

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Processes a query using the RAG-First, Tools-as-Fallback strategy.
        """
        start_time = datetime.now()
        # FINAL FIX #1: Count every query as soon as it comes in.
        self.session_metrics["queries_processed"] += 1

        try:
            # --- STEP 1: ATTEMPT TO ANSWER WITH RAG ---
            if self.retriever:
                print("🧠 Searching knowledge base...")
                retrieved_docs = self.retriever.get_relevant_documents(user_query)
                if retrieved_docs:
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    print("✅ Context found. Attempting to answer from knowledge base...")

                    rag_result = self.rag_answerer.invoke({"context": context, "question": user_query})
                    rag_answer = rag_result['text'].strip()

                    if "I DON'T KNOW" not in rag_answer.upper():
                        print("✅ Answer found in knowledge base.")

                        # --- FINAL FIX #2: Log metrics for the successful RAG path ---
                        self.session_metrics["successful_queries"] += 1
                        self.session_metrics["model_decisions"].append(
                            {"query": user_query, "decision": "rag_knowledge"})
                        self.session_metrics["reasoning_chains"].append(rag_result)  # Log the RAG chain result
                        # --- END OF FIX ---

                        return {
                            "query": user_query, "query_type": "rag_knowledge",
                            "model_used": self.llm.model, "response": rag_answer,
                            "processing_time_seconds": (datetime.now() - start_time).total_seconds(), "success": True,
                        }

            print("🟡 Knowledge base did not contain the answer. Falling back to tools.")

            # --- STEP 2: FALLBACK TO TOOL-USING AGENT ---
            query_type = self.classify_query(user_query)
            selected_agent_bundle = self.tool_agents[query_type.value]
            selected_agent = selected_agent_bundle["agent"]

            result = selected_agent.invoke({"input": user_query})

            # Log metrics for the tool-based path
            self.session_metrics["model_decisions"].append({"query": user_query, "decision": query_type.value})
            self.session_metrics["reasoning_chains"].append(result)

            raw_output = result.get('output', '')
            clean_response = raw_output
            try:
                json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                    clean_response = parsed_json.get("action_input", raw_output)
            except (json.JSONDecodeError, TypeError):
                clean_response = raw_output

            # Count success for the tool-based path
            self.session_metrics["successful_queries"] += 1

            processing_time = (datetime.now() - start_time).total_seconds()
            response = {
                "query": user_query,
                "query_type": query_type.value,
                "model_used": selected_agent_bundle["llm_model"],
                "response": clean_response,
                "processing_time_seconds": processing_time,
                "success": True,
            }
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return response

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # No success counter is incremented here, which is correct for an error.
            return {
                "query": user_query,
                "query_type": "error",
                "model_used": self.model_name,
                "response": error_msg,
                "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                "success": False
            }

    def get_academic_metrics(self) -> Dict[str, Any]:
        """Returns metrics for academic analysis."""
        api_metrics = self.api_client.metrics
        return {
            "session_overview": {
                "total_queries": self.session_metrics["queries_processed"],
                "successful_queries": self.session_metrics["successful_queries"],
                "success_rate": self.session_metrics["successful_queries"] / max(1, self.session_metrics[
                    "queries_processed"])
            },
            "api_performance": {
                "total_api_calls": api_metrics.total_calls,
                "successful_api_calls": api_metrics.successful_calls,
                "failed_api_calls": api_metrics.failed_calls,
                "api_success_rate": api_metrics.successful_calls / max(1, api_metrics.total_calls),
                "average_response_time": api_metrics.total_response_time / max(1, api_metrics.total_calls),
                "endpoints_used": api_metrics.endpoints_used
            },
            "model_info": {
                "base_model_name": self.model_name,
                "framework": "LangChain + Ollama",
                "api_source": "PokéAPI (https://pokeapi.co/)"
            }
        }