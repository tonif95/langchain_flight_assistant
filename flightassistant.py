from dotenv import load_dotenv
try:
    # Prefer stdlib when available (Python 3.8+)
    from typing import TypedDict
except Exception:
    # Fallback for older Python where TypedDict lives in typing_extensions
    from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import List, Any, Dict, Annotated
import uuid
from flightassistant_tools import ryanair_flight_search
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import asyncio

# Cargamos las variables de entorno
load_dotenv(override=True)


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]

class FlightAssistant:
    def __init__(self, memory):
        self.worker_llm_with_tools = None
        self.tools = None
        self.memory = memory
        self.graph = None
        self.flightassistant_id = str(uuid.uuid4())


    async def setup(self):
        # Traemos las herramientas del worker
        self.tools = [ryanair_flight_search]

        # Establecemos el modelo de lenguaje de nuestro worker y lo inicializamos con las herramientas
        worker_llm = ChatOpenAI(model="gpt-4o-mini")
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)

        # Construimos el grafo
        await self.build_graph()

    def worker(self, state: State) -> Dict[str, Any]:
        """
        Worker Node. His function is organizate the trip
        """

        # Estas son las instrucciones del worker
        system_message = f"""You are a helpful assistant equipped with web tools and FLIGHT search tools.
            The current date is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.

            **RULES FOR FLIGHT SEARCH:**
            1. If the user asks for flights, use the 'ryanair_flight_search' tool.
            2. You MUST convert city names to IATA codes yourself (e.g., Madrid -> MAD, London -> STN or LGW, Barcelona -> BCN).
            3. If the user doesn't specify a year, assume the upcoming date relative to today.
            4. Format dates strictly as YYYY-MM-DD.        
            """
        
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_message)] + messages
        else:
            messages[0].content = system_message
        response = self.worker_llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def worker_router(self, state: State) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END
    
    async def build_graph(self):
        graph_builder = StateGraph(State)
        # Añadir los nodos del sistema
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        # Añadimos los conectores para que el sistema funcione correctamente
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges("worker", self.worker_router, {"tools":"tools", END: END})
        graph_builder.add_edge(START, "worker")
        # Compilamos el grafo utilizando la base de datos como memoria
        self.graph = graph_builder.compile(checkpointer=self.memory)
    
    async def run_superstep(self, user_input:str):
        config = {"configurable": {"thread_id": self.flightassistant_id}}
        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=user_input)]}, # Revisar luego esto de aqui para entender el porque y como funciona
            config=config
            )
        final_response = result["messages"][-1]
        return final_response.content if isinstance(final_response, AIMessage) else "Sin respuesta de texto."
    
async def main():
    print("🤖 Iniciando Agente de Viajes ...")

    async with AsyncSqliteSaver.from_conn_string("memory.db") as checkpointer:

        flightassist = FlightAssistant(memory=checkpointer)
        await flightassist.setup()
        print(f"✅ Listo. Puedes pedirme que te de los precios de los vuelos según la fecha que me indiques")

        try:
            while True:
                try:
                    user_input = input("\n👤 Terricola: ")
                except EOFError: break

                if user_input.strip().lower() in ["salir", "exit"]: break
                if not user_input.strip(): continue

                print("⏳ Trabajando...")
                respuesta = await flightassist.run_superstep(user_input)
                print(f"🤖 Agente: {respuesta}")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrumpido.")
        

if __name__ == "__main__":
    asyncio.run(main())
            



    



        


    
