import requests
from langchain.tools import tool

@tool
def ryanair_flight_search(origen: str, destino: str, fecha: str, moneda: str = "EUR"):
    """
    Busca los vuelos más baratos en Ryanair para una fecha específica.
    ...
    """

    url = "https://ryanair-api-hx0t.onrender.com/api/search-fares"

    params = {
        "from": origen,
        "to": destino,
        "date": fecha,
        "currency": moneda
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    
    except requests.exception.RequesException as e:
        return {"error": f"Error de conexión: {str(e)}"}