import asyncio
import websockets
import json

async def handle_ping(websocket):
    try:
        async for message in websocket:
            # Recibimos el mensaje y lo decodificamos
            data = json.loads(message)
            print(f"Mensaje recibido: {data}")

            if data.get("message") == "ping":
                # Responder con "pong" si el mensaje es "ping"
                response = {"message": "pong"}
                await websocket.send(json.dumps(response))
                print("Enviado: pong")
            else:
                response = {"message": "unknown"}
                await websocket.send(json.dumps(response))
                print("Enviado: unknown")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    # Inicia el servidor WebSocket en el puerto 8765
    server = await websockets.serve(handle_ping, "localhost", 8765)
    print("Servidor WebSocket iniciado en ws://localhost:8765")
    
    # Mantiene el servidor en ejecución
    await server.wait_closed()

# Ejecuta el servidor WebSocket
if __name__ == "__main__":
    asyncio.run(main())
