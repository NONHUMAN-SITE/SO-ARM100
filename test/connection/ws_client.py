import asyncio
import websockets
import json

async def ping_pong():
    uri = "wss://e394-45-231-80-122.ngrok-free.app"

    try:
        async with websockets.connect(uri) as websocket:
            print("Conectado al servidor WebSocket.")

            while True:
                # Enviar un mensaje "ping" al servidor
                ping_message = {"message": "ping"}
                await websocket.send(json.dumps(ping_message))
                print(f"Enviado: {ping_message}")

                # Esperar la respuesta del servidor
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    data = json.loads(response)
                    print(f"Recibido: {data}")
                except asyncio.TimeoutError:
                    print("No se recibió respuesta en 5 segundos.")

                # Esperar 5 segundos antes de volver a enviar
                await asyncio.sleep(5)

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Conexión cerrada inesperadamente: {e}")
    except Exception as e:
        print(f"Error en el cliente: {e}")

# Ejecutar el cliente
if __name__ == "__main__":
    asyncio.run(ping_pong())
