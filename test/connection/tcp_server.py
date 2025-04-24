# tcp_server.py
import zmq
import time

def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:3000")  # o "tcp://*:3000"

    print(f"[SERVER] Escuchando en localhost:3000")

    while True:
        try:
            mensaje = socket.recv_string()
            print(f"[SERVER] Recibido: {mensaje}")

            if mensaje == "ping":
                respuesta = "pong"
            else:
                respuesta = "desconocido"

            socket.send_string(respuesta)
            print(f"[SERVER] Enviado: {respuesta}")
            time.sleep(1)
        except Exception as e:
            print(f"[SERVER] Error: {e}")
            break

if __name__ == "__main__":
    run_server()
