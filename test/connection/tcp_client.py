# tcp_client.py
import zmq
import time

REMOTE_HOST = 'tcp://127.0.0.1:3000'

def run_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(REMOTE_HOST)
    print(f"[CLIENT] Conectado a {REMOTE_HOST}")

    while True:
        try:
            mensaje = "ping"
            socket.send_string(mensaje)
            print(f"[CLIENT] Enviado: {mensaje}")

            respuesta = socket.recv_string()
            print(f"[CLIENT] Recibido: {respuesta}")
            
            time.sleep(1)
        except Exception as e:
            print(f"[CLIENT] Error: {e}")
            break

if __name__ == "__main__":
    run_client()
