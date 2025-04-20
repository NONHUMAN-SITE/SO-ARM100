import socket
import time

# ⚠️ CAMBIA esto por la dirección que te da Pinggy cuando hagas ssh
PINGGY_HOST = 'tupinggysubdominio.pinggy.io'
PORT = 443  # Puerto por defecto si usaste -p 443

def run_client():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((PINGGY_HOST, PORT))
        print("[CLIENT] Conectado al servidor")

        while True:
            mensaje = "ping"
            client_socket.sendall(mensaje.encode())
            print(f"[CLIENT] Enviado: {mensaje}")

            data = client_socket.recv(1024)
            print(f"[CLIENT] Recibido: {data.decode()}")

            time.sleep(5)

if __name__ == "__main__":
    run_client()
