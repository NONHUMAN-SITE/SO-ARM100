import socket

HOST = '0.0.0.0'  # Escucha en todas las interfaces
PORT = 8000       # Puerto local que será expuesto con Pinggy

def run_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"[SERVER] Escuchando en {HOST}:{PORT}")

        conn, addr = server_socket.accept()
        with conn:
            print(f"[SERVER] Conexión establecida con {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    print("[SERVER] Conexión cerrada por el cliente")
                    break

                mensaje = data.decode()
                print(f"[SERVER] Recibido: {mensaje}")

                if mensaje == "ping":
                    respuesta = "pong"
                else:
                    respuesta = "desconocido"

                conn.sendall(respuesta.encode())
                print(f"[SERVER] Enviado: {respuesta}")

if __name__ == "__main__":
    run_server()
