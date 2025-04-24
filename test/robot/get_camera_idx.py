import cv2
import time

def test_camera(camera_idx):
    try:
        # Intentar abrir la cámara
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            return False
        
        # Leer un frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False
        
        # Mostrar la información de la cámara
        print(f"\nCámara {camera_idx}:")
        print(f"- Resolución: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"- FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        # Mostrar la imagen
        window_name = f"Camera {camera_idx}"
        cv2.imshow(window_name, frame)
        print(f"Presiona cualquier tecla para continuar probando...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        
        # Liberar la cámara
        cap.release()
        return True
        
    except Exception as e:
        print(f"Error al probar cámara {camera_idx}: {str(e)}")
        return False

def main():
    print("Buscando cámaras disponibles...")
    cameras_found = []
    
    # Probar índices del 0 al 10
    for idx in range(1,10):
        print(f"\nProbando cámara con índice {idx}...")
        if test_camera(idx):
            cameras_found.append(idx)
            print(f"¡Cámara {idx} encontrada y funcional!")
        else:
            print(f"No se encontró cámara en el índice {idx}")
    
    print("\n=== Resumen ===")
    if cameras_found:
        print("Cámaras encontradas en los siguientes índices:", cameras_found)
        print("\nPuedes usar cualquiera de estos índices en tu configuración de SO100Robot")
        print("Ejemplo: robot_client = SO100Robot(enable_camera=True, cam_idx=<índice>)")
    else:
        print("No se encontraron cámaras disponibles")

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
