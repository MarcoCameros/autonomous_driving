import os

def borrar_archivos_con_2_png():
    for archivo in os.listdir():
        if archivo.endswith(" 2.png"):
            try:
                os.remove(archivo)
                print(f"Archivo eliminado: {archivo}")
            except Exception as e:
                print(f"Error al eliminar {archivo}: {e}")

if __name__ == "__main__":
    borrar_archivos_con_2_png()
