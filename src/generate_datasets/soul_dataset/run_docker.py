import os
import shutil
import subprocess
from PIL import Image  # Suponiendo que quieres aplicar un comando usando PIL (Python Imaging Library)

def main(source_folder, images_dir):
    # Directorio donde se copiarán las imágenes
    prueba_folder = os.path.join(source_folder, 'prueba')
    
    segs_dir = os.path.join(source_folder, images_dir)
    # Iterar sobre cada imagen en el directorio fuente
    i = 0
    for seg in os.listdir(segs_dir):
        if seg == '.DS_Store':
            continue

        if seg.lower().endswith(('.png')):  # Filtrar imágenes por extensión
            print(' --------------------- ')
            print('image number --> ', i)
            i +=1
            src_path = os.path.join(segs_dir, seg)
            dest_path = os.path.join(prueba_folder, seg)
            
            # Copiar la imagen a la carpeta 'prueba'
            shutil.copy(src_path, dest_path)
            
            # Procesar la imagen
            seg_dir = prueba_folder
            res_dir = os.path.join(source_folder, 'graph_extracted_full')
            
            # Run the Docker command using subprocess
            docker_command = [
                'docker', 'run', '--rm',
                '-v', f'{seg_dir}:/var/segmentations',
                '-v', f'{res_dir}:/var/results',
                'mariaromeo/octa-graph-extraction',
                'graph_extraction_full', '--generate_graph_file'
            ]
            subprocess.run(docker_command, check=True)
            
            # Borrar la imagen de la carpeta 'prueba'
            os.remove(dest_path)
            print(f"Removed {dest_path}")

            # Anadir la imagen a 'processed'
            #processed.append(file_name)

if __name__ == "__main__":
    # Source directory. Inside this directory, there must be a folder called as images and another called 'prueba'
    images_dir = 'transformed_images'
    source_folder = r'/Users/maria/Documents/GitHub/TFM/OCTA_time_series/GRAPH_matching/data/soul_dataset/'  # Cambia esto por la ruta a tu carpeta de imágenes
    main(source_folder, images_dir)