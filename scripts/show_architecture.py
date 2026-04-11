import base64
import requests
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def render_mermaid(mermaid_code, output_path="figures/architecture_auto.png"):
    """
    Transforme un code Mermaid en image via l'API Mermaid.ink 
    et l'affiche dans le notebook ou via matplotlib.
    """
    graphbytes = mermaid_code.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    
    url = "https://mermaid.ink/img/" + base64_string
    
    print(f"[*] Génération du diagramme via Mermaid.ink...")
    response = requests.get(url)
    
    if response.status_code == 200:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Image sauvée dans : {output_path}")
        
        # Affichage
        try:
            # Si on est dans un notebook
            display(Image(url=url))
        except:
            # Si on est en script standard
            img = mpimg.imread(output_path)
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.title("Architecture Globale HybridFace", fontsize=15, pad=20)
            plt.show()
    else:
        print(f"❌ Erreur lors de la génération : {response.status_code}")

# Le code Mermaid exact de ton projet
architecture_mermaid = """
flowchart TD
    classDef camera fill:#1e1e1e,stroke:#8b949e,color:#fff
    classDef detection fill:#1a4b28,stroke:#3fb950,color:#fff
    classDef hybrid fill:#0b4f6c,stroke:#00D7FF,color:#fff
    classDef aesthetic fill:#5a4600,stroke:#FFD700,color:#fff
    classDef output fill:#21262d,stroke:#c9d1d9,color:#fff

    Cam(📷 Flux Vidéo/Webcam):::camera --> Frame(Frame BGR):::camera
    
    Frame --> YOLO{YOLOv8n-Face\n+ Tracking ID}:::detection
    
    YOLO --> |Crop Visage| Engine[Moteur d'Analyse]
    
    subgraph "Pipe Démographique (Async)"
        Engine --> ViT[ViT ONNX\nÂge régression]:::hybrid
        Engine --> Caffe[Caffe CNN\nGenre classification]:::hybrid
        ViT & Caffe --> Fusion[Fusion & Smoother]:::hybrid
    end
    
    subgraph "Pipe Esthétique (Math)"
        Engine --> Mesh[MediaPipe\n468 Landmarks]:::aesthetic
        Mesh --> Phi[Nombre d'Or Φ]:::aesthetic
        Mesh --> Sym[Symétrie]:::aesthetic
        Mesh --> Reg[Regard/Teint]:::aesthetic
        Phi & Sym & Reg --> GS[Golden Score]:::aesthetic
    end
    
    Fusion --> Render(Overlay Visuel):::output
    GS --> Render
    Render --> Screen(🖥️ Écran):::output
"""

if __name__ == "__main__":
    render_mermaid(architecture_mermaid)
