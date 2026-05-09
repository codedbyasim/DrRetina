import os
import shutil
import glob
from PIL import Image
import torch

# Import backend functions to run prediction
from backend import predict, get_model, device

def main():
    source_dir = r"H:\RetinoAgent\grade_3_images"
    output_dir = r"H:\RetinoAgent\top_grade_3_images"
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(source_dir, "*.png"))
    print(f"Found {len(image_paths)} images to test.")
    
    results = []
    
    # Pre-load model to avoid reloading
    print("Loading model...")
    model = get_model()
    model.eval()
    
    print("Testing images...")
    for i, path in enumerate(image_paths):
        try:
            pil_img = Image.open(path).convert("RGB")
            # Predict returns: grade, probs, pil224, cam_pil
            grade, probs, _, _ = predict(pil_img)
            
            # Confidence for Grade 3 (index 3)
            conf_grade_3 = float(probs[3])
            
            results.append({
                "path": path,
                "name": os.path.basename(path),
                "predicted_grade": grade,
                "conf_3": conf_grade_3
            })
            
            if (i + 1) % 10 == 0:
                print(f"Tested {i+1}/{len(image_paths)} images...")
                
        except Exception as e:
            print(f"Error testing {path}: {e}")
            
    # Sort by Grade 3 confidence (highest first)
    results.sort(key=lambda x: x["conf_3"], reverse=True)
    
    # Select top 10 highest confidence images
    top_n = 10
    top_results = results[:top_n]
    
    print("\n--- Top Images by Confidence ---")
    for res in top_results:
        print(f"{res['name']} - Predicted: {res['predicted_grade']} - Conf for G3: {res['conf_3']*100:.2f}%")
        # Move image to the new folder
        shutil.copy2(res["path"], os.path.join(output_dir, res["name"]))
        
    print(f"\nSuccessfully copied top {top_n} Grade 3 images to {output_dir}")

if __name__ == "__main__":
    main()
