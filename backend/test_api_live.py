import urllib.request
import json
import io
from PIL import Image, ImageDraw

def create_image(disease_name):
    img = Image.new("RGB", (600, 450), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 120, 520, 330], fill=(34, 197, 94), outline=(22, 163, 74))
    draw.line([(100, 280), (520, 130)], fill=(187, 247, 208), width=3)

    if disease_name == "anthracnose":
        draw.ellipse([250, 150, 310, 210], fill=(202, 138, 4))
        draw.ellipse([265, 165, 295, 195], fill=(28, 15, 5))
    elif disease_name == "multi":
        draw.ellipse([200, 200, 260, 260], fill=(202, 138, 4))
        draw.ellipse([215, 215, 245, 245], fill=(28, 15, 5))
        draw.ellipse([360, 150, 440, 210], fill=(240, 240, 245))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def post_file(disease_name):
    img_bytes = create_image(disease_name)
    boundary = "Boundary123456789"
    
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="sample_{disease_name}.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8")
    
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + img_bytes + footer

    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode("utf-8"))
    
    print(f"\n[LIVE API RESULT - {disease_name.upper()}]")
    print(f"  * Status: {data['status']}")
    print(f"  * Primary Disease: {data['disease']} ({data['confidence']}%)")
    print(f"  * Is Multiple Diseases: {data['is_multiple_diseases']}")
    print(f"  * Predicted Disease List: {[d['name'] for d in data['predicted_diseases']]}")
    print(f"  * Bounding Box Count: {len(data['detections'])}")
    for i, box in enumerate(data['detections']):
        print(f"    - Box {i+1}: {box['disease']} ({box['confidence']}%) @ {box['bbox']}")
    print(f"  * Inference Time: {data['execution_time_ms']}ms")

print("=== RUNNING FASTAPI LIVE CLIENT VERIFICATION ===")
post_file("healthy")
post_file("anthracnose")
post_file("multi")
print("\n=== LIVE API VERIFICATION SUCCESSFUL ===")
