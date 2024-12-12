from easydict import EasyDict as edict
from PIL import Image
from models.llava import LLaVA

if __name__ == "__main__":

    # model download command: git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
    model_path = "/mnt/hdd/weights/llava-v1.5-7b"

    prompt = """
    Answer the following question about the image with yes or no: Does the abdominopelvic CT scan indicate a chronic obstructive process in the gastrointestinal tract?
    """

    image_file = "/media/yesindeed/DATADRIVE1/mount/remote_cse/datasets/LLaVA-Med/data/images/34630837_F2.jpg"
    img = Image.open(image_file).convert("RGB")

    """
    GT report:
    Abdominopelvic CT scan in axial view indicates significant distension of the stomach and intestines with marked luminal dilatation observed in the oesophagus, stomach, small, and large bowel, accompanied by faecal loading. Notably, the distended large bowel is positioned anterior to the liver, causing medial displacement of the liver, which suggests a possible chronic underlying condition. This constellation of findings points to a long-standing obstructive process in the gastrointestinal tract, necessitating further clinical correlation and potential intervention.
    """

    llava_model = LLaVA(model_args=edict(model_path=model_path, model_base=None))
    llava_model.load_from_pretrained(model_path=model_path)

    output = llava_model.infer_vision_language(img, prompt)
    print(output)
