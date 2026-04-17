from train import training
from video_generator import generate_video

if __name__ == "__main__":
    # --- MENU DI SCELTA ---
    # Imposta "train" per addestrare, "video" per vedere la regata

    print("Insert a mode: \n" 
    "- '1' to start training\n" 
    "- '2' to generate video\n"
    "- '3' to train and then generate video")
    MODE = input()
    
    if MODE == "1":
        model, env, callback = training()
    elif MODE == "2":
        generate_video()
    elif MODE == "3":

        model, env, callback = training()
        generate_video()

