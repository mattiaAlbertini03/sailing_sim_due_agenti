from train import training
from video_generator import generate_video

if __name__ == "__main__":
    print("Insert a mode: \n" 
          "- '1' to start TRAINING (Solo)\n" 
          "- '2' to start TRAINING (Self-Play)\n"
          "- '3' to generate video (Solo)\n"
          "- '4' to generate video (Self-Play)\n")
    
    MODE = input()
    
    if MODE == "1":
        model, env, callback = training(mode="solo")
    elif MODE == "2":
        model, env, callback = training(mode="self_play")
        model, env, callback = training(mode="self_play")
        model, env, callback = training(mode="self_play")
        generate_video(mode="self_play")
    elif MODE == "3":
        generate_video(mode="solo")
    elif MODE == "4":
        generate_video(mode="self_play")
