import os
import sys
import time
import subprocess
import glob
import re
from datetime import datetime

def get_latest_episode():
    """Find the latest episode number from model files or checkpoint files"""
    # Look for latest model
    if os.path.exists('models/BTCUSDT_actor_latest.pt'):
        # Try to find episode number from metrics files
        metrics_files = glob.glob('results/BTCUSDT_training_metrics_ep*.csv')
        if metrics_files:
            # Extract episode numbers from filenames
            episode_numbers = []
            for file in metrics_files:
                match = re.search(r'ep(\d+)\.csv$', file)
                if match:
                    episode_numbers.append(int(match.group(1)))
            if episode_numbers:
                return max(episode_numbers)
    
    # If no model file or couldn't determine episode, start from beginning
    return 0

def main():
    # Configuration parameters (could be moved to command line arguments)
    symbol = 'BTCUSDT'
    interval = '1h'
    start_date = '2023-05-29'
    end_date = '2025-05-12'
    episodes = 3000
    save_freq = 1  
    max_restarts = 20
    
    restart_count = 0
    current_episode = get_latest_episode()
    
    print("=" * 50)
    print(f"AUTO-RESUME TRAINING SCRIPT")
    print(f"Starting from episode {current_episode}/{episodes}")
    print("=" * 50)
    
    # Create a log file for the auto-resume process
    os.makedirs('logs', exist_ok=True)
    log_file = f"logs/auto_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    while current_episode < episodes and restart_count < max_restarts:
        # Log the restart
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now()}: Starting/Resuming training from episode {current_episode}\n")
        
        print(f"Starting training from episode {current_episode}")
        
        # Construct the command to run train_bot.py
        cmd = [
            sys.executable, "train_bot.py",
            "--symbol", symbol,
            "--interval", interval,
            "--start-date", start_date,
            "--end-date", end_date,
            "--episodes", str(episodes),
            "--save-freq", str(save_freq),
            "--resume-from", str(current_episode),
        ]
        
        try:
            # Run the training script
            process = subprocess.Popen(cmd)
            
            # Wait for it to complete
            process.wait()
            
            # Check if the process completed successfully
            if process.returncode == 0:
                print("Training completed successfully!")
                with open(log_file, 'a') as f:
                    f.write(f"{datetime.now()}: Training completed successfully!\n")
                break
            else:
                # The process crashed
                restart_count += 1
                print(f"Training crashed with return code {process.returncode}. Restart {restart_count}/{max_restarts}")
                with open(log_file, 'a') as f:
                    f.write(f"{datetime.now()}: Training crashed with return code {process.returncode}. Restart {restart_count}/{max_restarts}\n")
                
                # Get the latest episode number for resuming
                time.sleep(5)  # Wait a bit for files to be properly saved
                current_episode = get_latest_episode()
                print(f"Will resume from episode {current_episode}")
                
        except KeyboardInterrupt:
            # Handle manual interruption
            print("\nTraining manually interrupted by user. Exiting.")
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now()}: Training manually interrupted by user.\n")
            break
        except Exception as e:
            # Handle other exceptions
            restart_count += 1
            print(f"Exception occurred: {e}. Restart {restart_count}/{max_restarts}")
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now()}: Exception occurred: {e}. Restart {restart_count}/{max_restarts}\n")
            
            # Get the latest episode number for resuming
            time.sleep(5)  # Wait a bit for files to be properly saved
            current_episode = get_latest_episode()
            print(f"Will resume from episode {current_episode}")
    
    # Final report
    if current_episode >= episodes:
        print("Training completed successfully after all episodes!")
    elif restart_count >= max_restarts:
        print(f"Maximum number of restarts ({max_restarts}) reached. Stopping auto-resume.")
    
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()}: Auto-resume script terminated.\n")

if __name__ == "__main__":
    main() 