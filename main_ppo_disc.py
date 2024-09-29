import os

print("-------------PPO-DISC--------------------")
# Directory where results are saved
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

agent = "ppo_disc_01"

command_template = f"python rl.py --rla_names={agent} --epochs=210000 --do_train=1 --do_vis=1"
command = command_template
# Execute the command
os.system(command)

result_directory = "result_ppo_disc_01_new"
ensure_directory_exists(result_directory)

with open(os.path.join(result_directory, f"{agent}.txt"), "w") as f:
        f.write(os.popen(command).read())
