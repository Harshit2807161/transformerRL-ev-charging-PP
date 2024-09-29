import os

NEPO = [10,25]
NSTEPS = [1024,2048]
CLPRNGE = [0.4,0.2]

for i in range(len(NEPO)):
    print("-------------PPO-DISC--------------------")
    # Directory where results are saved

    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    agent = "ppo_disc_01"

    command_template = f"python rl.py --rla_names={agent} --epochs=210000 --do_train=1 --do_vis=1 --n_epochs={NEPO[i]} --n_steps={NSTEPS[i]} --clip_range={CLPRNGE[i]}"
    command = command_template
    # Execute the command
    os.system(command)

    result_directory = f"result_ppo_disc_01_{NSTEPS[i]}_{NEPO[i]}_{CLPRNGE[i]}"
    ensure_directory_exists(result_directory)

    with open(os.path.join(result_directory, f"{agent}.txt"), "w") as f:
            f.write(os.popen(command).read())
