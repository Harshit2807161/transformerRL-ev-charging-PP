import os

NEPO = [10,25,50]
NSTEPS = [256,1024,2048]
CLPRNGE = [0.2,0.4]

for i in range(len(NEPO)):
     for j in range(len(NSTEPS)):
          for k in range(len(CLPRNGE)):
                print("-------------PPO-DISC--------------------")
                # Directory where results are saved

                def ensure_directory_exists(directory):
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                agent = "ppo_disc_01"

                command_template = f"python rl.py --rla_names={agent} --epochs=210000 --do_train=1 --do_vis=1 --n_epochs={NEPO[i]} --n_steps={NSTEPS[j]} --clip_range={CLPRNGE[k]}"
                command = command_template
                # Execute the command
                os.system(command)

                result_directory = "result_ppo_disc_01"
                ensure_directory_exists(result_directory)

                with open(os.path.join(result_directory, f"{agent}.txt"), "w") as f:
                        f.write(os.popen(command).read())
