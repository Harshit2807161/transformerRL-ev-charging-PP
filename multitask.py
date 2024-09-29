import threading
import subprocess

def run_script(command):
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    command1 = "python rl.py --rla_names=ddpg_01 --epochs=210000 --do_train=1 --do_vis=1"
    command2 = "python rl.py --rla_names=dqn_01 --epochs=210000 --do_train=1 --do_vis=1"
    command3 = "python rl.py --rla_names=ppo_disc_01 --epochs=210000 --do_train=1 --do_vis=1"
    command4  = "python rl.py --rla_names=ppo_cont_01 --epochs=210000 --do_train=1 --do_vis=1"
    

    script1_thread = threading.Thread(target=run_script, args=(command1,))
    script2_thread = threading.Thread(target=run_script, args=(command2,))
    script3_thread = threading.Thread(target=run_script, args=(command3,)) 
    script4_thread = threading.Thread(target=run_script, args=(command4,))

    script1_thread.start()
    script2_thread.start()
    script3_thread.start()
    script4_thread.start()

    script1_thread.join()
    script2_thread.join()
    script3_thread.join()
    script4_thread.join()
    print("All scripts have finished executing.")
